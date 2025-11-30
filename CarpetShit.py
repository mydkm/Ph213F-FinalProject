import os
import sys
import argparse

import numpy as _np
try:
    import cupy as _cp
    xp = _cp
    ON_GPU = True
except Exception:
    xp = _np
    ON_GPU = False

import numpy as np
import matplotlib
if os.environ.get("DISPLAY", "") == "" and os.name != "nt":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Rectangle
from dataclasses import dataclass

# visualization settings
COLOR_MODE = "height"         # "height" (|u|) or "signed" (u)
COLORMAP   = "turbo"
VIDEO_W, VIDEO_H = 1280, 720
VIDEO_DPI = 160
CARPET_FACE_RGBA    = (0.12, 0.08, 0.20, 0.95)
CARPET_BORDER_COLOR = (1.0, 1.0, 1.0, 1.0)
CARPET_BORDER_LW    = 3.0

DTYPE = xp.float32
def to_cpu(a): return _cp.asnumpy(a) if ON_GPU else a

def sr_carpet_generation(
    N: int, n: int, base_len: float,
    Lx: float, Ly: float, dx: float, dy: float
):
    """
    Construct a growing solid "carpet":

      n = 1:
        - One centered solid square of side = base_len (in physical units).

      n >= 2:
        - Keep all existing squares.
        - For EACH square from level (n-1), add 8 new squares "around" it:
            * child side = parent_side / 3
            * child centers are at offsets (±parent_side, 0),
              (0, ±parent_side), (±parent_side, ±parent_side)
              relative to the parent center.
          This ensures the new squares do NOT touch the parent; there is a gap.

    Returns:
      open_mask  (xp.bool_ [N,N]): True = fluid, False = obstacle (carpet)
      rects_phys (list of (x0,y0,w,h)) for drawing borders for all squares
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if base_len <= 0.0:
        raise ValueError("base_len must be positive")

    open_mask  = xp.ones((N, N), dtype=xp.bool_)
    rects_phys = []

    # helper to rasterize one square
    def paint_square(cx, cy, side):
        sx = max(1, int(round(side / dx)))
        sy = max(1, int(round(side / dy)))
        i0 = int(round((cx - 0.5 * side) / dx)); i1 = i0 + sx
        j0 = int(round((cy - 0.5 * side) / dy)); j1 = j0 + sy
        i0 = max(0, min(N, i0)); i1 = max(0, min(N, i1))
        j0 = max(0, min(N, j0)); j1 = max(0, min(N, j1))
        if i1 > i0 and j1 > j0:
            open_mask[i0:i1, j0:j1] = False
            rects_phys.append((cx - 0.5 * side, cy - 0.5 * side, side, side))

    # Level-1: centered base square (side = base_len)
    cx0, cy0 = 0.5 * Lx, 0.5 * Ly
    level = [(cx0, cy0, base_len)]
    for px, py, ps in level:
        paint_square(px, py, ps)

    # Deeper levels: each parent spawns 8 neighbors around it, not touching
    for lev in range(2, n + 1):
        next_level = []
        for (px, py, ps) in level:
            child_side = ps / 3.0         # size shrinks by factor 3 each level
            offset     = ps               # center-to-center offset
            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    if ox == 0 and oy == 0:
                        continue
                    cx_child = px + ox * offset
                    cy_child = py + oy * offset
                    next_level.append((cx_child, cy_child, child_side))
        # Paint all new children and then treat them as the "outermost" layer
        for px, py, ps in next_level:
            paint_square(px, py, ps)
        level = next_level

    return open_mask, rects_phys

def sponge_damping(Nx, Ny, thickness=24, b_max=2.0):
    i = xp.arange(Nx, dtype=DTYPE)
    j = xp.arange(Ny, dtype=DTYPE)
    di = xp.minimum(i, (Nx - 1) - i)[:, None]
    dj = xp.minimum(j, (Ny - 1) - j)[None, :]
    def ramp(d):
        T = max(thickness, 1)
        r = (T - d) / T
        return xp.clip(r, 0.0, 1.0) ** 2
    return (b_max * xp.maximum(ramp(di), ramp(dj))).astype(DTYPE)


# waveconfig class
@dataclass
class WaveConfig:
    N: int = 540
    n: int = 1
    base_len: float = 0.3       # side length of the level-1 square
    Lx: float = 16/9
    Ly: float = 1.0
    c: float = 1.0
    CFL: float = 0.45
    T: float = 6.0
    sponge_thickness: int = 28
    sponge_strength: float = 2.0
    save_mp4: bool = True
    mp4_fname: str = "wave_centered_carpet_gpu_rect.mp4"
    fps: int = 60
    steps_per_frame: int = 4
    # Single pulse controls (physical units)
    pulse_x: float = 0.22
    pulse_y: float = 0.50
    pulse_sigma: float = 0.03
    pulse_amp: float = 1.0


# progress bar
def make_progress_callback():
    def _cb(curr, total):
        # curr is 0-based
        frac = (curr + 1) / total
        bar_len = 30
        filled = int(bar_len * frac)
        bar = "#" * filled + "-" * (bar_len - filled)
        percent = int(frac * 100)
        sys.stdout.write(
            f"\rRendering frames: |{bar}| {percent:3d}% ({curr+1}/{total})"
        )
        sys.stdout.flush()
        if curr + 1 >= total:
            sys.stdout.write("\n")
    return _cb
# -------------------------------------------------


def run_sim(cfg: WaveConfig):
    N, Lx, Ly, c = cfg.N, cfg.Lx, cfg.Ly, cfg.c

    x = xp.linspace(0, Lx, N, dtype=DTYPE)
    y = xp.linspace(0, Ly, N, dtype=DTYPE)
    dx = float(to_cpu(x[1] - x[0])); dy = float(to_cpu(y[1] - y[0]))
    X, Y = xp.meshgrid(x, y, indexing="ij")

    dt_stable = 1.0 / (c * ((1.0/dx**2 + 1.0/dy**2) ** 0.5))
    dt = cfg.CFL * dt_stable
    Nt = int(np.ceil(cfg.T / dt))

    # use the new fixed-size growing carpet
    open_mask, rects_phys = sr_carpet_generation(
        N, cfg.n, cfg.base_len, Lx, Ly, dx, dy
    )
    obstacle = ~open_mask
    maskF = open_mask.astype(DTYPE)
    b = sponge_damping(N, N, thickness=cfg.sponge_thickness, b_max=cfg.sponge_strength)

    # coefficients
    Cx2 = (c * dt / dx) ** 2
    Cy2 = (c * dt / dy) ** 2

    # fields
    u_nm1 = xp.zeros((N, N), dtype=DTYPE)
    u_n   = xp.zeros((N, N), dtype=DTYPE)

    # single circular pulse at t=0
    r2 = (X - cfg.pulse_x)**2 + (Y - cfg.pulse_y)**2
    u0 = (cfg.pulse_amp * xp.exp(-0.5 * r2 / (cfg.pulse_sigma**2))).astype(DTYPE)
    u0 *= maskF
    u_nm1[...] = u0

    # startup: u^1
    u0m = u0 * maskF
    u_xx0 = (u0m[2:, 1:-1] - 2.0 * u0m[1:-1, 1:-1] + u0m[:-2, 1:-1])
    u_yy0 = (u0m[1:-1, 2:] - 2.0 * u0m[1:-1, 1:-1] + u0m[1:-1, :-2])
    u_n[1:-1, 1:-1] = u0[1:-1, 1:-1] + 0.5 * (Cx2 * u_xx0 + Cy2 * u_yy0)
    u_n[0, :]=u_n[-1, :]=0.0; u_n[:, 0]=u_n[:, -1]=0.0
    u_n *= maskF

    # frameless canvas
    fig = plt.figure(figsize=(VIDEO_W / VIDEO_DPI, VIDEO_H / VIDEO_DPI), frameon=False)
    ax = plt.Axes(fig, [0, 0, 1, 1]); ax.set_axis_off(); fig.add_axes(ax)

    # initial frame (we'll immediately overwrite clim in update)
    if COLOR_MODE == "height":
        frame0 = to_cpu(xp.abs(u_n)).T
        vmin0, vmax0 = 0.0, float(np.max(frame0)) or 1e-9
    else:
        frame0 = to_cpu(u_n).T
        A0 = float(np.max(np.abs(frame0))) or 1e-9
        vmin0, vmax0 = -A0, A0

    im = ax.imshow(frame0, origin="lower", extent=[0, Lx, 0, Ly],
                   interpolation="bilinear", cmap=COLORMAP,
                   vmin=vmin0, vmax=vmax0, aspect="equal")

    # colored faces for obstacles
    obstacle_cpu = to_cpu(obstacle)
    overlay = np.zeros((N, N, 4), dtype=float)
    r,g,b_,a_ = CARPET_FACE_RGBA
    overlay[..., 3] = 0.0
    overlay[obstacle_cpu, 0] = r; overlay[obstacle_cpu, 1] = g
    overlay[obstacle_cpu, 2] = b_; overlay[obstacle_cpu, 3] = a_
    ax.imshow(overlay.transpose(1, 0, 2), origin="lower",
              extent=[0, Lx, 0, Ly], interpolation="nearest",
              zorder=10, aspect="equal")

    # white borders for every square
    for (x0p, y0p, wp, hp) in rects_phys:
        ax.add_patch(Rectangle((x0p, y0p), wp, hp, fill=False,
                               linewidth=CARPET_BORDER_LW,
                               edgecolor=CARPET_BORDER_COLOR, zorder=20))

    # time stepper (single pulse only)
    def step(u_nm1, u_n):
        u_m = u_n * maskF
        u_xx = (u_m[2:, 1:-1] - 2.0 * u_m[1:-1, 1:-1] + u_m[:-2, 1:-1])
        u_yy = (u_m[1:-1, 2:] - 2.0 * u_m[1:-1, 1:-1] + u_m[1:-1, :-2])
        denom = 1.0 / (1.0 + 0.5 * b[1:-1, 1:-1] * dt)
        core = ((0.5 * b[1:-1, 1:-1] * dt - 1.0) * u_nm1[1:-1, 1:-1] +
                2.0 * u_n[1:-1, 1:-1] + Cx2 * u_xx + Cy2 * u_yy)
        u_np1 = xp.empty_like(u_n)
        u_np1[1:-1, 1:-1] = denom * core
        z = xp.array(0.0, dtype=DTYPE)
        u_np1[0, :]=u_np1[-1, :]=z; u_np1[:, 0]=u_np1[:, -1]=z
        u_np1 *= maskF
        return u_np1

    frames = int(np.ceil(Nt / cfg.steps_per_frame))

    # color scaling
    def update(_k):
        nonlocal u_nm1, u_n
        for _ in range(cfg.steps_per_frame):
            u_np1 = step(u_nm1, u_n)
            u_nm1, u_n = u_n, u_np1

        if COLOR_MODE == "height":
            u_abs = to_cpu(np.abs(u_n))
            # Use robust upper bound so small spikes don't dominate
            A = float(np.percentile(u_abs, 99.0))  # 99th percentile of |u|
            A = max(A, 1e-9)
            im.set_data(u_abs.T)
            im.set_clim(0.0, A)
        else:
            u_cpu = to_cpu(u_n)
            A = float(np.percentile(np.abs(u_cpu), 99.0))
            A = max(A, 1e-9)
            im.set_data(u_cpu.T)
            im.set_clim(-A, A)
        return (im,)

    ani = animation.FuncAnimation(fig, update, frames=frames,
                                  interval=1000/cfg.fps, blit=False,
                                  cache_frame_data=False)

    if cfg.save_mp4:
        progress_cb = make_progress_callback()
        used = None
        for codec in ["h264_nvenc","hevc_nvenc","h264_qsv","hevc_qsv",
                      "h264_vaapi","hevc_vaapi","h264_amf","hevc_amf"]:
            try:
                writer = FFMpegWriter(fps=cfg.fps, codec=codec, bitrate=4500,
                                      extra_args=["-pix_fmt","yuv420p"])
                ani.save(cfg.mp4_fname, dpi=VIDEO_DPI, writer=writer,
                         progress_callback=progress_cb)
                used = codec
                break
            except Exception:
                continue
        if used is None:
            writer = FFMpegWriter(fps=cfg.fps, codec="libx264", bitrate=4500,
                                  extra_args=["-pix_fmt","yuv420p"])
            ani.save(cfg.mp4_fname, dpi=VIDEO_DPI, writer=writer,
                     progress_callback=progress_cb)
            used = "libx264"

        print("Saved:", cfg.mp4_fname, "| GPU compute:", ON_GPU, "| Encoder:", used)
    else:
        # don't know why anyone would need it, but wrote logic for it anyway
        print("Simulation finished (save_mp4=False, no video written).")


# input args
def parse_args():
    defaults = WaveConfig()  # dataclass instance with default values
    p = argparse.ArgumentParser(
        description="2D wave simulation with growing carpet obstacle."
    )

    p.add_argument("--N", type=int, default=defaults.N,
                   help=f"Grid size (NxN), default {defaults.N}")
    p.add_argument("--n", type=int, default=defaults.n,
                   help=f"Carpet depth level, default {defaults.n}")
    p.add_argument("--base-len", type=float, default=defaults.base_len,
                   help=f"Side length of level-1 square, default {defaults.base_len}")
    p.add_argument("--Lx", type=float, default=defaults.Lx,
                   help=f"Domain length in x, default {defaults.Lx}")
    p.add_argument("--Ly", type=float, default=defaults.Ly,
                   help=f"Domain length in y, default {defaults.Ly}")
    p.add_argument("--c", type=float, default=defaults.c,
                   help=f"Wave speed, default {defaults.c}")
    p.add_argument("--CFL", type=float, default=defaults.CFL,
                   help=f"CFL factor, default {defaults.CFL}")
    p.add_argument("--T", type=float, default=defaults.T,
                   help=f"Total simulation time, default {defaults.T}")
    p.add_argument("--sponge-thickness", type=int, default=defaults.sponge_thickness,
                   help=f"Sponge thickness (cells), default {defaults.sponge_thickness}")
    p.add_argument("--sponge-strength", type=float, default=defaults.sponge_strength,
                   help=f"Sponge damping strength, default {defaults.sponge_strength}")
    p.add_argument("--mp4-fname", type=str, default=defaults.mp4_fname,
                   help=f"Output MP4 filename, default '{defaults.mp4_fname}'")
    p.add_argument("--fps", type=int, default=defaults.fps,
                   help=f"Video frames per second, default {defaults.fps}")
    p.add_argument("--steps-per-frame", type=int, default=defaults.steps_per_frame,
                   help=f"Simulation steps per video frame, default {defaults.steps_per_frame}")
    p.add_argument("--pulse-x", type=float, default=defaults.pulse_x,
                   help=f"Pulse center x, default {defaults.pulse_x}")
    p.add_argument("--pulse-y", type=float, default=defaults.pulse_y,
                   help=f"Pulse center y, default {defaults.pulse_y}")
    p.add_argument("--pulse-sigma", type=float, default=defaults.pulse_sigma,
                   help=f"Pulse Gaussian width, default {defaults.pulse_sigma}")
    p.add_argument("--pulse-amp", type=float, default=defaults.pulse_amp,
                   help=f"Pulse amplitude, default {defaults.pulse_amp}")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = WaveConfig(
        N=args.N,
        n=args.n,
        base_len=args.base_len,
        Lx=args.Lx,
        Ly=args.Ly,
        c=args.c,
        CFL=args.CFL,
        T=args.T,
        sponge_thickness=args.sponge_thickness,
        sponge_strength=args.sponge_strength,
        save_mp4=bool(args.save_mp4),
        mp4_fname=args.mp4_fname,
        fps=args.fps,
        steps_per_frame=args.steps_per_frame,
        pulse_x=args.pulse_x,
        pulse_y=args.pulse_y,
        pulse_sigma=args.pulse_sigma,
        pulse_amp=args.pulse_amp,
    )
    run_sim(cfg)
