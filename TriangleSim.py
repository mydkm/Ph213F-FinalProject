import os
import sys
import argparse
import math

import numpy as np
import matplotlib

if os.environ.get("DISPLAY", "") == "" and os.name != "nt":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Polygon
from dataclasses import dataclass

# HI FROM BERTRAND

# -------------------------------------------------
# Visualization settings
# -------------------------------------------------
COLOR_MODE = "height"         # "height" (|u|) or "signed" (u)
COLORMAP   = "turbo"

VIDEO_W, VIDEO_H = 1920, 1080
VIDEO_DPI = 160

CARPET_FACE_RGBA    = (0.12, 0.08, 0.20, 0.95)
CARPET_BORDER_COLOR = (1.0, 1.0, 1.0, 1.0)
CARPET_BORDER_LW    = 3.0

# CPU-only dtype
DTYPE = np.float32


# -------------------------------------------------
# Sierpinski-style triangle generation
# -------------------------------------------------
def sr_triangle_generation(
    N: int, n: int, base_len: float,
    Lx: float, Ly: float, dx: float, dy: float
):
    """
    Construct a solid pattern of downward equilateral triangles.

    Returns
    -------
    open_mask : np.bool_ [N, N]
        True  -> fluid
        False -> obstacle

    tris_phys : list
        Physical vertices of all triangles.

    tris_side : list[float]
        Side length for each triangle.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if base_len <= 0.0:
        raise ValueError("base_len must be positive")

    open_mask = np.ones((N, N), dtype=np.bool_)
    tris_phys = []
    tris_side = []

    # Coordinate arrays for rasterization
    xs = np.arange(N, dtype=DTYPE) * dx
    ys = np.arange(N, dtype=DTYPE) * dy

    def rasterize_triangle(v1, v2, v3):
        x1, y1 = v1
        x2, y2 = v2
        x3, y3 = v3

        min_x = min(x1, x2, x3)
        max_x = max(x1, x2, x3)
        min_y = min(y1, y2, y3)
        max_y = max(y1, y2, y3)

        i0 = max(0, int(math.floor(min_x / dx)))
        i1 = min(N, int(math.ceil(max_x / dx)))
        j0 = max(0, int(math.floor(min_y / dy)))
        j1 = min(N, int(math.ceil(max_y / dy)))

        if i1 <= i0 or j1 <= j0:
            return

        Xi = xs[i0:i1][:, None]
        Yj = ys[j0:j1][None, :]

        Ax, Ay = x1, y1
        Bx, By = x2, y2
        Cx, Cy = x3, y3

        # Orientation-agnostic point-in-triangle test
        s1 = (Xi - Bx) * (Ay - By) - (Ax - Bx) * (Yj - By)
        s2 = (Xi - Cx) * (By - Cy) - (Bx - Cx) * (Yj - Cy)
        s3 = (Xi - Ax) * (Cy - Ay) - (Cx - Ax) * (Yj - Ay)

        cond1 = (s1 >= 0) & (s2 >= 0) & (s3 >= 0)
        cond2 = (s1 <= 0) & (s2 <= 0) & (s3 <= 0)
        inside = cond1 | cond2

        sub = open_mask[i0:i1, j0:j1]
        sub[inside] = False

    def add_triangle_from_center(cx, cy, side):
        """
        Downward triangle:
          - apex at the bottom
          - flat edge at the top
          - centroid at (cx, cy)
        """
        h = side * math.sqrt(3.0) / 2.0

        v_bottom = (cx,              cy - 2.0 * h / 3.0)
        v_left   = (cx - side / 2.0, cy +       h / 3.0)
        v_right  = (cx + side / 2.0, cy +       h / 3.0)

        rasterize_triangle(v_bottom, v_left, v_right)
        tris_phys.append((v_bottom, v_left, v_right))
        tris_side.append(side)

    # Level 1 triangle
    cx0, cy0 = 0.5 * Lx, 0.5 * Ly
    s0 = base_len

    all_tris = [(cx0, cy0, s0)]
    outer_layer = [(cx0, cy0, s0)]

    for _lev in range(2, n + 1):
        new_layer = []

        for cx, cy, s in outer_layer:
            child_side = s / 2.0
            h = s * math.sqrt(3.0) / 2.0

            c_top   = (cx,           cy + 2.0 * h / 3.0)
            c_left  = (cx - s / 2.0, cy -       h / 3.0)
            c_right = (cx + s / 2.0, cy -       h / 3.0)

            new_layer.append((c_top[0],   c_top[1],   child_side))
            new_layer.append((c_left[0],  c_left[1],  child_side))
            new_layer.append((c_right[0], c_right[1], child_side))

        all_tris.extend(new_layer)
        outer_layer = new_layer

    for cx, cy, s in all_tris:
        add_triangle_from_center(cx, cy, s)

    return open_mask, tris_phys, tris_side


# -------------------------------------------------
# Sponge damping
# -------------------------------------------------
def sponge_damping(Nx, Ny, thickness=24, b_max=2.0):
    i = np.arange(Nx, dtype=DTYPE)
    j = np.arange(Ny, dtype=DTYPE)

    di = np.minimum(i, (Nx - 1) - i)[:, None]
    dj = np.minimum(j, (Ny - 1) - j)[None, :]

    def ramp(d):
        T = max(thickness, 1)
        r = (T - d) / T
        return np.clip(r, 0.0, 1.0) ** 2

    return (b_max * np.maximum(ramp(di), ramp(dj))).astype(DTYPE)


# -------------------------------------------------
# Wave config
# -------------------------------------------------
@dataclass
class WaveConfig:
    N: int = 540
    n: int = 1

    base_len: float = 0.3

    Lx: float = 16 / 9
    Ly: float = 1.0

    c: float = 1.0
    CFL: float = 0.45
    T: float = 6.0

    sponge_thickness: int = 28
    sponge_strength: float = 2.0

    save_mp4: bool = True
    mp4_fname: str = "wave_sierpinski_triangle.mp4"
    fps: int = 60
    steps_per_frame: int = 4

    pulse_x: float = 0.22
    pulse_y: float = 0.50
    pulse_sigma: float = 0.03
    pulse_amp: float = 1.0


# -------------------------------------------------
# Progress bar
# -------------------------------------------------
def make_progress_callback():
    def _cb(curr, total):
        frac = (curr + 1) / total
        bar_len = 30
        filled = int(bar_len * frac)
        bar = "#" * filled + "-" * (bar_len - filled)
        percent = int(frac * 100)

        sys.stdout.write(
            f"\rRendering frames: |{bar}| {percent:3d}% ({curr + 1}/{total})"
        )
        sys.stdout.flush()

        if curr + 1 >= total:
            sys.stdout.write("\n")

    return _cb


# -------------------------------------------------
# Main simulation
# -------------------------------------------------
def run_sim(cfg: WaveConfig):
    N, Lx, Ly, c = cfg.N, cfg.Lx, cfg.Ly, cfg.c

    x = np.linspace(0, Lx, N, dtype=DTYPE)
    y = np.linspace(0, Ly, N, dtype=DTYPE)

    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])

    X, Y = np.meshgrid(x, y, indexing="ij")

    dt_stable = 1.0 / (c * ((1.0 / dx**2 + 1.0 / dy**2) ** 0.5))
    dt = cfg.CFL * dt_stable
    Nt = int(np.ceil(cfg.T / dt))

    # Obstacle geometry
    open_mask, tris_phys, tris_side = sr_triangle_generation(
        N, cfg.n, cfg.base_len, Lx, Ly, dx, dy
    )

    obstacle = ~open_mask
    maskF = open_mask.astype(DTYPE)

    b = sponge_damping(
        N,
        N,
        thickness=cfg.sponge_thickness,
        b_max=cfg.sponge_strength,
    )

    # Coefficients
    Cx2 = (c * dt / dx) ** 2
    Cy2 = (c * dt / dy) ** 2

    # Fields
    u_nm1 = np.zeros((N, N), dtype=DTYPE)
    u_n   = np.zeros((N, N), dtype=DTYPE)

    # Initial Gaussian pulse
    r2 = (X - cfg.pulse_x) ** 2 + (Y - cfg.pulse_y) ** 2
    u0 = (
        cfg.pulse_amp
        * np.exp(-0.5 * r2 / (cfg.pulse_sigma ** 2))
    ).astype(DTYPE)

    u0 *= maskF
    u_nm1[...] = u0

    # Startup step: compute u^1
    u0m = u0 * maskF

    u_xx0 = (
        u0m[2:, 1:-1]
        - 2.0 * u0m[1:-1, 1:-1]
        + u0m[:-2, 1:-1]
    )

    u_yy0 = (
        u0m[1:-1, 2:]
        - 2.0 * u0m[1:-1, 1:-1]
        + u0m[1:-1, :-2]
    )

    u_n[1:-1, 1:-1] = (
        u0[1:-1, 1:-1]
        + 0.5 * (Cx2 * u_xx0 + Cy2 * u_yy0)
    )

    u_n[0, :] = 0.0
    u_n[-1, :] = 0.0
    u_n[:, 0] = 0.0
    u_n[:, -1] = 0.0

    u_n *= maskF

    # Frameless canvas
    fig = plt.figure(
        figsize=(VIDEO_W / VIDEO_DPI, VIDEO_H / VIDEO_DPI),
        frameon=False,
    )

    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)

    # Initial frame
    if COLOR_MODE == "height":
        frame0 = np.abs(u_n).T
        vmin0 = 0.0
        vmax0 = float(np.max(frame0)) or 1e-9
    else:
        frame0 = u_n.T
        A0 = float(np.max(np.abs(frame0))) or 1e-9
        vmin0 = -A0
        vmax0 = A0

    im = ax.imshow(
        frame0,
        origin="lower",
        extent=[0, Lx, 0, Ly],
        interpolation="bilinear",
        cmap=COLORMAP,
        vmin=vmin0,
        vmax=vmax0,
        aspect="equal",
    )

    # Obstacle face overlay
    overlay = np.zeros((N, N, 4), dtype=float)
    r, g, b_, a_ = CARPET_FACE_RGBA

    overlay[..., 3] = 0.0
    overlay[obstacle, 0] = r
    overlay[obstacle, 1] = g
    overlay[obstacle, 2] = b_
    overlay[obstacle, 3] = a_

    ax.imshow(
        overlay.transpose(1, 0, 2),
        origin="lower",
        extent=[0, Lx, 0, Ly],
        interpolation="nearest",
        zorder=10,
        aspect="equal",
    )

    # Triangle borders
    for tri, side in zip(tris_phys, tris_side):
        v1, v2, v3 = tri

        lw = CARPET_BORDER_LW * (side / cfg.base_len)
        lw = max(lw, 0.5)

        ax.add_patch(
            Polygon(
                [v1, v2, v3],
                closed=True,
                fill=False,
                linewidth=lw,
                edgecolor=CARPET_BORDER_COLOR,
                zorder=20,
            )
        )

    # Time stepper
    def step(u_nm1, u_n):
        u_m = u_n * maskF

        u_xx = (
            u_m[2:, 1:-1]
            - 2.0 * u_m[1:-1, 1:-1]
            + u_m[:-2, 1:-1]
        )

        u_yy = (
            u_m[1:-1, 2:]
            - 2.0 * u_m[1:-1, 1:-1]
            + u_m[1:-1, :-2]
        )

        denom = 1.0 / (1.0 + 0.5 * b[1:-1, 1:-1] * dt)

        core = (
            (0.5 * b[1:-1, 1:-1] * dt - 1.0)
            * u_nm1[1:-1, 1:-1]
            + 2.0 * u_n[1:-1, 1:-1]
            + Cx2 * u_xx
            + Cy2 * u_yy
        )

        u_np1 = np.empty_like(u_n)

        u_np1[1:-1, 1:-1] = denom * core

        u_np1[0, :] = 0.0
        u_np1[-1, :] = 0.0
        u_np1[:, 0] = 0.0
        u_np1[:, -1] = 0.0

        u_np1 *= maskF

        return u_np1

    frames = int(np.ceil(Nt / cfg.steps_per_frame))

    def update(_k):
        nonlocal u_nm1, u_n

        for _ in range(cfg.steps_per_frame):
            u_np1 = step(u_nm1, u_n)
            u_nm1, u_n = u_n, u_np1

        if COLOR_MODE == "height":
            u_abs = np.abs(u_n)
            A = float(np.percentile(u_abs, 99.0))
            A = max(A, 1e-9)

            im.set_data(u_abs.T)
            im.set_clim(0.0, A)

        else:
            A = float(np.percentile(np.abs(u_n), 99.0))
            A = max(A, 1e-9)

            im.set_data(u_n.T)
            im.set_clim(-A, A)

        return (im,)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=1000 / cfg.fps,
        blit=False,
        cache_frame_data=False,
    )

    if cfg.save_mp4:
        progress_cb = make_progress_callback()

        # CPU-only software encoder.
        # Removed GPU/hardware encoders:
        # h264_nvenc, hevc_nvenc, h264_qsv, hevc_qsv,
        # h264_vaapi, hevc_vaapi, h264_amf, hevc_amf
        writer = FFMpegWriter(
            fps=cfg.fps,
            codec="libx264",
            bitrate=4500,
            extra_args=["-pix_fmt", "yuv420p"],
        )

        ani.save(
            cfg.mp4_fname,
            dpi=VIDEO_DPI,
            writer=writer,
            progress_callback=progress_cb,
        )

        print("Saved:", cfg.mp4_fname, "| Compute: CPU | Encoder: libx264")

    else:
        print("Simulation finished using CPU only. No video written.")


# -------------------------------------------------
# Input arguments
# -------------------------------------------------
def parse_args():
    defaults = WaveConfig()

    p = argparse.ArgumentParser(
        description="CPU-only 2D wave simulation with Sierpinski-style triangle obstacle."
    )

    p.add_argument("--N", type=int, default=defaults.N,
                   help=f"Grid size (NxN), default {defaults.N}")

    p.add_argument("--n", type=int, default=defaults.n,
                   help=f"Triangle depth level, default {defaults.n}")

    p.add_argument("--base-len", type=float, default=defaults.base_len,
                   help=f"Side length of level-1 triangle, default {defaults.base_len}")

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
                   help=f"Sponge thickness in cells, default {defaults.sponge_thickness}")

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

    p.add_argument("--save-mp4", dest="save_mp4", action="store_true",
                   default=defaults.save_mp4,
                   help="Save MP4 video. Default: enabled.")

    p.add_argument("--no-save-mp4", dest="save_mp4", action="store_false",
                   help="Disable MP4 saving.")

    return p.parse_args()


# -------------------------------------------------
# Entry point
# -------------------------------------------------
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