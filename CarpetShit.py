# wave_centered_carpet_gpu.py
# Requires: numpy, matplotlib, ffmpeg; optional: cupy-cudaXX (for NVIDIA GPUs)
import os

# ------- Array backend (CuPy -> GPU, else NumPy -> CPU) -------
import numpy as _np
try:
    import cupy as _cp
    xp = _cp
    ON_GPU = True
except Exception:
    xp = _np
    ON_GPU = False

import numpy as np  # host-only utilities (imshow, scalars)
import matplotlib
# Use GUI if available, otherwise render headlessly
if os.environ.get("DISPLAY", "") == "" and os.name != "nt":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
from dataclasses import dataclass

DTYPE = xp.float32

def to_cpu(a):
    """Return a NumPy array on host (no-op if already CPU)."""
    if ON_GPU:
        return _cp.asnumpy(a)
    return a

# ------------------------------
# Centered Sierpinski carpet (depth n) as an obstacle sub-square
# ------------------------------
def centered_carpet_open_mask(N: int, n: int, frac: float = 0.5):
    """
    Return xp.bool_ mask of shape (N,N): True=open, False=obstacle.
    A single n-depth Sierpinski carpet is centered; its side ~ frac*N,
    rounded down to a multiple of 3**n so the pattern is exact.
    """
    if not (0 < frac <= 1.0):
        raise ValueError("frac must be in (0,1].")
    base = 3**max(0, n)
    size = max(base, int(frac * N) // base * base)
    size = min(size, (N // base) * base)

    open_mask = xp.ones((N, N), dtype=xp.bool_)
    i0 = (N - size) // 2
    j0 = (N - size) // 2
    i1 = i0 + size
    j1 = j0 + size

    # Build solid carpet inside sub-square
    ii, jj = xp.meshgrid(xp.arange(size, dtype=xp.int32),
                         xp.arange(size, dtype=xp.int32), indexing="ij")
    solid = xp.ones((size, size), dtype=xp.bool_)
    for k in range(n):
        step = 3**k
        hole = ((ii // step) % 3 == 1) & ((jj // step) % 3 == 1)
        solid = solid & (~hole)
    # open where NOT solid (holes are open)
    open_inside = ~solid
    open_mask[i0:i1, j0:j1] = open_inside
    return open_mask

# ------------------------------
# Vectorized sponge damping near outer boundaries
# ------------------------------
def sponge_damping(Nx, Ny, thickness=24, b_max=2.0):
    # distance to nearest vertical/horizontal border
    i = xp.arange(Nx, dtype=DTYPE)
    j = xp.arange(Ny, dtype=DTYPE)
    di = xp.minimum(i, (Nx - 1) - i)[:, None]  # (Nx,1)
    dj = xp.minimum(j, (Ny - 1) - j)[None, :]  # (1,Ny)
    # ramp 0..1 inside sponge, quadratic profile
    def ramp(d):
        T = max(thickness, 1)
        r = (T - d) / T
        return xp.clip(r, 0.0, 1.0) ** 2
    b = b_max * xp.maximum(ramp(di), ramp(dj))
    return b.astype(DTYPE)

# ------------------------------
# Time-harmonic Gaussian source
# ------------------------------
@dataclass
class Beam:
    x0: float; y0: float; sigx: float; sigy: float; omega: float; amp: float
    def field(self, X, Y, t):
        gx = xp.exp(-0.5 * ((X - self.x0) / self.sigx) ** 2, dtype=DTYPE)
        gy = xp.exp(-0.5 * ((Y - self.y0) / self.sigy) ** 2, dtype=DTYPE)
        return (self.amp * xp.sin(self.omega * t)).astype(DTYPE) * gx * gy

# ------------------------------
# Config and solver
# ------------------------------
@dataclass
class WaveConfig:
    N: int = 486
    n: int = 1                # Sierpinski depth
    carpet_frac: float = 0.48 # fraction of domain width for the carpet square
    Lx: float = 1.0; Ly: float = 1.0
    c: float = 1.0
    CFL: float = 0.45
    T: float = 6.0
    sponge_thickness: int = 28
    sponge_strength: float = 2.0
    save_mp4: bool = True
    mp4_fname: str = "wave_centered_carpet_gpu.mp4"
    fps: int = 30
    steps_per_frame: int = 4

def run_sim(cfg: WaveConfig):
    N, Lx, Ly, c = cfg.N, cfg.Lx, cfg.Ly, cfg.c

    # Grid on device
    x = xp.linspace(0, Lx, N, dtype=DTYPE)
    y = xp.linspace(0, Ly, N, dtype=DTYPE)
    dx = float(to_cpu(x[1] - x[0])); dy = float(to_cpu(y[1] - y[0]))
    X, Y = xp.meshgrid(x, y, indexing="ij")

    # Stable dt (2D CFL): dt <= 1/(c*sqrt(1/dx^2 + 1/dy^2))
    dt_stable = 1.0 / (c * ( (1.0/dx**2 + 1.0/dy**2) ** 0.5 ))
    dt = cfg.CFL * dt_stable
    Nt = int(np.ceil(cfg.T / dt))

    # Masks and damping on device
    open_mask = centered_carpet_open_mask(N, cfg.n, cfg.carpet_frac)
    obstacle = ~open_mask
    maskF = open_mask.astype(DTYPE)  # multiply to zero-out obstacles
    b = sponge_damping(N, N, thickness=cfg.sponge_thickness, b_max=cfg.sponge_strength)

    # Coefficients
    Cx2 = (c * dt / dx) ** 2
    Cy2 = (c * dt / dy) ** 2
    dt2 = dt * dt

    # Fields
    u_nm1 = xp.zeros((N, N), dtype=DTYPE)
    u_n   = xp.zeros((N, N), dtype=DTYPE)

    # Source: left side, aimed at carpet
    beam = Beam(
        x0=0.18 * Lx, y0=0.5 * Ly,
        sigx=0.02 * Lx, sigy=0.09 * Ly,
        omega=2.0 * np.pi * 2.0, amp=45.0
    )

    # Startup (u^0=0 -> u^1) with forcing
    u_n[:] = 0.5 * dt2 * beam.field(X, Y, 0.0)
    u_n *= maskF  # clamp obstacles to zero

    # --- Visualization setup (host) ---
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    im = ax.imshow(to_cpu(u_n).T, origin="lower", extent=[0, Lx, 0, Ly],
                   interpolation="bilinear")
    ax.set_title(f"Wave vs Centered Sierpinski Carpet (n={cfg.n}, GPU={ON_GPU})")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    cb = fig.colorbar(im, ax=ax, shrink=0.8); cb.set_label("u(x,y,t)")

    # Draw obstacle overlay (static)
    obstacle_cpu = to_cpu(obstacle)
    overlay = np.zeros((N, N, 4), dtype=float)
    overlay[..., 3] = 0.0
    overlay[obstacle_cpu, :3] = 0.0
    overlay[obstacle_cpu, 3] = 0.35
    ax.imshow(overlay.transpose(1, 0, 2), origin="lower",
              extent=[0, Lx, 0, Ly], interpolation="nearest", zorder=10)

    # --- Time stepper on device ---
    def step(u_nm1, u_n, t):
        f = beam.field(X, Y, t)

        # Impose Dirichlet inside obstacles before Laplacian via multiply
        u_m = u_n * maskF

        # 5-point Laplacian
        u_xx = (u_m[2:, 1:-1] - 2.0 * u_m[1:-1, 1:-1] + u_m[:-2, 1:-1])
        u_yy = (u_m[1:-1, 2:] - 2.0 * u_m[1:-1, 1:-1] + u_m[1:-1, :-2])

        denom = 1.0 / (1.0 + 0.5 * b[1:-1, 1:-1] * dt)
        core = ((0.5 * b[1:-1, 1:-1] * dt - 1.0) * u_nm1[1:-1, 1:-1] +
                2.0 * u_n[1:-1, 1:-1] +
                Cx2 * u_xx + Cy2 * u_yy +
                dt2 * f[1:-1, 1:-1])

        u_np1 = xp.empty_like(u_n)
        u_np1[1:-1, 1:-1] = denom * core

        # Clamp outer borders and obstacles
        z = xp.array(0.0, dtype=DTYPE)
        u_np1[0, :] = u_np1[-1, :] = z
        u_np1[:, 0] = u_np1[:, -1] = z
        u_np1 *= maskF
        return u_np1

    frames = int(np.ceil(Nt / cfg.steps_per_frame))
    t = dt

    def update(_k):
        nonlocal u_nm1, u_n, t
        for _ in range(cfg.steps_per_frame):
            u_np1 = step(u_nm1, u_n, t)
            u_nm1, u_n = u_n, u_np1
            t += dt
        # copy one frame to host and autoscale colors
        A = float(to_cpu(xp.max(xp.abs(u_n))))
        A = max(A, 1e-9)
        im.set_data(to_cpu(u_n).T)
        im.set_clim(-A, A)
        ax.set_title(f"Wave vs Centered Sierpinski Carpet (n={cfg.n}, GPU={ON_GPU}) | t={t:.2f}")
        return (im,)

    ani = animation.FuncAnimation(
        fig, update, frames=frames, interval=1000/cfg.fps,
        blit=False, cache_frame_data=False
    )

    headless = matplotlib.get_backend().lower() == "agg"
    if cfg.save_mp4 or headless:
        # Try GPU encoders first; fall back to libx264
        gpu_codecs = [
            "h264_nvenc", "hevc_nvenc",      # NVIDIA
            "h264_qsv", "hevc_qsv",          # Intel QuickSync
            "h264_vaapi", "hevc_vaapi",      # Linux VAAPI
            "h264_amf", "hevc_amf"           # AMD AMF (Windows)
        ]
        used = None
        for codec in gpu_codecs:
            try:
                writer = FFMpegWriter(fps=cfg.fps, codec=codec, bitrate=4500,
                                      extra_args=["-pix_fmt", "yuv420p"])
                ani.save(cfg.mp4_fname, dpi=160, writer=writer)
                used = codec
                break
            except Exception:
                continue
        if used is None:
            writer = FFMpegWriter(fps=cfg.fps, codec="libx264", bitrate=4500,
                                  extra_args=["-pix_fmt", "yuv420p"])
            ani.save(cfg.mp4_fname, dpi=160, writer=writer)
            used = "libx264"
        print("Saved:", cfg.mp4_fname, "| GPU compute:", ON_GPU, "| Encoder:", used)
    else:
        plt.show()

if __name__ == "__main__":
    cfg = WaveConfig(
        N=486,     # for n up to ~4, multiples of 3**n look crisp
        n=1,       # <— change to 2,3,... for deeper carpets
        carpet_frac=0.48,
        T=6.0, steps_per_frame=4,
        save_mp4=True, mp4_fname="wave_centered_carpet_gpu.mp4"
    )
    run_sim(cfg)
