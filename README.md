# Ph213F-FinalProject

````markdown
# 2D Wave Scattering on a Growing Carpet Fractal

This repository contains a 2D finite-difference solver for the wave equation with:

- A **single point-source pulse** that generates a circular wave.
- A **growing, solid “carpet” obstacle** made of squares (a Sierpiński-style construction).
- **Optional GPU acceleration** via CuPy (falls back to NumPy on CPU).
- A **video renderer** (MP4 via FFmpeg) with a **terminal progress bar**.
- A **CLI interface** exposing all simulation parameters.

The main script numerically integrates the damped 2D wave equation, visualizes the resulting wave field, and writes an MP4 of the evolution.


## 0. Description of the Final

The “final” deliverable is a short MP4 video that shows:

1. A circular wave emitted from a localized Gaussian pulse (point source).
2. Propagation and interaction of the wave with a **square carpet fractal**:
   - At depth `n = 1` there is a single centered square of side length `base_len`.
   - At depth `n ≥ 2`, each square from the previous level spawns **8 smaller squares** around it (cardinal + diagonal directions), with a gap so they do not touch the parent square and side length reduced by a factor of 3 each level.
3. Outgoing waves are absorbed by a **sponge damping layer** near the domain boundaries to reduce reflections.
4. A color-mapped representation of the displacement magnitude `|u(x, y, t)|` (or signed field, depending on `COLOR_MODE`).

The simulation supports **time-adaptive color scaling** (based on the 99th percentile of |u|) so that high-amplitude features remain visible throughout the video, even as the wave disperses and interacts with the obstacle.


## 1. Project Setup

### 1.1. Requirements

- **Python** ≥ 3.9
- **NumPy**
- **Matplotlib**
- **CuPy** (optional, for GPU acceleration on NVIDIA hardware)
- **FFmpeg** (for MP4 encoding; must be available on your `PATH`)

### 1.2. Install FFmpeg

On Ubuntu / Debian:

```bash
sudo apt-get update
sudo apt-get install ffmpeg
````

On macOS (Homebrew):

```bash
brew install ffmpeg
```

On Windows, download FFmpeg from the official builds and add the `bin/` folder to your `PATH`.

### 1.3. Create and Activate a Virtual Environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows PowerShell / CMD
```

### 1.4. Install Python Dependencies

Minimal CPU-only setup:

```bash
pip install numpy matplotlib
```

Optional GPU acceleration (example for CUDA 12.x, adjust to your CUDA version):

```bash
pip install cupy-cuda12x
```

If CuPy fails to import, the code will automatically fall back to NumPy on CPU.

> **Note:** There is no hard dependency on CuPy. If it is not installed or fails to import, `ON_GPU` becomes `False` and the simulation runs entirely on the CPU.

## 2. Running the Code & Available Input Arguments

Assume the main script is named `wave_sim.py`. If you saved it under a different filename, just replace `wave_sim.py` below with your actual filename.

### 2.1. Basic Usage

```bash
python wave_sim.py
```

This runs the simulation with the default `WaveConfig` parameters and writes an MP4 file (by default `wave_centered_carpet_gpu_rect.mp4`). While encoding, you will see a **progress bar** in the terminal, e.g.:

```text
Rendering frames: |##########--------------------|  33% (120/360)
```

### 2.2. Example: Change Carpet Depth and Output Name

```bash
python wave_sim.py --n 2 --base-len 0.25 --mp4-fname carpet_depth2.mp4
```

### 2.3. Example: Disable Video Saving (simulation only)

```bash
python wave_sim.py --save-mp4 0
```

### 2.4. All Command-Line Arguments

All fields of the `WaveConfig` dataclass are exposed as CLI arguments. If an argument is omitted, the **default shown here** is used.

| Argument             | Type               | Default                               | Description                                                                            |
| -------------------- | ------------------ | ------------------------------------- | -------------------------------------------------------------------------------------- |
| `--N`                | `int`              | `540`                                 | Grid size (domain is `N × N` points). Larger = higher resolution and slower.           |
| `--n`                | `int`              | `1`                                   | Carpet depth level. `1` = single centered square; higher `n` = more fractal squares.   |
| `--base-len`         | `float`            | `0.3`                                 | Side length (in physical units) of the level-1 centered square.                        |
| `--Lx`               | `float`            | `16/9` ≈ `1.7778`                     | Physical length of the domain in x (horizontal) direction.                             |
| `--Ly`               | `float`            | `1.0`                                 | Physical length of the domain in y (vertical) direction.                               |
| `--c`                | `float`            | `1.0`                                 | Wave speed used in the PDE.                                                            |
| `--CFL`              | `float`            | `0.45`                                | CFL safety factor (time step = `CFL * dt_stable`). Must be < √0.5 for stability in 2D. |
| `--T`                | `float`            | `6.0`                                 | Total simulation time (in the same units as `c` and the domain).                       |
| `--sponge-thickness` | `int`              | `28`                                  | Thickness (in grid cells) of the boundary damping layer.                               |
| `--sponge-strength`  | `float`            | `2.0`                                 | Maximum damping coefficient in the sponge layer.                                       |
| `--save-mp4`         | `int` (`0` or `1`) | `1`                                   | Whether to save an MP4 file (`1`) or not (`0`).                                        |
| `--mp4-fname`        | `str`              | `"wave_centered_carpet_gpu_rect.mp4"` | Output MP4 filename.                                                                   |
| `--fps`              | `int`              | `60`                                  | Frames per second of the output video.                                                 |
| `--steps-per-frame`  | `int`              | `4`                                   | Number of simulation time steps per rendered video frame.                              |
| `--pulse-x`          | `float`            | `0.22`                                | x-coordinate of the pulse center (point source) in physical units.                     |
| `--pulse-y`          | `float`            | `0.50`                                | y-coordinate of the pulse center.                                                      |
| `--pulse-sigma`      | `float`            | `0.03`                                | Gaussian width of the initial pulse (controls spatial extent of the source).           |
| `--pulse-amp`        | `float`            | `1.0`                                 | Amplitude of the initial displacement pulse.                                           |

#### Notes

* The **progress bar** only appears when `--save-mp4 1` (the default), because it uses the frame encoding callback from Matplotlib’s `FFMpegWriter`.
* If FFmpeg is missing or a hardware codec is unavailable, the script automatically falls back to a CPU codec (`libx264`), as long as FFmpeg itself is installed.
* `COLOR_MODE` and `COLORMAP` are configured at the top of the script. If you’d like to experiment with signed fields (`u` instead of `|u|`) or different colormaps, edit those constants.

## 3. References

*(To be added.)*

```
::contentReference[oaicite:0]{index=0}
```
