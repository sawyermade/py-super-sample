# py-super-sample

Command-line image upscaler for Apple Silicon Macs. It runs
[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (`RealESRGAN_x4plus`) on the Mac's GPU
through PyTorch's MPS backend and saves the result right next to the original image.

```bash
upscale ~/Pictures/photo.jpg
# -> ~/Pictures/photo_upscaled_3840x2160.png
```

## Contents

- [Requirements](#requirements)
- [Install uv](#install-uv)
- [Set up the project (`uv sync`)](#set-up-the-project-uv-sync)
- [Ways to run it](#ways-to-run-it)
- [Output and options](#output-and-options)
- [Upscaling more than 4x (multiple passes)](#upscaling-more-than-4x-multiple-passes)
- [Updating and uninstalling](#updating-and-uninstalling)
- [Notes](#notes)

## Requirements

- A Mac with Apple Silicon (M1 or newer). It falls back to CUDA or CPU elsewhere, but CPU is slow.
- [uv](https://docs.astral.sh/uv/), which manages Python and all dependencies for you. You don't
  need to install Python, create a virtualenv, or run `pip`.

## Install uv

With [Homebrew](https://brew.sh):

```bash
brew install uv
```

Or with the official installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then open a new terminal and check it worked:

```bash
uv --version
```

## Set up the project (`uv sync`)

From this folder:

```bash
uv sync
```

This creates `.venv/` with Python 3.13 (the version pinned in `.python-version`) and installs the
dependencies (PyTorch, spandrel, Pillow, NumPy) plus this project itself. It's optional for most
of the ways to run it below, because `uv run` does the same thing automatically the first time you
use it. Run `uv sync` again after pulling changes or editing `pyproject.toml`.

The model weights (`RealESRGAN_x4plus.pth`, 67 MB) are not part of the install. The first time you
upscale an image, they're downloaded from the official Real-ESRGAN GitHub release into
`~/.cache/py-super-sample/`. After that it works offline.

## Ways to run it

All of these do exactly the same thing. Pick whichever suits you. `photo.jpg` is any image, and you
can add the [options](#options) to any of them.

### 1. Installed command (works from any folder)

Install it once, from this folder:

```bash
uv tool install --editable .
```

This puts an `upscale` command in `~/.local/bin`, in its own isolated environment, so you can use
it from anywhere:

```bash
upscale ~/Pictures/photo.jpg
```

If your shell says `command not found: upscale`, `~/.local/bin` isn't on your PATH yet. Run
`uv tool update-shell` and open a new terminal.

### 2. `uv run` from the project folder

No install needed. From inside this folder:

```bash
uv run upscale.py photo.jpg
```

or, using the command name the project defines:

```bash
uv run upscale photo.jpg
```

These only work from inside the project folder, and a relative image path is resolved from there.
Use an absolute path (e.g. `~/Pictures/photo.jpg`) to upscale something elsewhere.

### 3. `uv run --project` from any folder

Tell `uv` where the project is, and run it from wherever your image is. Relative image paths are
resolved from your current folder:

```bash
uv run --project ~/Git/py-super-sample ~/Git/py-super-sample/upscale.py ~/Pictures/photo.jpg
```

### 4. A shell function for #3

To avoid typing that every time, add this to `~/.zshrc` (adjust the path if you move the project)
and open a new terminal:

```zsh
upscale() { uv run --project ~/Git/py-super-sample ~/Git/py-super-sample/upscale.py "$@"; }
```

Then:

```bash
upscale ~/Pictures/photo.jpg
```

This is an alternative to the installed command in #1. If you have both, the shell function wins.
Use a different function name if you want to keep both.

### 5. The project's virtualenv directly

After `uv sync`, you can skip `uv run` and call the environment's Python yourself:

```bash
~/Git/py-super-sample/.venv/bin/python ~/Git/py-super-sample/upscale.py photo.jpg
```

or activate the environment first:

```bash
source ~/Git/py-super-sample/.venv/bin/activate
python ~/Git/py-super-sample/upscale.py photo.jpg
```

## Output and options

The upscaled image is saved in the same folder as the original, with
`_upscaled_<width>x<height>` added to the name. **The output is always a PNG**, whatever the input
format (JPEG, WebP, TIFF, BMP, ...):

| Input | Output |
|---|---|
| `photo.jpg` (960x540) | `photo_upscaled_3840x2160.png` |
| `logo.png` (200x100) | `logo_upscaled_800x400.png` |

The image is decoded once, upscaled in memory, and written once as a lossless PNG, so nothing is
re-compressed along the way, and the output of one run can be fed into another without any loss
(see [multiple passes](#upscaling-more-than-4x-multiple-passes)).

The script prints the output path when it finishes. Existing files with the same name are
overwritten. Because the extension is no longer part of the name, `in.jpg` and `in.webp` of the
same size map to the same output, `in_upscaled_<width>x<height>.png`, so the second run overwrites
the first.

### Options

| Option | Default | Description |
|---|---|---|
| `-s`, `--scale` | 4 per pass | Final size as a multiple of the original: 4 for one pass, 16 for two. Other values resample the result of the last pass, e.g. `-s 2` for a 2x upscale, or `-p 2 -s 8` for two passes resized down to 8x. |
| `-p`, `--passes` | `1` | Number of 4x model passes, run back to back in memory. Only the final image is written. See [multiple passes](#upscaling-more-than-4x-multiple-passes). |
| `-t`, `--tile` | `512` | Tile size in input pixels. The image is processed in tiles to limit memory use. Lower it if you run out of memory, or use `0` to process the whole image at once. |
| `-h`, `--help` | | Show the usage message. |

```bash
upscale photo.jpg --scale 2
upscale huge-scan.png --tile 256
upscale photo.jpg --passes 2 --scale 8
uv run upscale.py photo.jpg -s 2 -t 0
```

## Upscaling more than 4x (multiple passes)

One pass is a single 4x run of the model. `--passes` runs it again on its own output, all in
memory. Only the final image is written, named with its final size, and nothing is compressed or
saved between passes:

```bash
upscale photo.jpg --passes 2               # 16x
upscale photo.jpg --passes 2 --scale 8     # two passes, then resized to 8x
```

Each pass prints its sizes as it goes, for example:

```
photo.jpg: 758x2076 on mps
pass 1/2: 758x2076 (1.6 MP) -> 3032x8304 (25.2 MP)
pass 2/2: 3032x8304 (25.2 MP) -> 12128x33216 (402.8 MP)
```

Without `--scale`, the final size is 4x per pass: 16x for two passes, 64x for three. That gets huge
fast. A 758x2076 image at `--passes 2` is 12,128x33,216, about 403 MP. Use `--scale` to land on the
size you actually want.

Both passes always run at full size, and the result is then resized down to `--scale` (Lanczos). So
`--passes 2 --scale 8` costs about as much as `--passes 2`, plus the resize. It's sharper than
`--scale 8` with one pass, which only stretches a single 4x pass with plain resampling.

Measured on an M3 Ultra with 96 GB: a 758x2076 image with `--passes 2 --scale 8` took about 111
seconds and peaked at about 25 GB of memory (as reported by macOS), because the 403 MP intermediate
is held in memory. Lowering `--tile` shrinks the per-tile working set, but not the full-size images.
On a Mac with much less memory, large inputs with two passes may not fit.

Each extra pass is much slower than the last (it works on a much bigger image), and the returns
diminish: Real-ESRGAN starts amplifying its own invented texture, so the result can look plasticky.
Check the result at 100% before you print it.

## Updating and uninstalling

- **Editing the code:** the installed command is editable, so changes to `upscale.py` take effect
  immediately. If you move the project folder, run `uv tool install --editable .` again from its
  new location. If you change `pyproject.toml` (for example, to add a dependency), add
  `--reinstall`, and run `uv sync` to update the project's own `.venv/`.
- **Install a plain copy instead:** `uv tool install .` doesn't depend on this folder staying put.
  After code changes, run `uv tool install --reinstall .`.
- **Uninstall the command:** `uv tool uninstall py-super-sample`.
- **Remove the project environment:** delete `.venv/`. `uv sync` recreates it.
- **Remove the model:** the downloaded weights stay in `~/.cache/py-super-sample/` until you delete
  that folder, which frees the 67 MB.

## Notes

- **Speed:** on an M3 Ultra, a 2000x1500 image becomes 8000x6000 in about 11 seconds.
- **Formats:** reads anything Pillow can read (JPEG, PNG, WebP, TIFF, BMP, ...). HEIC isn't
  supported out of the box. Output is always PNG, and transparency is always kept. Camera rotation
  (EXIF orientation) is applied and the ICC color profile is carried over. Other EXIF metadata is
  not copied to the output. If the input is a JPEG, its compression artifacts are already in the
  source and the upscaler will enlarge them. Saving as PNG only avoids adding more.
- **Looks, not fidelity:** Real-ESRGAN is a GAN. It invents plausible texture, so results look sharp
  and natural, but the added detail isn't real. Don't rely on it where fidelity matters
  (documents, forensics, medical images).
- **Output size:** a 4x upscale has 16 times as many pixels as the input, and PNG is lossless, so
  large images produce large files. A smooth 48 MP test image came out at 41 MB, but a detailed
  photo compresses much worse, roughly 1.5 to 2 bytes per pixel (about 75 to 100 MB for 48 MP).
  Saving the PNG also adds a few seconds to big outputs.
- **Troubleshooting:** if you see memory errors, lower `--tile`. Operations MPS doesn't support
  fall back to the CPU automatically (`PYTORCH_ENABLE_MPS_FALLBACK=1` is set by the script).
- **Licensing:** the code and model come from the Real-ESRGAN project. See its repository for the
  license terms.
