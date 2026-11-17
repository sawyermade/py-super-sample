# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-file CLI that upscales images with Real-ESRGAN (`RealESRGAN_x4plus`) on Apple Silicon, using PyTorch's MPS backend. All logic lives in `upscale.py`. It is packaged as the `upscale` console script (`upscale = "upscale:main"`, built with hatchling, `only-include = ["upscale.py"]`). If you add another module, add it to `only-include` or the installed command won't ship it.

## Commands

```bash
uv sync                                    # create .venv (Python 3.13 via .python-version) and install deps
uv run upscale.py photo.jpg [-p N] [-s X] [-t N]   # run from the repo root (also: uv run upscale photo.jpg)
uv tool install --editable .               # global `upscale` command; code edits apply immediately
uv tool install --editable --reinstall .   # needed after changing pyproject.toml (deps, entry point, build config)
```

There are no tests, linter, or formatter configured. Verify changes by running the CLI on a real image. The output is written next to the input, so use copies outside the repo. For tiling changes, compare `upscale_rgb(model, rgb, device, 0)` (whole image) against `tile=512` on a lossless PNG. The max difference should stay at about 4 levels out of 255 or less.

## Architecture

Pipeline in `main()`: Pillow decode, `exif_transpose`, split alpha, then N passes of `upscale_rgb()`, an optional Lanczos resize, and a single PNG save.

- **Model loading:** weights are loaded through `spandrel` (`ModelLoader`), not the `realesrgan`/`basicsr` pip packages, which break on current torchvision. The weights are not in the repo. They are downloaded on first run to `~/.cache/py-super-sample/RealESRGAN_x4plus.pth` (honours `XDG_CACHE_HOME`), written as `.part` and renamed, so an interrupted download never leaves a half-file under the real name.
- **Tiling (`upscale_rgb`):** the image is processed in `--tile`-sized windows plus `TILE_PAD` (64) pixels of context on each side. The padding is cropped off after upscaling, and tiles are written into a preallocated uint8 output array. `TILE_PAD` was chosen by measurement against whole-image output: 32 left visible seams (max diff 42), 64 is effectively seamless (max diff 4). The uint8 to float conversion is done per tile, not for the whole image, to keep memory down on big inputs.
- **Passes and scale:** `--passes N` chains N 4x model passes entirely in memory. `--scale` defaults to `4**passes`. If it is set to anything else, the result of the last pass is resized with Lanczos, up or down. `-p 2 -s 8` therefore still computes the full 16x intermediate before shrinking it. For a 758x2076 input that is 403 MP, about 111 s and about 25 GB peak on an M3 Ultra.
- **Alpha, color and metadata:** only RGB goes through the model. Alpha is Lanczos-resized separately and re-attached. The ICC profile is kept, camera rotation is applied, and other EXIF is dropped.
- **Import order matters:** `PYTORCH_ENABLE_MPS_FALLBACK=1` is set before `import torch` on purpose (hence the `# noqa: E402`).

## Behaviors to preserve

- **Output is always one lossless PNG**, named `<stem>_upscaled_<W>x<H>.png` (final size) next to the input, whatever the input format. Never write intermediate files or use lossy formats anywhere in the pipeline. This is an explicit requirement. Same stem and size with a different extension collides and overwrites. This is documented, not handled.
- **stdout carries only the output path; everything else goes to stderr.** Progress counters print only when stderr is a TTY.

## Docs and repo conventions

- `README.md` deliberately documents every way to run the tool: installed command, `uv run upscale.py`, `uv run upscale`, `uv run --project ...` from any folder, a zsh function, and the venv's Python. The user asked for all of them to be kept. Update the README whenever CLI options or behavior change. The timing and memory figures in it come from real runs on an M3 Ultra (96 GB).
- `images/` is for local test images and is gitignored for all image file types (`images/.gitignore`). Its `README.md` is tracked.
- **Shell quirks on this machine:** `noclobber` is on, so `>` onto an existing file fails (use `>|`). zsh aborts a command when a glob matches nothing (e.g. `rm -f foo_*`).
