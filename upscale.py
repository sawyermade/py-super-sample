"""Upscale an image with Real-ESRGAN (x4plus) on Apple Silicon.

    upscale path/to/photo.jpg
    upscale path/to/photo.jpg -p 2 -s 8   # two model passes, final size 8x

Writes path/to/photo_upscaled_<W>x<H>.png next to the original and prints
that path. The output is always a lossless PNG, whatever the input format.
With several passes, everything stays in memory: only the final image is
written, named with its final size.
The model weights (~64 MB) are downloaded to ~/.cache/py-super-sample on
first run.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

# Let ops that MPS doesn't implement fall back to the CPU instead of crashing.
# Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402
from spandrel import ImageModelDescriptor, ModelLoader  # noqa: E402

MODEL_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
)
# Kept outside the project so the installed command works from anywhere.
CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "py-super-sample"
MODEL_PATH = CACHE_DIR / "RealESRGAN_x4plus.pth"

# Extra context pixels around each tile, cropped off after upscaling. Measured
# against whole-image output: 32 leaves faint seams, 64 is visually seamless.
TILE_PAD = 64


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def ensure_weights() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    log(f"Downloading {MODEL_PATH.name} from {MODEL_URL}")
    partial = MODEL_PATH.with_suffix(".part")

    def progress(blocks: int, block_size: int, total: int) -> None:
        if total > 0 and sys.stderr.isatty():
            pct = min(100, blocks * block_size * 100 // total)
            print(f"\r  {pct:3d}%", end="", file=sys.stderr, flush=True)

    urllib.request.urlretrieve(MODEL_URL, partial, reporthook=progress)
    partial.replace(MODEL_PATH)  # only appears under the real name once complete
    return MODEL_PATH


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(device: torch.device) -> ImageModelDescriptor:
    model = ModelLoader().load_from_file(ensure_weights())
    if not isinstance(model, ImageModelDescriptor):
        raise SystemExit(f"{MODEL_PATH.name} is not an image-to-image model")
    return model.to(device).eval()


@torch.inference_mode()
def upscale_rgb(
    model: ImageModelDescriptor, rgb: np.ndarray, device: torch.device, tile: int
) -> np.ndarray:
    """Upscale an HxWx3 uint8 array by model.scale, tile by tile to bound memory."""
    h, w, _ = rgb.shape
    s = model.scale
    # uint8 view; each tile is converted to float on its own (keeps memory low on big inputs).
    src = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    out = np.empty((h * s, w * s, 3), dtype=np.uint8)

    step = tile if tile > 0 else max(h, w)
    ys = range(0, h, step)
    xs = range(0, w, step)
    total, done = len(ys) * len(xs), 0

    for y0 in ys:
        for x0 in xs:
            y1, x1 = min(y0 + step, h), min(x0 + step, w)
            # Padded input window, clamped to the image.
            py0, py1 = max(y0 - TILE_PAD, 0), min(y1 + TILE_PAD, h)
            px0, px1 = max(x0 - TILE_PAD, 0), min(x1 + TILE_PAD, w)

            chunk = src[:, :, py0:py1, px0:px1].float().div_(255).to(device)
            up = model(chunk)

            # Crop the padding back off (in output pixels).
            cy, cx = (y0 - py0) * s, (x0 - px0) * s
            up = up[:, :, cy : cy + (y1 - y0) * s, cx : cx + (x1 - x0) * s]
            up = (up.clamp_(0, 1) * 255).round().byte()[0].permute(1, 2, 0)
            out[y0 * s : y1 * s, x0 * s : x1 * s] = up.cpu().numpy()

            done += 1
            if sys.stderr.isatty():
                print(f"\r  tile {done}/{total}", end="", file=sys.stderr, flush=True)
    if sys.stderr.isatty():
        print(file=sys.stderr)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Upscale an image with Real-ESRGAN.")
    parser.add_argument("image", type=Path, help="path to the image to upscale")
    parser.add_argument(
        "-p",
        "--passes",
        type=int,
        default=1,
        help="number of 4x model passes, run back to back in memory (default 1)",
    )
    parser.add_argument(
        "-s",
        "--scale",
        type=float,
        default=None,
        help="final size as a multiple of the original (default: 4 per pass, i.e. "
        "4 for one pass, 16 for two; other values resample the result of the last pass)",
    )
    parser.add_argument(
        "-t",
        "--tile",
        type=int,
        default=512,
        help="tile size in input pixels (default 512; 0 = whole image at once)",
    )
    args = parser.parse_args()

    src = args.image.expanduser()
    if not src.is_file():
        parser.error(f"not a file: {src}")
    if args.passes < 1:
        parser.error("--passes must be >= 1")
    if args.scale is not None and args.scale <= 0:
        parser.error("--scale must be > 0")
    if args.tile < 0:
        parser.error("--tile must be >= 0")

    with Image.open(src) as opened:
        icc = opened.info.get("icc_profile")
        img = ImageOps.exif_transpose(opened)  # honour camera rotation
        has_alpha = img.mode in ("RGBA", "LA", "PA") or (
            img.mode == "P" and "transparency" in img.info
        )
        alpha = img.convert("RGBA").getchannel("A") if has_alpha else None
        rgb = np.array(img.convert("RGB"))  # writable copy (torch.from_numpy needs it)

    device = pick_device()
    h0, w0 = rgb.shape[:2]
    log(f"{src.name}: {w0}x{h0} on {device}")

    start = time.perf_counter()
    model = load_model(device)

    # Every pass runs back to back in memory; nothing intermediate touches the disk.
    current = rgb
    for n in range(1, args.passes + 1):
        h, w = current.shape[:2]
        if args.passes > 1:
            s = model.scale
            log(
                f"pass {n}/{args.passes}: {w}x{h} ({w * h / 1e6:.1f} MP) -> "
                f"{w * s}x{h * s} ({w * h * s * s / 1e6:.1f} MP)"
            )
        current = upscale_rgb(model, current, device, args.tile)
    result = Image.fromarray(current)

    scale = args.scale if args.scale is not None else float(model.scale**args.passes)
    out_w, out_h = round(w0 * scale), round(h0 * scale)
    if result.size != (out_w, out_h):
        result = result.resize((out_w, out_h), Image.Resampling.LANCZOS)

    if alpha is not None:
        result.putalpha(alpha.resize(result.size, Image.Resampling.LANCZOS))

    # Always PNG: lossless, and it keeps alpha whatever the input format was.
    dest = src.with_name(f"{src.stem}_upscaled_{out_w}x{out_h}.png")
    result.save(dest, **({"icc_profile": icc} if icc else {}))

    log(f"Done in {time.perf_counter() - start:.1f}s -> {out_w}x{out_h}")
    print(dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
