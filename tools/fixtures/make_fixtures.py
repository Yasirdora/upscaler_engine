#!/usr/bin/env python3
"""Generate the synthetic fixtures the conversion pipeline and the device harness read.

Two kinds of fixture, both deterministic and both generated rather than committed:

  * **Test tiles** -- one PNG per conversion config in `tools/convert/models/`, written to the
    config's `test_tile` path at exactly the config's input shape. `convert.py` runs the PyTorch
    oracle and Core ML on this tile to measure Level A fidelity, so its content has to exercise
    what a super-resolution model is sensitive to: smooth gradients (banding), fine periodic detail
    (aliasing, ringing), hard edges and thin strokes (halos, text deformation) and noise-like
    texture (false detail). It does not need to be a photograph.
  * **Workloads** -- `Sources/TestFixtures/workloads/{A_1080p,B_12MP,C_48MP}.jpg`, the three
    certification spec §7.1 sizes. These drive memory, timing and survival runs, where content is
    irrelevant beyond not being trivially compressible, so they are montages of 512x512 synthetic
    cells, each with its own seed.

Every pixel is a function of a seed and a size: the same command on any machine with the pinned
numpy and Pillow produces the same files. Nothing here is a photograph, so nothing here carries
anyone's rights or likeness.

    python tools/fixtures/make_fixtures.py               # tiles and workloads, skip existing
    python tools/fixtures/make_fixtures.py --tiles       # tiles only
    python tools/fixtures/make_fixtures.py --force       # overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "tools" / "convert" / "models"
WORKLOAD_DIR = REPO_ROOT / "Sources" / "TestFixtures" / "workloads"

#: Certification spec §7.1 workloads: file name -> (width, height).
WORKLOADS = {
    "A_1080p.jpg": (1920, 1080),
    "B_12MP.jpg": (4032, 3024),
    "C_48MP.jpg": (8064, 6048),
}
WORKLOAD_CELL = 512
WORKLOAD_JPEG_QUALITY = 90


def log(message: str) -> None:
    print(message, file=sys.stderr)


def value_noise(width: int, height: int, rng: np.random.Generator, octaves: int = 5) -> np.ndarray:
    """Multi-octave value noise in [0, 1], HxW float32: random grids upsampled bilinearly."""
    out = np.zeros((height, width), dtype=np.float32)
    amplitude, total = 1.0, 0.0
    for octave in range(octaves):
        cells = 2 ** (octave + 2)
        grid = rng.random((cells + 1, cells + 1), dtype=np.float32)
        ys = np.linspace(0, cells, height, endpoint=False, dtype=np.float32)
        xs = np.linspace(0, cells, width, endpoint=False, dtype=np.float32)
        y0, x0 = ys.astype(np.int64), xs.astype(np.int64)
        fy, fx = (ys - y0)[:, None], (xs - x0)[None, :]
        top = grid[y0][:, x0] * (1 - fx) + grid[y0][:, x0 + 1] * fx
        bottom = grid[y0 + 1][:, x0] * (1 - fx) + grid[y0 + 1][:, x0 + 1] * fx
        out += amplitude * (top * (1 - fy) + bottom * fy)
        total += amplitude
        amplitude *= 0.5
    return out / total


def synthesize(width: int, height: int, seed: int) -> Image.Image:
    """One deterministic RGB image: gradient sky, gratings, texture, edges and strokes."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    u, v = xx / max(width - 1, 1), yy / max(height - 1, 1)

    # Smooth two-axis colour gradient: the banding probe.
    base = np.stack([0.25 + 0.5 * u, 0.35 + 0.4 * (1 - v), 0.55 + 0.35 * v * u], axis=-1)

    # Fine periodic detail in the lower-left quadrant: aliasing and ringing probe.
    frequency = rng.uniform(0.18, 0.32)
    grating = 0.5 + 0.5 * np.sin(2 * np.pi * frequency * (xx + 0.35 * yy))
    region = ((u < 0.5) & (v > 0.5)).astype(np.float32)[..., None]
    base = base * (1 - 0.45 * region) + 0.45 * region * grating[..., None]

    # Organic texture over the right half: false-detail probe.
    texture = value_noise(width, height, rng)
    tint = rng.uniform(0.6, 1.0, size=3).astype(np.float32)
    region = (u >= 0.5).astype(np.float32)[..., None]
    base = base * (1 - 0.5 * region) + 0.5 * region * texture[..., None] * tint

    pixels = np.clip(base * 255.0 + rng.normal(0.0, 2.0, size=base.shape), 0, 255).astype(np.uint8)
    image = Image.fromarray(pixels)

    # Hard-edged shapes and thin strokes: halo and text-deformation probe.
    draw = ImageDraw.Draw(image)
    unit = max(min(width, height) // 16, 4)
    for _ in range(6):
        x0, y0 = int(rng.integers(0, width - unit)), int(rng.integers(0, height - unit))
        size = int(rng.integers(unit, unit * 3))
        colour = tuple(int(c) for c in rng.integers(0, 256, size=3))
        if rng.random() < 0.5:
            draw.rectangle([x0, y0, x0 + size, y0 + size // 2], fill=colour)
        else:
            draw.ellipse([x0, y0, x0 + size, y0 + size], outline=colour, width=max(unit // 8, 1))
    line_y = int(height * 0.15)
    while line_y < height * 0.45:
        x = int(width * 0.05)
        while x < width * 0.45:
            glyph = int(rng.integers(2, 7))
            draw.line([x, line_y, x, line_y + unit // 2], fill=(20, 20, 20), width=1)
            draw.line([x, line_y, x + glyph, line_y], fill=(20, 20, 20), width=1)
            x += glyph + 3
        line_y += unit // 2 + 4
    return image


def test_tile_targets() -> list[tuple[Path, int, int]]:
    """(path, width, height) for every conversion config's `test_tile`, sorted for determinism."""
    targets = []
    for config_path in sorted(CONFIG_DIR.glob("*.yaml")):
        config = yaml.safe_load(config_path.read_text())
        _, _, height, width = config["conversion"]["input_shape"]
        targets.append((REPO_ROOT / config["test_tile"], width, height))
    return targets


def write_tiles(force: bool) -> int:
    written = 0
    for path, width, height in test_tile_targets():
        if path.exists() and not force:
            log(f"[tile] exists, kept: {path.relative_to(REPO_ROOT)}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        synthesize(width, height, seed=width * 1_000 + height).save(path, format="PNG")
        log(f"[tile] wrote {path.relative_to(REPO_ROOT)} ({width}x{height})")
        written += 1
    return written


def write_workloads(force: bool) -> int:
    written = 0
    WORKLOAD_DIR.mkdir(parents=True, exist_ok=True)
    for name, (width, height) in WORKLOADS.items():
        path = WORKLOAD_DIR / name
        if path.exists() and not force:
            log(f"[workload] exists, kept: {path.relative_to(REPO_ROOT)}")
            continue
        canvas = Image.new("RGB", (width, height))
        seed = 0
        for top in range(0, height, WORKLOAD_CELL):
            for left in range(0, width, WORKLOAD_CELL):
                canvas.paste(synthesize(WORKLOAD_CELL, WORKLOAD_CELL, seed=seed), (left, top))
                seed += 1
        canvas.save(path, format="JPEG", quality=WORKLOAD_JPEG_QUALITY)
        log(f"[workload] wrote {path.relative_to(REPO_ROOT)} ({width}x{height}, {seed} cells)")
        written += 1
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the synthetic test tiles and workloads")
    parser.add_argument("--tiles", action="store_true", help="generate conversion test tiles only")
    parser.add_argument("--workloads", action="store_true", help="generate workload images only")
    parser.add_argument("--force", action="store_true", help="overwrite files that already exist")
    args = parser.parse_args(argv)
    both = not args.tiles and not args.workloads
    written = 0
    if args.tiles or both:
        written += write_tiles(args.force)
    if args.workloads or both:
        written += write_workloads(args.force)
    log(f"[done] {written} file(s) written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
