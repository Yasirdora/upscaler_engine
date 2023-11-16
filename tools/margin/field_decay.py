"""How far does one input pixel actually MATTER — not how far can it reach.

`verify_geometry.py --check field` answers the reach question: perturb one input pixel, find the
outermost output pixel that moves at all. For RRDBNet that is 348.25 input px, and the frame-oracle
argument in `derive_context_margin_torch.py` was originally built on it — an oracle was held to need
a full receptive field of real content before it could stand in for a full-frame run.

`verify_geometry.py --check oracle` then measured the opposite: for RealESRGAN_x4plus at tile 256,
the whole decay curve stops moving once the oracle has 128 px of context, agreeing with a 384 px
oracle to 0.00017/255 over every swept margin. Reach and influence are not the same quantity.

This script measures the second one directly, so the disagreement is explained rather than merely
recorded: it perturbs one input pixel and reports the LARGEST output change at each input-pixel
distance from it. The result is a magnitude profile, in units of the criterion (x/255 on the 8-bit
product surface's scale), against which "128 px is enough oracle" can be read off rather than
inferred.

    python tools/margin/field_decay.py                       # both models
    python tools/margin/field_decay.py --model RealESRGAN_x4plus

Seeded and deterministic. No Upscayl or any GPL/AGPL source was consulted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "convert"))
from model_loader import build_model  # noqa: E402

#: Input-pixel distances the profile is reported at. 128 and 349 are the two numbers the oracle
#: argument turns on: the measured convergence point and the measured receptive-field radius.
DISTANCES = [1, 2, 4, 8, 16, 24, 32, 34, 48, 64, 96, 128, 160, 192, 256, 320, 348, 349]


def profile(model_id: str, canvas: int, device: str, seed: int) -> int:
    config = yaml.safe_load((ROOT / "tools" / "convert" / "models" / f"{model_id}.yaml").read_text())
    model, _ = build_model(config, ROOT / config["weights"]["path"])
    model = model.to(device).eval()
    scale = int(config["scale_factor"])

    torch.manual_seed(seed)
    x = torch.rand(1, 3, canvas, canvas, device=device)
    centre = canvas // 2
    perturbed = x.clone()
    perturbed[0, :, centre, centre] = 1.0 - perturbed[0, :, centre, centre]

    with torch.no_grad():
        clean = model(x).float().cpu().numpy()[0]
        if device == "mps":
            torch.mps.empty_cache()
        dirty = model(perturbed).float().cpu().numpy()[0]
        if device == "mps":
            torch.mps.empty_cache()
    # Reported on the criterion's scale: the model works in [0,1] and the criterion counts 8-bit
    # levels, so a change of 1/255 in model units is a change of 1.0 in the units of every table in
    moved = np.abs(clean - dirty).max(axis=0) * 255.0
    del clean, dirty

    height, width = moved.shape
    ys, xs = np.mgrid[0:height, 0:width]
    block = centre * scale
    # Chebyshev distance from the perturbed pixel's own 4x4 output block, back in INPUT pixels.
    dx = np.maximum(0, np.maximum(block - xs, xs - (block + scale - 1)))
    dy = np.maximum(0, np.maximum(block - ys, ys - (block + scale - 1)))
    distance_in = np.maximum(dx, dy) / scale

    clipped = bool(moved[0].any() or moved[-1].any() or moved[:, 0].any() or moved[:, -1].any())
    print(f"\n{model_id}  ({config['architecture']['type']}, canvas {canvas}px, scale x{scale})")
    if clipped:
        print("  WARNING: the disturbance reached the canvas edge — every number below is a "
              "lower bound, not the profile.")
    print(f"  {'distance (input px)':>21}   {'largest output change /255':>26}")
    for d in DISTANCES:
        selected = moved[distance_in >= d]
        if selected.size == 0:
            continue
        print(f"  {d:>21}   {float(selected.max()):>26.6f}")
    return 1 if clipped else 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", nargs="+", default=["realesr-general-x4v3", "RealESRGAN_x4plus"])
    p.add_argument("--canvas", type=int, default=800,
                   help="must clear the expected radius on every side, or the profile is clipped")
    p.add_argument("--device", default="mps")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    status = 0
    for model_id in args.model:
        status |= profile(model_id, args.canvas, args.device, args.seed)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
