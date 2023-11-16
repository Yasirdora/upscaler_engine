"""Receptive-field radius for RealESRGAN_x2plus, measured rather than derived.

PREDICTION, STATED BEFORE MEASURING so the result can falsify it: x2plus should have a receptive
field roughly TWICE x4plus's in input pixels, not half. The body is the identical 23-block RRDBNet,
so its radius in *body* pixels is the same ~348. But x2plus feeds the body a `pixel_unshuffle`d
input, where one body pixel spans a 2x2 block of input pixels — so the same body radius reaches
twice as far in input space. Expect ~696 input px against x4plus's 348.25.

If that holds it is a correctness finding, not a curiosity: x2plus would otherwise run on the
76 px margin derived for x4plus. A
larger field does not automatically mean a larger margin — influence decays with distance and the
criterion is where it falls below 1/255, not where it reaches zero — but it removes any argument
that x4plus's number is conservative for x2plus.

Method is `tools/margin/verify_geometry.py:check_receptive_field` verbatim: perturb one input
pixel, find how far the output moves. No layer arithmetic. The canvas must be comfortably wider
than the expected field or the number is a lower bound rather than the radius, and a clipped
measurement looks exactly like a small one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "convert"))
from model_loader import build_model  # noqa: E402


def measure(model_id: str, canvas: int, device: str = "mps") -> tuple[float, bool]:
    config = yaml.safe_load(
        (ROOT / "tools" / "convert" / "models" / f"{model_id}.yaml").read_text()
    )
    model, _ = build_model(config, ROOT / config["weights"]["path"])
    model = model.to(device).eval()
    scale = int(config["scale_factor"])

    torch.manual_seed(0)
    x = torch.rand(1, 3, canvas, canvas, device=device)
    centre = canvas // 2
    perturbed = x.clone()
    perturbed[0, :, centre, centre] = 1.0 - perturbed[0, :, centre, centre]

    # One forward at a time, reduced to host memory before the next: holding both large outputs
    # live is enough to get the process killed, which looks identical to a silently failing check.
    with torch.no_grad():
        clean = model(x).float().cpu().numpy()[0]
        if device == "mps":
            torch.mps.empty_cache()
        dirty = model(perturbed).float().cpu().numpy()[0]
        if device == "mps":
            torch.mps.empty_cache()

    moved = np.abs(clean - dirty).max(axis=0)
    del clean, dirty
    rows, columns = np.nonzero(moved > 0)
    top, bottom, left, right = rows.min(), rows.max(), columns.min(), columns.max()
    block_start, block_end = centre * scale, centre * scale + scale - 1
    radius_out = max(
        block_start - left, right - block_end, block_start - top, bottom - block_end
    )
    clipped = bool(
        top == 0 or left == 0 or bottom == canvas * scale - 1 or right == canvas * scale - 1
    )
    return radius_out / scale, clipped


def main() -> int:
    print("model                     scale  canvas   radius (input px)  clipped", flush=True)
    print("-" * 74, flush=True)
    for model_id, canvas in (("RealESRGAN_x4plus", 800), ("RealESRGAN_x2plus", 1800)):
        radius, clipped = measure(model_id, canvas)
        flag = "  ** LOWER BOUND, RERUN WIDER **" if clipped else ""
        scale = 2 if "x2" in model_id else 4
        print(
            f"{model_id:<26}{scale:>3}x{canvas:>8}{radius:>18.2f}   {clipped}{flag}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
