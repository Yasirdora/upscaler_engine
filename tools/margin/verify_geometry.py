"""Verification harness for `derive_context_margin_torch.py` — run this before trusting a number.

Four things have to be true before a context-margin derivation means anything, and none of them
is self-evident from reading the probe:

1. **The ring geometry is the shipped one.** The criterion is stated on the mean of a ring, so a
   ring that is off by one pixel is a different criterion. `check_rings` transcribes
   `ContextMargin.ringProfile` (`Sources/UpscaleCore/Tiling/ContextMargin.swift`) literally —
   the two-horizontal-edges-plus-two-strided-verticals loop, corner bookkeeping and all — and
   compares it against the probe's integral-image version on random fields.

2. **The receptive-field radii are what the frame-oracle argument assumes.** The probe replaces
   the full-frame oracle with a *frame* oracle, and that substitution is only exact if the frame
   supplies at least a receptive field of real context. `check_receptive_field` measures the radius
   by perturbing one input pixel and finding how far the output moves, rather than trusting the
   layer arithmetic.

3. **The product surface is the shipped one.** `check_surface` pins clamp-to-[0,1] then
   `floor(x*255 + 0.5)` against a table of exact boundary values.

4. **The frame oracle is enough oracle.** `check_oracle_ladder` takes real windows and recomputes
   the whole decay curve against oracles built from a ladder of context widths. If the curve stops
   moving once context reaches some C, then an oracle with C px of context is the full-frame oracle
   as far as this criterion can tell — and the derivation is entitled to use one. This is the check
   that decides how much context an x4plus run must pay for, so it is measured, not assumed.

    python tools/margin/verify_geometry.py                    # all checks
    python tools/margin/verify_geometry.py --check rings      # one check, one process

Checks are selectable because this measurement is memory-hungry: RRDBNet on a 1024px frame will
be killed on a 16GB host that is already swapping, and a killed check looks exactly like a passing
one if it is buried in a batch. Run them one at a time when the host is loaded.

Exit status is non-zero if any check fails. No Upscayl or any GPL/AGPL source was consulted.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "convert"))
sys.path.insert(0, str(ROOT / "tools" / "margin"))

from derive_context_margin_torch import RECEPTIVE_FIELD_RADIUS, ring_profile, to_product_surface  # noqa: E402
from model_loader import build_model  # noqa: E402


def swift_ring_profile(values: np.ndarray, width: int, height: int):
    """Literal transcription of `ContextMargin.ringProfile`, Swift control flow preserved.

    Deliberately unvectorised and deliberately ugly: its only job is to be obviously the same
    algorithm as the Swift, so that agreement with the fast version means something.
    """
    maximum_distance = min(width, height) // 2
    ring_sums = [0.0] * maximum_distance
    ring_counts = [0.0] * maximum_distance
    ring_maxima = [0.0] * maximum_distance
    row_samples = width * 3
    for distance in range(maximum_distance):
        low = distance
        high_row = height - 1 - distance
        high_column = width - 1 - distance
        span = width - 2 * distance
        if span <= 0:
            continue
        total = 0.0
        maximum = 0.0
        for row in (low, high_row):
            if not (row != high_row or low != high_row):
                continue
            start = row * row_samples + low * 3
            segment = values[start : start + span * 3]
            total += float(segment.sum())
            maximum = max(maximum, float(segment.max()) if segment.size else 0.0)
            ring_counts[distance] += span * 3
        if low == high_row:  # a one-row frame was counted once above
            ring_counts[distance] = float(span * 3)
        inner_rows = range(low + 1, high_row)
        if len(inner_rows) > 0:
            for column in (low, high_column):
                if not (column != high_column or low != high_column):
                    continue
                for channel in range(3):
                    start = (low + 1) * row_samples + column * 3 + channel
                    segment = values[start : start + len(inner_rows) * row_samples : row_samples]
                    total += float(segment.sum())
                    maximum = max(maximum, float(segment.max()) if segment.size else 0.0)
                    ring_counts[distance] += len(inner_rows)
        ring_sums[distance] = total
        ring_maxima[distance] = maximum
    ring_mean = [s / c if c > 0 else 0.0 for s, c in zip(ring_sums, ring_counts)]
    return ring_mean, ring_maxima


def check_rings() -> bool:
    """The probe's ring must be the Swift harness's ring, at every swept margin."""
    ok = True
    rng = np.random.default_rng(0)
    margins = list(range(0, 127, 2))
    for tile in (256, 384, 512):
        side = tile * 4
        diff = rng.integers(0, 64, size=(side, side, 3)).astype(np.uint8)
        reference_mean, reference_max = swift_ring_profile(
            diff.astype(np.float32).reshape(-1), side, side
        )
        measured = ring_profile(diff, margins, 4)
        worst_mean = max(abs(measured[m][0] - reference_mean[m * 4]) for m in measured)
        worst_max = max(abs(measured[m][1] - reference_max[m * 4]) for m in measured)
        good = worst_mean == 0.0 and worst_max == 0.0
        ok &= good
        print(
            f"  rings  tile {tile:>3} ({side}px output): max |ring-mean delta| {worst_mean:.3e}, "
            f"max |ring-max delta| {worst_max:.3e}  {'OK' if good else 'FAIL'}"
        )
    return ok


def check_surface() -> bool:
    """clamp to [0,1] then floor(x*255 + 0.5), on exact boundary values."""
    cases = [(-1.0, 0), (0.0, 0), (0.5 / 255, 1), (0.4 / 255, 0), (1.0 / 255, 1), (0.5, 128), (1.0, 255), (2.0, 255)]
    ok = True
    for value, expected in cases:
        t = torch.full((1, 3, 1, 1), value)
        got = int(to_product_surface(t)[0, 0, 0])
        good = got == expected
        ok &= good
        if not good:
            print(f"  surface x={value}: expected {expected}, got {got}  FAIL")
    print(f"  surface  clamp[0,1] + floor(255x+0.5) over {len(cases)} boundary values  {'OK' if ok else 'FAIL'}")
    return ok


def check_receptive_field(device: str = "mps") -> bool:
    """Perturb one input pixel; the radius is how far the output moves. No layer arithmetic.

    The measurement must be *unclipped* — if the disturbance reaches the canvas edge the number is
    a lower bound, not the radius — so each model is run on a canvas comfortably wider than its
    expected field: 800 px leaves RRDBNet's 348 px field 52 px of slack on every side.

    The two forwards are run and reduced to host memory one at a time. Holding both 4x outputs of
    an 800px canvas live at once is enough to get the process killed on a loaded machine, which
    looks exactly like a silently failing check.
    """
    ok = True
    for model_id, canvas in (("realesr-general-x4v3", 256), ("RealESRGAN_x4plus", 800)):
        config = yaml.safe_load((ROOT / "tools" / "convert" / "models" / f"{model_id}.yaml").read_text())
        model, _ = build_model(config, ROOT / config["weights"]["path"])
        model = model.to(device).eval()
        scale = int(config["scale_factor"])
        torch.manual_seed(0)
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
        moved = np.abs(clean - dirty).max(axis=0)
        del clean, dirty
        rows, columns = np.nonzero(moved > 0)
        top, bottom, left, right = rows.min(), rows.max(), columns.min(), columns.max()
        block_start, block_end = centre * scale, centre * scale + scale - 1
        radius_out = max(block_start - left, right - block_end, block_start - top, bottom - block_end)
        radius_in = radius_out / scale
        clipped = top == 0 or left == 0 or bottom == canvas * scale - 1 or right == canvas * scale - 1
        expected = RECEPTIVE_FIELD_RADIUS[config["architecture"]["type"]]
        good = (not clipped) and radius_in <= expected
        ok &= good
        print(
            f"  field  {model_id:<22} {config['architecture']['type']:<17} measured radius "
            f"{radius_in:7.2f} input px, probe assumes {expected:>4}"
            f"{', CLIPPED' if clipped else ''}  {'OK' if good else 'FAIL'}"
        )
    return ok


def check_oracle_ladder(
    model_id: str = "RealESRGAN_x4plus", tile: int = 256, device: str = "mps"
) -> bool:
    """Does the DECAY CURVE stop moving as the oracle is given more context?

    Windows are centred in their 512 montage cell, so context up to 128 px is continuous natural
    photograph and everything past that is a neighbouring cell — an unrelated picture. Two
    questions are answered at once: how much context an oracle needs, and whether the montage's
    foreign content beyond 128 px changes the answer.

    The reported quantity is the largest disagreement, over every swept margin, between the decay
    curve computed against a `C`-context oracle and the deepest rung. A value far below 1/255 means
    the criterion cannot tell the two oracles apart.
    """
    margins = [0, 4, 8, 12, 16, 20, 26, 32, 40, 48, 64, 80, 96, 112, 126]
    ladder = [64, 128, 192, 256, 384]
    config = yaml.safe_load((ROOT / "tools" / "convert" / "models" / f"{model_id}.yaml").read_text())
    model, _ = build_model(config, ROOT / config["weights"]["path"])
    model = model.to(device).eval()
    scale = int(config["scale_factor"])
    source = Image.open(ROOT / "Sources/TestFixtures/workloads/B_12MP.jpg").convert("RGB")
    inset = (512 - tile) // 2
    windows = [(512 * c + inset, 512 * r + inset) for c, r in [(1, 1), (5, 4)]]

    def infer(rect):
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        array = np.asarray(source.crop(rect), np.uint8).copy()
        tensor = torch.from_numpy(array).to(device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        with torch.no_grad():
            out = to_product_surface(model(tensor))
        del tensor
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        return out

    ok = True
    for wx, wy in windows:
        bare = infer((wx, wy, wx + tile, wy + tile))
        curves = {}
        for context in ladder:
            oracle = infer((wx - context, wy - context, wx + tile + context, wy + tile + context))
            region = oracle[context * scale : (context + tile) * scale,
                            context * scale : (context + tile) * scale]
            diff = np.abs(region.astype(np.int16) - bare.astype(np.int16)).astype(np.uint8)
            curves[context] = {m: v[0] for m, v in ring_profile(diff, margins, scale).items()}
            del oracle
        deepest = curves[ladder[-1]]
        print(f"  oracle {model_id} tile {tile} window ({wx},{wy}) — natural context 128 px")
        for context in ladder:
            drift = max(abs(curves[context][m] - deepest[m]) for m in deepest)
            good = context < 128 or drift < 0.05
            ok &= good
            note = "(oracle starved)" if context < 128 else ""
            print(
                f"    context {context:>3} px: max |curve - curve({ladder[-1]})| over all margins "
                f"= {drift:.5f}/255  {note}"
            )
    return ok


CHECKS = {
    "rings": check_rings,
    "surface": check_surface,
    "field": check_receptive_field,
    "oracle": check_oracle_ladder,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", nargs="+", choices=sorted(CHECKS), default=sorted(CHECKS))
    args = parser.parse_args()
    print("verifying derive_context_margin_torch.py against the shipped geometry\n")
    results = []
    for name in args.check:
        results.append(CHECKS[name]())
    print()
    if all(results):
        print(f"CHECKS PASSED: {', '.join(args.check)}")
        return 0
    print("VERIFICATION FAILED — do not trust a margin produced by this probe")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
