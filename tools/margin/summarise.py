"""Read the derivation JSONs and print the tables the report is built from.

Every aggregate is printed with its probe count, its distinct-image count AND its distinct-content
count. This project has been bitten twice by a number whose sample was not what its size implied —
once comparing a 9-probe figure against a 387-probe one, once by a "9-probe" figure that was nine
windows of a single image — so the denominators travel with the number everywhere.

Shapes are never blended: each tile size is summarised on its own.

    python tools/margin/summarise.py margin-data/x4plus-256-*.json --tile 256
    python tools/margin/summarise.py margin-data/*.json --split oracle_context
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

THRESHOLD_255 = 1.0
SAFETY_FACTOR = 2


def crossing(decay: dict[int, float]) -> tuple[int | None, int | None]:
    for margin in sorted(decay):
        if decay[margin] < THRESHOLD_255:
            return margin, margin * SAFETY_FACTOR
    return None, None


def decay_of(rows: list[dict], margins: list[int], reducer=np.mean) -> dict[int, float]:
    out = {}
    for m in margins:
        values = [r["rings"][str(m)][0] for r in rows if str(m) in r["rings"]]
        if values:
            out[m] = float(reducer(values))
    return out


def denominators(rows: list[dict]) -> str:
    return (
        f"{len(rows)} probes, {len({r['image'] for r in rows})} distinct images, "
        f"{len({r['content'] for r in rows})} distinct contents"
    )


def report_block(label: str, rows: list[dict], margins: list[int], show: list[int]) -> None:
    if not rows:
        return
    probe_mean = decay_of(rows, margins)
    worst = decay_of(rows, margins, np.max)
    images = sorted({r["image"] for r in rows})
    per_image = {n: decay_of([r for r in rows if r["image"] == n], margins) for n in images}
    image_mean = {
        m: float(np.mean([per_image[n][m] for n in images if m in per_image[n]])) for m in margins
    }
    print("=" * 92)
    print(f"{label} — {denominators(rows)}")
    print("=" * 92)
    print(f"{'margin':>7} {'probe-mean':>12} {'image-mean':>12} {'worst probe':>12}")
    for m in show:
        if m in probe_mean:
            print(f"{m:>7} {probe_mean[m]:>12.4f} {image_mean[m]:>12.4f} {worst[m]:>12.4f}")
    for name, decay in (("probe-mean", probe_mean), ("image-mean", image_mean), ("worst probe", worst)):
        c, mm = crossing(decay)
        verdict = f"crossing {c} -> DERIVED MARGIN {mm}" if c is not None else "NO CROSSING below 1/255"
        print(f"  {name:<12}: {verdict}")
    if len(images) > 1:
        print("  per image:")
        for n in images:
            c, mm = crossing(per_image[n])
            n_rows = sum(1 for r in rows if r["image"] == n)
            print(f"    {n:<30} n={n_rows:<4} crossing {str(c):<6} margin {str(mm):<6} "
                  f"floor {min(per_image[n].values()):.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="+", type=Path)
    ap.add_argument("--tile", type=int, nargs="+", default=[256, 384, 512])
    ap.add_argument("--show", type=int, nargs="+",
                    default=[0, 4, 8, 12, 16, 20, 24, 26, 32, 40, 48, 56, 64, 80, 96, 112, 126])
    ap.add_argument("--split", choices=["none", "seam", "oracle_context", "natural_context"],
                    default="none", help="also report the sample cut this way")
    args = ap.parse_args()

    for path in args.json:
        blob = json.loads(path.read_text())
        print()
        print("#" * 92)
        print(f"# {path.name}")
        print(f"# model {blob['model']} ({blob['architecture']}), receptive field "
              f"{blob['receptive_field_radius_input_px']} input px, scale x{blob['scale']}")
        print(f"# placement {blob['placement']}, oracle {blob['oracle']}")
        print(f"# measures  {blob['measures']}")
        print(f"# criterion {blob['criterion']['statistic']} < {blob['criterion']['threshold_255']}/255, "
              f"x{blob['criterion']['safety_factor']} safety ({blob['criterion']['source']})")
        print(f"# device {blob['environment']['device']}, torch {blob['environment']['torch']}, "
              f"seed {blob['environment']['seed']}")
        if blob.get("duplicate_windows_dropped"):
            print(f"# {blob['duplicate_windows_dropped']} duplicate-content windows dropped before running")
        print("#" * 92)

        for tile in args.tile:
            rows = [p for p in blob["probes"] if p["tile"] == tile]
            if not rows:
                continue
            margins = sorted({int(k) for r in rows for k in r["rings"]})
            report_block(f"TILE {tile}", rows, margins, args.show)

            if args.split == "seam":
                for label, subset in (
                    ("windows clear of any montage seam", [r for r in rows if not r["straddles_seam"]]),
                    ("windows straddling a montage seam", [r for r in rows if r["straddles_seam"]]),
                ):
                    report_block(f"TILE {tile} — {label}", subset, margins, args.show)
            elif args.split == "oracle_context":
                # How much context the ORACLE actually had on its most-starved side. The committed
                # harness gives an edge window zero context on two sides, so its "full-frame oracle"
                # is on those sides the tile itself, and the deficit it measures is understated.
                for low, high in ((0, 0), (1, 63), (64, 127), (128, 10**6)):
                    subset = [r for r in rows if low <= min(r["oracle_context"]) <= high]
                    if subset:
                        report_block(
                            f"TILE {tile} — oracle context {low}..{high} px on its thinnest side",
                            subset, margins, args.show)
            elif args.split == "natural_context":
                for low, high in ((0, 33), (34, 127), (128, 10**6), (-1, -1)):
                    subset = [r for r in rows if low <= r["natural_context"] <= high]
                    if subset:
                        name = ("whole source is one photograph" if low == -1
                                else f"natural context {low}..{high} px")
                        report_block(f"TILE {tile} — {name}", subset, margins, args.show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
