"""Context-margin derivation that measures context influence ALONE.

WHY THIS EXISTS
---------------
`UpscaleCLI derive-context-margin` compares `CoreML(tile)` against `PyTorch(full frame)`. That
difference carries two error terms:

    |CoreML(tile) - PyTorch(full)|  <=  |PyTorch(tile) - PyTorch(full)|  +  |CoreML(t) - PyTorch(t)|
                                        `--- context deficit --------'     `--- conversion ------'

Only the first is what a context margin is for; the second has a floor no margin can reduce. Margin
derivation should measure the first term alone. This probe keeps BOTH sides in pure PyTorch, so the
conversion term is absent from each and cancels exactly. Nothing else about the criterion is
changed:

  * ring-based comparison, ring N = the one-pixel frame N output pixels in from the window border.
    Verified bit-identical to `ContextMargin.ringProfile`
    (`Sources/UpscaleCore/Tiling/ContextMargin.swift`) by `tools/margin/verify_geometry.py`.
  * mean absolute error on the 8-bit product surface (clamp [0,1], round to 8 bits)
  * the 1/255 mean-ring criterion of certification spec §4, UNMOVED, with the x2 safety factor
  * oracle cropped at exactly `scale * (x0, y0)`, integer crops, no resampling anywhere

THE ORACLE, AND WHY A FRAME CAN STAND IN FOR A FULL FRAME
---------------------------------------------------------
A full-frame oracle is unaffordable at 12MP and impossible at 48MP, and it is also unnecessary.
A network's output at a pixel depends only on inputs within its receptive-field radius, so an
oracle run on a FRAME that surrounds the window with at least that much real content produces,
for that window's output region, exactly the pixels a full-frame run would.

Radii are MEASURED, not derived from layer arithmetic (`verify_geometry.py --check field`):
perturb one input pixel, find how far the output moves.

    SRVGGNetCompact (x4v3)   34.00 input px
    RRDBNet         (x4plus) 348.25 input px

THE THING THIS REPOSITORY CANNOT SUPPLY
---------------------------------------
No natural image here is larger than 517x512 (the corpus tops out there; `interior_screen`).
The only larger assets are `Sources/TestFixtures/workloads/{A,B,C}.jpg`, which their own README
records as MONTAGES: corpus photos tiled on a strict 512x512 grid, verified here by seam
detection. Content across a cell boundary is therefore an unrelated photograph.

Consequences, which differ sharply by architecture and are recorded per probe rather than
argued away:

  * x4v3 needs 34 px of context. A 256 window centred in a 512 cell has 128 px of natural
    context on every side, so its ENTIRE receptive field is genuine, continuous photograph.
    x4v3 can be derived on natural content here.
  * x4plus needs 348 px. 256 + 2*348 = 952 > 512, so NO window anywhere in this repository can
    be given a full natural receptive field. Every x4plus probe is compromised; the question is
    only by how much, which `--oracle-context` ladders answer empirically.

Every probe records `natural_context` (px to the nearest montage seam) and `straddles_seam`, and
every aggregate is reported with its probe count, distinct-image count AND distinct-content count.

USAGE
    python tools/margin/verify_geometry.py          # run this first, always
    python tools/margin/derive_context_margin_torch.py \
        --model RealESRGAN_x4plus --tile-size 256 \
        --sources workloads --placement cell-centred \
        --output margin-data/x4plus-256-cell-centred.json

No Upscayl or any GPL/AGPL source was consulted.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "convert"))
from model_loader import build_model  # noqa: E402

#: Receptive-field radius in INPUT pixels, per architecture. Measured by `verify_geometry.py`,
#: not asserted: perturbing one input pixel moves output pixels out to exactly this distance and
#: no further. RRDBNet's true radius is 348.25; 349 is the integer that covers it.
RECEPTIVE_FIELD_RADIUS = {"RRDBNet": 349, "SRVGGNetCompact": 34}

#: The montage cell of `Sources/TestFixtures/workloads/*.jpg`, measured by adjacent-column and
#: adjacent-row difference spikes: every 512 px, in all three images, with no other spacing.
MONTAGE_CELL = 512

#: The criterion. Certification spec §4. NOT tunable from the command line, on purpose.
THRESHOLD_255 = 1.0
SAFETY_FACTOR = 2

WORKLOADS = ["A_1080p", "B_12MP", "C_48MP"]


# --------------------------------------------------------------------------------------- geometry


#: Frames are trimmed to a multiple of this. 1 for x4 models — no constraint, and no change to
#: any derivation already committed. `run()` sets it to 2 for x2 models, whose `pixel_unshuffle`
#: rejects an odd dimension outright.
FRAME_ALIGNMENT = 1


@dataclass
class Frame:
    """One oracle evaluation: a rectangle of a source image, run whole.

    Callers align dimensions to `FRAME_ALIGNMENT` before constructing one. x2 models
    `pixel_unshuffle` by 2 before the body, so an odd frame throws — and evening it *here* would
    not be enough, because window origins are computed from the caller's own width and height, so
    the last window would sit one row past the oracle. That desynchronisation surfaces as a shape
    assertion rather than a wrong number, and only because the assertion exists.
    """

    image: str
    x: int
    y: int
    w: int
    h: int
    #: (tile, frame-local x, frame-local y)
    windows: list[tuple[int, int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Alignment is 1 for x4 models, so their frames are untouched and every number this tool
        # has already produced reproduces byte-identically. Only x2 sets it to 2.
        self.w -= self.w % FRAME_ALIGNMENT
        self.h -= self.h % FRAME_ALIGNMENT


def seam_geometry(origin: int, tile: int, cell: int | None) -> tuple[int, bool]:
    """Distance from a window's border to the nearest montage seam, and whether it straddles one.

    Returns `(natural_context, straddles)`. `natural_context` is how much genuinely continuous
    photograph surrounds the window before an unrelated one begins; for a non-montage source it is
    reported as -1, meaning "the whole source is one photograph".
    """
    if cell is None:
        return -1, False
    start_cell, end_cell = origin // cell, (origin + tile - 1) // cell
    if start_cell != end_cell:
        return 0, True
    before = origin - start_cell * cell
    after = (start_cell + 1) * cell - (origin + tile)
    return min(before, after), False


def cell_centred_windows(size: tuple[int, int], tile: int, cell: int) -> list[tuple[int, int]]:
    """One window per full montage cell, centred in it — the most natural context a cell can give.

    A cell is a whole photograph, so a window centred in it is surrounded by `(cell - tile) / 2` px
    of continuous, correlated content on every side. At tile 256 that is 128 px.
    """
    width, height = size
    if tile > cell:
        return []
    inset = (cell - tile) // 2
    return [
        (column * cell + inset, row * cell + inset)
        for row in range(height // cell)
        for column in range(width // cell)
    ]


def grid_window_origins(extent: int, tile: int, context: int) -> list[int]:
    """Non-overlapping window origins inside a frame, each with >= `context` px on both sides."""
    first, last = context, extent - tile - context
    if last < first:
        return []
    return [first + index * tile for index in range((last - first) // tile + 1)]


def plan_frames(
    image: str,
    size: tuple[int, int],
    tiles: list[int],
    context: int,
    frame_cap: tuple[int, int],
    placement: str,
    cell: int | None,
) -> list[Frame]:
    """Lay oracle frames over a source and place windows inside them.

    A frame is one oracle evaluation and frames may overlap — that only costs compute. The WINDOWS
    must not, because two overlapping windows are not two independent probes; the caller asserts
    that separately.
    """
    width, height = size
    frames: list[Frame] = []

    if placement == "cell-centred":
        if cell is None:
            return []
        # One frame per window, sized to give the window `context` px on every side. The frame is
        # clamped into the image, so windows near the source border get a frame that is shifted
        # rather than shrunk — the window keeps its full context either way.
        for tile in tiles:
            for wx, wy in cell_centred_windows(size, tile, cell):
                fx = min(max(0, wx - context), max(0, width - (tile + 2 * context)))
                fy = min(max(0, wy - context), max(0, height - (tile + 2 * context)))
                fw = min(width, tile + 2 * context)
                fh = min(height, tile + 2 * context)
                if wx < fx or wy < fy or wx + tile > fx + fw or wy + tile > fy + fh:
                    continue
                if min(wx - fx, wy - fy, fx + fw - wx - tile, fy + fh - wy - tile) < context:
                    continue
                frames.append(Frame(image, fx, fy, fw, fh, [(tile, wx - fx, wy - fy)]))
        return frames

    # placement == "grid": frames stride by exactly the span their windows occupy, so consecutive
    # frames contribute windows that abut rather than overlap.
    fw, fh = min(frame_cap[0], width), min(frame_cap[1], height)
    sx, sy = max(1, fw - 2 * context), max(1, fh - 2 * context)
    seen: set[tuple[int, int, int]] = set()
    for fy in range(0, height - fh + 1, sy):
        for fx in range(0, width - fw + 1, sx):
            frame = Frame(image, fx, fy, fw, fh)
            for tile in tiles:
                for wy in grid_window_origins(fh, tile, context):
                    for wx in grid_window_origins(fw, tile, context):
                        key = (tile, fx + wx, fy + wy)
                        if key in seen:
                            continue
                        seen.add(key)
                        frame.windows.append((tile, wx, wy))
            if frame.windows:
                frames.append(frame)
    return frames


def legacy_frames(image: str, size: tuple[int, int], tiles: list[int]) -> list[Frame]:
    """The Swift harness's own rule: whole image as the oracle, 3x3 left/centre/right windows.

    `ImageCommand.swift:562-566`. This reproduces the committed derivation's SAMPLE exactly, so
    that a context-only number can be compared against the committed one on identical windows.
    Its oracle is the whole source image, which for x4plus is smaller than one receptive field —
    that is a property of the committed derivation, faithfully reproduced, not a defect here.
    """
    # Aligned before any placement math: `Frame` also trims, but window origins are computed from
    # these locals, and evening only the frame would leave the last window one row past the oracle.
    width, height = (d - d % FRAME_ALIGNMENT for d in size)
    frame = Frame(image, 0, 0, width, height)
    for tile in tiles:
        if width < tile or height < tile:
            continue
        xs = sorted({0, (width - tile) // 2, width - tile})
        ys = sorted({0, (height - tile) // 2, height - tile})
        frame.windows.extend((tile, wx, wy) for wy in ys for wx in xs)
    return [frame] if frame.windows else []


# ------------------------------------------------------------------------------------ measurement


def to_product_surface(tensor: torch.Tensor) -> np.ndarray:
    """The product's own output stage: clamp to [0,1], round to 8 bits.

    Returns HxWx3 uint8. Both sides of every comparison pass through this, so what is measured is
    the surface the user actually sees, exactly as the Swift harness does it.
    """
    x = tensor.detach().float().clamp(0.0, 1.0)[0].permute(1, 2, 0)
    return (x * 255.0 + 0.5).floor().clamp(0, 255).to(torch.uint8).cpu().numpy()


def ring_profile(
    diff: np.ndarray, margins: list[int], scale: int
) -> dict[int, tuple[float, float, float]]:
    """Ring mean / ring max / retained-core mean at each margin, in INPUT-pixel margins.

    The ring at output distance d is the one-pixel frame d output pixels in from the border — the
    perimeter of `[d, H-1-d] x [d, W-1-d]`, which is the exact sample set `ContextMargin.ringProfile`
    sums. Margin is in input pixels, so d = margin * scale. Summed lazily through an integral image
    rather than four strided passes; `verify_geometry.py` pins the two against each other.
    """
    height, width, _ = diff.shape
    per_pixel = diff.astype(np.float64).sum(axis=2)
    integral = np.zeros((height + 1, width + 1), dtype=np.float64)
    np.cumsum(per_pixel, axis=0, out=integral[1:, 1:])
    np.cumsum(integral[1:, 1:], axis=1, out=integral[1:, 1:])

    def rect_sum(y0: int, x0: int, y1: int, x1: int) -> float:  # inclusive corners
        return float(
            integral[y1 + 1, x1 + 1] - integral[y0, x1 + 1] - integral[y1 + 1, x0] + integral[y0, x0]
        )

    out: dict[int, tuple[float, float, float]] = {}
    for margin in margins:
        d = margin * scale
        if d >= min(height, width) // 2:
            continue
        outer = rect_sum(d, d, height - 1 - d, width - 1 - d)
        span = min(height, width) - 2 * d
        inner = rect_sum(d + 1, d + 1, height - 2 - d, width - 2 - d) if span - 2 > 0 else 0.0
        ring_mean = (outer - inner) / ((4 * span - 4) * 3)
        core_mean = outer / (span * span * 3)
        ring_max = float(
            max(
                diff[d, d : width - d].max(),
                diff[height - 1 - d, d : width - d].max(),
                diff[d : height - d, d].max(),
                diff[d : height - d, width - 1 - d].max(),
            )
        )
        out[margin] = (ring_mean, ring_max, core_mean)
    return out


def crossing_and_margin(decay: dict[int, float]) -> tuple[int | None, int | None]:
    """First swept margin whose mean ring error is below 1/255, and that margin doubled.

    `ContextMargin.derive`, unchanged. The threshold is certification spec §4's and is not an
    argument.
    """
    for margin in sorted(decay):
        if decay[margin] < THRESHOLD_255:
            return margin, margin * SAFETY_FACTOR
    return None, None


# ------------------------------------------------------------------------------------------- main


def load_model(model_id: str, device: str):
    config = yaml.safe_load((ROOT / "tools" / "convert" / "models" / f"{model_id}.yaml").read_text())
    model, parameters = build_model(config, ROOT / config["weights"]["path"])
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, config, parameters


def image_tensor(source: Image.Image, rect: tuple[int, int, int, int], device: str):
    """Integer crop, no resampling, RGB float [0,1] NCHW — the pipeline's declared preprocessing."""
    x, y, w, h = rect
    array = np.asarray(source.crop((x, y, x + w, y + h)), dtype=np.uint8).copy()
    tensor = torch.from_numpy(array).to(device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return tensor, array


def resolve_sources(specs: list[str]) -> list[tuple[str, Path, int | None]]:
    """`(name, path, montage_cell_or_None)` for each requested source set or explicit path."""
    out: list[tuple[str, Path, int | None]] = []
    for spec in specs:
        if spec == "workloads":
            out += [(n, ROOT / f"Sources/TestFixtures/workloads/{n}.jpg", MONTAGE_CELL) for n in WORKLOADS]
        elif spec == "corpus":
            out += [
                (p.stem, p, None)
                for p in sorted((ROOT / "Sources/TestFixtures/corpus").glob("*.jpg"))
            ]
        else:
            name, _, rel = spec.partition("=")
            out.append((name, ROOT / (rel or name), None))
    return out


def run(args: argparse.Namespace) -> int:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    model, config, parameters = load_model(args.model, device)
    arch = config["architecture"]["type"]
    scale = int(config["scale_factor"])
    global FRAME_ALIGNMENT
    FRAME_ALIGNMENT = 2 if scale == 2 else 1
    radius = RECEPTIVE_FIELD_RADIUS[arch]

    if args.placement != "legacy" and args.oracle_context < radius:
        print(
            f"WARNING: oracle context {args.oracle_context} px < receptive-field radius {radius} px "
            f"for {arch}. The oracle is itself context-starved; the deficit is UNDERSTATED.",
            flush=True,
        )

    sources = resolve_sources(args.sources)
    frames: list[Frame] = []
    cells: dict[str, int | None] = {}
    for name, path, cell in sources:
        cells[name] = cell
        with Image.open(path) as im:
            size = im.size
        if args.placement == "legacy":
            frames.extend(legacy_frames(name, size, args.tile_size))
        else:
            frames.extend(
                plan_frames(name, size, args.tile_size, args.oracle_context,
                            (args.frame_w, args.frame_h), args.placement, cell)
            )

    # Two windows with byte-identical content are one probe, not two. C_48MP repeats 73 of its 165
    # montage cells verbatim, so without this the probe count overstates the evidence by 44% on
    # that image alone — the exact inflation this derivation exists to stop doing.
    duplicates = 0
    if args.dedupe_content:
        seen_content: set[str] = set()
        kept: list[Frame] = []
        for f in frames:
            with Image.open(next(p for n, p, _ in sources if n == f.image)) as source:
                source = source.convert("RGB")
                survivors = []
                for tile, wx, wy in f.windows:
                    box = (f.x + wx, f.y + wy, f.x + wx + tile, f.y + wy + tile)
                    digest = hashlib.sha1(np.asarray(source.crop(box), np.uint8).tobytes()).hexdigest()
                    if digest in seen_content:
                        duplicates += 1
                        continue
                    seen_content.add(digest)
                    survivors.append((tile, wx, wy))
            if survivors:
                kept.append(Frame(f.image, f.x, f.y, f.w, f.h, survivors))
        frames = kept
        print(f"[dedupe] dropped {duplicates} windows with byte-identical content", flush=True)

    if args.max_frames and len(frames) > args.max_frames:
        # Deterministic, seeded thinning — a subsample, never a cherry-pick. Recorded in the report.
        rng = np.random.default_rng(args.seed)
        frames = [frames[i] for i in sorted(rng.choice(len(frames), args.max_frames, replace=False))]

    total = sum(len(f.windows) for f in frames)
    if args.placement != "legacy":
        placed: dict[tuple[str, int], list[tuple[int, int]]] = {}
        for f in frames:
            for tile, wx, wy in f.windows:
                key = (f.image, tile)
                for px, py in placed.setdefault(key, []):
                    if abs((f.x + wx) - px) < tile and abs((f.y + wy) - py) < tile:
                        raise AssertionError(
                            f"overlapping windows in {f.image} tile {tile}: "
                            f"({f.x + wx},{f.y + wy}) vs ({px},{py})")
                placed[key].append((f.x + wx, f.y + wy))
    print(f"[plan] {len(frames)} frames, {total} windows, sources={[s[0] for s in sources]}", flush=True)
    if total == 0:
        print("REFUSED: no window can be placed under these constraints.")
        return 3

    margins = list(range(0, args.max_margin + 1, 2))
    probes: list[dict] = []
    started = time.time()
    opened: dict[str, Image.Image] = {}

    for index, frame in enumerate(frames):
        path = next(p for n, p, _ in sources if n == frame.image)
        if frame.image not in opened:
            opened.clear()
            opened[frame.image] = Image.open(path).convert("RGB")
        source = opened[frame.image]

        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        tensor, _ = image_tensor(source, (frame.x, frame.y, frame.w, frame.h), device)
        t0 = time.time()
        with torch.no_grad():
            oracle = to_product_surface(model(tensor))
        del tensor
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        print(
            f"[frame {index + 1}/{len(frames)}] {frame.image} @({frame.x},{frame.y}) "
            f"{frame.w}x{frame.h} oracle {time.time() - t0:.1f}s "
            f"({len(frame.windows)} windows, {time.time() - started:.0f}s elapsed)",
            flush=True,
        )

        for tile, wx, wy in frame.windows:
            window, array = image_tensor(source, (frame.x + wx, frame.y + wy, tile, tile), device)
            with torch.no_grad():
                predicted = to_product_surface(model(window))
            del window
            region = oracle[wy * scale : (wy + tile) * scale, wx * scale : (wx + tile) * scale]
            assert region.shape == predicted.shape, (region.shape, predicted.shape)
            diff = np.abs(region.astype(np.int16) - predicted.astype(np.int16)).astype(np.uint8)
            ox, oy = frame.x + wx, frame.y + wy
            cell = cells[frame.image]
            natural_x, straddle_x = seam_geometry(ox, tile, cell)
            natural_y, straddle_y = seam_geometry(oy, tile, cell)
            probes.append({
                "image": frame.image,
                "frame": [frame.x, frame.y, frame.w, frame.h],
                "origin": [ox, oy],
                "tile": tile,
                "oracle_context": [wx, frame.w - wx - tile, wy, frame.h - wy - tile],
                "natural_context": min(natural_x, natural_y) if cell else -1,
                "straddles_seam": bool(straddle_x or straddle_y),
                # Content identity, so an aggregate can report how many DISTINCT pictures it saw
                # rather than how many windows it drew. The montages repeat their cells.
                "content": hashlib.sha1(array.tobytes()).hexdigest()[:16],
                "rings": {str(m): v for m, v in ring_profile(diff, margins, scale).items()},
            })
        del oracle

    report: dict = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model,
        "architecture": arch,
        "parameters": parameters,
        "scale": scale,
        "receptive_field_radius_input_px": radius,
        "placement": args.placement,
        "oracle_context_px": None if args.placement == "legacy" else args.oracle_context,
        "oracle": (
            "legacy: whole source image (context-starved when the source is smaller than one "
            "receptive field)" if args.placement == "legacy"
            else f"frame supplying >= {args.oracle_context} px of context (receptive field {radius})"
        ),
        "montage_cell": MONTAGE_CELL,
        "measures": "PyTorch(tile) vs PyTorch(frame) — context influence alone, conversion cancels",
        "criterion": {
            "threshold_255": THRESHOLD_255, "statistic": "ring mean absolute error",
            "safety_factor": SAFETY_FACTOR, "surface": "8-bit product surface (clamp, floor(255x+0.5))",
            "source": "certification spec §4, unmoved",
        },
        "environment": {
            "torch": torch.__version__, "numpy": np.__version__,
            "python": platform.python_version(), "device": device,
            "platform": platform.platform(), "seed": args.seed,
        },
        "sources": [
            {"name": n, "path": str(p.relative_to(ROOT)), "size": list(Image.open(p).size),
             "montage_cell": c}
            for n, p, c in sources
        ],
        "argv": sys.argv[1:],
        "duplicate_windows_dropped": duplicates,
        "by_tile": {},
        "probes": probes,
    }

    for tile in args.tile_size:
        subset = [p for p in probes if p["tile"] == tile]
        if not subset:
            continue
        report["by_tile"][str(tile)] = summarise(subset, margins)
        block = report["by_tile"][str(tile)]
        print(
            f"[result] tile {tile}: {block['probes']} probes / {block['distinct_images']} images / "
            f"{block['distinct_contents']} distinct contents — crossing {block['crossing_mean']} "
            f"-> margin {block['derived_margin_mean']} "
            f"(worst-probe crossing {block['crossing_worst']} -> {block['derived_margin_worst']})",
            flush=True,
        )

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"[wrote] {os.path.relpath(out, ROOT)}  ({time.time() - started:.0f}s total)")
    return 0


def summarise(subset: list[dict], margins: list[int]) -> dict:
    """Aggregate one tile shape. Shapes are never blended; that is the caller's guarantee too."""
    images = sorted({p["image"] for p in subset})
    mean_decay, worst_decay = {}, {}
    for m in margins:
        values = [p["rings"][str(m)][0] for p in subset if str(m) in p["rings"]]
        if values:
            mean_decay[m] = float(np.mean(values))
            worst_decay[m] = float(np.max(values))
    crossing, margin = crossing_and_margin(mean_decay)
    worst_crossing, worst_margin = crossing_and_margin(worst_decay)
    per_image = {}
    for name in images:
        rows = [p for p in subset if p["image"] == name]
        decay = {
            m: float(np.mean([r["rings"][str(m)][0] for r in rows if str(m) in r["rings"]]))
            for m in margins if any(str(m) in r["rings"] for r in rows)
        }
        c, mm = crossing_and_margin(decay)
        per_image[name] = {"probes": len(rows), "crossing": c, "margin": mm,
                           "floor": min(decay.values()) if decay else None,
                           "decay": {str(k): v for k, v in decay.items()}}
    return {
        "probes": len(subset),
        "distinct_images": len(images),
        "distinct_contents": len({p["content"] for p in subset}),
        "images": images,
        "frames": len({(p["image"], tuple(p["frame"])) for p in subset}),
        "seam_straddling_probes": sum(1 for p in subset if p["straddles_seam"]),
        "min_natural_context": min((p["natural_context"] for p in subset), default=None),
        "mean_decay": {str(k): v for k, v in mean_decay.items()},
        "worst_decay": {str(k): v for k, v in worst_decay.items()},
        "crossing_mean": crossing, "derived_margin_mean": margin,
        "crossing_worst": worst_crossing, "derived_margin_worst": worst_margin,
        "per_image": per_image,
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="RealESRGAN_x4plus")
    p.add_argument("--tile-size", type=int, nargs="+", default=[256])
    p.add_argument("--max-margin", type=int, default=126,
                   help="sweep ceiling; 126 is the largest margin tile 256 can express "
                        "(MemoryBudget.plannedMargin caps at (tile-1)/2 = 127)")
    p.add_argument("--oracle-context", type=int, default=384,
                   help="context the oracle frame must supply around every window")
    p.add_argument("--placement", choices=["cell-centred", "grid", "legacy"], default="cell-centred")
    p.add_argument("--sources", nargs="+", default=["workloads"],
                   help="'workloads', 'corpus', or name=relative/path.jpg")
    p.add_argument("--frame-w", type=int, default=1024, help="grid placement: oracle frame cap")
    p.add_argument("--frame-h", type=int, default=1024)
    p.add_argument("--max-frames", type=int, default=0, help="seeded thinning; 0 = no cap")
    p.add_argument("--dedupe-content", action=argparse.BooleanOptionalAction, default=True,
                   help="drop windows whose pixels are byte-identical to one already planned")
    p.add_argument("--device", default="mps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
