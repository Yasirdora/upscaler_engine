#!/usr/bin/env python3
"""Comparison harness — Level A/B certification (certification spec §4 and §11).

What this tool answers: does the *product* reproduce the reference oracle closely enough to
certify? Not "can Python reproduce it" — the candidate is always the shipping path,
`UpscaleCLI upscale-tile … --emit-npy`, driven as a subprocess with its compute units pinned. A
Python-side coremltools comparison would certify a code path nobody ships.

Compute units are mandatory and recorded. Core ML numerics were measured diverging by backend
(cpuOnly differed from `all` by up to 14/255 on the same tile), so an unpinned run compares two
things that were never the same thing. `--compute-units` has no default, on purpose.

Scope: the converted artifact is fixed-shape (256×256), and this harness does not tile, so
this harness certifies **tile-level** equivalence. It derives a deterministic tile set from the
corpus, runs the oracle and the candidate over identical tile PNGs, and compares the tensors.
Whole-image certification arrives with tiling.

Geometry, precision and the pipeline come from `models/<id>/manifest.json`; thresholds come from
`config.json`. Neither is hardcoded here — a harness that carries its own idea of the model can
certify something the project never converted.

Import policy: numpy and the standard library at module scope. torch, PIL,
yaml and lpips are imported inside the functions that use them, so `pytest tools/` collects in an
environment that has only pytest and numpy.

Attribution: Real-ESRGAN is BSD-3-Clause. scikit-image (BSD-3-Clause) is the external oracle for
the metric tests, and `lpips` (BSD-2-Clause) is offline evaluation tooling that never ships. No
code from Upscayl (AGPL-3.0) or any GPL/AGPL source is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import plistlib
import random
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "tools" / "convert"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "reference"))

import manifest_emitter  # noqa: E402
import metrics  # noqa: E402
import run_reference  # noqa: E402

CONFIG_PATH = HERE / "config.json"
COMPUTE_UNITS = ("all", "cpuAndGPU", "cpuAndNeuralEngine", "cpuOnly")
COREML_VERSION_PLIST = Path("/System/Library/Frameworks/CoreML.framework/Resources/version.plist")

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_STRUCTURAL_ERROR = 2

#: Recomputation in `--check` runs on the same bytes, so it should reproduce the recorded value
#: exactly; the tolerance covers JSON float formatting only, not numeric drift.
RECHECK_TOLERANCE = 1e-9


class CertifyError(RuntimeError):
    """A condition that must stop the harness rather than be worked around."""


def log(message: str = "") -> None:
    print(message, flush=True)


def relative(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


# --- environment attribution --------------------------------------------------------------
def _sysctl(name: str) -> str:
    try:
        return subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def device_identity() -> str:
    return f"{_sysctl('hw.model')} ({platform.machine()})"


def os_identity() -> str:
    version = platform.mac_ver()[0] or platform.release()
    try:
        build = subprocess.run(
            ["sw_vers", "-buildVersion"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        build = "unknown"
    return f"macOS {version} ({build})"


def coreml_identity() -> str:
    """The Core ML runtime that produced the candidate — not coremltools, which converted it.

    Apple ships Core ML with the OS and a Core ML update can change numerics, so the framework's
    own version is the field that answers "why did this output change?" (certification spec §11).
    """
    try:
        data = plistlib.loads(COREML_VERSION_PLIST.read_bytes())
    except (OSError, plistlib.InvalidFileException):
        return "unknown"
    return (
        f"CoreML.framework {data.get('CFBundleVersion', '?')} "
        f"(build {data.get('BuildVersion', '?')})"
    )


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    commit = result.stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()
    return f"{commit}-dirty" if dirty else commit


def swift_identity() -> str:
    try:
        result = subprocess.run(
            ["swift", "--version"], capture_output=True, text=True, check=True, cwd=REPO_ROOT
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return " ".join(result.stdout.strip().splitlines()[:1])


# --- inputs -------------------------------------------------------------------------------
def load_config(path: Path = CONFIG_PATH) -> dict:
    if not path.exists():
        raise CertifyError(f"harness config not found at {relative(path)}")
    try:
        config = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise CertifyError(f"{relative(path)}: not valid JSON: {error}") from error
    if "thresholds" not in config:
        raise CertifyError(f"{relative(path)}: no 'thresholds' block")
    return config


def load_model_manifest(models_root: Path, model_id: str) -> dict:
    """The model's manifest is the authority on geometry, precision and pipeline."""
    import manifest as conversion_manifest  # noqa: PLC0415  (stdlib-only, but keep the seam local)

    path = models_root / model_id / "manifest.json"
    if not path.exists():
        raise CertifyError(f"{model_id}: no model manifest at {relative(path)}")
    try:
        return conversion_manifest.validate(path)
    except conversion_manifest.ManifestError as error:
        raise CertifyError(f"{model_id}: model manifest is invalid: {error}") from error


def geometry_from_manifest(model_manifest: dict) -> tuple[int, int, int]:
    """(tile size, scale factor, output size) — read, never assumed."""
    shape = model_manifest["conversion"]["input_shape"]
    output_shape = model_manifest["conversion"]["output_shape"]
    if shape[2] != shape[3]:
        raise CertifyError(f"non-square input shape {shape}: this harness derives square tiles")
    tile = shape[2]
    scale = model_manifest["scale_factor"]
    if output_shape[2] != tile * scale or output_shape[3] != tile * scale:
        raise CertifyError(
            f"manifest disagrees with itself: input {shape} x{scale} should give "
            f"{[1, 3, tile * scale, tile * scale]}, manifest says {output_shape}"
        )
    return tile, scale, tile * scale


def tile_set_label(sources: list[dict], entries_count: int) -> str:
    """A name for what was actually certified, recorded in the manifest.

    A run over a starter corpus must never be citable as full-corpus
    evidence, so the manifest states the tile set's size and how many source images it came from
    rather than leaving a reader to count entries.
    """
    contributing = sum(1 for source in sources if source["tile_count"] > 0)
    excluded = sum(1 for source in sources if source["tile_count"] == 0)
    label = (
        f"starter ({entries_count} tiles, {contributing} source image"
        f"{'' if contributing == 1 else 's'})"
    )
    if excluded:
        label += f"; {excluded} corpus image(s) excluded as smaller than one tile"
    return label


def corpus_version(images: list[Path]) -> str:
    """A content address for the corpus: the same images in any order give the same version, and
    one changed byte gives a different one."""
    digest = hashlib.sha256()
    for path in sorted(images):
        digest.update(relative(path).encode())
        digest.update(b"\0")
        digest.update(run_reference.sha256_of_file(path).encode())
    return f"sha256:{digest.hexdigest()[:16]} ({len(images)} image(s))"


# --- tile derivation ----------------------------------------------------------------------
def tile_positions(width: int, height: int, tile: int, stride: int) -> list[tuple[int, int]]:
    """Centre crop, four corners, and stride-`stride` interior crops where the image allows.

    Deduplicated and canonically sorted; an image smaller than one tile yields nothing, because
    padding it would certify a padding policy this project has not chosen yet (that is tiling,
    certification spec §5) and resizing it would certify an input the corpus does not contain.
    """
    if width < tile or height < tile:
        return []
    max_x, max_y = width - tile, height - tile
    positions = {
        (max_x // 2, max_y // 2),
        (0, 0),
        (max_x, 0),
        (0, max_y),
        (max_x, max_y),
    }
    positions.update(
        (x, y) for y in range(0, max_y + 1, stride) for x in range(0, max_x + 1, stride)
    )
    return sorted(positions)


def apply_postprocessing(array: np.ndarray, postprocessing: str) -> np.ndarray:
    """Put the oracle through the same postprocessing stage the product applies.

    The model manifest declares `postprocessing: "clamp [0,1]"`, and `CoreMLBackend` performs it
    before it hands back a pixel buffer. `run_reference.write_tensor` deliberately stores the raw
    forward pass, so comparing that tensor against the product's output scores the *clamp* as
    model error: this model overshoots [0,1] on a fraction of a percent of pixels, and those
    pixels then dominate max-abs identically on every backend — which is exactly what a real
    numerical divergence would not do. Both sides are compared after the same declared stage.
    """
    if postprocessing != run_reference.SUPPORTED_POSTPROCESSING:
        raise CertifyError(
            f"manifest declares postprocessing {postprocessing!r}, which this harness does not "
            f"implement (it implements {run_reference.SUPPORTED_POSTPROCESSING!r}). Comparing "
            "without applying the declared stage would score the pipeline, not the model."
        )
    return np.clip(array, 0.0, 1.0).astype(np.float32)


def oracle_tensor(model, tile_png: Path, postprocessing: str) -> tuple[np.ndarray, float]:
    """The oracle side, start to finish: forward pass, then the manifest's declared postprocessing.

    One named path exists so that "the pipeline the manifest declares" and "the pipeline the
    harness executes" cannot drift apart silently. `run_certify` never calls `run_reference.forward`
    itself, and a test asserts that; the executed clamp is verified by running this function, not
    by reading the source.
    """
    raw = run_reference.forward(model, run_reference.preprocess(tile_png))
    return apply_postprocessing(raw, postprocessing), out_of_range_fraction(raw)


def out_of_range_fraction(array: np.ndarray) -> float:
    """How much of the raw forward pass falls outside [0,1] — what postprocessing absorbs."""
    return float(((array < 0.0) | (array > 1.0)).mean())


def image_size(path: Path) -> tuple[int, int]:
    from PIL import Image  # noqa: PLC0415

    with Image.open(path) as image:
        return image.size


def derive_tiles(images: list[Path], tile: int, stride: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Returns (tile records in seeded order, per-source-image coverage records)."""
    records: list[dict] = []
    sources: list[dict] = []
    for path in sorted(images):
        width, height = image_size(path)
        positions = tile_positions(width, height, tile, stride)
        sources.append(
            {
                "path": relative(path),
                "sha256": run_reference.sha256_of_file(path),
                "width": width,
                "height": height,
                "tile_count": len(positions),
                "excluded_reason": (
                    None
                    if positions
                    else f"smaller than one {tile}x{tile} tile ({width}x{height}); "
                    "padding or resizing would certify a policy this project has not chosen"
                ),
            }
        )
        for x, y in positions:
            records.append({"source": path, "x": x, "y": y, "width": tile, "height": tile})

    # Seeded ordering: canonical first, then one deterministic shuffle. The order is recorded in
    # the manifest and asserted by the test suite, so a run is reproducible tile for tile.
    records.sort(key=lambda record: (relative(record["source"]), record["y"], record["x"]))
    random.Random(seed).shuffle(records)
    return records, sources


def tile_name(record: dict) -> str:
    return f"{record['source'].stem}_x{record['x']}_y{record['y']}"


def crop_tile(source: Path, x: int, y: int, size: int, destination: Path) -> None:
    """Both the oracle and the product read this PNG, so the crop happens exactly once."""
    from PIL import Image  # noqa: PLC0415

    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.convert("RGB").crop((x, y, x + size, y + size)).save(destination, format="PNG")


# --- candidate: the product path ----------------------------------------------------------
def candidate_command(
    mlpackage: Path, tile_png: Path, out_png: Path, out_npy: Path, units: str, tile_size: int
) -> list[str]:
    """The CLI invocation that produces the candidate tile.

    `--tile-size` is passed explicitly and is not optional. Omitting it let the CLI fall back to
    its 256 default, so certifying any other shape rejected every tile with exit 2 -- Level A at
    512 was unreachable for that reason alone, which is why the x4v3 bake-off could measure speed
    at 512 and fidelity only at 256.
    """
    return [
        "swift", "run", "UpscaleCLI", "upscale-tile",
        "--model", str(mlpackage),
        "--input", str(tile_png),
        "--output", str(out_png),
        "--compute-units", units,
        "--tile-size", str(tile_size),
        "--emit-npy", str(out_npy),
    ]


def run_candidate(command: list[str]) -> None:
    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise CertifyError(
            "the product path failed — certification cannot proceed\n"
            f"  command : {' '.join(command)}\n"
            f"  exit    : {result.returncode}\n"
            f"  stdout  : {result.stdout.strip()}\n"
            f"  stderr  : {result.stderr.strip()}"
        )


def weights_artifact_path(models_root: Path, model_id: str) -> Path:
    """Where the weights live, according to the fetcher that put them there.

    tools/artifacts/fetch.py owns this derivation (it comes from the manifest's URL); guessing
    `<model_id>.pth` here would be a second source of truth waiting to disagree with the first.
    """
    import fetch  # noqa: PLC0415  (stdlib-only, imported through run_reference's sys.path entry)

    manifest = fetch.read_manifest(models_root, model_id)
    url, _ = fetch.weights_spec(manifest, model_id)
    return fetch.weights_path(models_root, model_id, url)


def load_tensor(path: Path, expected_shape: list[int]) -> np.ndarray:
    if not path.exists():
        raise CertifyError(f"tensor missing: {relative(path)}")
    array = np.load(path, allow_pickle=False)
    if array.dtype != np.float32:
        raise CertifyError(f"{relative(path)}: expected float32, got {array.dtype}")
    if list(array.shape) != list(expected_shape):
        raise CertifyError(
            f"{relative(path)}: expected shape {expected_shape}, got {list(array.shape)}"
        )
    return array


# --- metrics ------------------------------------------------------------------------------
def tile_metrics(reference: np.ndarray, candidate: np.ndarray, tolerance: float) -> dict:
    value = metrics.psnr(reference, candidate)
    return {
        # Infinite PSNR means bit-identical tensors; recorded as null rather than a fabricated
        # ceiling that would be indistinguishable from a real measurement.
        "psnr_db": None if math.isinf(value) else value,
        "mae": metrics.mae(reference, candidate),
        "max_abs": metrics.max_abs_error(reference, candidate),
        "within_tolerance_fraction": metrics.within_tolerance_fraction(
            reference, candidate, tolerance
        ),
        "ssim": metrics.ssim(reference, candidate),
        "lpips": metrics.lpips_distance(reference, candidate),
    }


def _finite_psnr(entries: list[dict]) -> list[float]:
    return [e["metrics"]["psnr_db"] for e in entries if e["metrics"]["psnr_db"] is not None]


def aggregate(entries: list[dict], thresholds: dict) -> dict:
    """Level A and Level B aggregates, plus the verdict. Every tile is the same size by
    construction, so a mean of per-tile fractions is an exact pooled fraction."""
    if not entries:
        raise CertifyError(
            "no tiles were derived from the corpus — there is nothing to certify.\n"
            "  every corpus image is smaller than one tile; add corpus images at least as large "
            "as the model's fixed input shape"
        )

    finite = _finite_psnr(entries)
    exact = len(entries) - len(finite)
    psnr_median = statistics.median(finite) if finite else None

    fractions = [e["metrics"]["within_tolerance_fraction"] for e in entries]
    pooled_fraction = float(np.mean(fractions))
    ssims = [e["metrics"]["ssim"] for e in entries]
    lpipses = [e["metrics"]["lpips"] for e in entries]

    level_a_thresholds = thresholds["level_a"]
    level_b_thresholds = thresholds["level_b"]

    psnr_ok = psnr_median is None or psnr_median >= level_a_thresholds["psnr_db_median_min"]
    fraction_ok = pooled_fraction >= level_a_thresholds["within_tolerance_fraction_min"]
    passed = psnr_ok and fraction_ok

    level_a = {
        "thresholds": {
            "psnr_db_median_min": level_a_thresholds["psnr_db_median_min"],
            "max_abs_tolerance_255": level_a_thresholds["max_abs_tolerance_255"],
            "within_tolerance_fraction_min": level_a_thresholds["within_tolerance_fraction_min"],
        },
        "tile_count": len(entries),
        "exact_match_tiles": exact,
        "psnr_db_median": psnr_median,
        "psnr_db_min": min(finite) if finite else None,
        "mae_mean": float(np.mean([e["metrics"]["mae"] for e in entries])),
        "max_abs_worst": max(e["metrics"]["max_abs"] for e in entries),
        "max_abs_worst_255": max(e["metrics"]["max_abs"] for e in entries) * metrics.BYTE,
        "within_tolerance_fraction": pooled_fraction,
        "within_tolerance_fraction_worst_tile": min(fractions),
        "psnr_criterion_met": psnr_ok,
        "max_abs_criterion_met": fraction_ok,
        "status": "PASS" if passed else "FAIL",
    }
    level_b = {
        "gated": bool(level_b_thresholds.get("gated", False)),
        "reference_thresholds": {"ssim_median_min": level_b_thresholds["ssim_median_min"]},
        "ssim_median": statistics.median(ssims),
        "ssim_min": min(ssims),
        "ssim_reference_threshold_met": statistics.median(ssims)
        >= level_b_thresholds["ssim_median_min"],
        "lpips_net": metrics.LPIPS_NET,
        "lpips_median": statistics.median(lpipses),
        "lpips_max": max(lpipses),
        "status": "RECORDED",
    }
    return {"level_a": level_a, "level_b": level_b, "passed": passed}


def build_results(level_a: dict, level_b: dict) -> dict:
    return {
        "level_a": level_a,
        "level_b": level_b,
        "level_c": {
            "status": "NOT_AUTOMATED",
            "note": "Artifact inspection at 100/200/400% is a human checklist (certification "
            "spec §4). Explicitly out of scope for this harness.",
        },
        "performance": {
            "status": "NOT_MEASURED",
            "note": "Timings and thermals belong to benchmarking (certification spec §7), which "
            "own the methodology. No timing is recorded here.",
        },
        "energy": {"value": None, "kind": "unavailable"},
    }


# --- report -------------------------------------------------------------------------------
def _format_psnr(value: float | None) -> str:
    return "  exact  " if value is None else f"{value:8.3f}"


def _decibels(value: float | None) -> str:
    """Format a dB aggregate that may legitimately have no finite value.

    `psnr_db_median` is None when every tile is bit-identical to the oracle, and NaN when the
    diagnostic has nothing finite to take a median of. Both are real states of the system; a
    formatter that assumes a float turns them into a crash, and a formatter that prints 0.0 would
    turn them into a lie.
    """
    if value is None:
        return "exact (bit-identical to the oracle)"
    if math.isnan(value):
        return "unavailable"
    return f"{value:.4f} dB"


def _row(label: str, value: str) -> str:
    """One padding rule for every summary row, so the report's columns cannot drift apart."""
    return f"  {label:<26}: {value}"


def write_report(path: Path, context: dict) -> None:
    entries = context["entries"]
    level_a = context["results"]["level_a"]
    level_b = context["results"]["level_b"]
    lines: list[str] = []
    add = lines.append

    add("=" * 96)
    add(f"CERTIFICATION REPORT — {context['model_id']} — Level A/B (certification spec §4)")
    add("=" * 96)
    add(f"certification id : {context['certification_id']}")
    add(f"date             : {context['date']}")
    add(f"status           : {context['status']}")
    add("")
    add(f"compute units    : {context['compute_units']}   (pinned — backends diverge)")
    add(f"device           : {context['device']}")
    add(f"os               : {context['os_version']}")
    add(f"core ml          : {context['coreml_version']}")
    add(f"swift            : {context['swift']}")
    add(f"weights sha256   : {context['weights_sha256']}")
    add(f"mlpackage sha256 : {context['mlpackage_sha256']}")
    add(f"corpus           : {context['corpus_version']}")
    add(f"repo commit      : {context['commit']}")
    add("")
    add("candidate (the product path, one invocation per tile):")
    add(f"  {context['candidate_command']}")
    add("oracle:")
    add(f"  {context['reference_implementation']}")
    add("")

    add("-" * 96)
    add("CORPUS COVERAGE")
    add("-" * 96)
    for source in context["sources"]:
        if source["excluded_reason"]:
            add(f"  EXCLUDED  {source['path']}  {source['width']}x{source['height']}")
            add(f"            {source['excluded_reason']}")
        else:
            add(
                f"  {source['tile_count']:3d} tiles  {source['path']}  "
                f"{source['width']}x{source['height']}"
            )
    add("")

    add("-" * 96)
    add(f"PER-TILE METRICS  (tile order is seeded, seed={context['order_seed']})")
    add("-" * 96)
    add(
        f"{'#':>3}  {'tile':<34}{'PSNR dB':>9}  {'MAE/255':>8}  {'maxabs/255':>10}  "
        f"{'<=tol %':>8}  {'SSIM':>8}  {'LPIPS':>8}"
    )
    for entry in entries:
        values = entry["metrics"]
        add(
            f"{entry['index']:>3}  {Path(entry['tile_png']).stem:<34}"
            f"{_format_psnr(values['psnr_db'])}  "
            f"{values['mae'] * metrics.BYTE:8.4f}  {values['max_abs'] * metrics.BYTE:10.4f}  "
            f"{values['within_tolerance_fraction'] * 100:8.4f}  {values['ssim']:8.6f}  "
            f"{values['lpips']:8.6f}"
        )
    add("")

    add("-" * 96)
    add("LEVEL A — numerical correctness (gated)")
    add("-" * 96)
    tolerance_255 = level_a["thresholds"]["max_abs_tolerance_255"]
    add(_row("tiles compared", str(level_a["tile_count"])))
    add(
        _row(
            "PSNR median",
            f"{_decibels(level_a['psnr_db_median'])}   "
            f"(threshold >= {level_a['thresholds']['psnr_db_median_min']} dB)   "
            f"{'MET' if level_a['psnr_criterion_met'] else 'NOT MET'}",
        )
    )
    add(_row("PSNR worst tile", _decibels(level_a["psnr_db_min"])))
    add(_row("MAE mean", f"{level_a['mae_mean'] * metrics.BYTE:.4f}/255"))
    add(_row("max abs worst tile", f"{level_a['max_abs_worst_255']:.4f}/255"))
    add(
        _row(
            f"pixels within {tolerance_255}/255",
            f"{level_a['within_tolerance_fraction'] * 100:.4f} %   "
            f"(threshold >= {level_a['thresholds']['within_tolerance_fraction_min'] * 100} %)   "
            f"{'MET' if level_a['max_abs_criterion_met'] else 'NOT MET'}",
        )
    )
    add(
        _row(
            "worst tile within tol",
            f"{level_a['within_tolerance_fraction_worst_tile'] * 100:.4f} %",
        )
    )
    add(_row("LEVEL A", level_a["status"]))
    add("")

    add("-" * 96)
    add("LEVEL B — perceptual similarity (recorded, not gated)")
    add("-" * 96)
    add(
        _row(
            "SSIM median",
            f"{level_b['ssim_median']:.6f}   "
            f"(spec §4 reference >= {level_b['reference_thresholds']['ssim_median_min']})   "
            f"{'met' if level_b['ssim_reference_threshold_met'] else 'NOT met'}",
        )
    )
    add(_row("SSIM worst tile", f"{level_b['ssim_min']:.6f}"))
    add(_row(f"LPIPS ({level_b['lpips_net']}) median", f"{level_b['lpips_median']:.6f}"))
    add(_row(f"LPIPS ({level_b['lpips_net']}) worst tile", f"{level_b['lpips_max']:.6f}"))
    add("  LPIPS is a signal, never a verdict (certification spec §4). Level D is the authority.")
    add("")

    add("-" * 96)
    add("HUMAN VISUAL INSPECTION — required before a result is cited")
    add("-" * 96)
    worst = context["worst_tile"]
    add(f"  worst-PSNR tile : {worst['tile_png']}")
    add(f"  oracle  (8-bit) : {worst['reference_png']}")
    add(f"  product (8-bit) : {worst['candidate_png']}")
    add("  Open the last two at 100% and compare. Numbers do not settle Level C or D.")
    add("")
    add("  MANDATORY Level C inspection at 100% — the three worst tiles by max-abs error")
    add("  (the max-abs tail is a Level C assignment, not a failure):")
    for position, tile in enumerate(context["level_c_queue"], start=1):
        add(f"    {position}. {tile['name']}  max abs {tile['max_abs_255']:.4f}/255")
        add(f"       oracle  : {tile['reference_png']}")
        add(f"       product : {tile['candidate_png']}")
    add("")
    add("  A secondary bound on max-abs *magnitude* (the criterion bounds only how many pixels")
    add("  may deviate, not how far) is logged as material for a spec change. Thresholds tighten on")
    add("  evidence, never loosen.")
    add("")

    add("-" * 96)
    add("DIAGNOSTIC — candidate tensor precision (not a gated metric)")
    add("-" * 96)
    diagnostic = context["diagnostic"]
    add(
        f"  Both sides are compared after the manifest's declared postprocessing "
        f"({diagnostic['postprocessing']!r});"
    )
    add(
        f"  the raw forward pass leaves {diagnostic['overshoot'] * 100:.4f} % of values outside "
        "[0,1] on the median tile,"
    )
    add("  and that stage absorbs them on both sides rather than scoring them as model error.")
    add("")
    measured, quantised = diagnostic["as_measured"], diagnostic["quantised"]
    unavailable = measured is None or quantised is None or math.isnan(quantised)
    budget = None if unavailable else quantised - measured
    add(
        "  Level A thresholds apply to the 8-bit product surface; "
        + ("no finite budget to report" if budget is None else f"{budget:.2f} dB")
        + " of the observed"
    )
    add("  gap is attributable to the surface itself.")
    add("")
    add(
        "  `--emit-npy` dumps the 8-bit product surface widened to float32 — what"
        " `InferenceBackend.predict`"
    )
    add(
        "  returns and what the user receives (certifying that surface is"
        " certifying"
    )
    add(
        "  the product). The oracle is float32. Quantising the oracle the same way isolates the"
        " two:"
    )
    add("")
    add(f"  oracle vs product surface        : PSNR median {_decibels(measured)} (the gated figure)")
    add(
        f"  oracle quantised vs product      : PSNR median {_decibels(quantised)} "
        "(inference error alone)"
    )
    add(
        "  attributable to the 8-bit surface: "
        + ("unavailable" if budget is None else f"{budget:.4f} dB")
    )
    add("  method: the oracle is put through the product's own output stage — clamp to [0,1],")
    add("  round to 8 bits, widen back to float32 — and re-measured against the same candidate.")
    add("")
    add("=" * 96)
    add(f"STATUS: {context['status']}")
    add("=" * 96)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


# --- run ----------------------------------------------------------------------------------
def certification_paths(models_root: Path, model_id: str, units: str) -> dict:
    root = models_root / model_id / "certification"
    return {
        "root": root,
        "tiles": root / "tiles",
        "reference": root / "reference",
        "run": root / units,
        "candidate": root / units / "candidate",
        "manifest": root / units / "CERTIFICATION_MANIFEST.json",
        "report": root / units / "report.txt",
    }


def run_certify(model_id: str, units: str, models_root: Path, corpus_dir: Path, config: dict) -> int:
    model_manifest = load_model_manifest(models_root, model_id)
    tile_size, scale, output_size = geometry_from_manifest(model_manifest)
    _, postprocessing = run_reference.pipeline_from_manifest(model_manifest, model_id)

    mlpackage = models_root / model_id / "output" / f"{model_id}_{tile_size}x{tile_size}.mlpackage"
    if not mlpackage.exists():
        raise CertifyError(
            f"{model_id}: no converted artifact at {relative(mlpackage)}\n"
            "  run: python tools/convert/convert.py --model " + model_id
        )

    images = run_reference.discover_corpus(corpus_dir)
    if not images:
        raise CertifyError(
            f"corpus is empty: no images under {relative(corpus_dir)} — nothing to certify"
        )

    paths = certification_paths(models_root, model_id, units)
    thresholds = config["thresholds"]
    tolerance = thresholds["level_a"]["max_abs_tolerance_255"] / metrics.BYTE
    seed = config["tiles"]["order_seed"]
    stride = config["tiles"]["interior_stride"]

    log(f"[corpus] {len(images)} image(s) under {relative(corpus_dir)}")
    records, sources = derive_tiles(images, tile_size, stride, seed)
    for source in sources:
        if source["excluded_reason"]:
            log(f"  EXCLUDED {source['path']} ({source['width']}x{source['height']}): "
                f"{source['excluded_reason']}")
    log(f"[tiles] {len(records)} tile(s) derived, seeded order (seed={seed}, stride={stride})")
    if not records:
        raise CertifyError(
            "no tiles were derived from the corpus — there is nothing to certify.\n"
            f"  every corpus image is smaller than one {tile_size}x{tile_size} tile"
        )

    model, weights_sha, _, _ = run_reference.load_model_for(model_id)
    expected_shape = [1, 3, output_size, output_size]

    entries: list[dict] = []
    quantised_psnr: list[float] = []
    overshoot: list[float] = []
    for index, record in enumerate(records):
        name = tile_name(record)
        tile_png = paths["tiles"] / f"{name}.png"
        reference_npy = paths["reference"] / f"{name}.npy"
        reference_png = paths["reference"] / f"{name}.png"
        candidate_npy = paths["candidate"] / f"{name}.npy"
        candidate_png = paths["candidate"] / f"{name}.png"

        crop_tile(record["source"], record["x"], record["y"], tile_size, tile_png)

        reference, raw_overshoot = oracle_tensor(model, tile_png, postprocessing)
        overshoot.append(raw_overshoot)
        run_reference.write_outputs(reference, reference_npy, reference_png)

        candidate_npy.parent.mkdir(parents=True, exist_ok=True)
        run_candidate(
            candidate_command(
                mlpackage, tile_png, candidate_png, candidate_npy, units, tile_size
            )
        )

        reference_loaded = load_tensor(reference_npy, expected_shape)
        candidate = load_tensor(candidate_npy, expected_shape)
        values = tile_metrics(reference_loaded, candidate, tolerance)

        # Diagnostic only: the same comparison with the oracle put through the product's
        # 8-bit output stage, which isolates inference error from output quantisation.
        quantised = np.floor(np.clip(reference_loaded, 0.0, 1.0) * 255.0 + 0.5).astype(
            np.float32
        ) / np.float32(255.0)
        quantised_psnr.append(metrics.psnr(quantised, candidate))

        entries.append(
            {
                "index": index,
                "source_image": relative(record["source"]),
                "source_sha256": run_reference.sha256_of_file(record["source"]),
                "x": record["x"],
                "y": record["y"],
                "width": record["width"],
                "height": record["height"],
                "tile_png": relative(tile_png),
                "tile_sha256": run_reference.sha256_of_file(tile_png),
                "reference_npy": relative(reference_npy),
                "reference_sha256": run_reference.sha256_of_file(reference_npy),
                "candidate_npy": relative(candidate_npy),
                "candidate_sha256": run_reference.sha256_of_file(candidate_npy),
                "metrics": values,
            }
        )
        log(
            f"  [{index + 1:>3}/{len(records)}] {name:<34} "
            f"PSNR {_format_psnr(values['psnr_db']).strip():>8} dB  "
            f"maxabs {values['max_abs'] * metrics.BYTE:6.3f}/255  "
            f"SSIM {values['ssim']:.6f}  LPIPS {values['lpips']:.6f}"
        )

    summary = aggregate(entries, thresholds)
    results = build_results(summary["level_a"], summary["level_b"])
    status = "PASS" if summary["passed"] else "FAIL"

    candidate_hashes = [entry["candidate_sha256"] for entry in entries]
    identity = hashlib.sha256("".join(candidate_hashes).encode()).hexdigest()[:12]
    stamp = datetime.now(timezone.utc)
    reference_implementation = (
        "tools/reference/run_reference.py + tools/convert/model_loader.py "
        f"(RRDBNet, torch {run_reference.current_environment()['torch']}); "
        f"postprocessing {postprocessing!r} applied per models/{model_id}/manifest.json, "
        "matching the stage CoreMLBackend applies to the candidate"
    )

    manifest = manifest_emitter.build(
        certification_id=f"{model_id}-{units}-{stamp.strftime('%Y%m%dT%H%M%SZ')}-{identity}",
        date=stamp.isoformat(timespec="seconds"),
        model_id=model_id,
        weights_sha256=weights_sha,
        mlpackage_sha256=manifest_emitter.sha256_of_path(mlpackage),
        conversion={
            key: model_manifest["conversion"][key]
            for key in ("script_version", "coremltools", "python", "precision", "input_shape", "parameters")
        },
        reference={
            "implementation": reference_implementation,
            "commit": git_commit(),
            "output_hashes": [entry["reference_sha256"] for entry in entries],
        },
        coreml_output_hashes=candidate_hashes,
        corpus_version=corpus_version(images),
        device=device_identity(),
        os_version=os_identity(),
        coreml_version=coreml_identity(),
        compute_units=units,
        tiles={
            "tile_size": tile_size,
            "tile_set": tile_set_label(sources, len(entries)),
            "order_seed": seed,
            "derivation": (
                f"centre crop + four corners + stride-{stride} interior crops per corpus image, "
                "deduplicated, canonically sorted, then shuffled with the recorded seed"
            ),
            "source_images": sources,
            "entries": entries,
        },
        results=results,
        status=status,
    )
    manifest_emitter.write(paths["manifest"], manifest)

    worst_entry = min(
        entries,
        key=lambda entry: (
            math.inf if entry["metrics"]["psnr_db"] is None else entry["metrics"]["psnr_db"]
        ),
    )
    worst_name = Path(worst_entry["tile_png"]).stem
    worst_by_max_abs = sorted(entries, key=lambda entry: -entry["metrics"]["max_abs"])[:3]
    write_report(
        paths["report"],
        {
            "model_id": model_id,
            "certification_id": manifest["certification_id"],
            "date": manifest["date"],
            "status": status,
            "compute_units": units,
            "device": manifest["device"],
            "os_version": manifest["os_version"],
            "coreml_version": manifest["coreml_version"],
            "swift": swift_identity(),
            "weights_sha256": weights_sha,
            "mlpackage_sha256": manifest["mlpackage_sha256"],
            "corpus_version": manifest["corpus_version"],
            "commit": manifest["reference"]["commit"],
            "candidate_command": " ".join(
                candidate_command(
                    Path(relative(mlpackage)),
                    Path("<tile>.png"),
                    Path("<out>.png"),
                    Path("<out>.npy"),
                    units,
                    tile_size,
                )
            ),
            "reference_implementation": reference_implementation,
            "sources": sources,
            "entries": entries,
            "order_seed": seed,
            "results": results,
            "worst_tile": {
                "tile_png": worst_entry["tile_png"],
                "reference_png": relative(paths["reference"] / f"{worst_name}.png"),
                "candidate_png": relative(paths["candidate"] / f"{worst_name}.png"),
            },
            "level_c_queue": [
                {
                    "name": Path(entry["tile_png"]).stem,
                    "max_abs_255": entry["metrics"]["max_abs"] * metrics.BYTE,
                    "reference_png": relative(
                        paths["reference"] / f"{Path(entry['tile_png']).stem}.png"
                    ),
                    "candidate_png": relative(
                        paths["candidate"] / f"{Path(entry['tile_png']).stem}.png"
                    ),
                }
                for entry in worst_by_max_abs
            ],
            "diagnostic": {
                "postprocessing": postprocessing,
                "overshoot": statistics.median(overshoot),
                "as_measured": summary["level_a"]["psnr_db_median"],
                "quantised": statistics.median(
                    [value for value in quantised_psnr if not math.isinf(value)] or [math.nan]
                ),
            },
        },
    )

    log("")
    log(f"[manifest] {relative(paths['manifest'])}")
    log(f"[report]   {relative(paths['report'])}")
    log("")
    log("=" * 78)
    level_a = summary["level_a"]
    log(f"PSNR median   : {_decibels(level_a['psnr_db_median'])}   "
        f"(threshold >= {level_a['thresholds']['psnr_db_median_min']} dB)")
    log(f"within {level_a['thresholds']['max_abs_tolerance_255']}/255  : "
        f"{level_a['within_tolerance_fraction'] * 100:.4f} %   "
        f"(threshold >= {level_a['thresholds']['within_tolerance_fraction_min'] * 100} %)")
    log(f"SSIM median   : {summary['level_b']['ssim_median']:.6f}   (recorded, not gated)")
    log(f"LPIPS median  : {summary['level_b']['lpips_median']:.6f}   (recorded, not gated)")
    log(f"compute units : {units}")
    log(f"LEVEL A/B     : {status}")
    log("=" * 78)
    return EXIT_OK if summary["passed"] else EXIT_FAILED


# --- check --------------------------------------------------------------------------------
def run_check(model_id: str, units: str, models_root: Path, config: dict) -> int:
    paths = certification_paths(models_root, model_id, units)
    if not paths["manifest"].exists():
        raise CertifyError(
            f"no certification manifest at {relative(paths['manifest'])}\n"
            f"  run: python tools/certify/certify.py --model {model_id} --compute-units {units}"
        )

    try:
        manifest = manifest_emitter.load(paths["manifest"])
    except manifest_emitter.ManifestError as error:
        raise CertifyError(f"{relative(paths['manifest'])}: {error}") from error

    log(f"[check] {relative(paths['manifest'])}")
    failures: list[str] = []

    if manifest["model_id"] != model_id:
        failures.append(f"model_id: manifest says {manifest['model_id']!r}, asked for {model_id!r}")
    if manifest["compute_units"] != units:
        failures.append(
            f"compute_units: manifest records {manifest['compute_units']!r}, asked for {units!r} — "
            "manifests are per backend and are never comparable across them"
        )

    # 1. Are the artifacts on disk still the ones the manifest describes?
    entries = manifest["tiles"]["entries"]
    for entry in entries:
        for path_field, hash_field in (
            ("source_image", "source_sha256"),
            ("tile_png", "tile_sha256"),
            ("reference_npy", "reference_sha256"),
            ("candidate_npy", "candidate_sha256"),
        ):
            path = REPO_ROOT / entry[path_field]
            if not path.exists():
                failures.append(f"{entry[path_field]}: missing")
                continue
            measured = run_reference.sha256_of_file(path)
            if measured != entry[hash_field]:
                failures.append(
                    f"{entry[path_field]}: hash mismatch\n"
                    f"      manifest: {entry[hash_field]}\n"
                    f"      on disk : {measured}"
                )
    log(f"  {len(entries) * 4} recorded artifact(s) re-hashed")

    try:
        weights = weights_artifact_path(models_root, model_id)
    except Exception as error:  # noqa: BLE001  (fetch raises its own ArtifactError type)
        raise CertifyError(f"{model_id}: cannot locate the weights artifact: {error}") from error
    mlpackage = (
        models_root
        / model_id
        / "output"
        / f"{model_id}_{manifest['tiles']['tile_size']}x{manifest['tiles']['tile_size']}.mlpackage"
    )
    for label, path, recorded in (
        ("weights_sha256", weights, manifest["weights_sha256"]),
        ("mlpackage_sha256", mlpackage, manifest["mlpackage_sha256"]),
    ):
        if not path.exists():
            failures.append(f"{label}: {relative(path)} is missing, cannot verify the artifact")
            continue
        measured = manifest_emitter.sha256_of_path(path)
        if measured != recorded:
            failures.append(
                f"{label}: {relative(path)} does not match the manifest\n"
                f"      manifest: {recorded}\n      on disk : {measured}"
            )

    # 2. Do the tensors on disk still produce the metrics the manifest recorded?
    tolerance = config["thresholds"]["level_a"]["max_abs_tolerance_255"] / metrics.BYTE
    recomputed = 0
    for entry in entries:
        reference_path = REPO_ROOT / entry["reference_npy"]
        candidate_path = REPO_ROOT / entry["candidate_npy"]
        if not (reference_path.exists() and candidate_path.exists()):
            continue  # already reported as missing
        reference = np.load(reference_path, allow_pickle=False)
        candidate = np.load(candidate_path, allow_pickle=False)
        if reference.shape != candidate.shape:
            failures.append(
                f"{entry['candidate_npy']}: shape {candidate.shape} does not match the reference "
                f"{reference.shape}"
            )
            continue
        fresh = {
            "psnr_db": metrics.psnr(reference, candidate),
            "mae": metrics.mae(reference, candidate),
            "max_abs": metrics.max_abs_error(reference, candidate),
            "within_tolerance_fraction": metrics.within_tolerance_fraction(
                reference, candidate, tolerance
            ),
            "ssim": metrics.ssim(reference, candidate),
        }
        if math.isinf(fresh["psnr_db"]):
            fresh["psnr_db"] = None
        for name, value in fresh.items():
            recorded = entry["metrics"][name]
            if value is None or recorded is None:
                if value != recorded:
                    failures.append(
                        f"{entry['candidate_npy']}: {name} recomputes to {value}, "
                        f"manifest records {recorded}"
                    )
                continue
            if not math.isclose(value, recorded, rel_tol=RECHECK_TOLERANCE, abs_tol=RECHECK_TOLERANCE):
                failures.append(
                    f"{entry['candidate_npy']}: {name} recomputes to {value!r}, "
                    f"manifest records {recorded!r}"
                )
        recomputed += 1
    log(f"  {recomputed} tile(s) recomputed from their tensors (LPIPS excepted — it needs torch)")

    # 3. Does the recorded verdict follow from the recorded numbers?
    if entries and not failures:
        summary = aggregate(entries, config["thresholds"])
        expected_status = "PASS" if summary["passed"] else "FAIL"
        if expected_status != manifest["status"]:
            failures.append(
                f"status: the recorded metrics give {expected_status}, manifest says "
                f"{manifest['status']!r}"
            )
        for field in ("psnr_db_median", "within_tolerance_fraction"):
            recorded = manifest["results"]["level_a"][field]
            fresh = summary["level_a"][field]
            if recorded is None or fresh is None:
                if recorded != fresh:
                    failures.append(f"results.level_a.{field}: {fresh} vs recorded {recorded}")
                continue
            if not math.isclose(fresh, recorded, rel_tol=RECHECK_TOLERANCE, abs_tol=RECHECK_TOLERANCE):
                failures.append(
                    f"results.level_a.{field}: re-derives to {fresh!r}, manifest records {recorded!r}"
                )

    log("")
    for failure in failures:
        log(f"  FAILED {failure}")
    if failures:
        log("")
        log(f"CHECK FAILED ({len(failures)} problem(s))")
        return EXIT_FAILED
    log(f"OK ({relative(paths['manifest'])} is valid: artifacts, metrics and verdict all reproduce)")
    return EXIT_OK


# --- entry point --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Comparison harness — Level A/B certification"
    )
    parser.add_argument("--model", required=True, help="model id, matching models/<id>/manifest.json")
    parser.add_argument(
        "--compute-units",
        default=None,
        choices=COMPUTE_UNITS,
        help="REQUIRED. Core ML backend to pin the candidate to; recorded in the manifest.",
    )
    parser.add_argument(
        "--check", action="store_true", help="validate an existing manifest against current outputs"
    )
    parser.add_argument("--models-root", default=None, help=argparse.SUPPRESS)  # test seam
    parser.add_argument("--corpus-dir", default=None, help=argparse.SUPPRESS)  # test seam
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)  # test seam
    args = parser.parse_args(argv)

    if args.compute_units is None:
        log(
            "error: --compute-units is required and has no default.\n"
            "  Core ML numerics were measured diverging by backend (cpuOnly differed from `all`\n"
            "  by up to 14/255 on the same tile), so an unpinned run is not comparable and not\n"
            f"  certifiable. Choose one of: {', '.join(COMPUTE_UNITS)}."
        )
        return EXIT_STRUCTURAL_ERROR

    models_root = Path(args.models_root).resolve() if args.models_root else REPO_ROOT / "models"
    corpus_dir = Path(args.corpus_dir).resolve() if args.corpus_dir else run_reference.CORPUS_DIR
    config_path = Path(args.config).resolve() if args.config else CONFIG_PATH

    try:
        config = load_config(config_path)
        if args.check:
            return run_check(args.model, args.compute_units, models_root, config)
        return run_certify(args.model, args.compute_units, models_root, corpus_dir, config)
    except (
        CertifyError,
        run_reference.ReferenceError,
        metrics.MetricError,
        manifest_emitter.ManifestError,
    ) as error:
        log("")
        log(f"STOPPED: {type(error).__name__}: {error}")
        return EXIT_STRUCTURAL_ERROR


if __name__ == "__main__":
    sys.exit(main())
