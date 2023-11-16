"""CERTIFICATION_MANIFEST.json — schema, writer and validator (certification spec §11).

The manifest is the evidence a certification run happened and what it measured. Certification spec
§11 requires it to be complete enough to answer, six months later, "why did model X produce this
output?" -- including after Apple ships a Core ML or iOS update. So the schema is strict in both
directions, exactly as the conversion manifest is: a missing field means the identity is incomplete,
and an unexpected field means something wrote a shape the reviewer never agreed to. Both fail
loudly, naming the offending field by its dotted path.

Two things beyond the §11 sketch:

  * `tiles` -- tile derivation requires the coordinates and the source image hash to be recorded
    in the manifest, and `--check` needs the per-tile artifact hashes.
  * per-tile `metrics` inside `tiles.entries` -- the aggregate alone cannot be re-verified.

The strict validator itself is imported from `tools/convert/manifest.py` rather than copied. Two
schema engines would be two behaviours, and the one that drifts is always the one nobody reads.

Import policy: standard library and numpy-free. Nothing here needs torch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools" / "convert"))

import manifest as conversion_manifest  # noqa: E402

ManifestError = conversion_manifest.ManifestError
Sha256 = conversion_manifest.Sha256
Shape4 = conversion_manifest.Shape4
sha256_of_path = conversion_manifest.sha256_of_path

STATUS_VALUES = ("PASS", "FAIL")
ENERGY_KINDS = ("measured", "estimated", "unavailable")

#: Dotted-path schema, certification spec §11. Nested dicts recurse; every listed field is
#: mandatory.
SCHEMA: dict = {
    "certification_id": str,
    "date": str,
    "model_id": str,
    "weights_sha256": Sha256,
    "mlpackage_sha256": Sha256,
    "conversion": {
        "script_version": str,
        "coremltools": str,
        "python": str,
        "precision": str,
        "input_shape": Shape4,
        "parameters": dict,
    },
    "reference": {
        "implementation": str,
        "commit": str,
        "output_hashes": list,
    },
    "coreml_output_hashes": list,
    "corpus_version": str,
    "device": str,
    "os_version": str,
    "coreml_version": str,
    "compute_units": str,
    "tiles": {
        "tile_size": int,
        # Names what was actually certified, so a starter-corpus run can never be cited as
        # full-corpus evidence by someone reading only the summary.
        "tile_set": str,
        "order_seed": int,
        "derivation": str,
        "source_images": list,
        "entries": list,
    },
    "results": {
        "level_a": dict,
        "level_b": dict,
        "level_c": dict,
        "performance": dict,
        "energy": dict,
    },
    "status": str,
}

TILE_FIELDS = {
    "index": int,
    "source_image": str,
    "source_sha256": str,
    "x": int,
    "y": int,
    "width": int,
    "height": int,
    "tile_png": str,
    "tile_sha256": str,
    "reference_npy": str,
    "reference_sha256": str,
    "candidate_npy": str,
    "candidate_sha256": str,
    "metrics": dict,
}

SOURCE_IMAGE_FIELDS = {
    "path": str,
    "sha256": str,
    "width": int,
    "height": int,
    "tile_count": int,
    "excluded_reason": (str, type(None)),
}

METRIC_FIELDS = ("psnr_db", "mae", "max_abs", "within_tolerance_fraction", "ssim", "lpips")


def _fail(path: str, message: str) -> None:
    raise ManifestError(f"{path}: {message}")


def _validate_energy(energy: dict) -> None:
    if set(energy) != {"value", "kind"}:
        _fail("results.energy", f"expected exactly 'value' and 'kind', got {sorted(energy)}")
    if energy["kind"] not in ENERGY_KINDS:
        _fail("results.energy.kind", f"expected one of {ENERGY_KINDS}, got {energy['kind']!r}")
    value = energy["value"]
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
        _fail("results.energy.value", f"expected a number or null, got {type(value).__name__}")
    if energy["kind"] == "unavailable" and value is not None:
        _fail("results.energy", "kind 'unavailable' must carry a null value, not a number")


def _validate_tiles(tiles: dict) -> None:
    for position, image in enumerate(tiles["source_images"]):
        where = f"tiles.source_images[{position}]"
        if not isinstance(image, dict):
            _fail(where, f"expected an object, got {type(image).__name__}")
        for field, expected in SOURCE_IMAGE_FIELDS.items():
            if field not in image:
                _fail(f"{where}.{field}", "missing required field")
            if not isinstance(image[field], expected) or isinstance(image[field], bool):
                _fail(f"{where}.{field}", f"expected {expected}, got {type(image[field]).__name__}")
        for field in image:
            if field not in SOURCE_IMAGE_FIELDS:
                _fail(f"{where}.{field}", "unexpected field (schema is strict)")

    declared = sum(image["tile_count"] for image in tiles["source_images"])
    if declared != len(tiles["entries"]):
        _fail(
            "tiles",
            f"coverage disagrees with the tile set: source_images declare {declared} tile(s), "
            f"entries holds {len(tiles['entries'])} — an image was dropped without being recorded",
        )

    for position, entry in enumerate(tiles["entries"]):
        where = f"tiles.entries[{position}]"
        if not isinstance(entry, dict):
            _fail(where, f"expected an object, got {type(entry).__name__}")
        for field, expected in TILE_FIELDS.items():
            if field not in entry:
                _fail(f"{where}.{field}", "missing required field")
            if not isinstance(entry[field], expected) or isinstance(entry[field], bool):
                _fail(f"{where}.{field}", f"expected {expected.__name__}, got {type(entry[field]).__name__}")
        for field in entry:
            if field not in TILE_FIELDS:
                _fail(f"{where}.{field}", "unexpected field (schema is strict)")
        for metric in METRIC_FIELDS:
            if metric not in entry["metrics"]:
                _fail(f"{where}.metrics.{metric}", "missing required field")
        for metric, value in entry["metrics"].items():
            if metric not in METRIC_FIELDS:
                _fail(f"{where}.metrics.{metric}", "unexpected field (schema is strict)")
            # psnr_db is null exactly when the two tensors are bit-identical (infinite PSNR);
            # a made-up finite ceiling would be indistinguishable from a real measurement.
            if value is None and metric != "psnr_db":
                _fail(f"{where}.metrics.{metric}", "only psnr_db may be null (exact match)")
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
                _fail(f"{where}.metrics.{metric}", f"expected a number, got {type(value).__name__}")
        if entry["index"] != position:
            _fail(f"{where}.index", f"tile order is evidence: expected {position}, got {entry['index']}")


def _validate_cross_references(data: dict) -> None:
    """§11 records the two hash lists separately from the per-tile block. They must agree, or the
    manifest is telling two stories."""
    entries = data["tiles"]["entries"]
    for field, recorded, expected in (
        (
            "reference.output_hashes",
            data["reference"]["output_hashes"],
            [entry["reference_sha256"] for entry in entries],
        ),
        (
            "coreml_output_hashes",
            data["coreml_output_hashes"],
            [entry["candidate_sha256"] for entry in entries],
        ),
    ):
        if list(recorded) != expected:
            _fail(field, "does not match the per-tile hashes in tiles.entries (same order required)")


def validate(source: str | Path | dict) -> dict:
    """Validate a certification manifest given as a dict or a path to JSON."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise ManifestError(f"<file>: certification manifest not found at {path}")
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as error:
            raise ManifestError(f"<file>: {path} is not valid JSON: {error}") from error
    else:
        data = source

    conversion_manifest._check(data, SCHEMA, "")  # one strict engine, not two -- see module docstring
    if data["status"] not in STATUS_VALUES:
        _fail("status", f"expected one of {STATUS_VALUES}, got {data['status']!r}")
    _validate_energy(data["results"]["energy"])
    _validate_tiles(data["tiles"])
    _validate_cross_references(data)
    return data


def write(path: str | Path, manifest: dict) -> dict:
    """Validate, then write. An invalid manifest is never persisted."""
    validate(manifest)
    text = json.dumps(manifest, indent=2, sort_keys=False) + "\n"
    # json.dumps emits bare Infinity/NaN, which is not JSON and which no other tool will read.
    json.loads(text, parse_constant=_reject_constant)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    return manifest


def _reject_constant(name: str):
    raise ManifestError(f"<file>: refusing to write non-JSON constant {name!r} into the manifest")


def load(path: str | Path) -> dict:
    """Load and validate a manifest from disk."""
    return validate(path)


def build(
    *,
    certification_id: str,
    date: str,
    model_id: str,
    weights_sha256: str,
    mlpackage_sha256: str,
    conversion: dict,
    reference: dict,
    coreml_output_hashes: list,
    corpus_version: str,
    device: str,
    os_version: str,
    coreml_version: str,
    compute_units: str,
    tiles: dict,
    results: dict,
    status: str,
) -> dict:
    """Assemble a manifest in §11's field order. Validated by the caller through `write`."""
    return {
        "certification_id": certification_id,
        "date": date,
        "model_id": model_id,
        "weights_sha256": weights_sha256,
        "mlpackage_sha256": mlpackage_sha256,
        "conversion": conversion,
        "reference": reference,
        "coreml_output_hashes": coreml_output_hashes,
        "corpus_version": corpus_version,
        "device": device,
        "os_version": os_version,
        "coreml_version": coreml_version,
        "compute_units": compute_units,
        "tiles": tiles,
        "results": results,
        "status": status,
    }
