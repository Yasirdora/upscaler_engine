"""Manifest schema, writer and validator for converted model artifacts.

The manifest is the identity document for a `.mlpackage`: it must be sufficient to answer
"why did model X produce this output?" six months later, including after Apple ships a
Core ML or iOS update (docs/CERTIFICATION.md §11).

The schema is deliberately strict in both directions. A missing field means the identity is
incomplete; an unexpected field means something wrote a shape the reviewer never agreed to.
Both fail loudly, naming the offending field by its dotted path.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ManifestError(ValueError):
    """Raised when a manifest violates the schema. The message names the offending field."""


class Sha256:
    """Marker type: a string that must be 64 lowercase hex characters."""


class Shape4:
    """Marker type: a list of exactly four positive integers (NCHW)."""


# Dotted-path schema. Nested dicts recurse; every listed field is mandatory.
SCHEMA: dict[str, Any] = {
    "model_id": str,
    "scale_factor": int,
    "weights": {
        "url": str,
        "sha256": Sha256,
        "bytes": int,
        "state_dict_key": str,
        "parameters": int,
    },
    "source": {
        "repo": str,
        "commit_or_release": str,
        "license": str,
    },
    "conversion": {
        "script": str,
        "script_version": str,
        "coremltools": str,
        "torch": str,
        "python": str,
        "representation": str,
        "precision": str,
        "input_shape": Shape4,
        "output_shape": Shape4,
        "min_deployment": str,
        "compute_units": str,
        "parameters": dict,
    },
    "preprocessing": str,
    "postprocessing": str,
    "outputs": {
        "mlpackage_sha256": Sha256,
        "reference_metrics": {
            "psnr_db": float,
            "mae": float,
            "max_abs": float,
        },
    },
    "created_utc": str,
}


def _check(node: Any, spec: Any, path: str) -> None:
    if isinstance(spec, dict):
        if not isinstance(node, dict):
            raise ManifestError(f"{path or '<root>'}: expected an object, got {type(node).__name__}")
        for key, sub in spec.items():
            where = f"{path}.{key}" if path else key
            if key not in node:
                raise ManifestError(f"{where}: missing required field")
            _check(node[key], sub, where)
        for key in node:
            if key not in spec:
                where = f"{path}.{key}" if path else key
                raise ManifestError(f"{where}: unexpected field (schema is strict)")
        return

    if spec is Sha256:
        if not isinstance(node, str) or not SHA256_RE.match(node):
            raise ManifestError(f"{path}: expected 64 lowercase hex characters, got {node!r}")
        return

    if spec is Shape4:
        ok = (
            isinstance(node, list)
            and len(node) == 4
            and all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in node)
        )
        if not ok:
            raise ManifestError(f"{path}: expected four positive integers (NCHW), got {node!r}")
        return

    if spec is float:  # ints are acceptable where a float is expected; bools never are
        if isinstance(node, bool) or not isinstance(node, (int, float)):
            raise ManifestError(f"{path}: expected a number, got {type(node).__name__}")
        return

    if spec is int and isinstance(node, bool):
        raise ManifestError(f"{path}: expected int, got bool")

    if not isinstance(node, spec):
        raise ManifestError(f"{path}: expected {spec.__name__}, got {type(node).__name__}")


def validate(source: str | Path | dict) -> dict:
    """Validate a manifest given as a dict or a path to JSON. Returns it; raises ManifestError."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise ManifestError(f"<file>: manifest not found at {path}")
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as error:
            raise ManifestError(f"<file>: {path} is not valid JSON: {error}") from error
    else:
        data = source

    _check(data, SCHEMA, "")
    return data


def write(path: str | Path, manifest: dict) -> dict:
    """Validate then write. An invalid manifest is never persisted."""
    validate(manifest)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")
    return manifest


def load(path: str | Path) -> dict:
    """Load and validate a manifest from disk."""
    return validate(path)


def sha256_of_path(path: str | Path) -> str:
    """SHA256 of a file, or of a directory bundle such as a `.mlpackage`.

    Directory digest is order-independent by construction: relative POSIX paths are sorted and
    each contributes its path and its bytes, so the same tree always yields the same digest.
    """
    target = Path(path)
    digest = hashlib.sha256()
    if target.is_file():
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    if not target.is_dir():
        raise FileNotFoundError(f"cannot hash missing path: {target}")

    for file in sorted(p for p in target.rglob("*") if p.is_file()):
        digest.update(file.relative_to(target).as_posix().encode())
        digest.update(b"\0")
        with file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


# --- fidelity to the committed manifest ---------------------------------------------------
# Fields whose value IS the model's identity and certified behaviour. If a re-conversion changes
# one of these, the artifact is not the artifact the repository has been certifying, and the
# manifest recording that must not be overwritten before a human has seen the difference.
FIDELITY_FIELDS: tuple[tuple[str, ...], ...] = (
    ("model_id",),
    ("scale_factor",),
    ("weights", "sha256"),
    ("weights", "bytes"),
    ("weights", "parameters"),
    ("conversion", "representation"),
    ("conversion", "precision"),
    ("conversion", "input_shape"),
    ("conversion", "output_shape"),
    ("conversion", "min_deployment"),
    ("outputs", "reference_metrics", "psnr_db"),
    ("outputs", "reference_metrics", "mae"),
    ("outputs", "reference_metrics", "max_abs"),
)

# Expected to differ on any honest re-run, and excluded deliberately. `mlpackage_sha256` is the
# important one: tools/artifacts/README.md states Core ML compilation is non-deterministic and that
# a .mlpackage is "re-derived by running the pinned conversion, not by matching a hash". Comparing
# it would fail every re-conversion and teach the reader to ignore the comparison entirely.
VOLATILE_FIELDS: tuple[str, ...] = (
    "outputs.mlpackage_sha256", "created_utc", "conversion.coremltools", "conversion.torch",
    "conversion.python", "conversion.compute_units", "conversion.script_version",
)

_MISSING = object()


def _at(manifest: dict, path: tuple[str, ...]) -> Any:
    node: Any = manifest
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return _MISSING
        node = node[key]
    return node


def compare_to_prior(prior: dict, candidate: dict) -> list[str]:
    """Differences in the fields that carry fidelity. Empty means the re-conversion agrees.

    An absent field on either side is a difference, not a pass: a manifest that has lost a
    fidelity field cannot vouch for the artifact, and silently agreeing with it would be the
    same vacuous green this comparison exists to remove.
    """
    drift: list[str] = []
    for path in FIDELITY_FIELDS:
        was, now = _at(prior, path), _at(candidate, path)
        if was is not _MISSING and now is not _MISSING and was == now:
            continue
        shown_was = "<absent>" if was is _MISSING else was
        shown_now = "<absent>" if now is _MISSING else now
        drift.append(f"{'.'.join(path)}: manifest {shown_was} != this run {shown_now}")
    return drift


def drift_against_prior(path: str | Path, candidate: dict) -> list[str]:
    """`compare_to_prior` against a manifest on disk. Unreadable JSON raises rather than passing."""
    source = Path(path)
    try:
        prior = json.loads(source.read_text())
    except json.JSONDecodeError as error:
        raise ManifestError(f"{source}: not valid JSON, refusing to overwrite it: {error}") from error
    if not isinstance(prior, dict):
        raise ManifestError(f"{source}: expected a JSON object, refusing to overwrite it")
    return compare_to_prior(prior, candidate)
