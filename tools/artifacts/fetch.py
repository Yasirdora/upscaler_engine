#!/usr/bin/env python3
"""Fetch and verify model artifacts declared by manifests.

Identity policy (binding): verification targets the **weights
SHA256** -- a reproducible input. A `.mlpackage` is never hash-matched to decide identity,
because Core ML compilation is not deterministic: two identical conversion runs produced
different package hashes while their Level A metrics were bit-identical. A package hash pins
only the exact binary that was shipped; a rebuilt package is re-derived by running the pinned
conversion, not by matching a hash.

Verification happens *before* an artifact is visible under its final name: the download lands
in a temp file inside the destination directory, is hashed there, and is renamed into place
only on a match. A failed fetch therefore cannot leave a partial or unverified file behind.

Standard library only -- no third-party dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = REPO_ROOT / "models"
CHUNK = 1 << 20
TIMEOUT_SECONDS = 300

EXIT_OK = 0
EXIT_VERIFICATION_FAILED = 1
EXIT_STRUCTURAL_ERROR = 2


class ArtifactError(RuntimeError):
    """Structural problem: missing or malformed manifest, missing required field."""


class VerificationError(RuntimeError):
    """A hash did not match. Never worked around -- always fatal to the run."""


def log(message: str) -> None:
    print(message, flush=True)


def display(path: Path) -> str:
    """Repo-relative when possible, absolute otherwise -- a fixture tree may live anywhere."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_models(models_root: Path) -> list[str]:
    """Every directory under models/ carrying a manifest.json, sorted for determinism."""
    if not models_root.is_dir():
        return []
    return sorted(p.parent.name for p in models_root.glob("*/manifest.json"))


def read_manifest(models_root: Path, model_id: str) -> dict:
    path = models_root / model_id / "manifest.json"
    if not path.exists():
        raise ArtifactError(f"no manifest for '{model_id}': {path} does not exist")
    try:
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ArtifactError(f"{path}: not valid JSON: {error}") from error
    if not isinstance(manifest, dict):
        raise ArtifactError(f"{path}: expected a JSON object at the top level")
    return manifest


def weights_spec(manifest: dict, model_id: str) -> tuple[str, str]:
    """Extract the fetch target. Schema validation proper belongs to tools.convert.manifest."""
    weights = manifest.get("weights")
    if not isinstance(weights, dict):
        raise ArtifactError(f"{model_id}: manifest field 'weights' is missing or not an object")
    for field in ("url", "sha256"):
        if not isinstance(weights.get(field), str) or not weights[field]:
            raise ArtifactError(f"{model_id}: manifest field 'weights.{field}' is missing or not a string")
    return weights["url"], weights["sha256"]


def weights_path(models_root: Path, model_id: str, url: str) -> Path:
    filename = Path(urllib.request.url2pathname(url.split("?", 1)[0])).name
    if not filename:
        raise ArtifactError(f"{model_id}: cannot derive a filename from weights.url {url!r}")
    return models_root / model_id / "weights" / filename


def download_verified(url: str, expected_sha256: str, target: Path) -> str:
    """Download to a temp file in the destination dir, hash it there, rename only on match."""
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=target.parent, prefix=".download-", suffix=".part")
    temp_path = Path(temp_name)
    try:
        digest = hashlib.sha256()
        written = 0
        with os.fdopen(handle, "wb") as out, urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:
            for chunk in iter(lambda: response.read(CHUNK), b""):
                out.write(chunk)
                digest.update(chunk)
                written += len(chunk)
        measured = digest.hexdigest()
        if measured != expected_sha256:
            raise VerificationError(
                f"SHA256 mismatch for {url}\n"
                f"  expected (manifest): {expected_sha256}\n"
                f"  measured (download): {measured}\n"
                f"  {written:,} bytes discarded; nothing was written to {target}"
            )
        os.replace(temp_path, target)
        log(f"  fetched and verified: {target.name} ({written:,} bytes, sha256 {measured})")
        return measured
    finally:
        if temp_path.exists():
            temp_path.unlink()


def fetch_model(models_root: Path, model_id: str) -> None:
    manifest = read_manifest(models_root, model_id)
    url, expected = weights_spec(manifest, model_id)
    target = weights_path(models_root, model_id, url)

    if target.exists():
        measured = sha256_of_file(target)
        if measured == expected:
            log(f"  already present and verified: {display(target)}")
            return
        raise VerificationError(
            f"local artifact does not match the manifest: {target}\n"
            f"  expected (manifest): {expected}\n"
            f"  measured (on disk) : {measured}\n"
            "  refusing to overwrite it -- inspect the file, then delete it and re-run"
        )
    log(f"  fetching {url}")
    download_verified(url, expected, target)


def verify_model(models_root: Path, model_id: str) -> bool:
    """Verify what exists locally. An absent artifact is reported, never a failure:
    CI runs with a cold cache and must not go red because weights were never fetched."""
    manifest = read_manifest(models_root, model_id)
    url, expected = weights_spec(manifest, model_id)
    target = weights_path(models_root, model_id, url)

    if not target.exists():
        log(f"  absent, nothing to verify: {display(target)}")
        return True
    measured = sha256_of_file(target)
    if measured == expected:
        log(f"  verified: {display(target)} (sha256 {measured})")
        return True
    log(
        f"  FAILED: {display(target)}\n"
        f"    expected (manifest): {expected}\n"
        f"    measured (on disk) : {measured}"
    )
    return False


def run(models: list[str], models_root: Path, verify_only: bool) -> int:
    if not models:
        log("no manifests found under models/ -- nothing to do")
        return EXIT_OK

    failures: list[str] = []
    for model_id in models:
        log(f"[{model_id}]")
        try:
            if verify_only:
                if not verify_model(models_root, model_id):
                    failures.append(model_id)
            else:
                fetch_model(models_root, model_id)
        except VerificationError as error:
            log(f"  FAILED: {error}")
            failures.append(model_id)

    log("")
    if failures:
        log(f"VERIFICATION FAILED for: {', '.join(failures)}")
        return EXIT_VERIFICATION_FAILED
    log(f"OK ({'verified' if verify_only else 'fetched/verified'} {len(models)} model(s))")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch and verify model artifacts by manifest")
    parser.add_argument("--model", help="model id, matching models/<id>/manifest.json")
    parser.add_argument("--all", action="store_true", help="every manifest under models/")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify what exists locally; never downloads. Absent artifacts are not a failure.",
    )
    parser.add_argument(
        "--models-root",
        default=None,
        help=argparse.SUPPRESS,  # test seam: point the tool at a fixture tree
    )
    args = parser.parse_args(argv)

    if args.model and args.all:
        parser.error("--model and --all are mutually exclusive")
    if not args.model and not args.all and not args.verify_only:
        parser.error("specify --model <id> or --all")

    models_root = Path(args.models_root).resolve() if args.models_root else MODELS_ROOT
    try:
        models = [args.model] if args.model else discover_models(models_root)
        return run(models, models_root, args.verify_only)
    except ArtifactError as error:
        log(f"STOPPED: {error}")
        return EXIT_STRUCTURAL_ERROR


if __name__ == "__main__":
    sys.exit(main())
