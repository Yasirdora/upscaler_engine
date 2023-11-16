"""Tests for artifact fetch + verification.

Offline by construction: fixtures are served over `file://` URLs from a tempdir, so the suite
never touches the network.

Expected hashes are external constants, computed independently with `shasum -a 256` and pasted
here. They are never recomputed with the code under test -- otherwise a bug in the hashing path
would agree with itself and the suite would prove nothing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fetch  # noqa: E402

PAYLOAD = b"weights-payload-v1"
PAYLOAD_SHA256 = "867f60135f1078405c544b9162c1faced8d1560aefe2a12537a2e2fb25c21d58"
OTHER_PAYLOAD = b"corrupted-payload"
OTHER_SHA256 = "9de7806565707a4e33be533a7fce46cd425f30221f9aa3470585496323ec190f"


def make_tree(tmp_path: Path, *, declared_sha256: str, payload: bytes = PAYLOAD) -> tuple[Path, Path]:
    """Build models/<id>/manifest.json pointing at a file:// payload. Returns (models_root, payload)."""
    source = tmp_path / "server" / "RealESRGAN_x4plus.pth"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(payload)

    models_root = tmp_path / "models"
    model_dir = models_root / "TestModel"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "manifest.json").write_text(
        json.dumps({"weights": {"url": source.as_uri(), "sha256": declared_sha256}})
    )
    return models_root, source


def cached_file(models_root: Path) -> Path:
    return models_root / "TestModel" / "weights" / "RealESRGAN_x4plus.pth"


def leftovers(models_root: Path) -> list[Path]:
    weights_dir = models_root / "TestModel" / "weights"
    return list(weights_dir.iterdir()) if weights_dir.exists() else []


# --- fetching -----------------------------------------------------------------------------
def test_fetch_downloads_and_verifies(tmp_path):
    models_root, _ = make_tree(tmp_path, declared_sha256=PAYLOAD_SHA256)

    assert fetch.main(["--model", "TestModel", "--models-root", str(models_root)]) == fetch.EXIT_OK
    assert cached_file(models_root).read_bytes() == PAYLOAD


def test_hash_mismatch_is_refused(tmp_path):
    """The manifest declares a different artifact than the URL serves."""
    models_root, _ = make_tree(tmp_path, declared_sha256=OTHER_SHA256)

    assert fetch.main(["--model", "TestModel", "--models-root", str(models_root)]) == fetch.EXIT_VERIFICATION_FAILED


def test_no_partial_state_after_failed_fetch(tmp_path, capsys):
    models_root, _ = make_tree(tmp_path, declared_sha256=OTHER_SHA256)

    fetch.main(["--model", "TestModel", "--models-root", str(models_root)])

    assert not cached_file(models_root).exists(), "unverified artifact must not reach its final name"
    assert leftovers(models_root) == [], "temp download must not survive a failed verification"
    output = capsys.readouterr().out
    assert OTHER_SHA256 in output and PAYLOAD_SHA256 in output, "both hashes must be printed"


def test_fetch_is_idempotent(tmp_path):
    models_root, _ = make_tree(tmp_path, declared_sha256=PAYLOAD_SHA256)
    assert fetch.main(["--model", "TestModel", "--models-root", str(models_root)]) == fetch.EXIT_OK

    target = cached_file(models_root)
    before = target.stat()

    assert fetch.main(["--model", "TestModel", "--models-root", str(models_root)]) == fetch.EXIT_OK
    after = target.stat()
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns), "must not re-download"


def test_existing_file_with_wrong_hash_is_not_overwritten(tmp_path):
    models_root, _ = make_tree(tmp_path, declared_sha256=PAYLOAD_SHA256)
    target = cached_file(models_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(OTHER_PAYLOAD)

    assert fetch.main(["--model", "TestModel", "--models-root", str(models_root)]) == fetch.EXIT_VERIFICATION_FAILED
    assert target.read_bytes() == OTHER_PAYLOAD, "the suspect file is preserved for inspection"


# --- verify-only --------------------------------------------------------------------------
def test_verify_only_passes_when_nothing_is_cached(tmp_path):
    """CI runs with a cold cache: absent weights must not turn the build red."""
    models_root, _ = make_tree(tmp_path, declared_sha256=PAYLOAD_SHA256)

    assert fetch.main(["--verify-only", "--all", "--models-root", str(models_root)]) == fetch.EXIT_OK
    assert not cached_file(models_root).exists(), "--verify-only must never download"


def test_verify_only_detects_corruption(tmp_path):
    models_root, _ = make_tree(tmp_path, declared_sha256=PAYLOAD_SHA256)
    target = cached_file(models_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(OTHER_PAYLOAD)

    code = fetch.main(["--verify-only", "--all", "--models-root", str(models_root)])
    assert code == fetch.EXIT_VERIFICATION_FAILED


def test_verify_only_passes_on_a_good_cache(tmp_path):
    models_root, _ = make_tree(tmp_path, declared_sha256=PAYLOAD_SHA256)
    fetch.main(["--model", "TestModel", "--models-root", str(models_root)])

    assert fetch.main(["--verify-only", "--all", "--models-root", str(models_root)]) == fetch.EXIT_OK


def test_verify_only_with_no_manifests_at_all(tmp_path):
    empty = tmp_path / "models"
    empty.mkdir()

    assert fetch.main(["--verify-only", "--all", "--models-root", str(empty)]) == fetch.EXIT_OK


# --- structural errors --------------------------------------------------------------------
def test_missing_manifest_names_the_path(tmp_path, capsys):
    models_root = tmp_path / "models"
    models_root.mkdir()

    code = fetch.main(["--model", "Nonexistent", "--models-root", str(models_root)])
    output = capsys.readouterr().out

    assert code == fetch.EXIT_STRUCTURAL_ERROR
    assert str(models_root / "Nonexistent" / "manifest.json") in output


def test_malformed_manifest_names_the_path(tmp_path, capsys):
    models_root = tmp_path / "models"
    (models_root / "Broken").mkdir(parents=True)
    (models_root / "Broken" / "manifest.json").write_text("{not json")

    code = fetch.main(["--model", "Broken", "--models-root", str(models_root)])
    output = capsys.readouterr().out

    assert code == fetch.EXIT_STRUCTURAL_ERROR
    assert "manifest.json" in output and "not valid JSON" in output


@pytest.mark.parametrize("weights", [{}, {"url": "file:///x"}, {"sha256": PAYLOAD_SHA256}, {"url": 7, "sha256": 7}])
def test_missing_weights_fields_are_named(tmp_path, capsys, weights):
    models_root = tmp_path / "models"
    (models_root / "Partial").mkdir(parents=True)
    (models_root / "Partial" / "manifest.json").write_text(json.dumps({"weights": weights}))

    code = fetch.main(["--model", "Partial", "--models-root", str(models_root)])
    output = capsys.readouterr().out

    assert code == fetch.EXIT_STRUCTURAL_ERROR
    assert "weights." in output


def test_discovery_is_sorted_and_manifest_scoped(tmp_path):
    models_root = tmp_path / "models"
    for name in ("Zulu", "Alpha", "Mike"):
        (models_root / name).mkdir(parents=True)
        (models_root / name / "manifest.json").write_text("{}")
    (models_root / "NoManifest").mkdir()

    assert fetch.discover_models(models_root) == ["Alpha", "Mike", "Zulu"]


def test_cli_rejects_model_and_all_together(tmp_path):
    with pytest.raises(SystemExit):
        fetch.main(["--model", "X", "--all"])


def test_cli_requires_a_target_for_fetching(tmp_path):
    with pytest.raises(SystemExit):
        fetch.main([])
