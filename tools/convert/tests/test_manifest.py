"""Schema-enforcement and round-trip tests for the manifest module.

The manifest is an identity document: a silently-accepted malformed manifest is worse than a
missing one, because it looks authoritative. These tests assert that every violation fails and
that the failure names the offending field.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from manifest import ManifestError, sha256_of_path, validate, write  # noqa: E402

VALID: dict = {
    "model_id": "RealESRGAN_x4plus",
    "scale_factor": 4,
    "weights": {
        "url": "https://example.invalid/RealESRGAN_x4plus.pth",
        "sha256": "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
        "bytes": 67040989,
        "state_dict_key": "params_ema",
        "parameters": 16697987,
    },
    "source": {
        "repo": "https://github.com/xinntao/Real-ESRGAN",
        "commit_or_release": "v0.1.0",
        "license": "BSD-3-Clause",
    },
    "conversion": {
        "script": "tools/convert/convert.py",
        "script_version": "1.0.0",
        "coremltools": "8.3.0",
        "torch": "2.5.0",
        "python": "3.12.14",
        "representation": "mlprogram",
        "precision": "float16",
        "input_shape": [1, 3, 256, 256],
        "output_shape": [1, 3, 1024, 1024],
        "min_deployment": "macOS14",
        "compute_units": "ALL",
        "parameters": {"tracer": "torch.jit.trace", "seed": 0},
    },
    "preprocessing": "RGB float [0,1], NCHW",
    "postprocessing": "clamp [0,1]",
    "outputs": {
        "mlpackage_sha256": "0" * 64,
        "reference_metrics": {"psnr_db": 65.36, "mae": 0.0004, "max_abs": 0.009},
    },
    "created_utc": "2026-08-20T15:00:00+00:00",
}


def test_valid_manifest_passes():
    assert validate(copy.deepcopy(VALID)) == VALID


@pytest.mark.parametrize(
    "path",
    [
        ("model_id",),
        ("scale_factor",),
        ("weights", "sha256"),
        ("weights", "state_dict_key"),
        ("source", "license"),
        ("conversion", "python"),
        ("conversion", "coremltools"),
        ("conversion", "compute_units"),
        ("outputs", "mlpackage_sha256"),
        ("outputs", "reference_metrics", "psnr_db"),
        ("created_utc",),
    ],
)
def test_missing_required_field_is_named(path):
    manifest = copy.deepcopy(VALID)
    node = manifest
    for key in path[:-1]:
        node = node[key]
    del node[path[-1]]

    with pytest.raises(ManifestError) as error:
        validate(manifest)
    assert ".".join(path) in str(error.value)
    assert "missing required field" in str(error.value)


def test_unexpected_field_is_rejected_and_named():
    manifest = copy.deepcopy(VALID)
    manifest["conversion"]["sneaky_extra"] = "unreviewed"

    with pytest.raises(ManifestError) as error:
        validate(manifest)
    assert "conversion.sneaky_extra" in str(error.value)


@pytest.mark.parametrize(
    "path,bad_value",
    [
        (("scale_factor",), "4"),
        (("weights", "bytes"), "67040989"),
        (("weights", "parameters"), 16697987.5),
        (("conversion", "parameters"), "none"),
        (("outputs", "reference_metrics", "mae"), "0.0004"),
    ],
)
def test_wrong_type_is_named(path, bad_value):
    manifest = copy.deepcopy(VALID)
    node = manifest
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = bad_value

    with pytest.raises(ManifestError) as error:
        validate(manifest)
    assert ".".join(path) in str(error.value)


@pytest.mark.parametrize("bad", ["not-a-hash", "ABCDEF" * 10 + "abcd", "a" * 63, 12345])
def test_malformed_sha256_is_rejected(bad):
    manifest = copy.deepcopy(VALID)
    manifest["weights"]["sha256"] = bad

    with pytest.raises(ManifestError) as error:
        validate(manifest)
    assert "weights.sha256" in str(error.value)


@pytest.mark.parametrize("bad", [[1, 3, 256], [1, 3, 256, 256, 1], [1, 3, 0, 256], "1,3,256,256", [1, 3, 256, -256]])
def test_malformed_shape_is_rejected(bad):
    manifest = copy.deepcopy(VALID)
    manifest["conversion"]["input_shape"] = bad

    with pytest.raises(ManifestError) as error:
        validate(manifest)
    assert "conversion.input_shape" in str(error.value)


def test_booleans_are_not_accepted_as_numbers():
    manifest = copy.deepcopy(VALID)
    manifest["scale_factor"] = True

    with pytest.raises(ManifestError) as error:
        validate(manifest)
    assert "scale_factor" in str(error.value)


def test_round_trip_preserves_content(tmp_path):
    target = tmp_path / "manifest.json"
    write(target, copy.deepcopy(VALID))

    assert json.loads(target.read_text()) == VALID
    assert validate(target) == VALID


def test_invalid_manifest_is_never_persisted(tmp_path):
    target = tmp_path / "manifest.json"
    broken = copy.deepcopy(VALID)
    del broken["conversion"]["torch"]

    with pytest.raises(ManifestError):
        write(target, broken)
    assert not target.exists(), "an invalid manifest must not reach disk"


def test_missing_file_reports_clearly(tmp_path):
    with pytest.raises(ManifestError) as error:
        validate(tmp_path / "nope.json")
    assert "manifest not found" in str(error.value)


def test_malformed_json_reports_clearly(tmp_path):
    target = tmp_path / "manifest.json"
    target.write_text("{not json")

    with pytest.raises(ManifestError) as error:
        validate(target)
    assert "not valid JSON" in str(error.value)


def test_directory_digest_is_stable_and_content_sensitive(tmp_path):
    bundle = tmp_path / "fake.mlpackage"
    (bundle / "Data").mkdir(parents=True)
    (bundle / "Manifest.json").write_text("{}")
    (bundle / "Data" / "model.mil").write_bytes(b"weights")

    first = sha256_of_path(bundle)
    assert first == sha256_of_path(bundle), "same tree must hash identically"

    (bundle / "Data" / "model.mil").write_bytes(b"weightz")
    assert sha256_of_path(bundle) != first, "content change must change the digest"


def test_file_digest_matches_known_value(tmp_path):
    target = tmp_path / "x.bin"
    target.write_bytes(b"")
    # SHA256 of the empty input — a fixed external constant, not recomputed by the code under test.
    assert sha256_of_path(target) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
