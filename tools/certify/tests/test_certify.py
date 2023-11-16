"""Tests for the certification harness.

Offline, fixture-based, and model-free: no weights, no `.mlpackage`, no corpus, no Swift build,
no network. What the model computes is settled by the integration run; what these tests prove is
that the harness cannot silently mis-derive a tile set, mis-shape a manifest, or bless outputs
that no longer match the evidence.

Runs with pytest + numpy only -- CI's actual environment (see the import-policy tests). The `.npy`
and `.png` fixtures are written as plain bytes or seeded arrays; only their hashes and their tensor
content matter here, never their image validity.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import certify  # noqa: E402
import manifest_emitter  # noqa: E402
import metrics  # noqa: E402

MODEL_ID = "FixtureModel"
TILE = 8
SCALE = 4
OUTPUT = TILE * SCALE
SHA_A = "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1"


@pytest.fixture()
def config():
    return certify.load_config()


# --- tile derivation ----------------------------------------------------------------------
def test_a_tile_sized_image_yields_exactly_one_position():
    assert certify.tile_positions(256, 256, 256, 128) == [(0, 0)]


@pytest.mark.parametrize("size", [(255, 256), (256, 255), (300, 200), (1, 1)])
def test_an_image_smaller_than_one_tile_yields_nothing(size):
    """Padding would certify a padding policy nobody has chosen; resizing would certify
    an input the corpus does not contain. Yielding nothing is the honest answer."""
    assert certify.tile_positions(size[0], size[1], 256, 128) == []


def test_positions_include_the_centre_and_all_four_corners():
    positions = set(certify.tile_positions(517, 512, 256, 128))
    assert (0, 0) in positions and (261, 0) in positions
    assert (0, 256) in positions and (261, 256) in positions
    assert ((517 - 256) // 2, (512 - 256) // 2) in positions


def test_interior_crops_follow_the_stride_and_stay_inside_the_image():
    positions = certify.tile_positions(1024, 768, 256, 128)
    assert (128, 128) in positions and (256, 384) in positions
    for x, y in positions:
        assert 0 <= x <= 1024 - 256 and 0 <= y <= 768 - 256


def test_positions_are_deduplicated_and_sorted():
    positions = certify.tile_positions(768, 768, 256, 128)
    assert positions == sorted(set(positions))


def test_tile_order_is_deterministic_for_a_given_seed(tmp_path, monkeypatch):
    images = [tmp_path / "a.png", tmp_path / "b.png"]
    fake_sizes(monkeypatch, images, (600, 600))

    first, _ = certify.derive_tiles(images, 256, 128, 0)
    second, _ = certify.derive_tiles(images, 256, 128, 0)
    assert [(r["source"], r["x"], r["y"]) for r in first] == [
        (r["source"], r["x"], r["y"]) for r in second
    ]


def test_a_different_seed_gives_a_different_order(tmp_path, monkeypatch):
    images = [tmp_path / "a.png"]
    fake_sizes(monkeypatch, images, (900, 900))

    first, _ = certify.derive_tiles(images, 256, 128, 0)
    second, _ = certify.derive_tiles(images, 256, 128, 99)
    assert [(r["x"], r["y"]) for r in first] != [(r["x"], r["y"]) for r in second]
    assert sorted((r["x"], r["y"]) for r in first) == sorted((r["x"], r["y"]) for r in second)


def test_excluded_images_are_recorded_with_a_reason_not_dropped(tmp_path, monkeypatch):
    small, large = tmp_path / "small.png", tmp_path / "large.png"
    sizes = {small: (100, 100), large: (600, 600)}
    for path in sizes:
        path.write_bytes(b"image-fixture-" + path.name.encode())
    monkeypatch.setattr(certify, "image_size", lambda path: sizes[Path(path)])

    _, sources = certify.derive_tiles([small, large], 256, 128, 0)
    by_path = {Path(s["path"]).name: s for s in sources}
    assert by_path["small.png"]["tile_count"] == 0
    assert "smaller than one 256x256 tile" in by_path["small.png"]["excluded_reason"]
    assert by_path["large.png"]["excluded_reason"] is None


def fake_sizes(monkeypatch, paths, size):
    """Give the fixture files a size without decoding them: CI has numpy, not PIL. Patched through
    monkeypatch so the substitution cannot leak into the tests that follow."""
    for path in paths:
        path.write_bytes(b"image-fixture-" + path.name.encode())
    monkeypatch.setattr(certify, "image_size", lambda _path: size)


def test_tile_names_carry_their_coordinates(tmp_path):
    record = {"source": tmp_path / "shot.jpg", "x": 130, "y": 128}
    assert certify.tile_name(record) == "shot_x130_y128"


# --- geometry comes from the model manifest -----------------------------------------------
def model_manifest(tile=TILE, scale=SCALE, output=None):
    output = output or [1, 3, tile * scale, tile * scale]
    return {
        "scale_factor": scale,
        "conversion": {"input_shape": [1, 3, tile, tile], "output_shape": output},
    }


def test_geometry_is_read_from_the_manifest_not_assumed():
    assert certify.geometry_from_manifest(model_manifest()) == (TILE, SCALE, OUTPUT)


def test_a_manifest_that_disagrees_with_itself_stops_the_run():
    broken = model_manifest(output=[1, 3, 999, 999])
    with pytest.raises(certify.CertifyError) as error:
        certify.geometry_from_manifest(broken)
    assert "disagrees with itself" in str(error.value)


def test_a_non_square_input_shape_stops_the_run():
    manifest = model_manifest()
    manifest["conversion"]["input_shape"] = [1, 3, 256, 512]
    with pytest.raises(certify.CertifyError):
        certify.geometry_from_manifest(manifest)


# --- the declared postprocessing stage ---------------------------------------------------
def test_postprocessing_is_applied_not_merely_read():
    """The defect this test was written for: `run_reference.write_tensor` stores the *raw* forward
    pass, and the product clamps before it returns. Comparing the two directly scores the clamp as
    model error — on the real corpus that turned a 3.2/255 worst pixel into 32.2/255, identically
    on every backend, which is precisely what a genuine numerical divergence would not do."""
    raw = np.array([[[[-0.08, 0.5], [1.13, 1.0]]]], dtype=np.float32)

    applied = certify.apply_postprocessing(raw, "clamp [0,1]")

    assert applied.min() >= 0.0 and applied.max() <= 1.0
    assert applied.dtype == np.float32
    assert applied[0, 0, 0, 1] == np.float32(0.5), "in-range values must pass through untouched"


def test_an_undeclared_postprocessing_stops_the_run():
    with pytest.raises(certify.CertifyError) as error:
        certify.apply_postprocessing(np.zeros((1, 1, 2, 2), dtype=np.float32), "no clamp")
    assert "does not" in str(error.value)


def test_the_overshoot_that_postprocessing_absorbs_is_measured():
    raw = np.array([[[[-0.08, 0.5], [1.13, 1.0]]]], dtype=np.float32)
    assert certify.out_of_range_fraction(raw) == 0.5
    assert certify.out_of_range_fraction(np.zeros((1, 1, 4, 4), dtype=np.float32)) == 0.0


# --- thresholds -----------------------------------------------------------------------------
def test_thresholds_come_from_the_config_not_from_the_code(config):
    assert config["thresholds"]["level_a"]["psnr_db_median_min"] == 38.0
    assert config["thresholds"]["level_a"]["max_abs_tolerance_255"] == 4.0
    assert config["thresholds"]["level_a"]["within_tolerance_fraction_min"] == 0.99
    source = Path(certify.__file__).read_text()
    assert "38.0" not in source, "a threshold literal in the harness would outrank the config"


def test_a_config_without_thresholds_is_refused(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"tiles": {"order_seed": 0}}))
    with pytest.raises(certify.CertifyError) as error:
        certify.load_config(path)
    assert "thresholds" in str(error.value)


def test_a_below_threshold_run_aggregates_to_fail(config):
    """A red result must be reachable: the harness has to be able to say FAIL."""
    entries = [make_entry(index, psnr=20.0, fraction=0.5) for index in range(3)]
    summary = certify.aggregate(entries, config["thresholds"])
    assert summary["passed"] is False
    assert summary["level_a"]["status"] == "FAIL"
    assert summary["level_a"]["psnr_criterion_met"] is False
    assert summary["level_a"]["max_abs_criterion_met"] is False


def test_a_run_that_meets_both_criteria_passes(config):
    entries = [make_entry(index, psnr=58.0, fraction=1.0) for index in range(3)]
    summary = certify.aggregate(entries, config["thresholds"])
    assert summary["passed"] is True and summary["level_a"]["status"] == "PASS"


def test_psnr_alone_cannot_carry_a_run_that_misses_the_pixel_criterion(config):
    entries = [make_entry(index, psnr=58.0, fraction=0.90) for index in range(3)]
    assert certify.aggregate(entries, config["thresholds"])["passed"] is False


def test_an_empty_tile_set_is_refused_rather_than_certified_vacuously(config):
    with pytest.raises(certify.CertifyError) as error:
        certify.aggregate([], config["thresholds"])
    assert "nothing to certify" in str(error.value)


def make_entry(index: int, *, psnr: float, fraction: float) -> dict:
    return {
        "index": index,
        "source_image": "corpus/x.jpg",
        "source_sha256": SHA_A,
        "x": 0,
        "y": 0,
        "width": TILE,
        "height": TILE,
        "tile_png": f"tiles/t{index}.png",
        "tile_sha256": SHA_A,
        "reference_npy": f"reference/t{index}.npy",
        "reference_sha256": SHA_A,
        "candidate_npy": f"candidate/t{index}.npy",
        "candidate_sha256": SHA_A,
        "metrics": {
            "psnr_db": psnr,
            "mae": 0.001,
            "max_abs": 0.01,
            "within_tolerance_fraction": fraction,
            "ssim": 0.999,
            "lpips": 0.0001,
        },
    }


def test_tile_metrics_wires_every_field_through(monkeypatch):
    """LPIPS is stubbed: it needs torch, and this suite must collect with numpy alone."""
    monkeypatch.setattr(metrics, "lpips_distance", lambda a, b: 0.25)
    rng = np.random.default_rng(3)
    reference = rng.random((1, 3, 16, 16), dtype=np.float32)
    candidate = (reference + 0.001).astype(np.float32)

    values = certify.tile_metrics(reference, candidate, 4.0 / metrics.BYTE)

    assert set(values) == {"psnr_db", "mae", "max_abs", "within_tolerance_fraction", "ssim", "lpips"}
    assert values["lpips"] == 0.25
    assert values["psnr_db"] == pytest.approx(metrics.psnr(reference, candidate))
    assert values["within_tolerance_fraction"] == 1.0


def test_bit_identical_tensors_record_a_null_psnr_not_a_ceiling(monkeypatch):
    monkeypatch.setattr(metrics, "lpips_distance", lambda a, b: 0.0)
    array = np.random.default_rng(4).random((1, 3, 16, 16), dtype=np.float32)

    values = certify.tile_metrics(array, array, 4.0 / metrics.BYTE)

    assert values["psnr_db"] is None, "infinity is not a measurement and must not be invented"
    assert values["max_abs"] == 0.0


# --- manifest schema ------------------------------------------------------------------------
def build_certification(tmp_path: Path, tiles: int = 3, noise: float = 0.002) -> tuple[Path, dict]:
    """A complete, self-consistent certification tree: artifacts on disk plus its manifest."""
    config = certify.load_config()
    tolerance = config["thresholds"]["level_a"]["max_abs_tolerance_255"] / metrics.BYTE
    models_root = tmp_path / "models"
    model_dir = models_root / MODEL_ID
    weights = model_dir / "weights" / f"{MODEL_ID}.pth"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"weights-fixture")
    # `--check` asks tools/artifacts/fetch.py where the weights live rather than guessing, so the
    # fixture needs the manifest that derivation reads.
    (model_dir / "manifest.json").write_text(
        json.dumps({"weights": {"url": f"https://example.invalid/{MODEL_ID}.pth", "sha256": SHA_A}})
    )
    package = model_dir / "output" / f"{MODEL_ID}_{TILE}x{TILE}.mlpackage"
    package.mkdir(parents=True)
    (package / "coremldata.bin").write_bytes(b"mlpackage-fixture")

    paths = certify.certification_paths(models_root, MODEL_ID, "all")
    rng = np.random.default_rng(11)
    entries = []
    for index in range(tiles):
        name = f"fixture_x0_y{index}"
        source = tmp_path / "corpus" / "fixture.jpg"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"corpus-fixture")

        reference = rng.random((1, 3, OUTPUT, OUTPUT), dtype=np.float32)
        candidate = (reference + rng.normal(0, noise, reference.shape)).astype(np.float32)
        tile_png = paths["tiles"] / f"{name}.png"
        tile_png.parent.mkdir(parents=True, exist_ok=True)
        tile_png.write_bytes(b"tile-fixture-" + name.encode())
        reference_npy = paths["reference"] / f"{name}.npy"
        candidate_npy = paths["candidate"] / f"{name}.npy"
        for path, array in ((reference_npy, reference), (candidate_npy, candidate)):
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, array, allow_pickle=False)

        entries.append(
            {
                "index": index,
                "source_image": certify.relative(source),
                "source_sha256": certify.run_reference.sha256_of_file(source),
                "x": 0,
                "y": index,
                "width": TILE,
                "height": TILE,
                "tile_png": certify.relative(tile_png),
                "tile_sha256": certify.run_reference.sha256_of_file(tile_png),
                "reference_npy": certify.relative(reference_npy),
                "reference_sha256": certify.run_reference.sha256_of_file(reference_npy),
                "candidate_npy": certify.relative(candidate_npy),
                "candidate_sha256": certify.run_reference.sha256_of_file(candidate_npy),
                # Built field by field rather than through `certify.tile_metrics`, which also
                # computes LPIPS: that pulls in torch, and CI has numpy only (import policy).
                # `tile_metrics` is covered separately with LPIPS stubbed out.
                "metrics": {
                    "psnr_db": metrics.psnr(reference, candidate),
                    "mae": metrics.mae(reference, candidate),
                    "max_abs": metrics.max_abs_error(reference, candidate),
                    "within_tolerance_fraction": metrics.within_tolerance_fraction(
                        reference, candidate, tolerance
                    ),
                    "ssim": metrics.ssim(reference, candidate),
                    "lpips": 0.00012,  # fixture constant; the real value comes from the run
                },
            }
        )

    summary = certify.aggregate(entries, config["thresholds"])
    manifest = manifest_emitter.build(
        certification_id="fixture-all-20260821T000000Z-abcdef123456",
        date="2026-08-21T00:00:00+00:00",
        model_id=MODEL_ID,
        weights_sha256=certify.run_reference.sha256_of_file(weights),
        mlpackage_sha256=manifest_emitter.sha256_of_path(package),
        conversion={
            "script_version": "1.0.0",
            "coremltools": "8.3.0",
            "python": "3.12.14",
            "precision": "float16",
            "input_shape": [1, 3, TILE, TILE],
            "parameters": {"tracer": "torch.jit.trace", "seed": 0},
        },
        reference={
            "implementation": "fixture",
            "commit": "0" * 40,
            "output_hashes": [entry["reference_sha256"] for entry in entries],
        },
        coreml_output_hashes=[entry["candidate_sha256"] for entry in entries],
        corpus_version="sha256:deadbeefdeadbeef (1 image(s))",
        device="Fixture (arm64)",
        os_version="macOS 26.5.2 (25F84)",
        coreml_version="CoreML.framework 0.0.0 (build 0)",
        compute_units="all",
        tiles={
            "tile_size": TILE,
            "tile_set": certify.tile_set_label(
                [{"path": "x", "tile_count": tiles}], len(entries)
            ),
            "order_seed": 0,
            "derivation": "fixture",
            "source_images": [
                {
                    "path": certify.relative(tmp_path / "corpus" / "fixture.jpg"),
                    "sha256": entries[0]["source_sha256"],
                    "width": 64,
                    "height": 64,
                    "tile_count": tiles,
                    "excluded_reason": None,
                }
            ],
            "entries": entries,
        },
        results=certify.build_results(summary["level_a"], summary["level_b"]),
        status="PASS" if summary["passed"] else "FAIL",
    )
    manifest_emitter.write(paths["manifest"], manifest)
    return models_root, manifest


def test_a_freshly_built_manifest_validates(tmp_path):
    _, manifest = build_certification(tmp_path)
    assert manifest_emitter.validate(manifest)["status"] == "PASS"


def test_every_section_11_field_is_present(tmp_path):
    _, manifest = build_certification(tmp_path)
    for field in (
        "certification_id", "date", "model_id", "weights_sha256", "mlpackage_sha256",
        "conversion", "reference", "coreml_output_hashes", "corpus_version", "device",
        "os_version", "coreml_version", "compute_units", "results", "status",
    ):
        assert field in manifest, f"certification spec §11 field missing: {field}"
    for field in ("level_a", "level_b", "level_c", "performance", "energy"):
        assert field in manifest["results"]
    assert manifest["results"]["energy"] == {"value": None, "kind": "unavailable"}


def test_the_tile_set_is_named_so_a_starter_run_cannot_pose_as_gate_1_evidence(tmp_path):
    _, manifest = build_certification(tmp_path)
    assert manifest["tiles"]["tile_set"].startswith("starter (3 tiles, 1 source image)")


def test_the_tile_set_label_counts_contributors_and_names_exclusions():
    sources = [
        {"path": "big.jpg", "tile_count": 12},
        {"path": "small.jpg", "tile_count": 0},
    ]
    label = certify.tile_set_label(sources, 12)
    assert label.startswith("starter (12 tiles, 1 source image)")
    assert "1 corpus image(s) excluded" in label

    plural = certify.tile_set_label(
        [{"path": "a", "tile_count": 4}, {"path": "b", "tile_count": 8}], 12
    )
    assert plural == "starter (12 tiles, 2 source images)"


def test_a_manifest_without_a_tile_set_label_is_refused(tmp_path):
    _, manifest = build_certification(tmp_path)
    del manifest["tiles"]["tile_set"]
    with pytest.raises(manifest_emitter.ManifestError) as error:
        manifest_emitter.validate(manifest)
    assert "tile_set" in str(error.value)


def test_compute_units_are_always_recorded(tmp_path):
    _, manifest = build_certification(tmp_path)
    assert manifest["compute_units"] == "all"


@pytest.mark.parametrize(
    "field", ["certification_id", "model_id", "compute_units", "corpus_version", "status"]
)
def test_a_missing_field_is_named(tmp_path, field):
    _, manifest = build_certification(tmp_path)
    del manifest[field]
    with pytest.raises(manifest_emitter.ManifestError) as error:
        manifest_emitter.validate(manifest)
    assert field in str(error.value)


def test_an_unexpected_field_is_refused(tmp_path):
    _, manifest = build_certification(tmp_path)
    manifest["extra_credit"] = 1
    with pytest.raises(manifest_emitter.ManifestError) as error:
        manifest_emitter.validate(manifest)
    assert "extra_credit" in str(error.value) and "strict" in str(error.value)


def test_an_invented_status_is_refused(tmp_path):
    _, manifest = build_certification(tmp_path)
    manifest["status"] = "MOSTLY_PASS"
    with pytest.raises(manifest_emitter.ManifestError) as error:
        manifest_emitter.validate(manifest)
    assert "MOSTLY_PASS" in str(error.value)


def test_unavailable_energy_may_not_carry_a_number(tmp_path):
    _, manifest = build_certification(tmp_path)
    manifest["results"]["energy"] = {"value": 1.2, "kind": "unavailable"}
    with pytest.raises(manifest_emitter.ManifestError) as error:
        manifest_emitter.validate(manifest)
    assert "unavailable" in str(error.value)


def test_hash_lists_must_agree_with_the_per_tile_records(tmp_path):
    _, manifest = build_certification(tmp_path)
    manifest["coreml_output_hashes"][0] = SHA_A
    with pytest.raises(manifest_emitter.ManifestError) as error:
        manifest_emitter.validate(manifest)
    assert "coreml_output_hashes" in str(error.value)


def test_dropping_an_image_without_recording_it_is_refused(tmp_path):
    """Coverage and the tile set have to tell the same story, or an excluded image could vanish."""
    _, manifest = build_certification(tmp_path)
    manifest["tiles"]["source_images"][0]["tile_count"] += 1
    with pytest.raises(manifest_emitter.ManifestError) as error:
        manifest_emitter.validate(manifest)
    assert "coverage disagrees" in str(error.value)


def test_tile_order_is_part_of_the_evidence(tmp_path):
    _, manifest = build_certification(tmp_path)
    manifest["tiles"]["entries"][1]["index"] = 7
    with pytest.raises(manifest_emitter.ManifestError) as error:
        manifest_emitter.validate(manifest)
    assert "tile order is evidence" in str(error.value)


def test_an_infinite_metric_is_never_written_as_json(tmp_path):
    _, manifest = build_certification(tmp_path)
    manifest["tiles"]["entries"][0]["metrics"]["mae"] = float("inf")
    with pytest.raises(manifest_emitter.ManifestError) as error:
        manifest_emitter.write(tmp_path / "out.json", manifest)
    assert "Infinity" in str(error.value)


def test_only_psnr_may_be_null(tmp_path):
    _, manifest = build_certification(tmp_path)
    manifest["tiles"]["entries"][0]["metrics"]["ssim"] = None
    with pytest.raises(manifest_emitter.ManifestError) as error:
        manifest_emitter.validate(manifest)
    assert "psnr_db" in str(error.value)


# --- --check ---------------------------------------------------------------------------------
def test_check_validates_a_manifest_it_just_emitted(tmp_path, config, capsys):
    models_root, _ = build_certification(tmp_path)
    assert certify.run_check(MODEL_ID, "all", models_root, config) == certify.EXIT_OK
    assert "is valid" in capsys.readouterr().out


def test_a_byte_flipped_tensor_fails_the_check_and_is_named(tmp_path, config, capsys):
    models_root, manifest = build_certification(tmp_path)
    target = Path(manifest["tiles"]["entries"][1]["candidate_npy"])
    data = bytearray(target.read_bytes())
    data[-1] ^= 0x01  # one bit, in the last float of the tensor
    target.write_bytes(bytes(data))

    assert certify.run_check(MODEL_ID, "all", models_root, config) == certify.EXIT_FAILED
    output = capsys.readouterr().out
    assert target.name in output and "hash mismatch" in output


def test_a_deleted_tensor_fails_the_check_and_is_named(tmp_path, config, capsys):
    models_root, manifest = build_certification(tmp_path)
    target = Path(manifest["tiles"]["entries"][0]["reference_npy"])
    target.unlink()

    assert certify.run_check(MODEL_ID, "all", models_root, config) == certify.EXIT_FAILED
    assert target.name in capsys.readouterr().out


def test_a_tampered_metric_fails_the_check_even_with_intact_files(tmp_path, config, capsys):
    """Re-hashing alone would bless a manifest whose numbers were edited after the fact."""
    models_root, manifest = build_certification(tmp_path)
    paths = certify.certification_paths(models_root, MODEL_ID, "all")
    manifest["tiles"]["entries"][0]["metrics"]["ssim"] = 0.5
    manifest_emitter.write(paths["manifest"], manifest)

    assert certify.run_check(MODEL_ID, "all", models_root, config) == certify.EXIT_FAILED
    assert "ssim recomputes to" in capsys.readouterr().out


def test_a_forged_verdict_fails_the_check(tmp_path, config, capsys):
    models_root, manifest = build_certification(tmp_path)
    paths = certify.certification_paths(models_root, MODEL_ID, "all")
    manifest["results"]["level_a"]["psnr_db_median"] = 99.0
    manifest_emitter.write(paths["manifest"], manifest)

    assert certify.run_check(MODEL_ID, "all", models_root, config) == certify.EXIT_FAILED
    assert "psnr_db_median" in capsys.readouterr().out


def test_a_modified_weights_artifact_fails_the_check(tmp_path, config, capsys):
    models_root, _ = build_certification(tmp_path)
    (models_root / MODEL_ID / "weights" / f"{MODEL_ID}.pth").write_bytes(b"different weights")

    assert certify.run_check(MODEL_ID, "all", models_root, config) == certify.EXIT_FAILED
    assert "weights_sha256" in capsys.readouterr().out


def test_checking_the_wrong_backends_manifest_is_refused(tmp_path, config, capsys):
    """Manifests are per backend and never comparable across them."""
    models_root, _ = build_certification(tmp_path)
    paths = certify.certification_paths(models_root, MODEL_ID, "all")
    (models_root / MODEL_ID / "certification" / "cpuOnly").mkdir(parents=True)
    (models_root / MODEL_ID / "certification" / "cpuOnly" / "CERTIFICATION_MANIFEST.json").write_text(
        paths["manifest"].read_text()
    )

    assert certify.run_check(MODEL_ID, "cpuOnly", models_root, config) == certify.EXIT_FAILED
    assert "compute_units" in capsys.readouterr().out


def test_check_without_a_manifest_stops_with_the_command_to_run(tmp_path, config):
    with pytest.raises(certify.CertifyError) as error:
        certify.run_check(MODEL_ID, "all", tmp_path / "models", config)
    assert "certify.py" in str(error.value)


# --- the compute-units requirement ------------------------------------------------------------
def test_the_harness_refuses_to_run_without_compute_units(capsys):
    code = certify.main(["--model", MODEL_ID])
    assert code == certify.EXIT_STRUCTURAL_ERROR
    output = capsys.readouterr().out
    assert "--compute-units is required" in output
    assert "14/255" in output, "the refusal should name the finding that motivates it"


def test_an_unknown_backend_is_rejected_by_the_parser():
    with pytest.raises(SystemExit):
        certify.main(["--model", MODEL_ID, "--compute-units", "gpu"])


def test_every_backend_the_engine_supports_is_offered():
    assert certify.COMPUTE_UNITS == ("all", "cpuAndGPU", "cpuAndNeuralEngine", "cpuOnly")


# --- import policy ---------------------------------------------------------------------------
@pytest.mark.parametrize("module", ["certify", "manifest_emitter", "metrics"])
def test_heavy_imports_stay_inside_their_functions(module):
    import ast
    import importlib

    forbidden = {"torch", "lpips", "skimage", "PIL", "yaml", "scipy", "coremltools"}
    source = Path(importlib.import_module(module).__file__).read_text()
    offenders = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.Import):
            offenders += [n.name.split(".")[0] for n in node.names if n.name.split(".")[0] in forbidden]
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in forbidden:
            offenders.append(node.module)

    assert not offenders, f"{module}.py imports {offenders} at module scope (import policy)"


def test_the_strict_schema_engine_is_still_where_this_module_expects_it():
    """manifest_emitter reuses tools/convert/manifest.py's validator rather than duplicating it.
    If that private helper is ever renamed, fail here rather than at certification time."""
    assert callable(getattr(manifest_emitter.conversion_manifest, "_check", None))


# --- declared vs executed pipeline -------------------------------------------------------------
#
# A manifest can *declare* a pipeline; this suite exists because declaring
# is not executing. The harness read `postprocessing: "clamp [0,1]"` from the model manifest and
# compared against an unclamped oracle anyway, which scored the product's clamp as model error
# (29.45/255 where the true figure was 2.18/255). These tests close that gap structurally: they
# run the real code path with the model stubbed out, and fail if the declared stage stops being
# executed. Model-free and numpy-only, so CI runs them.

FULL_MODEL_MANIFEST = {
    "model_id": MODEL_ID,
    "scale_factor": SCALE,
    "weights": {
        "url": f"https://example.invalid/{MODEL_ID}.pth",
        "sha256": SHA_A,
        "bytes": 1,
        "state_dict_key": "params_ema",
        "parameters": 1,
    },
    "source": {"repo": "https://example.invalid", "commit_or_release": "v0", "license": "BSD-3-Clause"},
    "conversion": {
        "script": "tools/convert/convert.py",
        "script_version": "1.0.0",
        "coremltools": "8.3.0",
        "torch": "2.5.0",
        "python": "3.12.14",
        "representation": "mlprogram",
        "precision": "float16",
        "input_shape": [1, 3, TILE, TILE],
        "output_shape": [1, 3, OUTPUT, OUTPUT],
        "min_deployment": "macOS14",
        "compute_units": "ALL",
        "parameters": {"tracer": "torch.jit.trace", "seed": 0},
    },
    "preprocessing": "RGB float [0,1], NCHW",
    "postprocessing": "clamp [0,1]",
    "outputs": {
        "mlpackage_sha256": SHA_A,
        "reference_metrics": {"psnr_db": 65.0, "mae": 0.0004, "max_abs": 0.009},
    },
    "created_utc": "2026-08-21T00:00:00+00:00",
}

#: The oracle's raw forward pass overshoots [0,1] — the condition that exposed the defect.
OVERSHOOT_LOW, OVERSHOOT_HIGH = -0.076, 1.126


def stub_certification_run(tmp_path, monkeypatch, *, postprocessing="clamp [0,1]"):
    """Wire `run_certify` to a stubbed model and product, leaving every real code path in place."""
    models_root = tmp_path / "models"
    model_dir = models_root / MODEL_ID
    model_dir.mkdir(parents=True)
    manifest = json.loads(json.dumps(FULL_MODEL_MANIFEST))
    manifest["postprocessing"] = postprocessing
    (model_dir / "manifest.json").write_text(json.dumps(manifest))
    (model_dir / "weights").mkdir()
    weights = model_dir / "weights" / f"{MODEL_ID}.pth"
    weights.write_bytes(b"weights-fixture")
    # The stub below must report the hash the file actually has: `--check` compares the manifest
    # against the artifact on disk, and it is right to fail when those disagree.
    weights_sha = certify.run_reference.sha256_of_file(weights)
    package = model_dir / "output" / f"{MODEL_ID}_{TILE}x{TILE}.mlpackage"
    package.mkdir(parents=True)
    (package / "coremldata.bin").write_bytes(b"mlpackage-fixture")

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "big.png").write_bytes(b"big-image-fixture")
    (corpus / "tiny.png").write_bytes(b"tiny-image-fixture")
    sizes = {"big.png": (TILE, TILE), "tiny.png": (TILE - 1, TILE - 1)}

    raw = np.full((1, 3, OUTPUT, OUTPUT), 0.5, dtype=np.float32)
    raw[0, 0, 0, 0] = OVERSHOOT_HIGH        # above the declared range
    raw[0, 1, 0, 1] = OVERSHOOT_LOW         # below it

    def fake_candidate(command):
        # The product's own contract, asserted where it is actually used.
        assert "--emit-npy" in command, "the candidate must be asked for its tensor"
        assert "--compute-units" in command, "compute units must be pinned on every invocation"
        target = Path(command[command.index("--emit-npy") + 1])
        target.parent.mkdir(parents=True, exist_ok=True)
        # What the product returns is already clamped: it is an 8-bit surface widened to float32.
        np.save(target, np.clip(raw, 0.0, 1.0).astype(np.float32), allow_pickle=False)

    monkeypatch.setattr(certify, "image_size", lambda path: sizes[Path(path).name])
    monkeypatch.setattr(certify, "crop_tile", lambda s, x, y, n, dest: (
        dest.parent.mkdir(parents=True, exist_ok=True), dest.write_bytes(b"tile-fixture")))
    monkeypatch.setattr(certify, "run_candidate", fake_candidate)
    monkeypatch.setattr(certify, "swift_identity", lambda: "swift-fixture")
    monkeypatch.setattr(certify.run_reference, "load_model_for",
                        lambda model_id: (object(), weights_sha, "RGB float [0,1], NCHW",
                                          postprocessing))
    monkeypatch.setattr(certify.run_reference, "preprocess", lambda path: np.zeros((1, 3, TILE, TILE),
                                                                                  dtype=np.float32))
    monkeypatch.setattr(certify.run_reference, "forward", lambda model, tensor: raw)
    monkeypatch.setattr(certify.run_reference, "postprocess_to_png",
                        lambda array, path: path.write_bytes(b"png-view-fixture"))
    monkeypatch.setattr(certify.run_reference, "current_environment", lambda: {"torch": "2.5.0"})
    monkeypatch.setattr(certify.metrics, "lpips_distance", lambda a, b: 0.0001)
    return models_root, corpus


def test_the_declared_postprocessing_is_actually_executed_by_the_oracle_path(tmp_path, monkeypatch, config):
    """The regression proper: run the real `run_certify` and read the tensor it wrote to disk."""
    models_root, corpus = stub_certification_run(tmp_path, monkeypatch)

    code = certify.run_certify(MODEL_ID, "all", models_root, corpus, config)

    assert code == certify.EXIT_OK
    paths = certify.certification_paths(models_root, MODEL_ID, "all")
    written = list(paths["reference"].glob("*.npy"))
    assert written, "the oracle wrote no reference tensor"
    for path in written:
        tensor = np.load(path)
        assert tensor.min() >= 0.0 and tensor.max() <= 1.0, (
            f"{path.name}: the declared postprocessing was not executed — the reference still "
            "carries values outside [0,1], which the product's clamp would be blamed for"
        )


def test_the_run_records_that_it_applied_the_declared_stage(tmp_path, monkeypatch, config):
    models_root, corpus = stub_certification_run(tmp_path, monkeypatch)
    certify.run_certify(MODEL_ID, "all", models_root, corpus, config)

    manifest = manifest_emitter.load(
        certify.certification_paths(models_root, MODEL_ID, "all")["manifest"])
    assert "clamp [0,1]" in manifest["reference"]["implementation"]
    assert "postprocessing" in manifest["reference"]["implementation"]


def test_a_manifest_declaring_an_unimplemented_pipeline_stops_the_run(tmp_path, monkeypatch, config):
    """Declaring a pipeline the harness cannot execute must halt, never silently apply another."""
    models_root, corpus = stub_certification_run(tmp_path, monkeypatch, postprocessing="no clamp")

    with pytest.raises((certify.CertifyError, certify.run_reference.ReferenceError)):
        certify.run_certify(MODEL_ID, "all", models_root, corpus, config)


def test_the_oracle_forward_pass_has_exactly_one_call_site(tmp_path, monkeypatch):
    """`oracle_tensor` is the only place the forward pass is taken, so postprocessing cannot be
    bypassed by a future edit that calls `run_reference.forward` directly again."""
    import ast

    tree = ast.parse(Path(certify.__file__).read_text())
    callers = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr == "forward"
            ):
                callers.append(node.name)

    assert callers == ["oracle_tensor"], (
        f"the forward pass is taken in {callers}; it belongs only in oracle_tensor, which applies "
        "the manifest's declared postprocessing"
    )


def test_the_stubbed_run_reports_the_overshoot_the_clamp_absorbs(tmp_path, monkeypatch, config):
    models_root, corpus = stub_certification_run(tmp_path, monkeypatch)
    certify.run_certify(MODEL_ID, "all", models_root, corpus, config)

    report = certify.certification_paths(models_root, MODEL_ID, "all")["report"].read_text()
    assert "outside" in report and "[0,1]" in report
    assert "8-bit product surface" in report, "the surface's wording must appear in the report"
    assert "attributable to the surface itself" in report


def test_the_stubbed_run_names_its_tile_set_and_excluded_image(tmp_path, monkeypatch, config):
    models_root, corpus = stub_certification_run(tmp_path, monkeypatch)
    certify.run_certify(MODEL_ID, "all", models_root, corpus, config)

    manifest = manifest_emitter.load(
        certify.certification_paths(models_root, MODEL_ID, "all")["manifest"])
    assert manifest["tiles"]["tile_set"].startswith("starter (1 tiles, 1 source image)")
    assert "excluded" in manifest["tiles"]["tile_set"]
    assert manifest["compute_units"] == "all"


def test_a_stubbed_run_then_checks_clean(tmp_path, monkeypatch, config):
    """End to end: certify, then validate the manifest it just wrote against the files on disk."""
    models_root, corpus = stub_certification_run(tmp_path, monkeypatch)
    certify.run_certify(MODEL_ID, "all", models_root, corpus, config)

    assert certify.run_check(MODEL_ID, "all", models_root, config) == certify.EXIT_OK
