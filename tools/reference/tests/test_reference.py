"""Tests for the reference oracle.

Runs with only pytest + numpy installed -- CI's actual environment (see the import-policy tests). No
network, no corpus, no weights, no forward pass, and nothing here imports torch, PIL or yaml,
directly or transitively. What the model computes is verified by the integration run; what these
tests prove is that a reference output can never be silently unattributable, corrupted, or
non-reproducible in its serialisation.

Fixture files for the PNG view and the corpus input are written as plain bytes: every assertion here
is about hashing and attribution, and only their content matters, not their image validity. Rendering
real PNGs is PIL's job and is exercised by the integration run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_reference as ref  # noqa: E402

MODEL_ID = "TestModel"
WEIGHTS_SHA = "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1"
ENVIRONMENT = {"torch": "2.5.0", "python": "3.12.14"}


def synthetic_tensor(seed: int = 0) -> np.ndarray:
    """A seeded array, not a photo: the fixture must not depend on the corpus."""
    rng = np.random.default_rng(seed)
    return rng.random((1, 3, 8, 8), dtype=np.float32)


def make_entry(tmp_path: Path, array: np.ndarray | None = None) -> tuple[dict, Path, Path, Path]:
    array = synthetic_tensor() if array is None else array
    source = tmp_path / "corpus" / "fixture.png"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"corpus-input-fixture")

    npy_path = tmp_path / "references" / MODEL_ID / "fixture.npy"
    png_path = tmp_path / "references" / MODEL_ID / "fixture.png"
    ref.write_tensor(array, npy_path)
    png_path.write_bytes(b"derived-8bit-view-fixture")

    entry = ref.build_entry(
        model_id=MODEL_ID, weights_sha256=WEIGHTS_SHA, input_path=source,
        npy_path=npy_path, png_path=png_path, array=array,
        preprocessing=ref.SUPPORTED_PREPROCESSING, postprocessing=ref.SUPPORTED_POSTPROCESSING,
        environment=ENVIRONMENT,
    )
    return entry, source, npy_path, png_path


# --- serialisation determinism ------------------------------------------------------------
def test_same_tensor_serialises_to_the_same_hashes(tmp_path):
    array = synthetic_tensor()
    first, second = tmp_path / "a.npy", tmp_path / "b.npy"

    ref.write_tensor(array, first)
    ref.write_tensor(array, second)

    assert ref.sha256_of_file(first) == ref.sha256_of_file(second)


def test_different_tensors_serialise_differently(tmp_path):
    a_npy, b_npy = tmp_path / "a.npy", tmp_path / "b.npy"

    ref.write_tensor(synthetic_tensor(0), a_npy)
    ref.write_tensor(synthetic_tensor(1), b_npy)

    assert ref.sha256_of_file(a_npy) != ref.sha256_of_file(b_npy)


def test_npy_roundtrip_is_bit_exact_float32(tmp_path):
    array = synthetic_tensor()
    npy_path = tmp_path / "x.npy"
    ref.write_tensor(array, npy_path)

    reloaded = np.load(npy_path)
    assert reloaded.dtype == np.float32
    assert np.array_equal(reloaded, array), "the .npy must be the exact tensor, not a requantised view"


# --- manifest -----------------------------------------------------------------------------
def test_manifest_round_trip(tmp_path):
    entry, *_ = make_entry(tmp_path)
    manifest_path = tmp_path / "references_manifest.json"

    ref.save_manifest(manifest_path, {"entries": [entry]})
    reloaded = ref.load_manifest(manifest_path)

    assert reloaded == {"entries": [entry]}
    assert json.loads(manifest_path.read_text())["entries"][0]["model_id"] == MODEL_ID


def test_entry_records_full_attribution(tmp_path):
    entry, *_ = make_entry(tmp_path)

    for field in (
        "model_id", "weights_sha256", "input", "input_sha256", "preprocessing", "postprocessing",
        "output_npy", "output_npy_sha256", "output_png", "output_png_sha256",
        "shape", "dtype", "environment", "created_utc",
    ):
        assert field in entry, f"attribution field missing: {field}"
    assert entry["environment"] == ENVIRONMENT
    assert entry["dtype"] == "float32"
    assert entry["shape"] == [1, 3, 8, 8]
    assert set(entry["environment"]) == {"torch", "python"}


def test_missing_manifest_is_an_empty_manifest(tmp_path):
    assert ref.load_manifest(tmp_path / "absent.json") == {"entries": []}


def test_malformed_manifest_names_the_file(tmp_path):
    path = tmp_path / "references_manifest.json"
    path.write_text("{not json")

    with pytest.raises(ref.ReferenceError) as error:
        ref.load_manifest(path)
    assert "references_manifest.json" in str(error.value)


# --- corruption detection -----------------------------------------------------------------
def test_intact_references_verify_clean(tmp_path):
    entry, *_ = make_entry(tmp_path)
    assert ref.verify_recorded_files([entry], tmp_path) == []


def test_single_byte_flip_in_npy_is_caught_and_named(tmp_path):
    entry, _, npy_path, _ = make_entry(tmp_path)

    data = bytearray(npy_path.read_bytes())
    data[-1] ^= 0x01                      # one bit, in the last float of the tensor
    npy_path.write_bytes(bytes(data))

    failures = ref.verify_recorded_files([entry], tmp_path)
    assert len(failures) == 1
    assert entry["output_npy"] in failures[0]
    assert "hash mismatch" in failures[0]


def test_corrupted_png_is_caught_and_named(tmp_path):
    entry, _, _, png_path = make_entry(tmp_path)
    png_path.write_bytes(png_path.read_bytes() + b"\x00")

    failures = ref.verify_recorded_files([entry], tmp_path)
    assert len(failures) == 1 and entry["output_png"] in failures[0]


def test_modified_input_is_caught_and_named(tmp_path):
    entry, source, _, _ = make_entry(tmp_path)
    source.write_bytes(b"a different corpus image")

    failures = ref.verify_recorded_files([entry], tmp_path)
    assert any(entry["input"] in f for f in failures), "a changed input invalidates its reference"


def test_deleted_output_is_caught_and_named(tmp_path):
    entry, _, npy_path, _ = make_entry(tmp_path)
    npy_path.unlink()

    failures = ref.verify_recorded_files([entry], tmp_path)
    assert len(failures) == 1
    assert entry["output_npy"] in failures[0] and "missing" in failures[0]


# --- corpus discovery + empty corpus ------------------------------------------------------
def _generate(corpus, manifest_path, tmp_path):
    return ref.main([
        "--model", MODEL_ID,
        "--corpus-dir", str(corpus),
        "--references-dir", str(tmp_path / "references"),
        "--manifest", str(manifest_path),
    ])


def test_empty_corpus_is_refused_not_recorded_as_zero(tmp_path, capsys):
    """Replaces an earlier `test_empty_corpus_exits_zero_and_records_nothing`.

    That test asserted "an empty oracle is not a failure" and pinned the destructive behaviour
    below. The corpus is gitignored, so an empty one is what a fresh checkout *always* has.
    """
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    manifest_path = tmp_path / "references_manifest.json"

    code = _generate(corpus, manifest_path, tmp_path)

    assert code == ref.EXIT_STRUCTURAL_ERROR
    out = capsys.readouterr().out
    assert "corpus empty" in out
    assert "REFUSING to record" in out


def test_an_empty_corpus_does_not_delete_the_committed_oracle(tmp_path, capsys):
    """The defect this rule exists for, stated as the failure it caused.

    `run_reference.py --model X` on a fresh checkout deleted every entry for X and exited 0. The
    reproducibility check hit exactly this: 53 committed references erased by the
    record step, after which `--verify` truthfully reported nothing to verify -- and also exited 0.
    """
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    manifest_path = tmp_path / "references_manifest.json"
    committed = {"entries": [
        {"model_id": MODEL_ID, "variant": ref.VARIANT_FULL_FRAME, "input": "corpus/a.jpg"},
        {"model_id": MODEL_ID, "variant": ref.VARIANT_FULL_FRAME, "input": "corpus/b.jpg"},
        {"model_id": "other-model", "variant": ref.VARIANT_FULL_FRAME, "input": "corpus/c.jpg"},
    ]}
    ref.save_manifest(manifest_path, committed)

    code = _generate(corpus, manifest_path, tmp_path)

    assert code == ref.EXIT_STRUCTURAL_ERROR
    assert ref.load_manifest(manifest_path) == committed, "the manifest must be byte-for-byte intact"
    assert "2 entry(s) for" in capsys.readouterr().out, "the refusal names what it protected"


def test_absent_corpus_directory_is_also_empty(tmp_path):
    assert ref.discover_corpus(tmp_path / "does-not-exist") == []


def test_discovery_is_recursive_sorted_and_extension_filtered(tmp_path):
    corpus = tmp_path / "corpus"
    (corpus / "faces").mkdir(parents=True)
    for name in ("b.png", "a.jpg", "notes.txt", "faces/c.tiff"):
        (corpus / name).write_bytes(b"x")

    found = [p.name for p in ref.discover_corpus(corpus)]
    assert found == ["a.jpg", "b.png", "c.tiff"], "sorted, recursive, images only"


def test_stem_collision_is_refused_naming_both(tmp_path):
    corpus = tmp_path / "corpus"
    (corpus / "sub").mkdir(parents=True)
    (corpus / "shot.png").write_bytes(b"x")
    (corpus / "sub" / "shot.jpg").write_bytes(b"x")

    with pytest.raises(ref.ReferenceError) as error:
        ref.discover_corpus(corpus)
    assert "shot.png" in str(error.value) and "shot.jpg" in str(error.value)


# --- pipeline contract --------------------------------------------------------------------
def test_pipeline_is_read_from_the_model_manifest(tmp_path):
    manifest = {"preprocessing": ref.SUPPORTED_PREPROCESSING, "postprocessing": ref.SUPPORTED_POSTPROCESSING}
    assert ref.pipeline_from_manifest(manifest, MODEL_ID) == (
        ref.SUPPORTED_PREPROCESSING,
        ref.SUPPORTED_POSTPROCESSING,
    )


@pytest.mark.parametrize(
    "manifest",
    [
        {"preprocessing": "BGR float [0,255], NHWC", "postprocessing": ref.SUPPORTED_POSTPROCESSING},
        {"preprocessing": ref.SUPPORTED_PREPROCESSING, "postprocessing": "no clamp"},
        {},
    ],
)
def test_unimplemented_pipeline_stops_the_oracle(manifest):
    """A manifest that declares a different pipeline must halt, never be silently ignored."""
    with pytest.raises(ref.ReferenceError) as error:
        ref.pipeline_from_manifest(manifest, MODEL_ID)
    assert "does not implement" in str(error.value)


# --- import policy -----------------------------------------------------------------------
def test_runtime_stack_is_not_imported_at_module_scope():
    """The defect this suite was amended for: a module-scope `import torch` in run_reference.py
    made every suite in the repo uncollectable in CI. Checked by reading the source rather than
    sys.modules, so the guard holds even in an environment where the packages happen to exist."""
    import ast

    source = Path(ref.__file__).read_text()
    forbidden = {"torch", "PIL", "yaml"}
    offenders = []
    for node in ast.parse(source).body:  # module scope only -- nested imports are the point
        if isinstance(node, ast.Import):
            offenders += [a.name.split(".")[0] for a in node.names if a.name.split(".")[0] in forbidden]
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in forbidden:
            offenders.append(node.module)

    assert not offenders, f"run_reference.py imports {offenders} at module scope (import policy)"


# --- reflection-matched oracle -------------------------------------------------------------
#
# These pin the geometry and the padding convention the engine actually implements. The Swift side
# is pinned by "reflection repeats the edge line — symmetric, not reflect-101"; if these two ever
# disagree, the oracle would be measuring a difference of padding conventions rather than of
# tiling, which is the exact defect the reflection-matched oracle corrects.


def test_window_origin_matches_the_engine_for_a_sub_tile_image():
    """Hand-derived from TilePlanner.windowOrigin for desert_figure (300x200, tile 256)."""
    # Horizontal: the image is wider than a tile, so the window slides back inside it.
    assert ref.window_origin(0, 152, 256, 300) == 0
    assert ref.window_origin(152, 148, 256, 300) == 44
    # Vertical: the image is shorter than a tile, so the window is centred and hangs out both ends.
    assert ref.window_origin(0, 152, 256, 200) == -28
    assert ref.window_origin(152, 48, 256, 200) == -28


def test_window_origin_floors_an_odd_overhang():
    """floor((image - tile) / 2), which for an odd difference rounds away from zero."""
    assert ref.window_origin(0, 10, 256, 201) == -28
    assert ref.window_origin(0, 10, 256, 255) == -1


def test_the_window_union_for_the_corpus_sub_tile_image():
    union = ref.padded_window_union(300, 200, 256, 52)
    assert union == {"left": 0, "top": 28, "right": 0, "bottom": 28, "tiles": 4}


def test_an_image_that_tiles_normally_needs_no_padding_at_all():
    union = ref.padded_window_union(517, 512, 256, 52)
    assert (union["left"], union["top"], union["right"], union["bottom"]) == (0, 0, 0, 0)


@pytest.mark.parametrize(
    "width,height,expected",
    [(300, 200, True), (255, 300, True), (256, 256, False), (517, 512, False), (256, 255, True)],
)
def test_only_sub_tile_images_need_a_reflection_oracle(width, height, expected):
    assert ref.needs_reflection_oracle(width, height, 256) is expected


def test_reflect_pad_repeats_the_edge_line():
    """Symmetric, not reflect-101: row 0 appears at both sides of the boundary."""
    tensor = np.arange(5, dtype=np.float32).reshape(1, 1, 5, 1)
    padded = ref.reflect_pad(tensor, {"top": 2, "bottom": 2, "left": 0, "right": 0})

    assert padded.shape == (1, 1, 9, 1)
    # Hand-written: [1,0 | 0,1,2,3,4 | 4,3]
    assert list(padded[0, 0, :, 0]) == [1.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 3.0]


def test_reflect_pad_handles_both_axes():
    tensor = np.arange(6, dtype=np.float32).reshape(1, 1, 2, 3)
    padded = ref.reflect_pad(tensor, {"top": 1, "bottom": 0, "left": 1, "right": 0})

    assert padded.shape == (1, 1, 3, 4)
    assert list(padded[0, 0, 0, :]) == [0.0, 0.0, 1.0, 2.0]  # top row mirrors row 0, edge repeated
    assert list(padded[0, 0, :, 0]) == [0.0, 0.0, 3.0]


def test_a_pad_wider_than_the_image_is_refused_not_guessed():
    """The engine reflects what it can and then replicates the edge; numpy keeps mirroring. Rather
    than record a canvas the engine cannot reproduce, the tool stops."""
    tensor = np.zeros((1, 3, 10, 10), dtype=np.float32)
    with pytest.raises(ref.ReferenceError) as error:
        ref.reflect_pad(tensor, {"top": 20, "bottom": 0, "left": 0, "right": 0})
    assert "edge replication" in str(error.value)


def test_crop_to_image_inverts_the_pad():
    original = np.arange(1 * 3 * 4 * 5, dtype=np.float32).reshape(1, 3, 4, 5)
    padding = {"top": 2, "bottom": 1, "left": 1, "right": 3, "scale_factor": 1}
    padded = ref.reflect_pad(original, padding)

    restored = ref.crop_to_image(padded, padding, 1)

    assert restored.shape == original.shape
    assert np.array_equal(restored, original)


def test_crop_to_image_respects_the_scale_factor():
    padding = {"top": 2, "bottom": 2, "left": 0, "right": 0, "scale_factor": 4}
    # A x4 output over a canvas padded by 2 input rows top and bottom.
    output = np.arange(1 * 1 * 40 * 8, dtype=np.float32).reshape(1, 1, 40, 8)
    cropped = ref.crop_to_image(output, padding, 4)
    assert cropped.shape == (1, 1, 24, 8)
    assert np.array_equal(cropped[0, 0, 0, :], output[0, 0, 8, :])


# --- variants coexist -----------------------------------------------------------------------
def test_entries_written_before_variants_existed_are_full_frame():
    assert ref.entry_variant({"model_id": "X"}) == ref.VARIANT_FULL_FRAME
    assert ref.entry_variant({"variant": ref.VARIANT_REFLECT_PADDED}) == ref.VARIANT_REFLECT_PADDED


def test_saving_one_variant_leaves_the_other_alone(tmp_path):
    """The reflection-matched oracle sits *alongside* the full-frame one; neither may displace it."""
    manifest_path = tmp_path / "references_manifest.json"
    full = {"model_id": MODEL_ID, "variant": ref.VARIANT_FULL_FRAME, "input": "corpus/a.jpg"}
    reflected = {"model_id": MODEL_ID, "variant": ref.VARIANT_REFLECT_PADDED, "input": "corpus/a.jpg"}
    ref.save_manifest(manifest_path, {"entries": [full, reflected]})

    replacement = dict(full, note="regenerated")
    ref.save_entries(manifest_path, MODEL_ID, ref.VARIANT_FULL_FRAME, [replacement])

    entries = ref.load_manifest(manifest_path)["entries"]
    assert reflected in entries, "the reflection-matched entry must survive"
    assert replacement in entries and full not in entries


def test_saving_a_variant_does_not_disturb_other_models(tmp_path):
    manifest_path = tmp_path / "references_manifest.json"
    other = {"model_id": "OtherModel", "variant": ref.VARIANT_FULL_FRAME, "input": "corpus/b.jpg"}
    ref.save_manifest(manifest_path, {"entries": [other]})

    ref.save_entries(manifest_path, MODEL_ID, ref.VARIANT_FULL_FRAME, [])

    assert ref.load_manifest(manifest_path)["entries"] == [other]


def test_a_reflect_padded_entry_without_padding_cannot_be_verified():
    with pytest.raises(ref.ReferenceError) as error:
        ref.recompute(None, Path("x.png"), {"variant": ref.VARIANT_REFLECT_PADDED, "output_npy": "o.npy"})
    assert "padding" in str(error.value)


# --- Scoped generation (--images) -----------------------------------------------------------


def test_parse_images_filter_defaults_to_whole_corpus():
    assert ref.parse_images_filter(None) is None


def test_parse_images_filter_strips_and_deduplicates():
    assert ref.parse_images_filter("a, b ,c") == ["a", "b", "c"]
    assert ref.parse_images_filter("a,a") == ["a"]


@pytest.mark.parametrize("raw", ["", "a,,b", " , ", ","])
def test_parse_images_filter_refuses_empty_segments(raw):
    with pytest.raises(ref.ReferenceError, match="no empty segments"):
        ref.parse_images_filter(raw)


def test_select_images_keeps_discovery_order_not_the_filters(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        (corpus / name).write_bytes(b"x")
    discovered = ref.discover_corpus(corpus)

    selected = ref.select_images(discovered, ["c", "a"])

    assert [p.name for p in selected] == ["a.jpg", "c.jpg"], "recorded order, not the filter's"


def test_select_images_unmatched_stem_is_structural_and_named(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.jpg").write_bytes(b"x")

    with pytest.raises(ref.ReferenceError, match="ghost"):
        ref.select_images(ref.discover_corpus(corpus), ["a", "ghost"])


def test_scoped_save_replaces_in_set_and_preserves_everything_else(tmp_path):
    manifest_path = tmp_path / "references_manifest.json"
    a_full = {"model_id": MODEL_ID, "variant": ref.VARIANT_FULL_FRAME, "input": "corpus/a.jpg"}
    b_full = {"model_id": MODEL_ID, "variant": ref.VARIANT_FULL_FRAME, "input": "corpus/b.jpg"}
    a_refl = {"model_id": MODEL_ID, "variant": ref.VARIANT_REFLECT_PADDED, "input": "corpus/a.jpg"}
    other = {"model_id": "OtherModel", "variant": ref.VARIANT_FULL_FRAME, "input": "corpus/a.jpg"}
    ref.save_manifest(manifest_path, {"entries": [a_full, b_full, a_refl, other]})

    replacement = dict(a_full, note="regenerated")
    ref.save_entries(
        manifest_path, MODEL_ID, ref.VARIANT_FULL_FRAME, [replacement], only_stems={"a"}
    )

    entries = ref.load_manifest(manifest_path)["entries"]
    assert replacement in entries and a_full not in entries, "in-set entry replaced"
    assert b_full in entries, "out-of-set entry of the same variant survives"
    assert a_refl in entries, "the other variant survives"
    assert other in entries, "the other model survives"


def test_unscoped_save_still_replaces_the_whole_variant(tmp_path):
    """The no-filter path is byte-identical to before: scoped saving is opt-in, never ambient."""
    manifest_path = tmp_path / "references_manifest.json"
    a_full = {"model_id": MODEL_ID, "variant": ref.VARIANT_FULL_FRAME, "input": "corpus/a.jpg"}
    b_full = {"model_id": MODEL_ID, "variant": ref.VARIANT_FULL_FRAME, "input": "corpus/b.jpg"}
    ref.save_manifest(manifest_path, {"entries": [a_full, b_full]})

    ref.save_entries(manifest_path, MODEL_ID, ref.VARIANT_FULL_FRAME, [])

    assert ref.load_manifest(manifest_path)["entries"] == []


def test_images_flag_with_unmatched_stem_stops_before_writing(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.jpg").write_bytes(b"x")
    manifest_path = tmp_path / "references_manifest.json"

    code = ref.main([
        "--model", MODEL_ID,
        "--images", "a,ghost",
        "--corpus-dir", str(corpus),
        "--references-dir", str(tmp_path / "references"),
        "--manifest", str(manifest_path),
    ])

    assert code == ref.EXIT_STRUCTURAL_ERROR
    assert "ghost" in capsys.readouterr().out
    assert ref.load_manifest(manifest_path)["entries"] == [], "nothing was recorded"


def test_verify_with_unmatched_stem_is_structural_and_named(tmp_path):
    manifest_path = tmp_path / "references_manifest.json"
    entry = {"model_id": MODEL_ID, "variant": ref.VARIANT_FULL_FRAME, "input": "corpus/a.jpg"}
    ref.save_manifest(manifest_path, {"entries": [entry]})

    with pytest.raises(ref.ReferenceError, match="ghost"):
        ref.run_verify(MODEL_ID, tmp_path / "references", manifest_path, images_filter=["ghost"])


def test_verify_with_filter_on_an_empty_model_is_structural(tmp_path):
    """A scoped verify for a stem with no recorded reference is an error of intent — the
    unfiltered empty-model noop (below) is the only quiet path."""
    manifest_path = tmp_path / "references_manifest.json"
    ref.save_manifest(manifest_path, {"entries": []})

    with pytest.raises(ref.ReferenceError, match="a"):
        ref.run_verify(MODEL_ID, tmp_path / "references", manifest_path, images_filter=["a"])


def test_unfiltered_verify_on_an_empty_model_is_a_refusal(tmp_path, capsys):
    """Replaces an earlier `test_unfiltered_verify_on_an_empty_model_stays_a_noop`.

    A green exit code from a check that examined zero references is the one outcome a reader is
    guaranteed to misread. Certification spec §11 makes this oracle the evidentiary system of
    record.
    """
    manifest_path = tmp_path / "references_manifest.json"
    ref.save_manifest(manifest_path, {"entries": []})

    code = ref.run_verify(MODEL_ID, tmp_path / "references", manifest_path)

    assert code == ref.EXIT_STRUCTURAL_ERROR
    out = capsys.readouterr().out
    assert "VERIFIED 0 REFERENCES" in out
    assert "this is a refusal, not a pass" in out


def test_verify_of_a_model_with_entries_is_unaffected_by_the_refusal(tmp_path):
    """The refusal keys on *this model's* entries, not on the manifest being non-empty."""
    manifest_path = tmp_path / "references_manifest.json"
    ref.save_manifest(manifest_path, {"entries": [
        {"model_id": "other-model", "variant": ref.VARIANT_FULL_FRAME, "input": "corpus/c.jpg"},
    ]})

    assert ref.run_verify(MODEL_ID, tmp_path / "references", manifest_path) == ref.EXIT_STRUCTURAL_ERROR


def test_verify_flag_with_unmatched_stem_stops_before_loading_any_model(tmp_path, capsys):
    manifest_path = tmp_path / "references_manifest.json"
    entry = {"model_id": MODEL_ID, "variant": ref.VARIANT_FULL_FRAME, "input": "corpus/a.jpg"}
    ref.save_manifest(manifest_path, {"entries": [entry]})

    code = ref.main([
        "--model", MODEL_ID, "--verify", "--images", "ghost",
        "--references-dir", str(tmp_path / "references"),
        "--manifest", str(manifest_path),
    ])

    assert code == ref.EXIT_STRUCTURAL_ERROR
    assert "ghost" in capsys.readouterr().out
