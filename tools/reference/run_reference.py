#!/usr/bin/env python3
"""ReferenceEngine — the permanent PyTorch oracle.

Every Level A/B judgement in this project is made against outputs produced here (certification spec
§4), so a reference output is only worth as much as its attribution: which model, which weights,
which input, which environment. The manifest is the committed evidence; the tensors themselves are
regenerated on demand and never committed.

`--verify` answers two different questions, and needs both to be useful:

  1. Are the reference files on disk still the ones the manifest describes?  (bit-rot, edits)
  2. Does this environment still *produce* those files from the same input?  (drift, nondeterminism)

Checking only (1) would miss a torch upgrade silently changing the oracle; checking only (2) would
miss a corrupted `.npy` that the recomputation happens to overwrite in memory. Both are checked,
and any mismatch names the offending file.

A third answer is possible and is **not** success: there may be nothing recorded to check. Both
`--verify` and generation refuse (exit 2) rather than report a quiet zero -- the corpus and the
`.npy` outputs are gitignored, so "nothing here" is the *default* state of a fresh checkout, and it
must never be mistaken for a clean bill of health. This is deliberately stricter than
`tools/artifacts/fetch.py`, whose absent-artifact-is-green convention exists so a cold CI cache can
still build; nothing downstream rests on the fetcher the way certification spec §11 rests on this
oracle.

Weights are obtained through `tools/artifacts/fetch.py` (imported, not re-implemented — the
fetcher owns downloading). The architecture comes from `tools/convert/model_loader.py`, the same
definition the conversion pipeline uses, so the oracle and the converted model can never diverge
silently.

Attribution: Real-ESRGAN is BSD-3-Clause. No Upscayl (AGPL-3.0) or other GPL/AGPL source is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools" / "convert"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "artifacts"))

import fetch  # noqa: E402  (the fetcher: weights are never downloaded by this module)

# Import policy: torch, PIL and yaml are imported inside the functions that
# use them, never at module scope. The test suite imports this module, and CI runs it with only
# pytest + numpy installed -- a module-scope torch import turns a collection error into a red build
# for every suite in the repo. Nothing here needs the runtime stack until real work is done.

CORPUS_DIR = REPO_ROOT / "Sources" / "TestFixtures" / "corpus"
REFERENCES_DIR = REPO_ROOT / "Sources" / "TestFixtures" / "references"
MANIFEST_PATH = REPO_ROOT / "Sources" / "TestFixtures" / "references_manifest.json"
MODELS_ROOT = REPO_ROOT / "models"
CONFIG_DIR = REPO_ROOT / "tools" / "convert" / "models"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
SEED = 0

# --- reflection-matched oracle -------------------------------------------------------------
# An image smaller than one tile cannot be tiled without the engine reflecting past its edge
# (correct engine behaviour). The full-frame oracle saw the image simply end,
# so comparing the two measures a difference of border conditions rather than of tiling. For those
# images a second oracle is recorded, over a canvas padded exactly the way the engine pads.
VARIANT_FULL_FRAME = "full-frame"
VARIANT_REFLECT_PADDED = "reflect-padded"
REFLECT_SUFFIX = "-reflectpadded"

# The engine's geometry. These are not free parameters: they are the numbers UpscaleCore uses
# (`ModelDescriptor.inputTileSize`, `ImageCommand.derivedContextMargin`, `scaleFactor`), and
# `certify-tiling` re-derives the window union from the real planner and refuses to proceed if the
# recorded extents disagree, so the two cannot drift apart unnoticed.
DEFAULT_TILE_SIZE = 256
DEFAULT_CONTEXT_MARGIN = 52
DEFAULT_SCALE_FACTOR = 4

#: numpy's "symmetric" repeats the edge line (…c b a | a b c…). That is what `vImageVerticalReflect`
#: produces in `SequentialTilingEngine.fillPaddedTile`, pinned by the Swift test
#: "reflection repeats the edge line — symmetric, not reflect-101".
PADDING_POLICY = "symmetric reflection, edge line repeated (numpy mode='symmetric')"

# The pipeline this oracle implements. The model's manifest is the authority: if it ever declares
# something else, the oracle stops rather than quietly applying a different pipeline.
SUPPORTED_PREPROCESSING = "RGB float [0,1], NCHW"
SUPPORTED_POSTPROCESSING = "clamp [0,1]"

EXIT_OK = 0
EXIT_VERIFICATION_FAILED = 1
EXIT_STRUCTURAL_ERROR = 2


class ReferenceError(RuntimeError):
    """A condition that must stop the oracle rather than be worked around."""


def log(message: str) -> None:
    print(message, flush=True)


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


# --- corpus + pipeline --------------------------------------------------------------------
def discover_corpus(corpus_dir: Path) -> list[Path]:
    """Every image under the corpus root, recursively, in a deterministic order."""
    if not corpus_dir.is_dir():
        return []
    found = sorted(
        p for p in corpus_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    stems: dict[str, Path] = {}
    for path in found:
        if path.stem in stems:
            raise ReferenceError(
                f"corpus stem collision: {relative(path)} and {relative(stems[path.stem])} "
                "would write the same reference filename -- rename one"
            )
        stems[path.stem] = path
    return found


def window_origin(valid_origin: int, valid_extent: int, tile_size: int, image_extent: int) -> int:
    """Where the engine puts a fixed-size window along one axis.

    A transcription of `TilePlanner.windowOrigin`. It is duplicated rather than imported because
    the oracle is Python and the planner is Swift; the duplication is made safe by `certify-tiling`
    checking the recorded extents against the real planner before it uses this oracle.
    """
    if image_extent < tile_size:
        return -((tile_size - image_extent + 1) // 2)  # floor((image - tile) / 2) for negatives
    centred = valid_origin + valid_extent // 2 - tile_size // 2
    return min(max(centred, 0), image_extent - tile_size)


def padded_window_union(
    width: int, height: int, tile_size: int, context_margin: int
) -> dict:
    """The union of every padded tile window for an image, as pad extents around it."""
    core = max(1, tile_size - 2 * context_margin)
    columns = (width + core - 1) // core
    rows = (height + core - 1) // core

    origins = []
    for row in range(rows):
        for column in range(columns):
            valid_x, valid_y = column * core, row * core
            origins.append(
                (
                    window_origin(valid_x, min(core, width - valid_x), tile_size, width),
                    window_origin(valid_y, min(core, height - valid_y), tile_size, height),
                )
            )
    min_x = min(origin[0] for origin in origins)
    min_y = min(origin[1] for origin in origins)
    max_x = max(origin[0] for origin in origins) + tile_size
    max_y = max(origin[1] for origin in origins) + tile_size

    return {
        "left": max(0, -min_x),
        "top": max(0, -min_y),
        "right": max(0, max_x - width),
        "bottom": max(0, max_y - height),
        "tiles": len(origins),
    }


def needs_reflection_oracle(width: int, height: int, tile_size: int) -> bool:
    """Only images smaller than one tile force the engine to reflect past their own edge."""
    return width < tile_size or height < tile_size


def pipeline_from_manifest(model_manifest: dict, model_id: str) -> tuple[str, str]:
    """Read the pipeline definition from the model manifest; refuse anything unimplemented."""
    pre = model_manifest.get("preprocessing")
    post = model_manifest.get("postprocessing")
    if pre != SUPPORTED_PREPROCESSING or post != SUPPORTED_POSTPROCESSING:
        raise ReferenceError(
            f"{model_id}: manifest declares a pipeline this oracle does not implement.\n"
            f"  manifest preprocessing : {pre!r}\n  supported              : {SUPPORTED_PREPROCESSING!r}\n"
            f"  manifest postprocessing: {post!r}\n  supported              : {SUPPORTED_POSTPROCESSING!r}\n"
            "  update the oracle deliberately rather than letting the two definitions drift"
        )
    return pre, post


def preprocess(path: Path) -> np.ndarray:
    """Per manifest: RGB float [0,1], NCHW."""
    from PIL import Image

    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.uint8).astype(np.float32) / 255.0
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None, ...])


def postprocess_to_png(array: np.ndarray, path: Path) -> None:
    """Per manifest: clamp [0,1], then an 8-bit view for human inspection."""
    from PIL import Image

    hwc = np.clip(array[0].transpose(1, 2, 0), 0.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.floor(hwc * 255.0 + 0.5).astype(np.uint8)).save(path)


def current_environment() -> dict:
    """Attribution for the environment that produced a reference. Imports torch on purpose:
    callers that only manipulate manifests never reach this."""
    import torch

    return {"torch": torch.__version__, "python": platform.python_version()}


def forward(model, tensor: np.ndarray) -> np.ndarray:
    """Run the oracle, padding to the model's input alignment when it has one.

    x2 models `pixel_unshuffle` by 2 before the body and reject an odd input dimension outright.
    Upstream handles this in its `RealESRGANer` wrapper, not in the network, and so do we: the
    architecture stays a faithful reimplementation and the padding lives with the caller that can
    be handed an odd image.

    **Replicate-pad, then crop the output by `scale * pad`.** Replicate rather than reflect because
    a single duplicated edge row is the mildest assumption available, and the padded region is
    removed from the result either way -- it exists only so the tensor has an even shape. The
    returned tensor is exactly the size an unpadded run would have produced.

    THIS IS ALSO A PRODUCT REQUIREMENT, recorded here because this is where it surfaced: the
    shipping pipeline must do the same before handing an arbitrary photograph to an x2 model. The
    engine's own tiles are 256/384/512 and always even, so the tiled path is safe today -- but a
    whole-image or preview path that feeds a raw decode is not.
    """
    import torch

    alignment = 2 if getattr(model, "scale", 4) == 2 else 1
    height, width = tensor.shape[-2:]
    pad_h, pad_w = (-height) % alignment, (-width) % alignment

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    with torch.no_grad():
        source = torch.from_numpy(tensor)
        if pad_h or pad_w:
            source = torch.nn.functional.pad(source, (0, pad_w, 0, pad_h), mode="replicate")
        output = model(source).numpy().astype(np.float32)

    if pad_h or pad_w:
        scale = output.shape[-2] // source.shape[-2]
        output = output[..., : output.shape[-2] - pad_h * scale, : output.shape[-1] - pad_w * scale]
    return output


def write_tensor(array: np.ndarray, npy_path: Path) -> None:
    """The reference proper: the exact float32 tensor. numpy only, deliberately."""
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, array, allow_pickle=False)


def write_outputs(array: np.ndarray, npy_path: Path, png_path: Path) -> None:
    """`.npy` is the exact float32 tensor; the PNG is a derived 8-bit view for human eyes."""
    write_tensor(array, npy_path)
    postprocess_to_png(array, png_path)


# --- manifest -----------------------------------------------------------------------------
def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"entries": []}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ReferenceError(f"{relative(path)}: not valid JSON: {error}") from error
    if not isinstance(data, dict) or not isinstance(data.get("entries"), list):
        raise ReferenceError(f"{relative(path)}: expected an object with an 'entries' list")
    return data


def save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def build_entry(
    *,
    model_id: str,
    weights_sha256: str,
    input_path: Path,
    npy_path: Path,
    png_path: Path,
    array: np.ndarray,
    preprocessing: str,
    postprocessing: str,
    environment: dict,
    variant: str = VARIANT_FULL_FRAME,
) -> dict:
    return {
        "model_id": model_id,
        "variant": variant,
        "weights_sha256": weights_sha256,
        "input": relative(input_path),
        "input_sha256": sha256_of_file(input_path),
        "preprocessing": preprocessing,
        "postprocessing": postprocessing,
        "output_npy": relative(npy_path),
        "output_npy_sha256": sha256_of_file(npy_path),
        "output_png": relative(png_path),
        "output_png_sha256": sha256_of_file(png_path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "environment": environment,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def verify_recorded_files(entries: list[dict], root: Path) -> list[str]:
    """Question 1: do the files on disk still match what the manifest recorded?"""
    failures: list[str] = []
    for entry in entries:
        for path_field, hash_field in (
            ("input", "input_sha256"),
            ("output_npy", "output_npy_sha256"),
            ("output_png", "output_png_sha256"),
        ):
            path = root / entry[path_field]
            if not path.exists():
                failures.append(f"{entry[path_field]}: missing")
                continue
            measured = sha256_of_file(path)
            if measured != entry[hash_field]:
                failures.append(
                    f"{entry[path_field]}: hash mismatch\n"
                    f"      manifest: {entry[hash_field]}\n"
                    f"      on disk : {measured}"
                )
    return failures


# --- commands -----------------------------------------------------------------------------
def load_model_for(model_id: str):
    """Weights via the artifact fetcher; architecture via the shared conversion loader."""
    log(f"[weights] ensuring {model_id} via tools/artifacts/fetch.py")
    if fetch.main(["--model", model_id]) != 0:
        raise ReferenceError(f"{model_id}: fetch.py could not provide verified weights")

    import yaml

    from model_loader import ModelLoadError, build_model

    config_path = CONFIG_DIR / f"{model_id}.yaml"
    if not config_path.exists():
        raise ReferenceError(f"{model_id}: no conversion config at {relative(config_path)}")
    config = yaml.safe_load(config_path.read_text())

    manifest_path = MODELS_ROOT / model_id / "manifest.json"
    if not manifest_path.exists():
        raise ReferenceError(f"{model_id}: no model manifest at {relative(manifest_path)}")
    model_manifest = json.loads(manifest_path.read_text())

    pre, post = pipeline_from_manifest(model_manifest, model_id)
    weights_path = REPO_ROOT / config["weights"]["path"]
    try:
        model, _ = build_model(config, weights_path)
    except ModelLoadError as error:
        raise ReferenceError(f"{model_id}: {error}") from error
    return model, sha256_of_file(weights_path), pre, post


def parse_images_filter(raw: str | None) -> list[str] | None:
    """Parse `--images`: comma-separated corpus stems, no empty segments (mirrors the
    certify-tiling filter). `None` means the whole corpus — the historical
    default, and the only mode certification sweeps should use."""
    if raw is None:
        return None
    stems = [segment.strip() for segment in raw.split(",")]
    if any(not stem for stem in stems):
        raise ReferenceError("--images expects comma-separated stems, no empty segments")
    return list(dict.fromkeys(stems))


def select_images(images: list[Path], stems: list[str] | None) -> list[Path]:
    """Narrow a discovered corpus to the requested stems, keeping discovery order — the
    recorded order, not the filter's. A stem that matches nothing is structural: the caller
    asked for evidence that cannot exist, and a silent partial run would look complete."""
    if stems is None:
        return images
    available = {path.stem for path in images}
    unmatched = [stem for stem in stems if stem not in available]
    if unmatched:
        raise ReferenceError(f"--images: no corpus image matches: {', '.join(unmatched)}")
    wanted = set(stems)
    return [path for path in images if path.stem in wanted]


def run_generate(
    model_id: str,
    corpus_dir: Path,
    references_dir: Path,
    manifest_path: Path,
    images_filter: list[str] | None = None,
) -> int:
    images = select_images(discover_corpus(corpus_dir), images_filter)
    if not images:
        # This branch used to prune the model's entries and exit 0, which made
        # an absent corpus -- the normal state of a fresh checkout, since the corpus is gitignored
        # -- *delete the committed oracle* and report success. `--verify` then had nothing left to
        # check and greened on the wreckage. Both halves of that were exit 0.
        recorded = [e for e in load_manifest(manifest_path)["entries"] if e["model_id"] == model_id]
        log(f"corpus empty: no images under {relative(corpus_dir)}")
        log(f"REFUSING to record: {relative(manifest_path)} is unchanged"
            f" ({len(recorded)} entry(s) for {model_id} left intact)")
        log("  the corpus is gitignored; fetch or restore it before regenerating the oracle")
        return EXIT_STRUCTURAL_ERROR

    log(f"[corpus] {len(images)} image(s) under {relative(corpus_dir)}")
    model, weights_sha, pre, post = load_model_for(model_id)

    entries: list[dict] = []
    for image_path in images:
        tensor = preprocess(image_path)
        output = forward(model, tensor)
        npy_path = references_dir / model_id / f"{image_path.stem}.npy"
        png_path = references_dir / model_id / f"{image_path.stem}.png"
        write_outputs(output, npy_path, png_path)
        entry = build_entry(
            model_id=model_id, weights_sha256=weights_sha, input_path=image_path,
            npy_path=npy_path, png_path=png_path, array=output,
            preprocessing=pre, postprocessing=post, environment=current_environment(),
        )
        entries.append(entry)
        log(
            f"  {image_path.name} {tuple(tensor.shape[2:])} -> {tuple(output.shape[2:])}  "
            f"npy sha256 {entry['output_npy_sha256'][:16]}..."
        )

    save_entries(
        manifest_path, model_id, VARIANT_FULL_FRAME, entries,
        only_stems=set(images_filter) if images_filter else None,
    )
    log(f"[manifest] {len(entries)} reference(s) recorded in {relative(manifest_path)}")
    return EXIT_OK


def entry_variant(entry: dict) -> str:
    """Entries written before variants existed carry no variant; they are all full-frame."""
    return entry.get("variant", VARIANT_FULL_FRAME)


def save_entries(
    manifest_path: Path,
    model_id: str,
    variant: str,
    entries: list[dict],
    only_stems: set[str] | None = None,
) -> None:
    """Replace this model's entries *of this variant only*.

    The reflection-matched oracle sits alongside the full-frame one and neither may silently
    displace the other, so regenerating one variant must leave the other untouched.

    With `only_stems` (the `--images` path) the replacement is further scoped to entries whose
    input stem is in the set: a scoped generation must never delete oracles it did not
    regenerate.
    """
    manifest = load_manifest(manifest_path)
    manifest["entries"] = [
        e
        for e in manifest["entries"]
        if not (
            e["model_id"] == model_id
            and entry_variant(e) == variant
            and (only_stems is None or Path(e["input"]).stem in only_stems)
        )
    ] + entries
    manifest["entries"].sort(key=lambda e: (e["model_id"], entry_variant(e), e["input"]))
    save_manifest(manifest_path, manifest)


def reflect_pad(tensor: np.ndarray, padding: dict) -> np.ndarray:
    """Pad an NCHW tensor the way the engine pads a tile window.

    Refuses the case the engine handles differently: when a pad strip is wider than the image, the
    engine reflects what it can and then replicates the outermost line, while numpy keeps mirroring.
    Producing a canvas the engine would not produce would make the oracle wrong in a way the
    certification could not see, so this stops instead.
    """
    height, width = tensor.shape[2], tensor.shape[3]
    for axis, (near, far, extent) in {
        "vertical": (padding["top"], padding["bottom"], height),
        "horizontal": (padding["left"], padding["right"], width),
    }.items():
        if max(near, far) > extent:
            raise ReferenceError(
                f"{axis} pad ({max(near, far)} px) exceeds the image extent ({extent} px). The "
                "engine falls back to edge replication there and numpy keeps mirroring, so the two "
                "canvases would differ. Refusing to record an oracle the engine cannot reproduce."
            )
    return np.pad(
        tensor,
        ((0, 0), (0, 0), (padding["top"], padding["bottom"]), (padding["left"], padding["right"])),
        mode="symmetric",
    )


def crop_to_image(array: np.ndarray, padding: dict, scale_factor: int) -> np.ndarray:
    """Cut the model's output back to the pixels the image itself owns."""
    top = padding["top"] * scale_factor
    left = padding["left"] * scale_factor
    height = array.shape[2] - (padding["top"] + padding["bottom"]) * scale_factor
    width = array.shape[3] - (padding["left"] + padding["right"]) * scale_factor
    return np.ascontiguousarray(array[:, :, top : top + height, left : left + width])


def run_reflect_padded(
    model_id: str,
    corpus_dir: Path,
    references_dir: Path,
    manifest_path: Path,
    tile_size: int,
    context_margin: int,
    scale_factor: int,
    images_filter: list[str] | None = None,
) -> int:
    """Record a reflection-matched oracle for every corpus image smaller than one tile."""
    images = select_images(discover_corpus(corpus_dir), images_filter)
    only_stems = set(images_filter) if images_filter else None
    if not images:
        log(f"corpus empty: no images under {relative(corpus_dir)} -- nothing to reflect-pad")
        save_entries(manifest_path, model_id, VARIANT_REFLECT_PADDED, [], only_stems=only_stems)
        return EXIT_OK

    candidates = []
    for image_path in images:
        tensor = preprocess(image_path)
        height, width = tensor.shape[2], tensor.shape[3]
        if needs_reflection_oracle(width, height, tile_size):
            candidates.append((image_path, width, height))
        else:
            log(
                f"  {image_path.name} ({width}x{height}) tiles without leaving the image -- the "
                "full-frame oracle already matches the engine's border condition"
            )

    if not candidates:
        log("no corpus image is smaller than one tile -- no reflection-matched oracle is needed")
        save_entries(manifest_path, model_id, VARIANT_REFLECT_PADDED, [], only_stems=only_stems)
        return EXIT_OK

    log(f"[reflect] {len(candidates)} image(s) smaller than one {tile_size}px tile")
    model, weights_sha, pre, post = load_model_for(model_id)

    entries: list[dict] = []
    for image_path, width, height in candidates:
        padding = padded_window_union(width, height, tile_size, context_margin)
        tensor = preprocess(image_path)
        canvas = reflect_pad(tensor, padding)
        output = crop_to_image(forward(model, canvas), padding, scale_factor)

        npy_path = references_dir / model_id / f"{image_path.stem}{REFLECT_SUFFIX}.npy"
        png_path = references_dir / model_id / f"{image_path.stem}{REFLECT_SUFFIX}.png"
        write_outputs(output, npy_path, png_path)

        entry = build_entry(
            model_id=model_id, weights_sha256=weights_sha, input_path=image_path,
            npy_path=npy_path, png_path=png_path, array=output,
            preprocessing=pre, postprocessing=post, environment=current_environment(),
            variant=VARIANT_REFLECT_PADDED,
        )
        entry["padding"] = {
            "policy": PADDING_POLICY,
            "left": padding["left"], "top": padding["top"],
            "right": padding["right"], "bottom": padding["bottom"],
            "canvas": [width + padding["left"] + padding["right"],
                       height + padding["top"] + padding["bottom"]],
            "tile_size": tile_size,
            "context_margin": context_margin,
            "scale_factor": scale_factor,
            "tiles_in_union": padding["tiles"],
            "edge_replication_fallback": False,
        }
        entries.append(entry)
        log(
            f"  {image_path.name} ({width}x{height}) -> canvas "
            f"{entry['padding']['canvas'][0]}x{entry['padding']['canvas'][1]} "
            f"(pad l{padding['left']} t{padding['top']} r{padding['right']} b{padding['bottom']}) "
            f"-> {tuple(output.shape[2:])}  npy sha256 {entry['output_npy_sha256'][:16]}..."
        )

    save_entries(manifest_path, model_id, VARIANT_REFLECT_PADDED, entries, only_stems=only_stems)
    log(f"[manifest] {len(entries)} reflection-matched reference(s) recorded")
    return EXIT_OK


def recompute(model, input_path: Path, entry: dict) -> np.ndarray:
    """Reproduce one recorded reference, through whichever pipeline produced it.

    A reflection-matched entry that was verified as if it were a full-frame one would fail every
    time and tell nobody why, so the variant selects the pipeline.
    """
    if entry_variant(entry) != VARIANT_REFLECT_PADDED:
        return forward(model, preprocess(input_path))

    padding = entry.get("padding")
    if not isinstance(padding, dict):
        raise ReferenceError(
            f"{entry['output_npy']}: a reflect-padded entry carries no 'padding' block, so the "
            "canvas that produced it cannot be reconstructed"
        )
    canvas = reflect_pad(preprocess(input_path), padding)
    return crop_to_image(forward(model, canvas), padding, padding["scale_factor"])


def run_verify(
    model_id: str,
    references_dir: Path,
    manifest_path: Path,
    images_filter: list[str] | None = None,
) -> int:
    manifest = load_manifest(manifest_path)
    entries = [e for e in manifest["entries"] if e["model_id"] == model_id]
    if images_filter is not None:
        wanted = set(images_filter)
        entries = [e for e in entries if Path(e["input"]).stem in wanted]
        found = {Path(e["input"]).stem for e in entries}
        unmatched = [stem for stem in images_filter if stem not in found]
        if unmatched:
            raise ReferenceError(
                f"--images: no recorded reference matches: {', '.join(unmatched)}"
            )
    if not entries:
        # Exiting 0 here would mean a CI job could run the project's strongest-sounding
        # check against nothing at all and go green. A verify that verifies zero references has not
        # answered the question it was asked; it has declined to.
        log(f"VERIFIED 0 REFERENCES for {model_id} -- nothing was checked")
        log(f"  {relative(manifest_path)} records no entries for this model")
        log("  this is a refusal, not a pass")
        return EXIT_STRUCTURAL_ERROR

    log(f"[verify] {len(entries)} recorded reference(s) for {model_id}")
    failures = verify_recorded_files(entries, REPO_ROOT)
    for failure in failures:
        log(f"  FAILED {failure}")

    model, weights_sha, _, _ = load_model_for(model_id)
    for entry in entries:
        if entry["weights_sha256"] != weights_sha:
            failures.append(
                f"{entry['input']}: weights changed since recording\n"
                f"      manifest: {entry['weights_sha256']}\n      on disk : {weights_sha}"
            )
            continue
        input_path = REPO_ROOT / entry["input"]
        if not input_path.exists():
            continue  # already reported by verify_recorded_files
        output = recompute(model, input_path, entry)
        with tempfile.TemporaryDirectory() as tmp:
            probe = Path(tmp) / "recomputed.npy"
            np.save(probe, output, allow_pickle=False)
            measured = sha256_of_file(probe)
        if measured != entry["output_npy_sha256"]:
            failures.append(
                f"{entry['output_npy']}: recomputation does not reproduce the recorded tensor\n"
                f"      manifest   : {entry['output_npy_sha256']}\n      recomputed : {measured}"
            )
            log(f"  FAILED {failures[-1]}")
        else:
            log(f"  reproduced: {entry['input']} -> {entry['output_npy']}")

    log("")
    if failures:
        log(f"VERIFICATION FAILED ({len(failures)} problem(s))")
        return EXIT_VERIFICATION_FAILED
    log(f"OK (verified {len(entries)} reference(s), files and recomputation)")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ReferenceEngine — the permanent PyTorch oracle")
    parser.add_argument("--model", required=True, help="model id, matching models/<id>/manifest.json")
    parser.add_argument("--verify", action="store_true", help="recompute and compare against the manifest")
    parser.add_argument(
        "--reflect-padded",
        action="store_true",
        help="record a reflection-matched oracle for corpus images smaller than one tile",
    )
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--context-margin", type=int, default=DEFAULT_CONTEXT_MARGIN)
    parser.add_argument("--scale-factor", type=int, default=DEFAULT_SCALE_FACTOR)
    parser.add_argument("--corpus-dir", default=None, help=argparse.SUPPRESS)      # test seam
    parser.add_argument("--references-dir", default=None, help=argparse.SUPPRESS)  # test seam
    parser.add_argument("--manifest", default=None, help=argparse.SUPPRESS)        # test seam
    parser.add_argument(
        "--images",
        default=None,
        help="comma-separated corpus stems for scoped generation (no empty segments); "
        "saving then preserves this model's oracle entries outside the set",
    )
    args = parser.parse_args(argv)

    corpus_dir = Path(args.corpus_dir).resolve() if args.corpus_dir else CORPUS_DIR
    references_dir = Path(args.references_dir).resolve() if args.references_dir else REFERENCES_DIR
    manifest_path = Path(args.manifest).resolve() if args.manifest else MANIFEST_PATH

    try:
        images_filter = parse_images_filter(args.images)
        if args.verify:
            return run_verify(
                args.model, references_dir, manifest_path, images_filter=images_filter
            )
        if args.reflect_padded:
            return run_reflect_padded(
                args.model, corpus_dir, references_dir, manifest_path,
                args.tile_size, args.context_margin, args.scale_factor,
                images_filter=images_filter,
            )
        return run_generate(
            args.model, corpus_dir, references_dir, manifest_path, images_filter=images_filter
        )
    except ReferenceError as error:
        log("")
        log(f"STOPPED: {type(error).__name__}: {error}")
        return EXIT_STRUCTURAL_ERROR


if __name__ == "__main__":
    sys.exit(main())
