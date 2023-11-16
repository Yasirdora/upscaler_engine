#!/usr/bin/env python3
"""Config-driven, reproducible PyTorch -> Core ML conversion pipeline.

One command turns a per-model YAML config into a `.mlpackage` plus the identity documents
that make it auditable later: `manifest.json` (certification spec §11) and `conversion/report.txt`.

Attribution: the RRDBNet architecture is reimplemented from the public description in the
Real-ESRGAN project (BSD-3-Clause, https://github.com/xinntao/Real-ESRGAN); module and
parameter names follow upstream so the published checkpoint loads with strict=True. No code
from Upscayl (AGPL-3.0) or any GPL/AGPL source was consulted or copied.
"""

from __future__ import annotations

import argparse
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

import coremltools as ct

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import (  # noqa: E402
    FIDELITY_FIELDS, VOLATILE_FIELDS, ManifestError, drift_against_prior, sha256_of_path,
    validate, write,
)
from model_loader import ModelLoadError, RRDBNet, build_model  # noqa: E402

SCRIPT_VERSION = "1.0.0"
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(__file__).resolve().parent / "models"

COMPUTE_UNITS = {
    "ALL": ct.ComputeUnit.ALL,
    "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
    "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
    "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
}
PRECISIONS = {"float16": ct.precision.FLOAT16, "float32": ct.precision.FLOAT32}
DEPLOYMENT_TARGETS = {"macOS14": ct.target.macOS14, "macOS15": ct.target.macOS15}


class ConversionError(RuntimeError):
    """Any condition that must stop the pipeline rather than be worked around."""


# --- Pipeline pieces ----------------------------------------------------------------------
def log(message: str) -> None:
    print(message, flush=True)


def load_config(model_id: str) -> dict:
    path = CONFIG_DIR / f"{model_id}.yaml"
    if not path.exists():
        raise ConversionError(f"no conversion config for '{model_id}' at {path}")
    return yaml.safe_load(path.read_text())


def verify_weights(config: dict) -> tuple[Path, str, int]:
    """Verify the local checkpoint against the config hash. Mismatch stops the pipeline."""
    path = REPO_ROOT / config["weights"]["path"]
    if not path.exists():
        raise ConversionError(
            f"weights not found at {path}. Fetch them first with "
            "`python tools/artifacts/fetch.py --model <id>`."
        )
    measured = sha256_of_path(path)
    expected = config["weights"]["sha256"]
    if measured != expected:
        raise ConversionError(
            "weights SHA256 mismatch -- refusing to convert an unidentified artifact.\n"
            f"  expected (config): {expected}\n  measured (on disk): {measured}"
        )
    log(f"[weights] verified sha256 {measured}")
    return path, measured, path.stat().st_size


def load_tile(config: dict) -> np.ndarray:
    path = REPO_ROOT / config["test_tile"]
    if not path.exists():
        raise ConversionError(f"test tile not found: {path}")
    _, _, height, width = config["conversion"]["input_shape"]
    image = Image.open(path).convert("RGB")
    if image.size != (width, height):
        raise ConversionError(f"test tile must be {width}x{height}, got {image.size}")
    array = np.asarray(image, dtype=np.uint8).astype(np.float32) / 255.0
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None, ...])


def to_png(array: np.ndarray, path: Path) -> None:
    hwc = np.clip(array[0].transpose(1, 2, 0), 0.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.floor(hwc * 255.0 + 0.5).astype(np.uint8)).save(path)


def level_a(reference: np.ndarray, candidate: np.ndarray) -> dict:
    ref = np.clip(reference, 0.0, 1.0).astype(np.float64)
    cand = np.clip(candidate, 0.0, 1.0).astype(np.float64)
    diff = np.abs(ref - cand)
    mse = float(np.mean((ref - cand) ** 2))
    return {
        "psnr_db": float("inf") if mse == 0.0 else float(10.0 * np.log10(1.0 / mse)),
        "mae": float(diff.mean()),
        "max_abs": float(diff.max()),
        "pixels_within_4_255_pct": float(np.mean(diff * 255.0 <= 4.0) * 100.0),
    }


def reference_pass(model: RRDBNet, tile: np.ndarray, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    with torch.no_grad():
        return model(torch.from_numpy(tile)).numpy()


def coreml_predict(mlpackage: Path, tile: np.ndarray, units: str) -> np.ndarray:
    model = ct.models.MLModel(str(mlpackage), compute_units=COMPUTE_UNITS[units])
    return np.asarray(model.predict({"input": tile})["output"], dtype=np.float32)


def write_report(path: Path, rows: list[tuple[str, str]], metrics: dict, config: dict) -> None:
    width = max(len(k) for k, _ in rows)
    lines = [
        "CONVERSION REPORT",
        "=" * 78,
        *[f"{k.ljust(width)} : {v}" for k, v in rows],
        "",
        "-- Level A (Core ML vs PyTorch oracle regenerated this run, clamped [0,1]) --",
        f"PSNR                 : {metrics['psnr_db']:.2f} dB  (threshold >= "
        f"{config['thresholds']['psnr_db']:.0f} dB, spec §4)",
        f"MAE                  : {metrics['mae']:.8f}  ({metrics['mae'] * 255:.4f}/255)",
        f"max abs error        : {metrics['max_abs']:.8f}  ({metrics['max_abs'] * 255:.4f}/255)",
        f"pixels within 4/255  : {metrics['pixels_within_4_255_pct']:.4f} %  (spec §4 wants >= 99%)",
        "",
        "-- Sources (licence discipline) --------------------------------------------",
        f"{config['source']['repo']} ({config['source']['license']}), "
        f"release {config['source']['commit_or_release']}",
        "  RRDBNet reimplemented from the public architecture description, with attribution.",
        "NO code from Upscayl (AGPL-3.0) or any GPL/AGPL source was consulted or copied.",
        "",
        "-- Scope -------------------------------------------------------------------",
        "Fixed-shape single-tile conversion. No tiling, no artifact fetching,",
        "no additional models, no benchmarking -- no timings are recorded, deliberately.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    log(f"[report] wrote {path.relative_to(REPO_ROOT)}")


# --- Commands -----------------------------------------------------------------------------
# --- fidelity to the committed manifest ---------------------------------------------------
def check_against_prior(manifest_path: Path, candidate: dict) -> list[str]:
    """Compare a freshly derived manifest against the one already on disk, BEFORE overwriting.

    `--check` alone cannot do this. It reads the manifest from disk and compares the artifact to
    it, so once a conversion has rewritten the manifest, `--check` is comparing the artifact to a
    hash written moments earlier by the same run -- self-consistency, tautologically true. The one
    moment fidelity to the committed evidence can be tested without reaching into git is here,
    while the prior manifest still exists.
    """
    if not manifest_path.exists():
        log("[fidelity] no prior manifest -- first conversion of this model, nothing to compare")
        return []
    drift = drift_against_prior(manifest_path, candidate)
    if drift:
        return drift
    log(f"[fidelity] agrees with the existing manifest on all {len(FIDELITY_FIELDS)} fidelity fields")
    log(f"[fidelity] not compared (expected to vary): {', '.join(VOLATILE_FIELDS)}")
    return []


def run_convert(model_id: str, units: str | None) -> int:
    config = load_config(model_id)
    conv = config["conversion"]
    units = units or conv["compute_units"]
    model_dir = REPO_ROOT / "models" / model_id
    output_dir, conversion_dir = model_dir / "output", model_dir / "conversion"

    weights_path, weights_sha, weights_bytes = verify_weights(config)
    model, parameters = build_model(config, weights_path)
    tile = load_tile(config)

    log("[reference] PyTorch fp32 forward pass ...")
    reference = reference_pass(model, tile, conv["seed"])

    log("[convert] tracing and converting (coremltools PyTorch frontend, no ONNX) ...")
    shape = tuple(conv["input_shape"])
    traced = torch.jit.trace(model, torch.from_numpy(tile))
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=shape, dtype=np.float32)],
        outputs=[ct.TensorType(name="output", dtype=np.float32)],
        convert_to=conv["representation"],
        compute_precision=PRECISIONS[conv["precision"]],
        minimum_deployment_target=DEPLOYMENT_TARGETS[conv["min_deployment"]],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    mlpackage = output_dir / f"{model_id}_{shape[2]}x{shape[3]}.mlpackage"
    mlmodel.save(str(mlpackage))
    log(f"[convert] saved {mlpackage.relative_to(REPO_ROOT)}")

    candidate = coreml_predict(mlpackage, tile, units)
    if candidate.shape != reference.shape:
        raise ConversionError(f"shape mismatch: {reference.shape} vs {candidate.shape}")

    metrics = level_a(reference, candidate)
    to_png(reference, conversion_dir / "reference.png")
    to_png(candidate, conversion_dir / "coreml.png")

    resolved = dict(config)
    resolved["conversion"] = {**conv, "compute_units": units}
    (conversion_dir / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))

    manifest = {
        "model_id": config["model_id"],
        "scale_factor": config["scale_factor"],
        "weights": {
            "url": config["weights"]["url"],
            "sha256": weights_sha,
            "bytes": weights_bytes,
            "state_dict_key": config["weights"]["state_dict_key"],
            "parameters": parameters,
        },
        "source": dict(config["source"]),
        "conversion": {
            "script": "tools/convert/convert.py",
            "script_version": SCRIPT_VERSION,
            "coremltools": ct.__version__,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "representation": conv["representation"],
            "precision": conv["precision"],
            "input_shape": list(conv["input_shape"]),
            "output_shape": list(reference.shape),
            "min_deployment": conv["min_deployment"],
            "compute_units": units,
            "parameters": {"tracer": "torch.jit.trace", "seed": conv["seed"]},
        },
        "preprocessing": config["preprocessing"],
        "postprocessing": config["postprocessing"],
        "outputs": {
            "mlpackage_sha256": sha256_of_path(mlpackage),
            "reference_metrics": {
                "psnr_db": metrics["psnr_db"],
                "mae": metrics["mae"],
                "max_abs": metrics["max_abs"],
            },
        },
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    manifest_path = model_dir / "manifest.json"
    drift = check_against_prior(manifest_path, manifest)
    if drift:
        candidate_path = model_dir / "manifest.candidate.json"
        write(candidate_path, manifest)
        log("")
        log("=" * 78)
        log(f"FIDELITY DRIFT ({len(drift)} field(s)) -- models/{model_id}/manifest.json NOT overwritten")
        for item in drift:
            log(f"  {item}")
        log(f"  this run's manifest written to models/{model_id}/manifest.candidate.json")
        log("  the committed manifest is intact; a human rules on the difference")
        log("=" * 78)
        return 1

    write(manifest_path, manifest)
    log(f"[manifest] wrote and validated models/{model_id}/manifest.json")

    threshold = config["thresholds"]["psnr_db"]
    passed = metrics["psnr_db"] >= threshold
    write_report(
        conversion_dir / "report.txt",
        [
            ("generated (UTC)", manifest["created_utc"]),
            ("mode", "convert"),
            ("verdict", "PASS" if passed else "FAIL"),
            ("platform", f"{platform.system()} {platform.mac_ver()[0]} ({platform.machine()})"),
            ("python", platform.python_version()),
            ("torch", torch.__version__),
            ("coremltools", ct.__version__),
            ("script_version", SCRIPT_VERSION),
            ("weights sha256", weights_sha),
            ("test tile", config["test_tile"]),
            ("test tile sha256", sha256_of_path(REPO_ROOT / config["test_tile"])),
            ("input shape", str(tuple(conv["input_shape"]))),
            ("output shape", str(tuple(reference.shape))),
            ("precision", conv["precision"]),
            ("representation", conv["representation"]),
            ("min deployment", conv["min_deployment"]),
            ("compute units", units),
            ("mlpackage", str(mlpackage.relative_to(REPO_ROOT))),
            ("mlpackage sha256", manifest["outputs"]["mlpackage_sha256"]),
        ],
        metrics,
        config,
    )
    summarise(metrics, threshold, passed)
    return 0 if passed else 1


def run_check(model_id: str, units: str | None) -> int:
    """Verification only: no reconversion, no writes. Hashes must verify and PSNR must hold."""
    config = load_config(model_id)
    model_dir = REPO_ROOT / "models" / model_id
    manifest = validate(model_dir / "manifest.json")
    log("[manifest] schema valid")
    log("[scope] this compares the artifact on disk against the manifest on disk. If a conversion")
    log("[scope] has run since the manifest was committed, both sides are that run's output and")
    log("[scope] this proves self-consistency, not fidelity to the committed evidence. Fidelity is")
    log("[scope] enforced at conversion time, by comparing to the prior manifest before overwriting.")

    units = units or manifest["conversion"]["compute_units"]
    failures: list[str] = []

    weights_path = REPO_ROOT / config["weights"]["path"]
    if not weights_path.exists():
        raise ConversionError(f"weights not found at {weights_path}")
    measured = sha256_of_path(weights_path)
    if measured != manifest["weights"]["sha256"]:
        failures.append(f"weights sha256: manifest {manifest['weights']['sha256']} != disk {measured}")
    else:
        log(f"[verify] weights sha256 matches manifest ({measured[:16]}...)")

    shape = manifest["conversion"]["input_shape"]
    mlpackage = model_dir / "output" / f"{model_id}_{shape[2]}x{shape[3]}.mlpackage"
    if not mlpackage.exists():
        raise ConversionError(f"no .mlpackage at {mlpackage} -- run without --check first")
    measured_pkg = sha256_of_path(mlpackage)
    if measured_pkg != manifest["outputs"]["mlpackage_sha256"]:
        failures.append(
            f"mlpackage sha256: manifest {manifest['outputs']['mlpackage_sha256']} != disk {measured_pkg}"
        )
    else:
        log(f"[verify] mlpackage sha256 matches manifest ({measured_pkg[:16]}...)")

    weights_path, _, _ = verify_weights(config)
    model, _ = build_model(config, weights_path)
    tile = load_tile(config)
    log("[reference] regenerating PyTorch oracle for this check ...")
    reference = reference_pass(model, tile, config["conversion"]["seed"])
    candidate = coreml_predict(mlpackage, tile, units)
    metrics = level_a(reference, candidate)

    threshold = config["thresholds"]["psnr_db"]
    passed = metrics["psnr_db"] >= threshold and not failures
    for failure in failures:
        log(f"[verify] FAIL {failure}")
    summarise(metrics, threshold, passed, mode="check", units=units)
    return 0 if passed else 1


def summarise(metrics: dict, threshold: float, passed: bool, mode: str = "convert", units: str = "") -> None:
    log("")
    log("=" * 78)
    log(f"PSNR          : {metrics['psnr_db']:.2f} dB   (threshold >= {threshold:.0f} dB)")
    log(f"MAE           : {metrics['mae']:.8f}  ({metrics['mae'] * 255:.4f}/255)")
    log(f"max abs error : {metrics['max_abs']:.8f}  ({metrics['max_abs'] * 255:.4f}/255)")
    if units:
        log(f"compute units : {units}")
    log(f"{mode.upper():<13} : {'PASS' if passed else 'FAIL'}")
    log("=" * 78)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproducible PyTorch -> Core ML conversion pipeline"
    )
    parser.add_argument("--model", required=True, help="model id, matching tools/convert/models/<id>.yaml")
    parser.add_argument("--check", action="store_true", help="verify only: no reconversion, no writes")
    parser.add_argument(
        "--compute-units",
        choices=sorted(COMPUTE_UNITS),
        default=None,
        help="Core ML compute units; recorded in the manifest either way (default: from config/manifest)",
    )
    args = parser.parse_args()
    try:
        return run_check(args.model, args.compute_units) if args.check else run_convert(args.model, args.compute_units)
    except (ConversionError, ManifestError, ModelLoadError) as error:
        log("")
        log("=" * 78)
        log(f"STOPPED: {type(error).__name__}: {error}")
        log("=" * 78)
        return 2


if __name__ == "__main__":
    sys.exit(main())
