# tools/convert — model conversion

`convert.py` turns a per-model YAML config into a fixed-shape `.mlpackage` plus the identity
documents that make it auditable.

## Setup — Python 3.12 is the supported interpreter line

    python3.12 -m venv .venv && source .venv/bin/activate
    pip install -r tools/convert/requirements.txt

## convert.py — reproducible conversion

    python tools/convert/convert.py --model RealESRGAN_x4plus
    python tools/convert/convert.py --model RealESRGAN_x4plus --check
    python tools/convert/convert.py --model RealESRGAN_x4plus --compute-units CPU_ONLY

Reads `tools/convert/models/<id>.yaml`, verifies the local weights SHA256 against the config,
reconstructs the architecture, converts (coremltools PyTorch frontend, no ONNX; `mlprogram`;
fp16; fixed shape), then runs the PyTorch oracle and Core ML on the config's test tile (generate the
tiles first with `python tools/fixtures/make_fixtures.py --tiles`):

- `models/<id>/manifest.json` — the identity document, validated before it is written
- `models/<id>/conversion/` — `config.yaml` (committed); `report.txt`, `input_tile.png`, `reference.png`
  and `coreml.png` are local by-products and gitignored
- `models/<id>/output/<id>_HxW.mlpackage` — gitignored

`--check` verifies without reconverting and without writing anything: manifest schema, weights
and `.mlpackage` hashes, and Level A metrics against a freshly regenerated PyTorch oracle.
Exit 0 iff all hold; exit 1 on a failed threshold or hash; exit 2 when the pipeline stops
deliberately (missing weights, missing `.mlpackage`, unidentified artifact).

A conversion whose fidelity fields disagree with an existing manifest never overwrites it: it
reports `FIDELITY DRIFT`, writes `manifest.candidate.json` beside it, and exits 1 so a person
decides whether the change is legitimate (for example, a new test tile) before accepting it.

Compute units affect fp16 numerics, so the setting used is recorded in the manifest and the
report. `--check` defaults to whatever the manifest recorded.

`.mlpackage` bytes are **not** reproducible across runs — Core ML compilation is not
guaranteed deterministic — so reproducibility is asserted at Level A against the oracle,
never by bitwise identity of the package.

## Tests

    pytest tools/convert/tests/

Licence: Real-ESRGAN is BSD-3-Clause and is reimplemented with attribution; reports carry the
attribution. No Upscayl (AGPL-3.0) or other GPL/AGPL source is used.
