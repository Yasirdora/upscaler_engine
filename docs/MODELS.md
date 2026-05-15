# Models

The engine runs **fixed-shape** Core ML models: each `.mlpackage` accepts exactly one input tile
size, and arbitrary images are handled by the tile planner. A model is identified by its
`models/<id>/manifest.json`, which pins the weights by SHA256 and records how the package was
converted and how faithful the conversion is.

No weights and no converted packages are committed. You fetch the weights, generate the test tiles,
and convert locally; every step verifies its inputs.

## Supported models

| Model id | Architecture | Scale | Tile shapes with configs | Context margin | Receptive field |
|---|---|---:|---|---:|---:|
| `RealESRGAN_x4plus` | RRDBNet, 23 blocks, 16.7 M params | ×4 | 256, 384, 512, 768, 1024 | 76 px | 348.25 px |
| `RealESRGAN_x2plus` | RRDBNet + `pixel_unshuffle` | ×2 | 256, 512 | 76 px (derived 108) | 697.50 px |
| `realesr-general-x4v3` | SRVGGNetCompact, 1.2 M params | ×4 | 256, 512 | 16 px | 34.00 px |

- **Committed manifests:** `RealESRGAN_x4plus` (256) and `RealESRGAN_x4plus_384`. The other configs
  in `tools/convert/models/` produce their manifest on first conversion. Converting the 512 and
  larger shapes of RRDBNet needs a lot of memory: it has been observed to exceed a 16 GB Mac.
- **Context margins** are measured per model, never copied from upstream (certification spec §5.1).
  The engine refuses a model with no derived margin rather than run it on another model's number:
  under-padding throws nothing, it produces seams. `RealESRGAN_x2plus` runs at 76 px, below its
  derived 108 px, because seam certification at both margins found no visible difference; see
  `ContextMargin.derivedButNotAdopted`.
- **Variants** carry their shape in the id (`RealESRGAN_x4plus_384`), because a manifest describes
  exactly one shape. All x4plus variants read the same weights file.

## What to expect from the models

`RealESRGAN_x4plus` is trained on heavily degraded images, and it shows: it recovers edges, frames
and regular structure convincingly, and it smooths what it cannot resolve rather than inventing
detail. Two failure modes are worth knowing before you pick it:

- **Very fine ornament is flattened.** Filigree a few pixels wide in the input comes back as smooth
  shapes, not as the original pattern.
- **Small, saturated mosaics lose colour.** A tiled wall of tiny coloured cells tends to come back
  as a sharp, mostly neutral grid.

`realesr-general-x4v3` needs about 14× fewer operations and behaves differently; compare both on your own
images (certification spec §9) before choosing.

## Workflow

Set up the Python tooling once (Python 3.12):

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r tools/convert/requirements.txt
```

Then, per model:

```bash
# 1. Fetch the weights pinned in the manifest (SHA256-verified; nothing unverified is kept).
python tools/artifacts/fetch.py --model RealESRGAN_x4plus

# 2. Generate the synthetic test tiles (and the workload images used on device).
python tools/fixtures/make_fixtures.py

# 3. Convert PyTorch -> Core ML and measure Level A fidelity against the PyTorch oracle.
python tools/convert/convert.py --model RealESRGAN_x4plus

# 4. Verify an existing conversion without rewriting anything.
python tools/convert/convert.py --model RealESRGAN_x4plus --check
```

The package lands at `models/<id>/output/<id>_<N>x<N>.mlpackage`, which is where `UpscaleCLI`
resolves a bare model id:

```bash
swift run UpscaleCLI upscale-image --model RealESRGAN_x4plus \
    --input photo.png --output photo_x4.png --method centerValidRegion
```

For a model without a committed manifest, download the weights from the `weights.url` in its
`tools/convert/models/<id>.yaml` to the `weights.path` it names; `convert.py` verifies the SHA256
before it converts anything.

`.mlpackage` bytes are not reproducible run to run — Core ML compilation is not deterministic — so
identity is the weights SHA256, and a rebuilt package is verified at Level A against the oracle,
never by its hash.

## Certification

Beyond the single-tile conversion check, the full Level A/B and tiling certification runs against
a local image corpus (`Sources/TestFixtures/corpus/README.md`):

```bash
python tools/reference/run_reference.py --model RealESRGAN_x4plus      # PyTorch oracle outputs
python tools/certify/certify.py --model RealESRGAN_x4plus --compute-units all
swift run UpscaleCLI certify-tiling --model RealESRGAN_x4plus --output tiling_report.txt
swift run UpscaleCLI derive-context-margin --model RealESRGAN_x4plus --output margin_report.txt
```

The thresholds and methodology are in [`CERTIFICATION.md`](CERTIFICATION.md).

## Licensing

The weights are published by the Real-ESRGAN project and are not part of this repository. Their
terms, and the terms of the datasets they were trained on, are set by their publishers; see
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) before redistributing weights or converted
models.
