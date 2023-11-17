# Example: upscale a photograph

A small command-line program that upscales one image end to end with the engine's public API. It
is a separate package that depends on the engine by product, exactly as an app would.

[`Sources/UpscaleImage/main.swift`](Sources/UpscaleImage/main.swift) walks through the steps in
order:

1. decode the file into the engine's input contract, keeping its colour profile and metadata;
2. plan the colour policy;
3. normalise the EXIF orientation;
4. split off alpha, if any — only colour goes through the model;
5. load the converted Core ML model and upscale tile by tile;
6. upscale alpha separately and recombine;
7. encode, embedding the planned profile and the original metadata.

## Run it

Convert a model first (see [`docs/MODELS.md`](../../docs/MODELS.md)), then from the repository
root:

```bash
swift run --package-path Examples/UpscaleImage upscale-image \
    --model models/RealESRGAN_x4plus \
    --input docs/images/chaukhandi-input.jpg \
    --output chaukhandi_x4.png
```

```
colour: source profile preserved end to end: sRGB IEC61966-2.1
upscaling 1280×883 ×4 with RealESRGAN_x4plus
  done in 75.2 sec
wrote chaukhandi_x4.png (5120×3532)
```

That is the image behind the comparison at the top of the main README (timing from an M-series
Mac).

Options:

- `--model` — a model directory holding `manifest.json` and
  `output/<id>_<N>x<N>.mlpackage`, as written by `tools/convert/convert.py`.
- `--output` — the format follows the extension: `.png`, `.jpg`, `.heic` or `.tiff`.
- `--method` — `centerValidRegion` (default), `hardCrop`, `linearFeather` or `cosineFeather`.

The example keeps the whole output image in memory. For very large images, use the band-spooled
path (`SequentialTilingEngine.upscale(_:model:backend:method:spoolingTo:progress:)`) or the
laddered path with a `MemoryBudget`; `UpscaleCLI upscale-image --spool` shows the former.
