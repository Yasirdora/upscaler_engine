# Drop the artifacts here

This folder is copied into the app bundle **verbatim** (a folder reference), so each `.mlpackage`
reaches the device uncompiled — the engine compiles it itself, exactly as it does on macOS.

Copy in, from the repository root:

    cp -R models/RealESRGAN_x4plus/output/RealESRGAN_x4plus_256x256.mlpackage DeviceHarnessApp/HarnessArtifacts/
    cp    models/RealESRGAN_x4plus/conversion/input_tile.png                  DeviceHarnessApp/HarnessArtifacts/

Both are gitignored build artifacts, never committed. Generate them with
`python tools/fixtures/make_fixtures.py` and `python tools/convert/convert.py --model RealESRGAN_x4plus`.

The app reports a clear "Model not bundled" message rather than crashing when they are absent.

**Other shapes:** the 384×384 and 512×512 packages drop in alongside, plus their shape-suffixed
tiles (`input_tile_384x384.png`, `input_tile_512x512.png`). File names are load-bearing — the
harness reads the shape off the artifact name (`<identifier>_<N>x<N>.mlpackage`). See
`DeviceHarnessApp/README.md` for the copy commands.
