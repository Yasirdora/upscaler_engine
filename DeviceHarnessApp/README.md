# DeviceHarnessApp

A minimal iOS shell over `UpscaleCore` and `DeviceHarness`: one screen, one button, one result
panel. It exists only because SwiftPM cannot produce an `.app`; all logic lives in the
`DeviceHarness` package target, where it is unit-tested on the host.

**1. Drop in the artifacts** — gitignored, so this step is manual. Missing ones are reported on
screen ("Model not bundled. Copy `<name>` …"), never a crash. The 256 pair alone runs the
single-tile check:

    cp -R models/RealESRGAN_x4plus/output/RealESRGAN_x4plus_256x256.mlpackage DeviceHarnessApp/HarnessArtifacts/
    cp    models/RealESRGAN_x4plus/conversion/input_tile.png                  DeviceHarnessApp/HarnessArtifacts/

Add the 384 or 512 variant and the app grows a shape picker — the harness reads the shape off the
artifact's own name, so these names are load-bearing:

    cp -R models/RealESRGAN_x4plus_384/output/RealESRGAN_x4plus_384_384x384.mlpackage DeviceHarnessApp/HarnessArtifacts/RealESRGAN_x4plus_384x384.mlpackage
    cp    models/RealESRGAN_x4plus_384/conversion/input_tile.png                      DeviceHarnessApp/HarnessArtifacts/input_tile_384x384.png
    cp -R models/RealESRGAN_x4plus_512/output/RealESRGAN_x4plus_512_512x512.mlpackage DeviceHarnessApp/HarnessArtifacts/RealESRGAN_x4plus_512x512.mlpackage
    cp    models/RealESRGAN_x4plus_512/conversion/input_tile.png                      DeviceHarnessApp/HarnessArtifacts/input_tile_512x512.png

Inside `HarnessArtifacts/`, models are named `<family>_<N>x<N>.mlpackage` (the family id, such as
`RealESRGAN_x4plus`, plus the shape — which is why the commands above rename the variants); tiles
are `input_tile_<N>x<N>.png`, except the 256 tile, which is the unsuffixed `input_tile.png`. For whole-image runs, also copy the workload images
from `Sources/TestFixtures/workloads/` (generate them with `python tools/fixtures/make_fixtures.py`).

**2. Build**

    xcodebuild -project DeviceHarnessApp/DeviceHarnessApp.xcodeproj -scheme DeviceHarnessApp -destination 'generic/platform=iOS' build CODE_SIGNING_ALLOWED=NO

**3. Run on a physical iPhone** (free provisioning is fine): open the project in Xcode →
DeviceHarnessApp scheme → select your phone → Signing & Capabilities → Team = your Apple ID, and
change the bundle identifier `com.example.upscaler.harness` to one of your own → Run. On first
launch, trust the developer under Settings → General → VPN & Device Management.

**4. Record:** device and chip, iOS version, compute-units setting, tile shape, SUCCESS/FAILED,
output size, and a screenshot of the result panel.

**Launch arguments.** `-memoryprobe` runs the allocate-until-killed memory probe. `-wholeimage`
runs a whole-image series (`-workload A|B|C`, `-runs N`, `-state cold|warm`, …; see
`WholeImageSeriesRequest`) and writes the record to the app's Documents folder after every run.

Wall clock and peak memory are **informational only** — they are not benchmark results and should
not be quoted as such (certification spec §7.3). Wall clock includes Core ML model compilation on a
cold start.
