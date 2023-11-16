import DeviceHarness
import SwiftUI

/// Device harness shell. One screen, one button, one result panel — the engine runs headless in
/// `UpscaleCore`/`DeviceHarness`, and this target exists only because SwiftPM cannot produce an iOS
/// `.app`. Deliberately no design pass, no icon, no branding.
///
/// Two diagnostics share the shell, both reached only by a launch argument:
///
/// - `-memoryprobe` swaps the root view for the allocate-until-killed memory probe.
/// - `-wholeimage` runs a whole-image series (`WholeImageRunner`) against the bundled workload,
///   printing the record and writing it to Documents. **It adds no view and no user-facing state**,
///   so the integration is one `.task` over the single-tile screen.
///
/// Without either argument this target is the single-tile harness, unchanged.
@main
struct DeviceHarnessAppMain: App {
    var body: some Scene {
        WindowGroup {
            if MemoryProbe.isRequested() {
                MemoryProbeView()
            } else {
                ContentView()
                    // Returns immediately when `-wholeimage` is absent. The series is a headless
                    // `nonisolated` call, so the tiling work runs off the main actor even though
                    // `.task` starts on it.
                    .task { await WholeImageRunner.runSeriesIfRequested() }
            }
        }
    }
}
