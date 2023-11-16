import Foundation
import Testing
import UpscaleCore

@testable import DeviceHarness

/// Model-free tests for the device harness.
///
/// Nothing here loads the `.mlpackage` or touches a device: these cover result formatting, error
/// mapping and resource resolution — the parts that decide whether a human staring at a phone can
/// tell what happened.
struct SingleTileRunnerTests {
    static func result(
        success: Bool = true,
        size: CGSize = CGSize(width: 1024, height: 1024),
        seconds: TimeInterval = 1.5,
        memory: UInt64? = 512 * 1024 * 1024,
        footprint: UInt64? = 192 * 1024 * 1024,
        units: String = "all",
        error: String? = nil
    ) -> SingleTileResult {
        SingleTileResult(
            success: success, outputSize: size, wallClockSeconds: seconds,
            peakMemoryBytes: memory, peakFootprintBytes: footprint, computeUnits: units,
            errorDescription: error)
    }

    // MARK: - Formatting

    @Test("a successful run shows status, units, output size and every measurement")
    func successRows() {
        let rows = DeviceHarness.displayRows(for: Self.result())
        let labels = rows.map(\.label)

        // The load/inference split and the cold warning exist because the total alone would
        // let a cold 512 run be compared against a warm 256 one as though the difference were the
        // model. `Self.result()` defaults to a cold run, hence the comparability row.
        #expect(
            labels == [
                "Status", "Compute units", "Output size", "Wall clock", "  ├ model load",
                "  └ inference", "  ⚠︎ comparability", "Peak memory", "Peak footprint",
            ])
        #expect(rows.first { $0.label == "Status" }?.value == "SUCCESS")
        #expect(rows.first { $0.label == "Output size" }?.value == "1024 × 1024")
    }

    @Test("a failed run omits output size and shows the error last")
    func failureRows() {
        let rows = DeviceHarness.displayRows(
            for: Self.result(success: false, size: .zero, error: "model not found at /x"))

        #expect(rows.map(\.label).contains("Output size") == false)
        #expect(rows.last?.label == "Error")
        #expect(rows.last?.value.contains("/x") == true)
        #expect(rows.first?.value == "FAILED")
    }

    @Test("measurements are labelled in the UI, not presented bare")
    func measurementsCarryTheirCaveat() {
        let rows = DeviceHarness.displayRows(for: Self.result())

        #expect(rows.first { $0.label == "Wall clock" }?.value.contains("informational") == true)
        #expect(rows.first { $0.label == "Peak memory" }?.value.contains("best-effort") == true)
        #expect(DeviceHarness.measurementDisclaimer.contains("never quote"))
    }

    @Test("sub-second timings read in milliseconds, longer ones in seconds")
    func timeFormatting() {
        #expect(DeviceHarness.format(seconds: 0.25) == "250 ms")
        #expect(DeviceHarness.format(seconds: 12.345) == "12.35 s")
    }

    @Test("unavailable memory says so rather than showing zero")
    func memoryFormatting() {
        #expect(DeviceHarness.format(bytes: nil) == "unavailable")
        #expect(DeviceHarness.format(bytes: 1024 * 1024) == "1.0 MB")
        #expect(DeviceHarness.format(bytes: 1_610_612_736) == "1536.0 MB")
    }

    // MARK: - Footprint alongside RSS

    /// The point of the footprint row, locked as a test: the RSS row is *added to*, never replaced.
    ///
    /// Earlier single-tile records are written in this exact field, this exact unit string and this
    /// exact value. A change to any of the three silently breaks comparability with every earlier
    /// device number, so all three are asserted literally rather than by shape.
    @Test("adding the footprint row leaves the RSS row's label, units and value untouched")
    func residentSizeRowIsUnchanged() {
        // 512 × 1024² = 536,870,912 B, hand-computed, formatted by the pre-existing MB path.
        let rows = DeviceHarness.displayRows(for: Self.result(memory: 536_870_912))
        let rss = rows.first { $0.label == "Peak memory" }

        #expect(rss?.value == "512.0 MB  (best-effort)")
        // The row's absolute position depends on the rows above it; what this test protects
        // is that the RSS row still sits immediately before the footprint row, unchanged in label,
        // units and value. An absolute index was never the property that mattered.
        let rssIndex = rows.firstIndex { $0.label == "Peak memory" }
        let footprintIndex = rows.firstIndex { $0.label == "Peak footprint" }
        #expect(rssIndex != nil)
        #expect(footprintIndex == rssIndex.map { $0 + 1 }, "RSS still immediately precedes footprint")
    }

    /// A reader who mistakes a high-water mark for a point sample misreads every memory result. The
    /// row has to say which it is on its face, without the doc comment.
    @Test("the footprint row names its metric and declares itself a high-water mark")
    func footprintRowIsSelfDescribing() {
        // 192 × 1024² = 201,326,592 B, hand-computed.
        let value = DeviceHarness.displayRows(for: Self.result(footprint: 201_326_592))
            .first { $0.label == "Peak footprint" }?.value

        #expect(value?.hasPrefix("192.0 MiB") == true)
        #expect(value?.contains("phys_footprint") == true, "the enforced metric is named")
        #expect(value?.contains("high-water") == true, "never mistakable for a point sample")
        #expect(value?.contains("best-effort") == true, "same contract as the RSS reading")
    }

    /// MiB, not the "MB" the adjacent row uses for the same divisor — the footprint figures are
    /// read against ceilings quoted in MiB, and a new field inherits no naming defect. Expectations
    /// are hand-computed powers of two, not recomputed by the code.
    @Test("footprint formats as MiB and says so")
    func footprintFormatting() {
        #expect(DeviceHarness.format(footprintBytes: nil) == "unavailable")
        #expect(DeviceHarness.format(footprintBytes: 1_048_576) == "1.0 MiB")
        #expect(DeviceHarness.format(footprintBytes: 1_572_864) == "1.5 MiB")
        #expect(DeviceHarness.format(footprintBytes: 3_221_225_472) == "3072.0 MiB")
    }

    @Test("an unavailable footprint reads unavailable, never a zero that looks measured")
    func unavailableFootprintIsHonest() {
        let rows = DeviceHarness.displayRows(for: Self.result(footprint: nil))

        #expect(
            rows.first { $0.label == "Peak footprint" }?.value.hasPrefix("unavailable") == true)
        // The RSS row is independent: one reading failing must not blank the other.
        #expect(rows.first { $0.label == "Peak memory" }?.value.hasPrefix("512.0 MB") == true)
    }

    @Test("peak footprint is either a real figure or nil — never fabricated")
    func peakFootprintIsBestEffort() {
        if let bytes = SingleTileRunner.peakFootprintBytes() {
            #expect(bytes > 0, "a reported figure must be a real measurement")
        }
    }

    @Test("a failed run still carries both memory readings, so a crash-adjacent run is legible")
    func failureCarriesBothReadings() {
        let rows = DeviceHarness.displayRows(
            for: SingleTileRunner.failure("no model", computeUnits: .cpuOnly).result)

        #expect(rows.contains { $0.label == "Peak memory" })
        #expect(rows.contains { $0.label == "Peak footprint" })
    }

    // MARK: - Error mapping

    @Test("typed engine errors keep their own description")
    func mapsInferenceErrors() {
        let url = URL(fileURLWithPath: "/tmp/absent.mlpackage")
        #expect(SingleTileRunner.describe(InferenceError.modelNotFound(url)).contains("absent.mlpackage"))
        #expect(
            SingleTileRunner.describe(InferenceError.invalidTileSize(expected: 256, got: 512))
                .contains("512"))
    }

    @Test(
        "missing resources produce an instruction naming the file, not a stack trace",
        arguments: [
            SingleTileRunner.ResourceError.modelMissing(fileName: "RealESRGAN_x4plus_512x512.mlpackage"),
            .tileMissing(fileName: "input_tile_512x512.png"),
        ])
    func resourceErrorsAreActionable(error: SingleTileRunner.ResourceError) {
        let message = SingleTileRunner.describe(error)
        #expect(message.contains("DeviceHarnessApp/HarnessArtifacts/"))
        #expect(message.contains("rebuild"))
        #expect(message.contains("512x512"), "the message must say which artifact is missing")
    }

    // MARK: - Behaviour reachable without a model

    @Test("a nil model URL fails with the copy-it-in instruction, not a crash")
    func missingModelIsReported() async {
        let run = await DeviceHarness.runSingleTile(modelURL: nil, tileURL: nil, computeUnits: .cpuOnly)

        #expect(run.result.success == false)
        #expect(run.outputPNG == nil)
        #expect(run.result.computeUnits == "cpuOnly")
        #expect(run.result.errorDescription?.contains("Model not bundled") == true)
    }

    @Test("a bundle without the resource folder resolves to nil rather than a bogus URL")
    func bundleResolutionIsHonest() {
        // The test runner's own bundle has no Resources/ folder — exactly the state of an app
        // built before the human copies the artifacts in.
        #expect(DeviceHarness.bundledModelURL(in: .main) == nil)
        #expect(DeviceHarness.bundledTileURL(in: .main) == nil)
    }

    @Test("peak memory is either a real figure or nil — never fabricated")
    func peakMemoryIsBestEffort() {
        if let bytes = SingleTileRunner.peakMemoryBytes() {
            #expect(bytes > 0, "a reported figure must be a real measurement")
        }
    }
}
