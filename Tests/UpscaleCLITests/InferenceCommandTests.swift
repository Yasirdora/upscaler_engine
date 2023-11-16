import Foundation
import Testing
import UpscaleCore

@testable import UpscaleCLI

/// Argument-parsing and validation tests for `upscale-tile`.
///
/// Model-free by design: parsing is a pure function, so every case here runs without the
/// `.mlpackage`. The one execution test uses a path that deliberately does not exist — proving the
/// exit-code contract needs the model's *absence*, not its presence.
struct InferenceCommandTests {
    static let minimal = ["--model", "m.mlpackage", "--input", "in.png", "--output", "out.png"]

    @Test("a minimal invocation parses, defaulting compute units to all")
    func minimalInvocation() throws {
        let options = try InferenceCommand.parse(Self.minimal)
        #expect(options.model == "m.mlpackage")
        #expect(options.input.lastPathComponent == "in.png")
        #expect(options.output.lastPathComponent == "out.png")
        #expect(options.computeUnits == .all)
    }

    @Test("all four compute-unit spellings are accepted", arguments: ComputeUnits.allCases)
    func everyComputeUnitValueParses(units: ComputeUnits) throws {
        let options = try InferenceCommand.parse(Self.minimal + ["--compute-units", units.rawValue])
        #expect(options.computeUnits == units)
    }

    @Test("flags may appear in any order")
    func flagOrderIsIrrelevant() throws {
        let reordered = [
            "--compute-units", "cpuOnly", "--output", "out.png", "--input", "in.png", "--model", "m",
        ]
        #expect(try InferenceCommand.parse(reordered).computeUnits == .cpuOnly)
        #expect(try InferenceCommand.parse(reordered).model == "m")
    }

    @Test("an unknown compute-unit value is rejected and named")
    func invalidComputeUnits() {
        #expect(throws: InferenceCommand.UsageError.invalidComputeUnits("gpu")) {
            try InferenceCommand.parse(Self.minimal + ["--compute-units", "gpu"])
        }
    }

    @Test(
        "each required flag is reported by name when missing", arguments: ["--model", "--input", "--output"])
    func missingRequiredFlag(flag: String) {
        var arguments = Self.minimal
        if let index = arguments.firstIndex(of: flag) {
            arguments.removeSubrange(index...arguments.index(after: index))
        }
        #expect(throws: InferenceCommand.UsageError.missingRequired(flag: flag)) {
            try InferenceCommand.parse(arguments)
        }
    }

    @Test(
        "a flag without a value is a usage error, not a silent default",
        arguments: [
            "--model", "--input", "--output", "--compute-units",
        ])
    func flagWithoutValue(flag: String) {
        #expect(throws: InferenceCommand.UsageError.missingValue(flag: flag)) {
            try InferenceCommand.parse([flag])
        }
    }

    @Test("unknown arguments are rejected rather than ignored")
    func unknownArgument() {
        #expect(throws: InferenceCommand.UsageError.unknownArgument("--frobnicate")) {
            try InferenceCommand.parse(Self.minimal + ["--frobnicate", "512"])
        }
    }

    @Test("--tile-size defaults to the 256 fixed shape when absent")
    func tileSizeDefaults() throws {
        let options = try InferenceCommand.parse(Self.minimal)
        #expect(options.tileSize == 256, "every pre-variant invocation is byte-identical")
    }

    @Test("--tile-size selects the converted variant and shapes the descriptor")
    func tileSizeSelectsVariant() throws {
        let options = try InferenceCommand.parse([
            "--model", "RealESRGAN_x4plus", "--input", "i", "--output", "o", "--tile-size", "384",
        ])
        #expect(options.tileSize == 384)
        let descriptor = InferenceCommand.descriptor(for: options)
        #expect(descriptor.inputTileSize == 384)
        #expect(descriptor.identifier == "RealESRGAN_x4plus_384x384")
    }

    @Test(
        "invalid --tile-size values are rejected and named",
        arguments: ["0", "-1", "abc", "256.5"])
    func invalidTileSize(value: String) {
        #expect(throws: InferenceCommand.UsageError.invalidTileSize(value)) {
            try InferenceCommand.parse(Self.minimal + ["--tile-size", value])
        }
    }

    @Test("usage errors carry a message naming the problem")
    func usageErrorMessages() {
        #expect(InferenceCommand.UsageError.missingRequired(flag: "--model").message.contains("--model"))
        #expect(InferenceCommand.UsageError.unknownArgument("--nope").message.contains("--nope"))
        #expect(InferenceCommand.UsageError.invalidComputeUnits("gpu").message.contains("cpuAndNeuralEngine"))
    }

    // MARK: - Model resolution

    @Test("a bare model id resolves to the conversion pipeline's artifact layout")
    func bareIdentifierResolution() {
        let url = InferenceCommand.resolveModelURL("RealESRGAN_x4plus")
        #expect(url.path.hasSuffix("models/RealESRGAN_x4plus/output/RealESRGAN_x4plus_256x256.mlpackage"))
    }

    @Test(
        "an explicit path is used as given",
        arguments: [
            "models/X/output/Y.mlpackage", "/tmp/custom.mlpackage", "./relative/thing.mlpackage",
        ])
    func explicitPathResolution(path: String) {
        #expect(
            InferenceCommand.resolveModelURL(path).path.hasSuffix(
                URL(fileURLWithPath: path).lastPathComponent))
    }

    @Test("the descriptor takes its identifier from the package name")
    func descriptorIdentifier() throws {
        let options = try InferenceCommand.parse([
            "--model", "RealESRGAN_x4plus", "--input", "i", "--output", "o",
        ])
        let descriptor = InferenceCommand.descriptor(for: options)
        #expect(descriptor.identifier == "RealESRGAN_x4plus_256x256")
        #expect(descriptor.inputTileSize == 256)
        #expect(descriptor.scaleFactor == 4)
        #expect(descriptor.precision == .float16)
    }

    // MARK: - Exit-code contract

    @Test("usage errors exit 2 and print the usage block")
    func usageErrorExitsTwo() async {
        var errors: [String] = []
        let status = await InferenceCommand.run(
            arguments: ["--model", "m"], out: { _ in }, err: { errors.append($0) })

        #expect(status == 2)
        #expect(errors.contains { $0.contains("--input") })
        #expect(errors.contains { $0.contains("USAGE:") })
    }

    @Test("a missing model exits 3, not 2 — it is an inference failure, not misuse")
    func missingModelExitsThree() async {
        let absent = NSTemporaryDirectory() + "definitely-absent.mlpackage"
        var errors: [String] = []
        let status = await InferenceCommand.run(
            arguments: ["--model", absent, "--input", "in.png", "--output", "out.png"],
            out: { _ in }, err: { errors.append($0) }
        )

        #expect(status == 3)
        #expect(errors.contains { $0.contains("inference error") })
    }
}
