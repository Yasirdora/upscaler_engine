import CoreGraphics
import Foundation
import ImageIO
import Testing
import UpscaleCore

@testable import UpscaleCLI

/// Argument-parsing tests for `upscale-image`.
///
/// Model-free by design: parsing is a pure function, so every case here runs without the
/// `.mlpackage`. The engine's behaviour is covered by `TilingTests` against a synthetic backend,
/// and the real model is exercised by `certify-tiling`.
struct ImageCommandTests {
    static let minimal = [
        "--model", "m.mlpackage", "--input", "in.png", "--output", "out.png",
        "--method", "centerValidRegion",
    ]

    @Test("a minimal invocation parses")
    func minimalInvocation() throws {
        let options = try ImageCommand.parse(Self.minimal)
        #expect(options.model == "m.mlpackage")
        #expect(options.input.lastPathComponent == "in.png")
        #expect(options.output.lastPathComponent == "out.png")
        #expect(options.method == .centerValidRegion)
        #expect(options.tileSize == 256, "the fixed shape of the converted artifact")
    }

    @Test("every stitch method is accepted by name", arguments: StitchMethod.allCases)
    func everyMethodParses(method: StitchMethod) throws {
        var arguments = Self.minimal
        arguments[arguments.count - 1] = method.rawValue
        #expect(try ImageCommand.parse(arguments).method == method)
    }

    @Test("--method is required — there is no silent default")
    func methodIsRequired() {
        let arguments = Array(Self.minimal.prefix(6))
        #expect(throws: ImageCommand.UsageError.missingRequired(flag: "--method")) {
            try ImageCommand.parse(arguments)
        }
    }

    @Test("an unknown method is rejected and the valid ones are listed")
    func invalidMethod() {
        var arguments = Self.minimal
        arguments[arguments.count - 1] = "blend"
        #expect(throws: ImageCommand.UsageError.invalidMethod("blend")) {
            try ImageCommand.parse(arguments)
        }
        #expect(ImageCommand.UsageError.invalidMethod("blend").message.contains("cosineFeather"))
    }

    @Test("each required flag is named when missing", arguments: ["--model", "--input", "--output"])
    func missingRequiredFlag(flag: String) {
        var arguments = Self.minimal
        if let index = arguments.firstIndex(of: flag) {
            arguments.removeSubrange(index...arguments.index(after: index))
        }
        #expect(throws: ImageCommand.UsageError.missingRequired(flag: flag)) {
            try ImageCommand.parse(arguments)
        }
    }

    @Test(
        "a flag without a value is a usage error, not a silent default",
        arguments: ["--model", "--input", "--output", "--method", "--tile-size"])
    func flagWithoutValue(flag: String) {
        #expect(throws: ImageCommand.UsageError.missingValue(flag: flag)) {
            try ImageCommand.parse([flag])
        }
    }

    @Test("unknown arguments are rejected rather than ignored")
    func unknownArgument() {
        #expect(throws: ImageCommand.UsageError.unknownArgument("--overlap")) {
            try ImageCommand.parse(Self.minimal + ["--overlap", "8"])
        }
    }

    @Test("a non-positive tile size is refused", arguments: ["0", "-256", "abc"])
    func invalidTileSize(value: String) {
        #expect(throws: ImageCommand.UsageError.invalidTileSize(value)) {
            try ImageCommand.parse(Self.minimal + ["--tile-size", value])
        }
    }

    @Test("--tile-size selects the artifact for that shape")
    func tileSizeSelectsTheArtifact() throws {
        let options = try ImageCommand.parse(
            [
                "--model", "RealESRGAN_x4plus", "--input", "i", "--output", "o", "--method", "hardCrop",
                "--tile-size", "512",
            ])
        let descriptor = ImageCommand.descriptor(for: options)

        #expect(descriptor.inputTileSize == 512)
        #expect(descriptor.packageURL.path.hasSuffix("RealESRGAN_x4plus_512x512.mlpackage"))
    }

    @Test("the descriptor carries the converted artifact's geometry")
    func descriptorGeometry() throws {
        let descriptor = ImageCommand.descriptor(for: try ImageCommand.parse(Self.minimal))
        #expect(descriptor.scaleFactor == 4)
        #expect(descriptor.precision == .float16)
    }

    // MARK: - --spool

    @Test("the default destination is the in-memory buffer — --spool is opt-in")
    func spoolDefaultsOff() throws {
        #expect(try ImageCommand.parse(Self.minimal).spool == false)
    }

    @Test("--spool selects the band-spooled destination")
    func spoolParses() throws {
        #expect(try ImageCommand.parse(Self.minimal + ["--spool"]).spool == true)
    }

    @Test("--spool changes nothing except the destination")
    func spoolChangesNothingElse() throws {
        var expected = try ImageCommand.parse(Self.minimal)
        let spooled = try ImageCommand.parse(Self.minimal + ["--spool"])

        // The flag is additive on the condition that every existing invocation keeps its exact
        // current meaning. Equality after flipping the one field is the mechanical form of that
        // condition: any other drift makes the two Options unequal and fails here.
        expected.spool = true
        #expect(spooled == expected)
    }

    @Test("--spool takes no value and does not swallow the next flag")
    func spoolTakesNoValue() throws {
        let options = try ImageCommand.parse(
            [
                "--model", "m.mlpackage", "--input", "in.png", "--output", "out.png",
                "--spool", "--method", "hardCrop", "--tile-size", "512",
            ])

        #expect(options.spool == true)
        #expect(options.method == .hardCrop, "the flag after --spool is still parsed as a flag")
        #expect(options.tileSize == 512)
    }

    @Test("--spool alone is still missing every required flag")
    func spoolAloneIsIncomplete() {
        #expect(throws: ImageCommand.UsageError.missingRequired(flag: "--model")) {
            try ImageCommand.parse(["--spool"])
        }
    }

    @Test(
        "a near-miss of --spool is rejected, not prefix-matched",
        arguments: ["--spooled", "--spo", "-spool"])
    func spoolIsNotPrefixMatched(argument: String) {
        #expect(throws: ImageCommand.UsageError.unknownArgument(argument)) {
            try ImageCommand.parse(Self.minimal + [argument])
        }
    }

    @Test("the usage block documents the flag and that it is off by default")
    func usageDocumentsSpool() {
        #expect(ImageCommand.usage.contains("--spool"))
        #expect(ImageCommand.usage.contains("Off by default."))
    }

    // MARK: - Encoding a spooled raster

    /// The spool's own format, restated from `SpooledImage`: tightly packed BGRA, top row first.
    /// Values are chosen so a channel swap or a row flip cannot pass — every component differs.
    static let spoolPixels: [[UInt8]] = [
        [10, 20, 30, 0], [40, 50, 60, 0],  // row 0: (B,G,R,A)
        [70, 80, 90, 0], [100, 110, 120, 0],  // row 1
    ]

    @Test("a spooled raster encodes to PNG with its channels and rows in the right order")
    func spooledRasterEncodes() throws {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("spool-encode-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let spoolURL = directory.appendingPathComponent("raster.bgra")
        try Data(Self.spoolPixels.flatMap { $0 }).write(to: spoolURL)
        let pngURL = directory.appendingPathComponent("raster.png")

        try ImageCommand.writePNG(
            SpooledImage(url: spoolURL, width: 2, height: 2, bytesPerRow: 8), to: pngURL)

        // Expectations are the literals above read back as RGB — BGRA byte 0 is blue, byte 2 is
        // red — never a value this test recomputed with the encoder under test.
        let decoded = try Self.decodeRGB(pngURL, width: 2, height: 2)
        #expect(decoded[0] == [30, 20, 10], "top-left")
        #expect(decoded[1] == [60, 50, 40], "top-right")
        #expect(decoded[2] == [90, 80, 70], "bottom-left")
        #expect(decoded[3] == [120, 110, 100], "bottom-right")
    }

    @Test("a spool shorter than its declared geometry is refused, never encoded")
    func shortSpoolIsRefused() throws {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("spool-short-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let spoolURL = directory.appendingPathComponent("raster.bgra")
        try Data(Self.spoolPixels.flatMap { $0 }).write(to: spoolURL)  // 16 B, one row short of 3 rows
        let pngURL = directory.appendingPathComponent("raster.png")

        #expect(throws: (any Error).self) {
            try ImageCommand.writePNG(
                SpooledImage(url: spoolURL, width: 2, height: 3, bytesPerRow: 8), to: pngURL)
        }
        #expect(
            FileManager.default.fileExists(atPath: pngURL.path) == false,
            "a truncated spool must not leave a plausible-looking PNG behind")
    }

    @Test("an absent spool file is an error, not an empty image")
    func absentSpoolIsRefused() {
        let absent = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("absent-\(UUID().uuidString).bgra")
        let png = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("absent-\(UUID().uuidString).png")

        #expect(throws: (any Error).self) {
            try ImageCommand.writePNG(
                SpooledImage(url: absent, width: 2, height: 2, bytesPerRow: 8), to: png)
        }
    }

    /// Read a PNG back as row-major RGB triples, through a path that shares no code with the
    /// encoder under test beyond Core Graphics itself.
    static func decodeRGB(_ url: URL, width: Int, height: Int) throws -> [[UInt8]] {
        guard
            let source = CGImageSourceCreateWithURL(url as CFURL, nil),
            let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            throw CocoaError(.fileReadCorruptFile)
        }
        var buffer = [UInt8](repeating: 0, count: width * height * 4)
        buffer.withUnsafeMutableBytes { raw in
            let context = CGContext(
                data: raw.baseAddress, width: width, height: height, bitsPerComponent: 8,
                bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
            context?.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        return (0..<(width * height)).map { Array(buffer[($0 * 4)..<($0 * 4 + 3)]) }
    }

    // MARK: - The derived margin

    /// The CLI holds no constant of its own. The margin is per model, so what this asserts is that
    /// the CLI reads the shared table rather than keeping a number of its own — a second copy is
    /// how the two would drift apart.
    @Test("the CLI resolves each model's own margin, and invents none")
    func contextMarginIsPerModel() {
        #expect(ImageCommand.derivedMargin(for: "RealESRGAN_x4plus") == 76)
        // 76, not its derived 108 — see `ContextMargin.derivedButNotAdopted`.
        #expect(ImageCommand.derivedMargin(for: "RealESRGAN_x2plus") == 76)
        #expect(ImageCommand.derivedMargin(for: "realesr-general-x4v3") == 16)
        // Upstream's tile_pad is 10 and is a reference point only.
        #expect(ImageCommand.derivedMargin(for: "RealESRGAN_x4plus") != 10)
        // No default: an underived model is refused, never run on another model's number.
        #expect(ImageCommand.derivedMargin(for: "not-a-model") == nil)
    }

    @Test(
        "an artifact stem resolves to the model family its margin is keyed on",
        arguments: [
            ("RealESRGAN_x4plus_256x256", "RealESRGAN_x4plus"),
            ("RealESRGAN_x4plus_384_384x384", "RealESRGAN_x4plus"),
            ("realesr-general-x4v3_512x512", "realesr-general-x4v3"),
            ("RealESRGAN_x2plus", "RealESRGAN_x2plus"),
            ("realesr-general-x4v3", "realesr-general-x4v3"),
        ])
    func artifactStemsResolveToTheirFamily(stem: String, family: String) {
        #expect(ImageCommand.modelFamily(of: stem) == family)
    }

    @Test("a bare model id run through upscale-image finds its margin")
    func bareModelIdFindsItsMargin() throws {
        let options = try ImageCommand.parse([
            "--model", "RealESRGAN_x4plus", "--input", "in.png", "--output", "out.png",
            "--method", "centerValidRegion",
        ])
        let descriptor = ImageCommand.descriptor(for: options)
        #expect(descriptor.identifier == "RealESRGAN_x4plus_256x256")
        let family = ImageCommand.modelFamily(of: descriptor.identifier)
        #expect(family == "RealESRGAN_x4plus")
        #expect(ImageCommand.derivedMargin(for: family) == 76)
    }

    // MARK: - Exit-code contract

    @Test("usage errors exit 2 and print the usage block")
    func usageErrorExitsTwo() async {
        var errors: [String] = []
        let status = await ImageCommand.run(
            arguments: ["--model", "m"], out: { _ in }, err: { errors.append($0) })

        #expect(status == 2)
        #expect(errors.contains { $0.contains("--input") })
        #expect(errors.contains { $0.contains("USAGE:") })
    }

    @Test("a missing model exits 3 — an engine failure, not misuse")
    func missingModelExitsThree() async {
        let absent = NSTemporaryDirectory() + "definitely-absent.mlpackage"
        var errors: [String] = []
        let status = await ImageCommand.run(
            arguments: [
                "--model", absent, "--input", "in.png", "--output", "out.png",
                "--method", "centerValidRegion",
            ], out: { _ in }, err: { errors.append($0) })

        #expect(status == 3)
        #expect(errors.contains { $0.contains("error") })
    }

    // MARK: - Report formatting

    @Test("report columns pad without truncating, in both directions")
    func reportColumns() {
        #expect(TilingCertificationCommand.column("abc", 6) == "abc   ")
        #expect(TilingCertificationCommand.column("abc", 6, right: true) == "   abc")
        #expect(
            TilingCertificationCommand.column("a-very-long-value", 6) == "a-very-long-value ",
            "an over-long value keeps its content and a separator rather than being cut")
    }
}
