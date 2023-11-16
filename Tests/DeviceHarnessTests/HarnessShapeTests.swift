import Foundation
import Testing

@testable import DeviceHarness

/// Model-free tests for fixed-shape selection.
///
/// The harness reads a shape off an artifact's file name and offers whatever was dropped in.
/// Nothing here loads a `.mlpackage`: what is under test is the naming convention and the discovery
/// around it — the parts that decide *which* weights a device run actually measured.
struct HarnessShapeTests {
    static let shipped = [256, 384, 512]

    // MARK: - Parsing the convention

    @Test("the conversion pipeline's names parse into family plus geometry", arguments: shipped)
    func parsesShippedNames(size: Int) {
        let shape = HarnessShape.parse(modelFileName: "RealESRGAN_x4plus_\(size)x\(size).mlpackage")

        #expect(shape?.identifier == "RealESRGAN_x4plus")
        #expect(shape?.tileSize == size)
    }

    @Test("only the geometry is split off — an identifier keeps its own underscores")
    func identifierKeepsUnderscores() {
        let shape = HarnessShape.parse(modelFileName: "Real_ESRGAN_x4_plus_384x384.mlpackage")

        #expect(shape?.identifier == "Real_ESRGAN_x4_plus")
        #expect(shape?.tileSize == 384)
    }

    @Test(
        "a name the convention would not have produced is rejected, never guessed",
        arguments: [
            "RealESRGAN_x4plus_384x512.mlpackage",  // not square
            "RealESRGAN_x4plus_384x384x384.mlpackage",  // not two sides
            "RealESRGAN_x4plus.mlpackage",  // no geometry
            "RealESRGAN_x4plus_384x384.mlmodelc",  // compiled, not the package
            "RealESRGAN_x4plus_384x384",  // no extension
            "RealESRGAN_x4plus_axa.mlpackage",  // not numeric
            "RealESRGAN_x4plus_0384x0384.mlpackage",  // padded
            "RealESRGAN_x4plus_+384x+384.mlpackage",  // signed
            "RealESRGAN_x4plus_-384x-384.mlpackage",  // negative
            "RealESRGAN_x4plus_0x0.mlpackage",  // degenerate
            "_256x256.mlpackage",  // no identifier
            "256x256.mlpackage",  // no separator
            ".mlpackage",
            "",
        ])
    func rejectsNamesOutsideTheConvention(fileName: String) {
        #expect(HarnessShape.parse(modelFileName: fileName) == nil)
    }

    @Test("a shape's own file name parses back into the same shape", arguments: shipped)
    func roundTripsThroughTheFileName(size: Int) {
        let shape = HarnessShape(identifier: "RealESRGAN_x4plus", tileSize: size)

        #expect(HarnessShape.parse(modelFileName: shape.modelFileName) == shape)
    }

    // MARK: - The 256 path is unchanged

    @Test("the default shape is exactly the original 256 artifact")
    func defaultShapeIsThe256Artifact() {
        #expect(HarnessShape.default256.modelFileName == DeviceHarness.modelFileName)
        #expect(HarnessShape.default256.tileFileName == DeviceHarness.tileFileName)
        #expect(HarnessShape.default256.tileSize == SingleTileRunner.tileSize)
        #expect(HarnessShape.default256.identifier == "RealESRGAN_x4plus")
    }

    @Test("256 resolves the unsuffixed tile first; the other shapes never accept it")
    func tileNamesPerShape() {
        #expect(
            HarnessShape.default256.tileFileNames == ["input_tile.png", "input_tile_256x256.png"])
        #expect(
            HarnessShape(identifier: "RealESRGAN_x4plus", tileSize: 384).tileFileNames
                == ["input_tile_384x384.png"])
        #expect(
            HarnessShape(identifier: "RealESRGAN_x4plus", tileSize: 512).tileFileNames
                == ["input_tile_512x512.png"])
    }

    @Test("the 256 descriptor is the family one, shape or no shape")
    func descriptorForDefaultShapeIsUnchanged() {
        let url = URL(fileURLWithPath: "/tmp/RealESRGAN_x4plus_256x256.mlpackage")

        #expect(
            SingleTileRunner.descriptor(modelURL: url)
                == SingleTileRunner.descriptor(shape: .default256, modelURL: url))
        #expect(SingleTileRunner.descriptor(modelURL: url).identifier == "RealESRGAN_x4plus")
        #expect(SingleTileRunner.descriptor(modelURL: url).inputTileSize == 256)
    }

    // MARK: - Descriptors for the new shapes

    @Test("a shape's descriptor carries its geometry and nothing else changes", arguments: shipped)
    func descriptorCarriesTheShape(size: Int) {
        let shape = HarnessShape(identifier: "RealESRGAN_x4plus", tileSize: size)
        let descriptor = SingleTileRunner.descriptor(
            shape: shape, modelURL: URL(fileURLWithPath: "/tmp/\(shape.modelFileName)"))

        #expect(descriptor.inputTileSize == size)
        #expect(descriptor.scaleFactor == 4)
        #expect(descriptor.precision == .float16)
        #expect(descriptor.packageURL.lastPathComponent == shape.modelFileName)
    }

    @Test("every shape gets its own descriptor identifier, and 256 keeps the family name")
    func descriptorIdentifiersAreUniquePerShape() {
        let shapes = Self.shipped.map { HarnessShape(identifier: "RealESRGAN_x4plus", tileSize: $0) }
        let identifiers = shapes.map(\.descriptorIdentifier)

        // Unique because the frozen backend caches compiled models under this name: a shared name
        // is a race in which one shape loads the other's weights.
        #expect(Set(identifiers).count == identifiers.count)
        #expect(HarnessShape.default256.descriptorIdentifier == "RealESRGAN_x4plus")
        #expect(
            HarnessShape(identifier: "RealESRGAN_x4plus", tileSize: 384).descriptorIdentifier
                == "RealESRGAN_x4plus_384x384")
    }

    // MARK: - Discovery in the drop-in folder

    @Test("discovery offers the variants actually present, smallest tile first")
    func discoversDroppedInVariants() throws {
        let directory = try Self.makeArtifactsDirectory([
            "RealESRGAN_x4plus_512x512.mlpackage",
            "RealESRGAN_x4plus_256x256.mlpackage",
            "RealESRGAN_x4plus_384x384.mlpackage",
            "input_tile.png",
            "input_tile_384x384.png",
            "PLACEHOLDER.md",
            "RealESRGAN_x4plus_384x385.mlpackage",  // off-convention: not offered
        ])
        defer { try? FileManager.default.removeItem(at: directory) }

        let shapes = DeviceHarness.shapes(inArtifactsDirectory: directory)

        #expect(shapes.map(\.tileSize) == [256, 384, 512])
        #expect(shapes.allSatisfy { $0.identifier == "RealESRGAN_x4plus" })
    }

    @Test("an empty or absent drop-in folder offers nothing rather than inventing a shape")
    func discoversNothingWhenTheFolderIsEmptyOrAbsent() throws {
        let directory = try Self.makeArtifactsDirectory([])
        defer { try? FileManager.default.removeItem(at: directory) }

        #expect(DeviceHarness.shapes(inArtifactsDirectory: directory).isEmpty)
        #expect(DeviceHarness.shapes(inArtifactsDirectory: directory.appendingPathComponent("nope")).isEmpty)
        #expect(DeviceHarness.shapes(inArtifactsDirectory: nil).isEmpty)
        // The test bundle has no HarnessArtifacts/ — the state of an app built before the copy step.
        #expect(DeviceHarness.availableShapes(in: .main).isEmpty)
    }

    @Test("a shape whose artifacts were never copied in resolves to nil, not a bogus URL")
    func resolutionIsHonestForEveryShape() {
        let shape = HarnessShape(identifier: "RealESRGAN_x4plus", tileSize: 512)

        #expect(DeviceHarness.bundledModelURL(for: shape, in: .main) == nil)
        #expect(DeviceHarness.bundledTileURL(for: shape, in: .main) == nil)
    }

    // MARK: - What the human sees when a shape's artifacts are missing

    @Test("a missing 512 model names the 512 artifact, not the 256 one")
    func missingModelNamesTheShape() async {
        let shape = HarnessShape(identifier: "RealESRGAN_x4plus", tileSize: 512)
        let run = await DeviceHarness.runSingleTile(shape: shape, in: .main, computeUnits: .cpuOnly)

        #expect(run.result.success == false)
        #expect(run.outputPNG == nil)
        #expect(
            run.result.errorDescription
                == "Model not bundled. Copy RealESRGAN_x4plus_512x512.mlpackage into "
                + "DeviceHarnessApp/HarnessArtifacts/ and rebuild.")
    }

    @Test("a missing 384 tile asks for the shape-suffixed tile")
    func missingTileNamesTheShape() async {
        let shape = HarnessShape(identifier: "RealESRGAN_x4plus", tileSize: 384)
        let run = await SingleTileRunner.run(
            shape: shape, modelURL: URL(fileURLWithPath: "/tmp/absent.mlpackage"), tileURL: nil,
            computeUnits: .cpuOnly)

        #expect(run.result.success == false)
        #expect(
            run.result.errorDescription
                == "Test tile not bundled. Copy input_tile_384x384.png into "
                + "DeviceHarnessApp/HarnessArtifacts/ and rebuild.")
    }

    @Test("the 256 messages keep their original wording, character for character")
    func defaultShapeMessagesAreUnchanged() async {
        let run = await DeviceHarness.runSingleTile(modelURL: nil, tileURL: nil, computeUnits: .cpuOnly)
        let tileMissing = await SingleTileRunner.run(
            modelURL: URL(fileURLWithPath: "/tmp/absent.mlpackage"), tileURL: nil, computeUnits: .cpuOnly)

        #expect(
            run.result.errorDescription
                == "Model not bundled. Copy RealESRGAN_x4plus_256x256.mlpackage into "
                + "DeviceHarnessApp/HarnessArtifacts/ and rebuild.")
        #expect(
            tileMissing.result.errorDescription
                == "Test tile not bundled. Copy input_tile.png into "
                + "DeviceHarnessApp/HarnessArtifacts/ and rebuild.")
    }

    // MARK: - Helpers

    /// A stand-in for the drop-in folder: names only, since discovery reads names and the
    /// `.mlpackage`s are directories anyway.
    static func makeArtifactsDirectory(_ entries: [String]) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("HarnessArtifacts-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        for entry in entries where entry.hasSuffix(".mlpackage") {
            try FileManager.default.createDirectory(
                at: directory.appendingPathComponent(entry, isDirectory: true),
                withIntermediateDirectories: true)
        }
        for entry in entries where !entry.hasSuffix(".mlpackage") {
            try Data().write(to: directory.appendingPathComponent(entry))
        }
        return directory
    }
}
