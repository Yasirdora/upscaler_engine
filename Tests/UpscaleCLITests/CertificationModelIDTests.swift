import Foundation
import Testing

@testable import UpscaleCLI

/// Tests for the certification oracle-pairing derivation (a trap caught in review: an
/// explicit path into the family's shared runtime directory silently paired the 256 oracle set —
/// a plausible-looking, wrong certification report).
struct CertificationModelIDTests {
    private func id(_ path: String, _ tileSize: Int) throws -> String {
        try ImageCommand.certificationModelID(
            resolvedURL: URL(fileURLWithPath: path), tileSize: tileSize)
    }

    @Test("the bare-id layout resolves to its own id at every shipped shape")
    func perIdLayout() throws {
        #expect(
            try id("models/RealESRGAN_x4plus/output/RealESRGAN_x4plus_256x256.mlpackage", 256)
                == "RealESRGAN_x4plus")
        #expect(
            try id("models/RealESRGAN_x4plus_384/output/RealESRGAN_x4plus_384_384x384.mlpackage", 384)
                == "RealESRGAN_x4plus_384")
        #expect(
            try id("models/RealESRGAN_x4plus_512/output/RealESRGAN_x4plus_512_512x512.mlpackage", 512)
                == "RealESRGAN_x4plus_512")
    }

    @Test("the shared runtime directory resolves to the shape's oracle set, not the family's")
    func sharedRuntimeDirectoryTrap() throws {
        // The case a margin sweep caught: same file name, one directory level up.
        #expect(
            try id("models/RealESRGAN_x4plus/output/RealESRGAN_x4plus_384x384.mlpackage", 384)
                == "RealESRGAN_x4plus_384")
        #expect(
            try id("models/RealESRGAN_x4plus/output/RealESRGAN_x4plus_512x512.mlpackage", 512)
                == "RealESRGAN_x4plus_512")
    }

    @Test("a package whose geometry contradicts --tile-size is refused, not mispaired")
    func shapeMismatch() {
        #expect(throws: ImageCommand.UsageError.self) {
            try id("models/RealESRGAN_x4plus/output/RealESRGAN_x4plus_512x512.mlpackage", 384)
        }
    }

    @Test("a name without square geometry is refused, not guessed")
    func unparseableGeometry() {
        #expect(throws: ImageCommand.UsageError.self) {
            try id("/tmp/some-model.mlpackage", 256)
        }
        #expect(throws: ImageCommand.UsageError.self) {
            try id("models/RealESRGAN_x4plus/output/RealESRGAN_x4plus_384x512.mlpackage", 384)
        }
    }
}
