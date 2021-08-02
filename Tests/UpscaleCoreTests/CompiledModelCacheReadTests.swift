import Foundation
import Testing

@testable import UpscaleCore

/// The cache **read** path.
///
/// A correct cache key is worthless if nothing reads it back: `load` would recompile on every
/// process launch and `moveIntoCaches`' own promise would never hold. These tests cover the guards
/// on the read, which is where the risk lives: a miss costs one recompile, but a **wrong hit**
/// loads the wrong weights or a half-written bundle.
///
/// They exist because a mutation deleting the `coremldata.bin` check once survived the entire
/// suite. Code with no coverage is code whose guards are decorative.
///
/// File-backed and model-free: real directories under a temporary root, no `.mlpackage`, no Core
/// ML. Expected paths are written out as literals rather than recomputed from the code under test.
struct CompiledModelCacheReadTests {
    static func descriptor(
        identifier: String = "RealESRGAN_x4plus", tile: Int = 512
    ) -> ModelDescriptor {
        ModelDescriptor(
            identifier: identifier, scaleFactor: 4, contextMargin: 76,
            packageURL: URL(fileURLWithPath: "/models/x.mlpackage"),
            inputTileSize: tile, precision: .float16)
    }

    /// A temporary caches root, optionally containing a compiled-model directory for `name`.
    /// `complete` controls whether `coremldata.bin` is present — the difference between a usable
    /// bundle and the debris a killed compile leaves behind.
    static func makeRoot(containing name: String?, complete: Bool = true) throws -> URL {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("cacheread-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        if let name {
            let dir =
                root
                .appendingPathComponent("com.tbd.upscaler/CompiledModels", isDirectory: true)
                .appendingPathComponent(name, isDirectory: true)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            if complete {
                try Data("stub".utf8).write(to: dir.appendingPathComponent("coremldata.bin"))
            }
        }
        return root
    }

    @Test("a complete cached bundle is a hit, at the geometry-keyed path")
    func completeBundleHits() throws {
        let root = try Self.makeRoot(containing: "RealESRGAN_x4plus_512x512.mlmodelc")
        defer { try? FileManager.default.removeItem(at: root) }

        let hit = CoreMLBackend.cachedModelURL(for: Self.descriptor(), in: root)

        #expect(hit != nil, "a complete bundle must be reused — this is the whole point of the cache")
        #expect(hit?.lastPathComponent == "RealESRGAN_x4plus_512x512.mlmodelc")
    }

    @Test("an empty cache is a miss, not a crash")
    func emptyCacheMisses() throws {
        let root = try Self.makeRoot(containing: nil)
        defer { try? FileManager.default.removeItem(at: root) }

        #expect(CoreMLBackend.cachedModelURL(for: Self.descriptor(), in: root) == nil)
    }

    /// The guard that mutation deleted. A compile killed by jetsam or a crash leaves the directory
    /// but not its payload; trusting it would hand Core ML a broken bundle.
    @Test("a partial bundle — directory present, coremldata.bin absent — is a MISS")
    func partialBundleMisses() throws {
        let root = try Self.makeRoot(
            containing: "RealESRGAN_x4plus_512x512.mlmodelc", complete: false)
        defer { try? FileManager.default.removeItem(at: root) }

        #expect(
            CoreMLBackend.cachedModelURL(for: Self.descriptor(), in: root) == nil,
            "a directory without coremldata.bin is debris from a killed compile, never a hit")
    }

    /// The geometry-bearing key is what makes reading the cache safe at all. With the identifier
    /// alone as the key, a hit could have been another shape's weights — a cross-instantiation
    /// reproduced on the host. This pins that the read inherits that protection.
    @Test("a different shape under the same identifier does not hit")
    func otherShapeDoesNotHit() throws {
        let root = try Self.makeRoot(containing: "RealESRGAN_x4plus_512x512.mlmodelc")
        defer { try? FileManager.default.removeItem(at: root) }

        #expect(CoreMLBackend.cachedModelURL(for: Self.descriptor(tile: 384), in: root) == nil)
        #expect(CoreMLBackend.cachedModelURL(for: Self.descriptor(tile: 256), in: root) == nil)
        #expect(CoreMLBackend.cachedModelURL(for: Self.descriptor(tile: 512), in: root) != nil)
    }

    @Test("a different identifier at the same shape does not hit")
    func otherIdentifierDoesNotHit() throws {
        let root = try Self.makeRoot(containing: "RealESRGAN_x4plus_512x512.mlmodelc")
        defer { try? FileManager.default.removeItem(at: root) }

        #expect(
            CoreMLBackend.cachedModelURL(for: Self.descriptor(identifier: "BSRGAN"), in: root)
                == nil)
    }

    @Test("a caches root that does not exist is a miss, never a throw")
    func absentRootMisses() {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("cacheread-absent-\(UUID().uuidString)", isDirectory: true)

        #expect(CoreMLBackend.cachedModelURL(for: Self.descriptor(), in: root) == nil)
    }

    /// A file where the bundle directory should be. `.mlmodelc` is a directory; a regular file at
    /// that path is not a usable model and must not be handed to Core ML.
    @Test("a regular file at the bundle path is a miss")
    func fileInsteadOfDirectoryMisses() throws {
        let root = try Self.makeRoot(containing: nil)
        defer { try? FileManager.default.removeItem(at: root) }
        let dir = root.appendingPathComponent("com.tbd.upscaler/CompiledModels", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data("not a bundle".utf8)
            .write(to: dir.appendingPathComponent("RealESRGAN_x4plus_512x512.mlmodelc"))

        #expect(CoreMLBackend.cachedModelURL(for: Self.descriptor(), in: root) == nil)
    }
}
