import Foundation
import Testing

@testable import UpscaleCore

/// Regression tests for the compiled-model cache key.
///
/// The defect these guard against: caching at `CompiledModels/\(identifier).mlmodelc`, deleting
/// whatever sat there first. A key with no geometry makes a multi-shape runtime evict the previous
/// shape on every switch and lets it instantiate the wrong shape's weights. The key is identifier
/// **and** input geometry, derived by `compiledModelName(for:)`.
///
/// Model-free and I/O-free: the derivation is a pure function, so these exercise it directly — no
/// `.mlpackage`, no `FileManager`, nothing that touches `Caches/`. Every expected name below is
/// written out as a literal rather than recomputed from the descriptor, so the assertions cannot
/// agree with the code under test by construction.
struct CompiledModelCacheTests {
    static let packageURL = URL(fileURLWithPath: "/models/RealESRGAN_x4plus/output/x.mlpackage")

    static func descriptor(
        identifier: String = "RealESRGAN_x4plus",
        scaleFactor: Int = 4,
        packageURL: URL = CompiledModelCacheTests.packageURL,
        tile: Int = 256,
        precision: Precision = .float16
    ) -> ModelDescriptor {
        ModelDescriptor(
            identifier: identifier, scaleFactor: scaleFactor, contextMargin: 76,
            packageURL: packageURL, inputTileSize: tile, precision: precision
        )
    }

    // MARK: - The defect: one identifier, several shapes

    @Test("two shapes sharing an identifier do not share a cache name")
    func shapesSharingAnIdentifierDoNotCollide() {
        let small = CoreMLBackend.compiledModelName(for: Self.descriptor(tile: 384))
        let large = CoreMLBackend.compiledModelName(for: Self.descriptor(tile: 512))

        #expect(small != large, "identifier-only keying is back: 384 and 512 would evict each other")
        #expect(small == "RealESRGAN_x4plus_384x384.mlmodelc")
        #expect(large == "RealESRGAN_x4plus_512x512.mlmodelc")
    }

    @Test("all three ladder shapes are pairwise distinct under one identifier")
    func ladderShapesArePairwiseDistinct() {
        let names = [256, 384, 512].map { CoreMLBackend.compiledModelName(for: Self.descriptor(tile: $0)) }

        #expect(
            names == [
                "RealESRGAN_x4plus_256x256.mlmodelc",
                "RealESRGAN_x4plus_384x384.mlmodelc",
                "RealESRGAN_x4plus_512x512.mlmodelc",
            ])
        #expect(Set(names).count == 3, "a ladder switching rungs would recompile on every switch")
    }

    // MARK: - The other half: same model must still hit the same cache entry

    @Test("same identifier and same shape produce the same name")
    func sameIdentifierAndShapeShareAName() {
        let first = CoreMLBackend.compiledModelName(for: Self.descriptor(tile: 256))
        let second = CoreMLBackend.compiledModelName(for: Self.descriptor(tile: 256))

        #expect(first == second, "the cache would never hit — every load would recompile")
        #expect(first == "RealESRGAN_x4plus_256x256.mlmodelc")
    }

    @Test("the derived name is stable across repeated calls")
    func derivationIsStable() {
        let subject = Self.descriptor(tile: 512)
        let names = (0..<8).map { _ in CoreMLBackend.compiledModelName(for: subject) }

        #expect(Set(names) == ["RealESRGAN_x4plus_512x512.mlmodelc"])
    }

    // MARK: - What the key is, and what it deliberately is not

    @Test("different identifiers at one shape stay separate")
    func identifiersRemainPartOfTheKey() {
        let first = CoreMLBackend.compiledModelName(for: Self.descriptor(identifier: "RealESRGAN_x4plus"))
        let second = CoreMLBackend.compiledModelName(for: Self.descriptor(identifier: "BSRGAN"))

        #expect(first == "RealESRGAN_x4plus_256x256.mlmodelc")
        #expect(second == "BSRGAN_256x256.mlmodelc")
    }

    @Test("the key is identifier plus geometry — not the artifact's location")
    func keyIsSemanticIdentityNotLocation() {
        let baseline = CoreMLBackend.compiledModelName(for: Self.descriptor())
        let elsewhere = CoreMLBackend.compiledModelName(
            for: Self.descriptor(packageURL: URL(fileURLWithPath: "/elsewhere/copy.mlpackage")))

        // Keying on `packageURL` is rejected: two copies of the same weights at different paths
        // would double-cache. Identifier + geometry is the identity the cache exists for.
        #expect(baseline == "RealESRGAN_x4plus_256x256.mlmodelc")
        #expect(elsewhere == baseline)

        // `scaleFactor` and `precision` are absent from the key too, but deliberately *not*
        // asserted: pinning them would ratify a decision nobody has made, and `Precision.float32`
        // exists in the enum — a float32 artifact sharing an identifier and shape with its float16
        // sibling would collide on one cache directory, the same defect the geometry key removes.
        // Revisit the day a second precision or scale factor ships.
    }

    @Test("the CLI's already-suffixed stems double the shape, deliberately")
    func suffixedIdentifiersDoubleDeliberately() {
        // `Sources/UpscaleCLI/InferenceCommand.swift` uses the artifact stem, which already ends in
        // `_<N>x<N>`. Doubled but unique, stable, and opaque to users — stripping the suffix would
        // reintroduce the ambiguity the key exists to remove, so this name is pinned, not a bug.
        let name = CoreMLBackend.compiledModelName(
            for: Self.descriptor(identifier: "RealESRGAN_x4plus_256x256"))

        #expect(name == "RealESRGAN_x4plus_256x256_256x256.mlmodelc")
    }

    @Test("the name is a compiled-model directory name")
    func nameIsAnMLModelCDirectory() {
        let name = CoreMLBackend.compiledModelName(for: Self.descriptor(tile: 384))

        #expect(name.hasSuffix(".mlmodelc"), "Core ML loads a compiled model from an .mlmodelc directory")
        #expect(!name.contains("/"), "the name is one path component, never a nested path")
    }
}
