import CoreGraphics
import CoreVideo
import Testing

@testable import UpscaleCore

/// The preview entry point.
///
/// **The criterion is pixel equality, so that is what these assert.** A plan can be correct in
/// every structural respect and still produce a wrong picture through a bad write offset; the plan
/// tests in `CroppedTilePlanTests` would all pass. So the central test here upscales an image
/// whole, upscales a region of it, and requires the two to agree byte for byte.
///
/// `NearestNeighbourBackend` stands in for Core ML: deterministic, model-free, and it exercises the
/// real tiling and stitching path, which is where such a bug would live.
struct PreviewPathTests {
    static let scale = 4
    static let tile = 64
    static let margin = 8

    static func descriptor() -> ModelDescriptor {
        ModelDescriptor(
            identifier: "stub", scaleFactor: scale, contextMargin: margin,
            packageURL: URL(fileURLWithPath: "/stub.mlpackage"),
            inputTileSize: tile, precision: .float16)
    }

    /// Structured noise: every pixel distinguishable, so a misplaced copy cannot coincidentally
    /// match. A flat fill would let a wrong offset pass.
    static func image(width: Int, height: Int) throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        let attributes: [CFString: Any] = [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary]
        guard
            CVPixelBufferCreate(
                kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA,
                attributes as CFDictionary, &buffer) == kCVReturnSuccess,
            let buffer
        else { throw TilingError.allocationFailed("test image") }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        let base = try #require(CVPixelBufferGetBaseAddress(buffer))
            .assumingMemoryBound(to: UInt8.self)
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        for y in 0..<height {
            for x in 0..<width {
                let i = y * stride + x * 4
                base[i] = UInt8((x &* 7 &+ y &* 3) & 0xFF)
                base[i + 1] = UInt8((x &* 13 &+ y &* 29) & 0xFF)
                base[i + 2] = UInt8((x &+ y &* 5) & 0xFF)
                base[i + 3] = 255
            }
        }
        return buffer
    }

    static func pixel(_ buffer: CVPixelBuffer, _ x: Int, _ y: Int) -> [UInt8] {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return [] }
        let p = base.assumingMemoryBound(to: UInt8.self)
        let i = y * CVPixelBufferGetBytesPerRow(buffer) + x * 4
        return [p[i], p[i + 1], p[i + 2], p[i + 3]]
    }

    // MARK: - The acceptance criterion

    /// The whole point of the preview path. A preview that disagrees with the final render is worse
    /// than none: the user commits to a render on the strength of it.
    @Test(
        "a preview region is bit-identical to the same region of a whole-image upscale",
        arguments: [
            CGRect(x: 64, y: 64, width: 96, height: 96),  // interior, straddles tiles
            CGRect(x: 0, y: 0, width: 80, height: 80),  // touches the image's own border
            CGRect(x: 40, y: 130, width: 70, height: 50),  // non-square, unaligned to the grid
        ])
    func previewMatchesWholeImage(rect: CGRect) async throws {
        let engine = SequentialTilingEngine(contextMargin: Self.margin)
        let source = try Self.image(width: 256, height: 192)
        let model = Self.descriptor()

        let whole = try await engine.upscale(
            source, model: model, backend: NearestNeighbourBackend(scaleFactor: Self.scale),
            method: .centerValidRegion, progress: nil)
        let preview = try await engine.upscale(
            source, in: rect, model: model,
            backend: NearestNeighbourBackend(scaleFactor: Self.scale),
            method: .centerValidRegion, progress: nil)

        #expect(CVPixelBufferGetWidth(preview) == Int(rect.width) * Self.scale)
        #expect(CVPixelBufferGetHeight(preview) == Int(rect.height) * Self.scale)

        let originX = Int(rect.origin.x) * Self.scale
        let originY = Int(rect.origin.y) * Self.scale
        var mismatches = 0
        var firstMismatch: String?
        for y in 0..<CVPixelBufferGetHeight(preview) {
            for x in 0..<CVPixelBufferGetWidth(preview) {
                let a = Self.pixel(preview, x, y)
                let b = Self.pixel(whole, originX + x, originY + y)
                if a != b {
                    mismatches += 1
                    if firstMismatch == nil { firstMismatch = "(\(x),\(y)) \(a) vs \(b)" }
                }
            }
        }
        #expect(mismatches == 0, "\(mismatches) differing pixels; first \(firstMismatch ?? "-")")
    }

    // MARK: - Geometry and refusals

    @Test("a region larger than the image is clamped, not refused")
    func oversizedRegionIsClamped() async throws {
        let engine = SequentialTilingEngine(contextMargin: Self.margin)
        let source = try Self.image(width: 128, height: 128)

        let preview = try await engine.upscale(
            source, in: CGRect(x: 64, y: 64, width: 999, height: 999), model: Self.descriptor(),
            backend: NearestNeighbourBackend(scaleFactor: Self.scale),
            method: .centerValidRegion, progress: nil)

        #expect(CVPixelBufferGetWidth(preview) == 64 * Self.scale)
    }

    @Test("a region entirely outside the image is refused rather than returning an empty buffer")
    func outsideRegionIsRefused() async throws {
        let engine = SequentialTilingEngine(contextMargin: Self.margin)
        let source = try Self.image(width: 128, height: 128)

        await #expect(throws: TilingError.emptyImage) {
            _ = try await engine.upscale(
                source, in: CGRect(x: 500, y: 500, width: 32, height: 32),
                model: Self.descriptor(),
                backend: NearestNeighbourBackend(scaleFactor: Self.scale),
                method: .centerValidRegion, progress: nil)
        }
    }

    /// A preview must cost a fraction of a whole render, or it is not a preview.
    @Test("a small region runs far fewer tiles than the whole image")
    func previewRunsFewerTiles() async throws {
        let engine = SequentialTilingEngine(contextMargin: Self.margin)
        let source = try Self.image(width: 256, height: 192)
        let wholeBackend = NearestNeighbourBackend(scaleFactor: Self.scale)
        let previewBackend = NearestNeighbourBackend(scaleFactor: Self.scale)

        _ = try await engine.upscale(
            source, model: Self.descriptor(), backend: wholeBackend,
            method: .centerValidRegion, progress: nil)
        _ = try await engine.upscale(
            source, in: CGRect(x: 96, y: 64, width: 48, height: 48), model: Self.descriptor(),
            backend: previewBackend, method: .centerValidRegion, progress: nil)

        #expect(previewBackend.predictionCount < wholeBackend.predictionCount)
        #expect(previewBackend.predictionCount > 0)
    }
}
