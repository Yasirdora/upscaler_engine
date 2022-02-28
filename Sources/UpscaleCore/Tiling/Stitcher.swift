import Accelerate
import CoreGraphics
import CoreVideo
import Foundation

/// Writes one tile's model output into the assembled image.
///
/// Crop and blend are *fused into this write*: a stitcher points vImage at a sub-rectangle of the
/// tile and a sub-rectangle of the output and moves the pixels once. No cropped intermediate is
/// materialised, and there is no second full-image pass.
public protocol Stitching: Sendable {
    var method: StitchMethod { get }

    func write(
        tile: CVPixelBuffer,
        region: TileRegion,
        grid: TileGrid,
        scaleFactor: Int,
        into output: CVPixelBuffer
    ) throws
}

/// A stitcher that can write to any `DestinationSurface`, not only a whole-image buffer.
///
/// This refines `Stitching` rather than replacing it, deliberately: the public `write(…into:
/// CVPixelBuffer)` requirement is satisfied below by a forwarding default, so the public API is
/// unchanged and every existing caller keeps compiling untouched. Both paths run the *same*
/// stitcher code, which is what makes the spooled path's pixel-equivalence claim checkable rather
/// than assumed.
protocol SurfaceStitching: Stitching {
    func write(
        tile: CVPixelBuffer,
        region: TileRegion,
        grid: TileGrid,
        scaleFactor: Int,
        into surface: DestinationSurface
    ) throws
}

extension SurfaceStitching {
    /// The whole-image entry point, satisfied once for all stitchers by wrapping the buffer.
    func write(
        tile: CVPixelBuffer, region: TileRegion, grid: TileGrid, scaleFactor: Int,
        into output: CVPixelBuffer
    ) throws {
        try write(
            tile: tile, region: region, grid: grid, scaleFactor: scaleFactor,
            into: PixelBufferSurface(output) as DestinationSurface)
    }
}

public enum Stitcher {
    /// The stitcher for a method. `centerValidRegion` is the contract default; which method
    /// actually ships is decided by the seam certification, not here.
    public static func make(_ method: StitchMethod) -> Stitching {
        makeSurfaceStitcher(method)
    }

    /// The same stitcher, typed so the engine can hand it a band-spooled destination.
    static func makeSurfaceStitcher(_ method: StitchMethod) -> SurfaceStitching {
        switch method {
        case .hardCrop: return HardCropStitcher()
        case .centerValidRegion: return CenterValidRegionStitcher()
        case .linearFeather: return FeatherStitcher(method: .linearFeather)
        case .cosineFeather: return FeatherStitcher(method: .cosineFeather)
        }
    }
}

// MARK: - Buffer views

/// A non-owning view of an 8-bit BGRA image, and the sub-rect arithmetic the stitchers need.
///
/// Everything here is pointer arithmetic on row strides; no pixel is ever touched by Swift.
struct PixelView {
    let base: UnsafeMutableRawPointer
    let width: Int
    let height: Int
    let rowBytes: Int
    /// Absolute image row that `base`'s first row represents.
    ///
    /// Zero for a whole-image buffer. A band-spooled destination (`DestinationSurface`) holds only
    /// a sliding window of rows, and sets this to the window's top so that `sub` can keep taking
    /// **absolute** output coordinates. That is what lets the four stitchers stay byte-for-byte
    /// unchanged across destinations — they are the numerically certified part of this path (Level
    /// A/B evidence, seam metrics), so not editing them is a correctness argument, not a
    /// convenience.
    let originY: Int

    static let bytesPerPixel = 4

    init?(_ buffer: CVPixelBuffer) {
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }
        self.base = base
        self.width = CVPixelBufferGetWidth(buffer)
        self.height = CVPixelBufferGetHeight(buffer)
        self.rowBytes = CVPixelBufferGetBytesPerRow(buffer)
        self.originY = 0
    }

    init(base: UnsafeMutableRawPointer, width: Int, height: Int, rowBytes: Int, originY: Int = 0) {
        self.base = base
        self.width = width
        self.height = height
        self.rowBytes = rowBytes
        self.originY = originY
    }

    /// A view of a sub-rectangle, in **absolute** image coordinates. The returned view shares
    /// storage — this is the "no copy" in zero-copy: cropping is an origin and a stride, never a
    /// memcpy.
    ///
    /// `y` is resolved against `originY`, so a windowed destination and a whole-image buffer are
    /// addressed identically by callers. Rows outside the window are a programmer error, not a
    /// runtime condition: the engine closes rows only once no future tile can write them.
    ///
    /// **The two axes follow different conventions, deliberately.** `y` is absolute — it is
    /// resolved against `originY`, and the returned view carries `originY: y`, so `sub` composes: a
    /// sub-view of a sub-view still takes absolute rows. `x` has no such origin and is therefore
    /// **relative to this view's left edge**, so nesting `sub` would offset x twice. Nothing nests
    /// `sub` today — every call site in `Tiling/` and `Inference/` takes one from a top-level view
    /// — which is why `originX` is not carried: it would be an unused axis. Adding a second level
    /// of `sub` means adding `originX` first, or the x offset is silently wrong.
    func sub(x: Int, y: Int, width: Int, height: Int) -> PixelView {
        PixelView(
            base: base.advanced(by: (y - originY) * rowBytes + x * Self.bytesPerPixel),
            width: width, height: height, rowBytes: rowBytes, originY: y
        )
    }

    var vImage: vImage_Buffer {
        vImage_Buffer(
            data: base, height: vImagePixelCount(height), width: vImagePixelCount(width),
            rowBytes: rowBytes
        )
    }
}

/// Integer rectangle arithmetic, so geometry decisions are never made in floating point.
struct IntRect: Equatable {
    var x: Int
    var y: Int
    var width: Int
    var height: Int

    init(x: Int, y: Int, width: Int, height: Int) {
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    }

    init(_ rect: CGRect) {
        self.init(
            x: Int(rect.minX.rounded()), y: Int(rect.minY.rounded()),
            width: Int(rect.width.rounded()), height: Int(rect.height.rounded())
        )
    }

    var maxX: Int { x + width }
    var maxY: Int { y + height }
    var isEmpty: Bool { width <= 0 || height <= 0 }

    func intersection(_ other: IntRect) -> IntRect {
        let minX = max(x, other.x)
        let minY = max(y, other.y)
        return IntRect(
            x: minX, y: minY,
            width: max(0, min(maxX, other.maxX) - minX),
            height: max(0, min(maxY, other.maxY) - minY)
        )
    }

    func inset(dx: Int, dy: Int) -> IntRect {
        IntRect(x: x + dx, y: y + dy, width: width - 2 * dx, height: height - 2 * dy)
    }
}

// MARK: - Shared geometry

extension Stitching {
    /// Output-space rect the tile's model output occupies, and the output image's own bounds.
    func geometry(
        region: TileRegion, grid: TileGrid, scaleFactor: Int, output: PixelView
    ) -> (tileRect: IntRect, bounds: IntRect) {
        let padded = IntRect(region.paddedInputRect)
        return (
            IntRect(
                x: padded.x * scaleFactor, y: padded.y * scaleFactor,
                width: grid.tileSize * scaleFactor, height: grid.tileSize * scaleFactor
            ),
            IntRect(x: 0, y: 0, width: output.width, height: output.height)
        )
    }

    /// One vImage copy from a sub-rect of the tile straight into a sub-rect of the output.
    func copy(
        destination: IntRect, tileRect: IntRect, tile: PixelView, output: PixelView
    ) throws {
        guard !destination.isEmpty else { return }
        var source = tile.sub(
            x: destination.x - tileRect.x, y: destination.y - tileRect.y,
            width: destination.width, height: destination.height
        ).vImage
        var target = output.sub(
            x: destination.x, y: destination.y, width: destination.width, height: destination.height
        ).vImage
        let error = vImageCopyBuffer(&source, &target, PixelView.bytesPerPixel, vImage_Flags(kvImageNoFlags))
        guard error == kvImageNoError else {
            throw TilingError.stitchFailed("vImageCopyBuffer failed with \(error)")
        }
    }

    /// Lock the tile, open the destination's window, and hand both to `body` as views.
    ///
    /// The destination's locking is its own business now: `PixelBufferSurface` locks and unlocks
    /// the `CVPixelBuffer` around the window exactly as this function used to, while
    /// `BandSpoolSurface` owns plain memory and has nothing to lock. Every stitcher body below is
    /// unchanged by that — they still call `locked(tile, output)` and receive two `PixelView`s in
    /// absolute coordinates.
    func locked<T>(
        _ tile: CVPixelBuffer, _ output: DestinationSurface,
        _ body: (PixelView, PixelView) throws -> T
    )
        throws -> T
    {
        CVPixelBufferLockBaseAddress(tile, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(tile, .readOnly) }
        guard let tileView = PixelView(tile) else {
            throw TilingError.stitchFailed("could not lock the tile buffer")
        }
        return try output.withOpenWindow { outputView in try body(tileView, outputView) }
    }
}

// MARK: - Non-blending stitchers

/// Keeps every tile's full output; later tiles overwrite earlier ones.
///
/// The naive baseline, present so the bake-off has a control: it writes the contaminated context
/// border straight into the image, which is exactly what the seam metric should catch.
struct HardCropStitcher: SurfaceStitching {
    let method = StitchMethod.hardCrop

    func write(
        tile: CVPixelBuffer, region: TileRegion, grid: TileGrid, scaleFactor: Int,
        into output: DestinationSurface
    ) throws {
        try locked(tile, output) { tileView, outputView in
            let (tileRect, bounds) = geometry(
                region: region, grid: grid, scaleFactor: scaleFactor, output: outputView)
            try copy(
                destination: tileRect.intersection(bounds), tileRect: tileRect,
                tile: tileView, output: outputView)
        }
    }
}

/// Keeps only the core the tile owns, discarding the context border on every side.
///
/// The contract default, and the one certification spec §5.2 expects to win: no blending means no
/// softening.
struct CenterValidRegionStitcher: SurfaceStitching {
    let method = StitchMethod.centerValidRegion

    func write(
        tile: CVPixelBuffer, region: TileRegion, grid: TileGrid, scaleFactor: Int,
        into output: DestinationSurface
    ) throws {
        try locked(tile, output) { tileView, outputView in
            let (tileRect, bounds) = geometry(
                region: region, grid: grid, scaleFactor: scaleFactor, output: outputView)
            try copy(
                destination: IntRect(region.validRect).intersection(bounds), tileRect: tileRect,
                tile: tileView, output: outputView)
        }
    }
}

// MARK: - Feather stitchers

/// Blends the overlap between a tile and the tiles already written above and to its left.
///
/// Raster order makes the blend a plain "over" composite: a tile is fully opaque except along the
/// edges it shares with already-written neighbours, where its alpha ramps 0 → 1 across a band
/// centred on the shared valid boundary. The neighbour underneath was itself written at full
/// opacity there, so the result is exactly `w·new + (1-w)·old` — no accumulation error, and no
/// normalisation pass.
///
/// The weight is separable, so it is built as the outer product of two one-dimensional ramps
/// (vDSP), converted to an 8-bit alpha plane, written into the tile's alpha channel, premultiplied
/// and composited — all Accelerate, no per-pixel Swift.
final class FeatherStitcher: SurfaceStitching, @unchecked Sendable {
    let method: StitchMethod

    private struct RampKey: Hashable {
        let width: Int
        let height: Int
        let leftBand: Int
        let topBand: Int
        let cosine: Bool
    }

    private let lock = NSLock()
    private var cache: [RampKey: [UInt8]] = [:]

    init(method: StitchMethod) {
        self.method = method
    }

    func write(
        tile: CVPixelBuffer, region: TileRegion, grid: TileGrid, scaleFactor: Int,
        into output: DestinationSurface
    ) throws {
        try locked(tile, output) { tileView, outputView in
            let (tileRect, bounds) = geometry(
                region: region, grid: grid, scaleFactor: scaleFactor, output: outputView)
            let valid = IntRect(region.validRect)
            let feather = grid.contextMargin * scaleFactor

            // The tile may contribute anywhere its data is trustworthy: its own core plus the
            // context margin around it, clipped to the tile's own output and to the image.
            let footprint = valid.inset(dx: -feather, dy: -feather)
                .intersection(tileRect)
                .intersection(bounds)
            guard !footprint.isEmpty else { return }

            // Ramp only against neighbours that have already been written (row-major order).
            let leftBand = valid.x > 0 ? min(2 * feather, footprint.width) : 0
            let topBand = valid.y > 0 ? min(2 * feather, footprint.height) : 0

            if leftBand == 0 && topBand == 0 {
                try copy(destination: footprint, tileRect: tileRect, tile: tileView, output: outputView)
                return
            }

            var alpha = try alphaPlane(
                width: footprint.width, height: footprint.height,
                leftBand: leftBand, topBand: topBand, cosine: method == .cosineFeather)

            let tileSub = tileView.sub(
                x: footprint.x - tileRect.x, y: footprint.y - tileRect.y,
                width: footprint.width, height: footprint.height)
            let outputSub = outputView.sub(
                x: footprint.x, y: footprint.y, width: footprint.width, height: footprint.height)

            try alpha.withUnsafeMutableBufferPointer { plane in
                guard let planeBase = plane.baseAddress else {
                    throw TilingError.stitchFailed("empty alpha plane")
                }
                var planar = vImage_Buffer(
                    data: planeBase, height: vImagePixelCount(footprint.height),
                    width: vImagePixelCount(footprint.width), rowBytes: footprint.width)
                var top = tileSub.vImage
                var bottom = outputSub.vImage

                // BGRA in memory: alpha is the last byte, so the ARGB-named mask bit is 0x1.
                var error = vImageOverwriteChannels_ARGB8888(
                    &planar, &top, &top, 0x1, vImage_Flags(kvImageNoFlags))
                guard error == kvImageNoError else {
                    throw TilingError.stitchFailed("vImageOverwriteChannels failed with \(error)")
                }
                // `vImagePremultiplyData_BGRA8888` is a function-like macro aliasing this symbol,
                // and Swift does not import those. Alpha.h states the RGBA-named entry point does
                // not have to be RGBA: it multiplies the first three bytes by the fourth, which is
                // exactly BGRA's layout.
                error = vImagePremultiplyData_RGBA8888(&top, &top, vImage_Flags(kvImageNoFlags))
                guard error == kvImageNoError else {
                    throw TilingError.stitchFailed("vImagePremultiplyData failed with \(error)")
                }
                error = vImagePremultipliedAlphaBlend_BGRA8888(
                    &top, &bottom, &bottom, vImage_Flags(kvImageNoFlags))
                guard error == kvImageNoError else {
                    throw TilingError.stitchFailed("vImagePremultipliedAlphaBlend failed with \(error)")
                }
            }
        }
    }

    /// The separable weight map as an 8-bit alpha plane. Cached: the grid produces only a handful
    /// of distinct footprint/band combinations per image.
    func alphaPlane(
        width: Int, height: Int, leftBand: Int, topBand: Int, cosine: Bool
    ) throws
        -> [UInt8]
    {
        let key = RampKey(
            width: width, height: height, leftBand: leftBand, topBand: topBand, cosine: cosine)
        if let cached = lock.withLock({ cache[key] }) { return cached }

        let columns = Self.ramp(length: width, band: leftBand, cosine: cosine)
        let rows = Self.ramp(length: height, band: topBand, cosine: cosine)

        var plane = [Float](repeating: 0, count: width * height)
        // Outer product: rows (height x 1) * columns (1 x width) -> height x width.
        vDSP_mmul(rows, 1, columns, 1, &plane, 1, vDSP_Length(height), vDSP_Length(width), 1)

        var scale: Float = 255.0
        vDSP_vsmul(plane, 1, &scale, &plane, 1, vDSP_Length(plane.count))
        var bytes = [UInt8](repeating: 0, count: plane.count)
        vDSP_vfixru8(plane, 1, &bytes, 1, vDSP_Length(plane.count))

        lock.withLock { cache[key] = bytes }
        return bytes
    }

    /// A one-dimensional weight: zero at the outer edge, one after `band` samples, one thereafter.
    static func ramp(length: Int, band: Int, cosine: Bool) -> [Float] {
        var weights = [Float](repeating: 1.0, count: max(length, 1))
        guard band > 1, length > 0 else { return weights }
        let count = min(band, length)

        var ramp = [Float](repeating: 0, count: count)
        var start: Float = 0
        var step = Float(1.0) / Float(count - 1)
        vDSP_vramp(&start, &step, &ramp, 1, vDSP_Length(count))

        if cosine {
            // Raised cosine: 0.5 - 0.5·cos(πt) — flat at both ends, so the blend has no visible
            // first derivative discontinuity where it meets the untouched core.
            var cosines = [Float](repeating: 0, count: count)
            var elements = Int32(count)
            vvcospif(&cosines, ramp, &elements)
            var negativeHalf: Float = -0.5
            var half: Float = 0.5
            vDSP_vsmsa(cosines, 1, &negativeHalf, &half, &ramp, 1, vDSP_Length(count))
        }
        weights.replaceSubrange(0..<count, with: ramp)
        return weights
    }
}
