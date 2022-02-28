import Accelerate
import CoreGraphics
import CoreVideo
import Foundation
import Testing

@testable import UpscaleCore

/// Tiling engine tests.
///
/// The planner half is pure geometry and runs without a model. The stitcher half runs on synthetic
/// buffers whose expected values are computed by hand in the test, never by the code under test.
struct TilePlannerTests {
    static let scale = 4
    static let planner = TilePlanner(scaleFactor: scale)

    /// Every edge of every valid region must have either `margin` pixels of context inside the
    /// window the model sees, or a window edge sitting exactly on the image boundary. The second
    /// case is not a weaker one: there, the tile sees precisely the boundary condition the
    /// full-frame oracle sees, because no context exists beyond an image edge for anyone. This is
    /// the whole correctness argument for tiling — if it holds, a tile's core cannot differ from
    /// the untiled result by more than the model's own numerical noise.
    static func assertContextInvariant(_ grid: TileGrid, imageSize: CGSize) {
        let width = Int(imageSize.width)
        let height = Int(imageSize.height)
        for tile in grid.tiles {
            let window = tile.paddedInputRect
            let valid = CGRect(
                x: tile.validRect.minX / CGFloat(scale), y: tile.validRect.minY / CGFloat(scale),
                width: tile.validRect.width / CGFloat(scale), height: tile.validRect.height / CGFloat(scale)
            )
            let margin = CGFloat(grid.contextMargin)
            #expect(
                valid.minX - window.minX >= margin || window.minX <= 0,
                "left edge of \(valid) has neither context nor the image border in \(window)")
            #expect(
                window.maxX - valid.maxX >= margin || window.maxX >= CGFloat(width),
                "right edge of \(valid) has neither context nor the image border in \(window)")
            #expect(
                valid.minY - window.minY >= margin || window.minY <= 0,
                "top edge of \(valid) has neither context nor the image border in \(window)")
            #expect(
                window.maxY - valid.maxY >= margin || window.maxY >= CGFloat(height),
                "bottom edge of \(valid) has neither context nor the image border in \(window)")
        }
    }

    /// The valid regions must tile the output exactly: no gap (a hole in the image) and no overlap
    /// (two tiles claiming ownership of one pixel).
    static func assertExactCover(_ grid: TileGrid, imageSize: CGSize) {
        let outputWidth = Int(imageSize.width) * scale
        let outputHeight = Int(imageSize.height) * scale
        var covered = [Int](repeating: 0, count: outputWidth * outputHeight)
        for tile in grid.tiles {
            for y in Int(tile.validRect.minY)..<Int(tile.validRect.maxY) {
                for x in Int(tile.validRect.minX)..<Int(tile.validRect.maxX) {
                    covered[y * outputWidth + x] += 1
                }
            }
        }
        #expect(covered.allSatisfy { $0 == 1 }, "valid regions must partition the output exactly")
    }

    // MARK: - Determinism and ordering

    @Test("the same request always produces the same grid")
    func determinism() {
        let first = Self.planner.plan(
            imageSize: CGSize(width: 517, height: 512), tileSize: 256, contextMargin: 32)
        let second = Self.planner.plan(
            imageSize: CGSize(width: 517, height: 512), tileSize: 256, contextMargin: 32)
        #expect(first == second)
    }

    @Test("tiles are emitted row-major")
    func rowMajorOrdering() {
        let grid = Self.planner.plan(
            imageSize: CGSize(width: 600, height: 400), tileSize: 256, contextMargin: 32)
        let origins = grid.tiles.map { ($0.validRect.minY, $0.validRect.minX) }
        let sorted = origins.sorted { $0 < $1 }
        #expect(origins.map(\.0) == sorted.map(\.0) && origins.map(\.1) == sorted.map(\.1))
    }

    // MARK: - Geometry

    @Test("an exact multiple of the core tiles without a remainder")
    func exactMultiple() {
        // core = 256 - 2*32 = 192; 576 = 3 * 192
        let size = CGSize(width: 576, height: 384)
        let grid = Self.planner.plan(imageSize: size, tileSize: 256, contextMargin: 32)

        #expect(grid.validCoreSize == 192)
        #expect(grid.tiles.count == 3 * 2)
        Self.assertExactCover(grid, imageSize: size)
        Self.assertContextInvariant(grid, imageSize: size)
    }

    @Test(
        "one pixel short of, equal to, and one pixel over a core multiple all tile correctly",
        arguments: [575, 576, 577]
    )
    func offByOneDimensions(width: Int) {
        let size = CGSize(width: width, height: 300)
        let grid = Self.planner.plan(imageSize: size, tileSize: 256, contextMargin: 32)

        #expect(grid.tiles.allSatisfy { $0.paddedInputRect.width == 256 })
        Self.assertExactCover(grid, imageSize: size)
        Self.assertContextInvariant(grid, imageSize: size)
    }

    @Test("an image smaller than one tile is a single reflection-padded tile")
    func subTileImage() {
        let size = CGSize(width: 100, height: 80)
        let grid = Self.planner.plan(imageSize: size, tileSize: 256, contextMargin: 32)

        #expect(grid.tiles.count == 1)
        let tile = grid.tiles[0]
        #expect(tile.validRect == CGRect(x: 0, y: 0, width: 400, height: 320))
        // The window is centred and hangs outside the image on both sides; reflection fills it.
        #expect(tile.paddedInputRect == CGRect(x: -78, y: -88, width: 256, height: 256))
        Self.assertExactCover(grid, imageSize: size)
        Self.assertContextInvariant(grid, imageSize: size)
    }

    @Test("an image exactly one tile wide needs no padding at all")
    func exactlyOneTile() {
        let size = CGSize(width: 256, height: 256)
        let grid = Self.planner.plan(imageSize: size, tileSize: 256, contextMargin: 32)

        #expect(grid.tiles.count == 4, "a 256 image with a 192 core still needs a 2x2 grid")
        #expect(grid.tiles.allSatisfy { $0.paddedInputRect.minX >= 0 && $0.paddedInputRect.minY >= 0 })
        #expect(grid.tiles.allSatisfy { $0.paddedInputRect.maxX <= 256 && $0.paddedInputRect.maxY <= 256 })
        Self.assertExactCover(grid, imageSize: size)
    }

    @Test(
        "extreme aspect ratios tile along the long axis only",
        arguments: [CGSize(width: 20, height: 400), CGSize(width: 400, height: 20)])
    func extremeAspectRatio(size: CGSize) {
        let grid = Self.planner.plan(imageSize: size, tileSize: 256, contextMargin: 32)

        let longSide = Int(max(size.width, size.height))
        #expect(grid.tiles.count == (longSide + 191) / 192)
        #expect(grid.tiles.allSatisfy { $0.paddedInputRect.width == 256 })
        Self.assertExactCover(grid, imageSize: size)
        Self.assertContextInvariant(grid, imageSize: size)
    }

    @Test("a zero-margin plan is a plain non-overlapping grid")
    func zeroMargin() {
        let size = CGSize(width: 512, height: 512)
        let grid = Self.planner.plan(imageSize: size, tileSize: 256, contextMargin: 0)

        #expect(grid.tiles.count == 4)
        #expect(grid.validCoreSize == 256)
        Self.assertExactCover(grid, imageSize: size)
    }

    // MARK: - Refusals and clamping

    @Test("a margin that would leave no core is clamped to the largest that still tiles")
    func marginClamping() {
        let grid = Self.planner.plan(
            imageSize: CGSize(width: 400, height: 400), tileSize: 256, contextMargin: 200)

        #expect(grid.contextMargin == 127, "(256 - 1) / 2 — the largest margin leaving a core")
        #expect(grid.validCoreSize == 2)
        #expect(!grid.tiles.isEmpty)
    }

    @Test(
        "a degenerate image produces no tiles rather than a trap",
        arguments: [
            CGSize(width: 0, height: 100), CGSize(width: 100, height: 0), CGSize(width: 0, height: 0),
        ])
    func degenerateImage(size: CGSize) {
        #expect(Self.planner.plan(imageSize: size, tileSize: 256, contextMargin: 32).tiles.isEmpty)
    }

    // MARK: - The cost of overlap

    @Test("the planner states the redundancy overlap costs")
    func redundancyIsReported() {
        let size = CGSize(width: 1920, height: 1080)
        let grid = Self.planner.plan(imageSize: size, tileSize: 256, contextMargin: 32)
        let redundancy = Self.planner.redundancy(for: grid, imageSize: size)

        // 10 x 6 tiles of 256^2 input against 1920x1080 of input: hand-computed, not code-derived.
        #expect(grid.tiles.count == 60)
        #expect(abs(redundancy - (60.0 * 256.0 * 256.0) / (1920.0 * 1080.0)) < 1e-9)
        #expect(redundancy > 1.0, "overlap is never free")
    }
}

// MARK: - Test doubles

/// A model with **no receptive field**: every output pixel depends on exactly one input pixel.
///
/// That property is what makes it a precise oracle for tiling. Context padding, valid-region
/// cropping and stitching are all supposed to be lossless rearrangements of the model's work, so
/// with a context-free model the tiled result must equal the untiled result *exactly* — for every
/// stitch method, including the feathers, because blending two identical values is a no-op. Any
/// off-by-one in the grid, the padding or the stitch shows up as a mismatched pixel.
///
/// Nearest-neighbour scaling is done by CoreGraphics with interpolation disabled, so this file
/// contains no per-pixel loop either.
final class NearestNeighbourBackend: InferenceBackend, @unchecked Sendable {
    let scaleFactor: Int
    private let lock = NSLock()
    private var calls = 0
    /// Fails the prediction at this index, to exercise the engine's error path.
    var failAt: Int?
    /// Cancels the surrounding task from inside this prediction, so the engine's cancellation
    /// check is exercised mid-run with no timing race.
    var cancelSelfAt: Int?

    init(scaleFactor: Int) { self.scaleFactor = scaleFactor }

    var predictionCount: Int { lock.withLock { calls } }

    func load(_ model: ModelDescriptor, context: InferenceContext) async throws {}

    func predict(_ tile: CVPixelBuffer) async throws -> CVPixelBuffer {
        let index = lock.withLock { () -> Int in
            defer { calls += 1 }
            return calls
        }
        if let failAt, index == failAt {
            throw InferenceError.predictionFailed("synthetic failure at tile \(index)")
        }
        if let cancelSelfAt, index == cancelSelfAt {
            withUnsafeCurrentTask { $0?.cancel() }
        }
        return try TilingTestSupport.nearestNeighbour(tile, scaleFactor: scaleFactor)
    }
}

enum TilingTestSupport {
    static func makeBuffer(width: Int, height: Int) throws -> CVPixelBuffer {
        try SequentialTilingEngine.allocate(width: width, height: height)
    }

    /// A deterministic, non-uniform test image: a diagonal gradient rendered by CoreGraphics.
    static func gradientImage(width: Int, height: Int) throws -> CVPixelBuffer {
        let buffer = try makeBuffer(width: width, height: height)
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard
            let context = CGContext(
                data: CVPixelBufferGetBaseAddress(buffer), width: width, height: height,
                bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
                    | CGBitmapInfo.byteOrder32Little.rawValue),
            let gradient = CGGradient(
                colorsSpace: CGColorSpaceCreateDeviceRGB(),
                colors: [
                    CGColor(red: 0.05, green: 0.4, blue: 0.9, alpha: 1),
                    CGColor(red: 0.95, green: 0.7, blue: 0.1, alpha: 1),
                ] as CFArray, locations: [0, 1])
        else { throw TilingError.allocationFailed("could not build the gradient fixture") }
        context.drawLinearGradient(
            gradient, start: .zero, end: CGPoint(x: width, y: height), options: [])
        return buffer
    }

    /// A fixture whose adjacent rows and columns always differ, so a test can tell a padding
    /// convention that repeats the edge line from one that skips it. A smooth gradient cannot:
    /// neighbouring rows quantise to the same byte.
    static func stripedImage(width: Int, height: Int) throws -> CVPixelBuffer {
        let buffer = try makeBuffer(width: width, height: height)
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let view = PixelView(buffer) else { return buffer }
        // One vImage fill per row; a loop over lines, never over pixels.
        for row in 0..<height {
            var pixel: [UInt8] = [
                UInt8((row &* 37) % 251), UInt8((row &* 91) % 253), UInt8((row &* 17) % 249), 255,
            ]
            var strip = view.sub(x: 0, y: row, width: width, height: 1).vImage
            vImageBufferFill_ARGB8888(&strip, &pixel, vImage_Flags(kvImageNoFlags))
        }
        return buffer
    }

    static func fill(_ buffer: CVPixelBuffer, blue: UInt8, green: UInt8, red: UInt8) {
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let view = PixelView(buffer) else { return }
        var pixel: [UInt8] = [blue, green, red, 255]
        var target = view.vImage
        vImageBufferFill_ARGB8888(&target, &pixel, vImage_Flags(kvImageNoFlags))
    }

    /// Nearest-neighbour magnification via CoreGraphics with interpolation switched off.
    static func nearestNeighbour(_ source: CVPixelBuffer, scaleFactor: Int) throws -> CVPixelBuffer {
        let width = CVPixelBufferGetWidth(source)
        let height = CVPixelBufferGetHeight(source)
        let output = try makeBuffer(width: width * scaleFactor, height: height * scaleFactor)

        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(output, [])
        defer {
            CVPixelBufferUnlockBaseAddress(output, [])
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
        }
        let bitmapInfo = CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        guard
            let sourceContext = CGContext(
                data: CVPixelBufferGetBaseAddress(source), width: width, height: height,
                bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(source),
                space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: bitmapInfo),
            let image = sourceContext.makeImage(),
            let destination = CGContext(
                data: CVPixelBufferGetBaseAddress(output), width: width * scaleFactor,
                height: height * scaleFactor, bitsPerComponent: 8,
                bytesPerRow: CVPixelBufferGetBytesPerRow(output),
                space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: bitmapInfo)
        else { throw TilingError.allocationFailed("could not build the nearest-neighbour context") }

        destination.interpolationQuality = .none
        destination.draw(
            image,
            in: CGRect(x: 0, y: 0, width: width * scaleFactor, height: height * scaleFactor))
        return output
    }

    /// Worst absolute channel difference between two buffers of the same size.
    ///
    /// vImage has no absolute-difference primitive, so this converts a row at a time with vDSP and
    /// reduces. One pass per row, never per pixel.
    static func maxDifference(_ lhs: CVPixelBuffer, _ rhs: CVPixelBuffer) -> Int {
        let width = min(CVPixelBufferGetWidth(lhs), CVPixelBufferGetWidth(rhs))
        let height = min(CVPixelBufferGetHeight(lhs), CVPixelBufferGetHeight(rhs))
        CVPixelBufferLockBaseAddress(lhs, .readOnly)
        CVPixelBufferLockBaseAddress(rhs, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(rhs, .readOnly)
            CVPixelBufferUnlockBaseAddress(lhs, .readOnly)
        }
        guard let left = PixelView(lhs), let right = PixelView(rhs) else { return .max }

        let samples = width * 4  // BGRA; the alpha channel is compared too, deliberately
        var a = [Float](repeating: 0, count: samples)
        var b = [Float](repeating: 0, count: samples)
        var difference = [Float](repeating: 0, count: samples)
        var worst: Float = 0
        for row in 0..<height {
            let leftRow = left.base.advanced(by: row * left.rowBytes).assumingMemoryBound(to: UInt8.self)
            let rightRow = right.base.advanced(by: row * right.rowBytes)
                .assumingMemoryBound(to: UInt8.self)
            vDSP_vfltu8(leftRow, 1, &a, 1, vDSP_Length(samples))
            vDSP_vfltu8(rightRow, 1, &b, 1, vDSP_Length(samples))
            vDSP_vsub(b, 1, a, 1, &difference, 1, vDSP_Length(samples))
            vDSP_vabs(difference, 1, &difference, 1, vDSP_Length(samples))
            var rowMaximum: Float = 0
            vDSP_maxv(difference, 1, &rowMaximum, vDSP_Length(samples))
            worst = max(worst, rowMaximum)
        }
        return Int(worst)
    }

    static func pixel(_ buffer: CVPixelBuffer, x: Int, y: Int) -> (b: UInt8, g: UInt8, r: UInt8) {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let view = PixelView(buffer) else { return (0, 0, 0) }
        let pointer = view.base.advanced(by: y * view.rowBytes + x * 4)
            .assumingMemoryBound(to: UInt8.self)
        return (pointer[0], pointer[1], pointer[2])
    }

    static let descriptor = ModelDescriptor(
        identifier: "SyntheticNearestNeighbour", scaleFactor: 4, contextMargin: 16,
        packageURL: URL(fileURLWithPath: "/dev/null"), inputTileSize: 64, precision: .float32)
}

// MARK: - Stitcher

struct StitcherTests {
    @Test("a linear ramp is exactly the values arithmetic says it is")
    func linearRamp() {
        let ramp = FeatherStitcher.ramp(length: 8, band: 5, cosine: false)
        // t = i/4 for i in 0...4, then flat.
        let expected: [Float] = [0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0]
        #expect(zip(ramp, expected).allSatisfy { abs($0 - $1) < 1e-6 })
    }

    @Test("a cosine ramp is the raised cosine, computed independently")
    func cosineRamp() {
        let ramp = FeatherStitcher.ramp(length: 6, band: 5, cosine: true)
        // 0.5 - 0.5·cos(π·t), t = i/4 — evaluated here, not by the code under test.
        let expected: [Float] = (0..<5).map { 0.5 - 0.5 * Foundation.cos(Float.pi * Float($0) / 4.0) } + [1.0]
        #expect(zip(ramp, expected).allSatisfy { abs($0 - $1) < 1e-6 })
        #expect(ramp[0] == 0.0 && ramp[4] == 1.0)
    }

    @Test("a ramp with no band is fully opaque", arguments: [0, 1])
    func degenerateRamp(band: Int) {
        #expect(FeatherStitcher.ramp(length: 5, band: band, cosine: false).allSatisfy { $0 == 1.0 })
    }

    @Test("both ramps rise monotonically and share their endpoints", arguments: [false, true])
    func rampsAreMonotone(cosine: Bool) {
        let ramp = FeatherStitcher.ramp(length: 32, band: 16, cosine: cosine)
        #expect(zip(ramp, ramp.dropFirst()).allSatisfy { $0 <= $1 })
        #expect(ramp.first == 0.0 && ramp.last == 1.0)
    }

    @Test("the alpha plane is the outer product of the two ramps")
    func alphaPlaneIsSeparable() throws {
        let stitcher = FeatherStitcher(method: .linearFeather)
        let plane = try stitcher.alphaPlane(width: 4, height: 3, leftBand: 4, topBand: 3, cosine: false)

        // Hand-computed: columns [0, 1/3, 2/3, 1], rows [0, 0.5, 1], product scaled by 255.
        // Compared to within one least-significant bit: vDSP_vfixru8's tie-breaking at exact .5
        // boundaries is not documented, and 0.5 x 1/3 x 255 lands on one. The endpoints are exact,
        // which is what the blend actually depends on.
        let columns: [Float] = [0, 1.0 / 3.0, 2.0 / 3.0, 1]
        let rows: [Float] = [0, 0.5, 1]
        for (y, rowWeight) in rows.enumerated() {
            for (x, columnWeight) in columns.enumerated() {
                let ideal = Double(rowWeight) * Double(columnWeight) * 255.0
                #expect(abs(Double(plane[y * 4 + x]) - ideal) <= 1.0, "alpha at (\(x),\(y))")
            }
        }
        #expect(plane[0] == 0, "a zero weight must be exactly transparent")
        #expect(plane[2 * 4 + 3] == 255, "a unit weight must be exactly opaque")
    }

    @Test("the feather blend is a monotone interpolation with the right endpoints")
    func featherBlendInterpolates() throws {
        // A tile of one flat colour blended over an output of another: the band must sweep from
        // the old value to the new one, monotonically, with no overshoot at either end.
        let scale = 1
        let tileSize = 32
        let margin = 4
        let tile = try TilingTestSupport.makeBuffer(width: tileSize, height: tileSize)
        let output = try TilingTestSupport.makeBuffer(width: tileSize, height: tileSize)
        TilingTestSupport.fill(tile, blue: 240, green: 240, red: 240)
        TilingTestSupport.fill(output, blue: 40, green: 40, red: 40)

        let grid = TileGrid(tileSize: tileSize, contextMargin: margin, tiles: [])
        let region = TileRegion(
            validRect: CGRect(x: 8, y: 0, width: 16, height: 32),
            paddedInputRect: CGRect(x: 0, y: 0, width: tileSize, height: tileSize))

        try FeatherStitcher(method: .linearFeather).write(
            tile: tile, region: region, grid: grid, scaleFactor: scale, into: output)

        // Geometry, derived here rather than read back from the engine: the footprint is the valid
        // rect grown by one margin on each side (x: 8-4 = 4, width 16+8 = 24), and the ramp covers
        // its first 2*margin = 8 columns, so image columns 4...11 carry weights i/7.
        let band = (0..<16).map { Int(TilingTestSupport.pixel(output, x: $0, y: 16).b) }
        #expect(band[0...3].allSatisfy { $0 == 40 }, "outside the footprint the old value must survive")
        #expect(band[4] == 40, "the ramp starts at zero weight")
        #expect(band[11] == 240, "the ramp reaches full weight at the end of the band")
        #expect(band[12...15].allSatisfy { $0 == 240 }, "past the band the tile is fully opaque")
        #expect(zip(band, band.dropFirst()).allSatisfy { $0 <= $1 }, "a feather never oscillates")
        for index in 4...11 {
            let weight = Double(index - 4) / 7.0
            let ideal = 40.0 + weight * (240.0 - 40.0)
            #expect(abs(Double(band[index]) - ideal) <= 2.0, "column \(index): \(band[index]) vs \(ideal)")
        }
    }
}

// MARK: - Engine

struct TilingEngineTests {
    static let descriptor = TilingTestSupport.descriptor

    /// The central correctness claim: with a model that has no receptive field, tiling is a
    /// lossless rearrangement, so every stitch method must reproduce the untiled result exactly.
    @Test(
        "a context-free model tiles to a bit-exact match of the untiled result",
        arguments: StitchMethod.allCases
    )
    func tilingIsLosslessForAContextFreeModel(method: StitchMethod) async throws {
        let image = try TilingTestSupport.gradientImage(width: 100, height: 70)
        let backend = NearestNeighbourBackend(scaleFactor: 4)
        let engine = SequentialTilingEngine(contextMargin: 8)

        let tiled = try await engine.upscale(
            image, model: Self.descriptor, backend: backend, method: method, progress: nil)
        let untiled = try TilingTestSupport.nearestNeighbour(image, scaleFactor: 4)

        #expect(CVPixelBufferGetWidth(tiled) == 400 && CVPixelBufferGetHeight(tiled) == 280)
        #expect(
            TilingTestSupport.maxDifference(tiled, untiled) == 0,
            "\(method.rawValue) changed pixels a context-free model cannot have changed")
    }

    @Test(
        "images that are not tile multiples still reproduce exactly",
        arguments: [
            CGSize(width: 64, height: 64), CGSize(width: 65, height: 63), CGSize(width: 30, height: 20),
        ]
    )
    func awkwardSizesAreExact(size: CGSize) async throws {
        let image = try TilingTestSupport.gradientImage(width: Int(size.width), height: Int(size.height))
        let engine = SequentialTilingEngine(contextMargin: 8)

        let tiled = try await engine.upscale(
            image, model: Self.descriptor, backend: NearestNeighbourBackend(scaleFactor: 4),
            method: .centerValidRegion, progress: nil)

        #expect(
            TilingTestSupport.maxDifference(
                tiled, try TilingTestSupport.nearestNeighbour(image, scaleFactor: 4)) == 0)
    }

    @Test("progress is reported once per tile, monotonically, ending at exactly 1.0")
    func progressReporting() async throws {
        let image = try TilingTestSupport.gradientImage(width: 100, height: 70)
        let backend = NearestNeighbourBackend(scaleFactor: 4)
        let engine = SequentialTilingEngine(contextMargin: 8)
        let reported = Reported()

        _ = try await engine.upscale(
            image, model: Self.descriptor, backend: backend, method: .centerValidRegion,
            progress: { reported.append($0) })

        let values = reported.values
        #expect(values.count == backend.predictionCount)
        #expect(zip(values, values.dropFirst()).allSatisfy { $0 < $1 })
        #expect(values.last == 1.0)
        #expect(values.allSatisfy { $0 > 0.0 && $0 <= 1.0 })
    }

    @Test("a cancelled task stops before doing any work")
    func cancellationBeforeStart() async throws {
        let backend = NearestNeighbourBackend(scaleFactor: 4)
        let engine = SequentialTilingEngine(contextMargin: 8)

        // The image is built inside the task: `CVPixelBuffer` is `@_nonSendable`, so neither it nor
        // the engine's result may cross the task boundary. The task yields a count instead.
        let task = Task { () -> Int in
            let image = try TilingTestSupport.gradientImage(width: 100, height: 70)
            _ = try await engine.upscale(
                image, model: TilingTestSupport.descriptor, backend: backend,
                method: .centerValidRegion, progress: nil)
            return backend.predictionCount
        }
        task.cancel()

        await #expect(throws: CancellationError.self) { _ = try await task.value }
        #expect(backend.predictionCount == 0)
    }

    @Test("cancellation mid-run stops at the next tile boundary")
    func cancellationMidRun() async throws {
        let image = try TilingTestSupport.gradientImage(width: 200, height: 200)
        let backend = NearestNeighbourBackend(scaleFactor: 4)
        backend.cancelSelfAt = 2
        let engine = SequentialTilingEngine(contextMargin: 8)

        await #expect(throws: CancellationError.self) {
            _ = try await engine.upscale(
                image, model: Self.descriptor, backend: backend, method: .centerValidRegion,
                progress: nil)
        }
        let planned = TilePlanner(scaleFactor: 4)
            .plan(imageSize: CGSize(width: 200, height: 200), tileSize: 64, contextMargin: 8).tiles.count
        #expect(backend.predictionCount == 3, "it must stop at the boundary, not finish the grid")
        #expect(planned > 3, "the fixture must have more tiles than were run for this to mean anything")
    }

    @Test("the output image is allocated exactly once per image")
    func singleOutputAllocation() async throws {
        let image = try TilingTestSupport.gradientImage(width: 200, height: 200)
        let backend = NearestNeighbourBackend(scaleFactor: 4)
        let engine = SequentialTilingEngine(contextMargin: 8)

        _ = try await engine.upscale(
            image, model: Self.descriptor, backend: backend, method: .centerValidRegion, progress: nil)

        #expect(backend.predictionCount > 1, "a single-tile image would make this vacuous")
        #expect(engine.outputAllocations == 1, "one image, one output buffer")

        _ = try await engine.upscale(
            image, model: Self.descriptor, backend: backend, method: .centerValidRegion, progress: nil)
        #expect(engine.outputAllocations == 2, "a second image is a second buffer, from the same pool")
    }

    @Test("a backend failure propagates instead of producing a half-finished image")
    func backendFailurePropagates() async throws {
        let image = try TilingTestSupport.gradientImage(width: 200, height: 200)
        let backend = NearestNeighbourBackend(scaleFactor: 4)
        backend.failAt = 1
        let engine = SequentialTilingEngine(contextMargin: 8)

        await #expect(throws: InferenceError.self) {
            _ = try await engine.upscale(
                image, model: Self.descriptor, backend: backend, method: .centerValidRegion,
                progress: nil)
        }
    }

    @Test("an unsupported pixel format is refused by name")
    func unsupportedFormat() async throws {
        var buffer: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, 64, 64, kCVPixelFormatType_32ARGB, nil, &buffer)
        let image = try #require(buffer)
        let engine = SequentialTilingEngine(contextMargin: 8)

        // kCVPixelFormatType_32ARGB is the number 32, not a four-character code, so the message
        // reports the value rather than four unprintable bytes.
        await #expect(throws: TilingError.unsupportedPixelFormat("32")) {
            _ = try await engine.upscale(
                image, model: Self.descriptor, backend: NearestNeighbourBackend(scaleFactor: 4),
                method: .centerValidRegion, progress: nil)
        }
    }

    // MARK: - Context padding

    @Test("reflection padding mirrors the image across its edge")
    func reflectionPadding() throws {
        let image = try TilingTestSupport.gradientImage(width: 40, height: 40)
        let tile = try TilingTestSupport.makeBuffer(width: 64, height: 64)
        // Window hangs 12 px off the left and top of a 40x40 image.
        try SequentialTilingEngine.fillPaddedTile(
            source: image, window: IntRect(x: -12, y: -12, width: 64, height: 64), into: tile)

        // Column 11 of the tile is image column 0 mirrored; column 12 is image column 0 itself.
        for offset in 1...12 {
            let mirrored = TilingTestSupport.pixel(tile, x: 12 - offset, y: 30)
            let original = TilingTestSupport.pixel(tile, x: 12 + offset - 1, y: 30)
            #expect(mirrored == original, "reflection is wrong \(offset) px outside the left edge")
        }
    }

    /// The oracle's canvas has to be padded the same way the engine pads, or the comparison
    /// measures the difference between two padding conventions instead of the tiling. This pins
    /// which convention the engine actually implements: **symmetric** reflection, in which the edge
    /// row is repeated (…c b a | a b c…), as opposed to reflect-101 (…c b | a b c…).
    /// `tools/reference/run_reference.py` pads with numpy's `mode="symmetric"` on the strength of
    /// this test.
    @Test("reflection repeats the edge line — symmetric, not reflect-101")
    func reflectionConventionIsSymmetric() throws {
        let image = try TilingTestSupport.stripedImage(width: 40, height: 40)
        let tile = try TilingTestSupport.makeBuffer(width: 64, height: 64)
        try SequentialTilingEngine.fillPaddedTile(
            source: image, window: IntRect(x: -12, y: -12, width: 64, height: 64), into: tile)

        // Tile row 12 is image row 0. Symmetric padding puts image row 0 at tile row 11 as well;
        // reflect-101 would put image row 1 there.
        let firstRealRow = TilingTestSupport.pixel(tile, x: 30, y: 12)
        let lastPaddedRow = TilingTestSupport.pixel(tile, x: 30, y: 11)
        let secondRealRow = TilingTestSupport.pixel(tile, x: 30, y: 13)

        #expect(firstRealRow != secondRealRow, "the fixture must distinguish adjacent rows")
        #expect(lastPaddedRow == firstRealRow, "the edge line is repeated: symmetric padding")
        #expect(lastPaddedRow != secondRealRow, "reflect-101 would have skipped the edge line")

        // The whole padded strip, checked against the symmetric mapping row by row: padded row
        // (12 - 1 - k) must hold image row k.
        for k in 0..<12 {
            #expect(
                TilingTestSupport.pixel(tile, x: 30, y: 11 - k)
                    == TilingTestSupport.pixel(tile, x: 30, y: 12 + k),
                "padded row \(11 - k) should mirror image row \(k)")
        }
        // And at the far edge: image ends at tile row 51, so row 52 repeats it.
        #expect(TilingTestSupport.pixel(tile, x: 30, y: 52) == TilingTestSupport.pixel(tile, x: 30, y: 51))
    }

    @Test("a window entirely outside the image is refused rather than filled with noise")
    func windowOutsideImage() throws {
        let image = try TilingTestSupport.gradientImage(width: 40, height: 40)
        let tile = try TilingTestSupport.makeBuffer(width: 64, height: 64)

        #expect(throws: TilingError.self) {
            try SequentialTilingEngine.fillPaddedTile(
                source: image, window: IntRect(x: 500, y: 500, width: 64, height: 64), into: tile)
        }
    }
}

/// Progress values are delivered from the engine's task; this collects them safely.
final class Reported: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [Double] = []
    func append(_ value: Double) { lock.withLock { storage.append(value) } }
    var values: [Double] { lock.withLock { storage } }
}

// MARK: - Measurement

struct ContextMarginTests {
    /// Two flat buffers differing by a known amount: the metrics are arithmetic, checkable by hand.
    @Test("metrics on a known constant difference are exactly the arithmetic")
    func metricsOnConstantDifference() throws {
        let candidate = try TilingTestSupport.makeBuffer(width: 16, height: 16)
        let reference = try TilingTestSupport.makeBuffer(width: 16, height: 16)
        TilingTestSupport.fill(candidate, blue: 110, green: 110, red: 110)
        TilingTestSupport.fill(reference, blue: 100, green: 100, red: 100)

        let (values, width, height) = try ContextMargin.absoluteDifference(candidate, reference)
        let metrics = ContextMargin.metrics(values: values, mask: nil)

        #expect(width == 16 && height == 16)
        #expect(values.count == 16 * 16 * 3, "alpha is excluded from the comparison")
        #expect(abs(metrics.meanAbs255 - 10.0) < 1e-6)
        #expect(abs(metrics.maxAbs255 - 10.0) < 1e-6)
        // PSNR = 10·log10(255² / 10²), evaluated here rather than by the code under test.
        #expect(abs(metrics.psnr - 10.0 * log10((255.0 * 255.0) / 100.0)) < 1e-6)
        #expect(metrics.withinToleranceFraction == 0.0, "10/255 is outside a 4/255 tolerance")
    }

    /// Certification spec §4 states the Level A criterion as "max-abs **<= 4/255** on >= 99% of
    /// pixels". A sample sitting exactly on the tolerance is therefore inside it. This test exists
    /// because an earlier instrument counted exact-boundary samples as exceeding, which understated
    /// the printed column on real data by 0.54 pp.
    @Test("samples sitting exactly at the tolerance count as within it")
    func toleranceIsInclusiveOfTheThreshold() {
        // Hand-counted: 0, 2, 4, 4, 4 are within 4/255; 5, 8, 100 are not. Five of eight.
        let values: [Float] = [0, 2, 4, 4, 4, 5, 8, 100]

        let metrics = ContextMargin.metrics(values: values, mask: nil, tolerance255: 4.0)

        #expect(
            metrics.withinToleranceFraction == 0.625,
            "three of these samples sit exactly at 4/255 and the spec counts them as within")
        #expect(metrics.sampleCount == 8)
    }

    @Test("a field sitting entirely on the tolerance is entirely within it")
    func aFieldExactlyAtToleranceIsFullyWithin() {
        let values = [Float](repeating: 4.0, count: 64)
        #expect(
            ContextMargin.metrics(values: values, mask: nil, tolerance255: 4.0).withinToleranceFraction == 1.0
        )
    }

    @Test("one least-significant step past the tolerance is outside it")
    func justPastToleranceIsOutside() {
        let values = [Float](repeating: 5.0, count: 64)
        #expect(
            ContextMargin.metrics(values: values, mask: nil, tolerance255: 4.0).withinToleranceFraction == 0.0
        )
    }

    @Test("the inclusive count survives masking")
    func inclusiveCountUnderAMask() {
        let values: [Float] = [4, 4, 9, 9]
        let mask: [Float] = [1, 1, 1, 0]
        // Selected samples are 4, 4, 9: two of three are within an inclusive 4/255 tolerance.
        let metrics = ContextMargin.metrics(values: values, mask: mask, tolerance255: 4.0)
        #expect(abs(metrics.withinToleranceFraction - 2.0 / 3.0) < 1e-6)
        #expect(metrics.sampleCount == 3)
    }

    @Test("identical buffers have no error and infinite PSNR")
    func metricsOnIdenticalBuffers() throws {
        let candidate = try TilingTestSupport.gradientImage(width: 32, height: 32)
        let reference = try TilingTestSupport.gradientImage(width: 32, height: 32)

        let (values, _, _) = try ContextMargin.absoluteDifference(candidate, reference)
        let metrics = ContextMargin.metrics(values: values, mask: nil)

        #expect(metrics.meanAbs255 == 0 && metrics.maxAbs255 == 0)
        #expect(metrics.psnr == .infinity)
        #expect(metrics.withinToleranceFraction == 1.0)
    }

    /// The seam detector has to be able to *find* a seam, so this plants one.
    @Test("an error confined to the tile boundaries produces a large boundary/interior ratio")
    func seamsAreDetected() throws {
        let size = 64
        let candidate = try TilingTestSupport.makeBuffer(width: size, height: size)
        let reference = try TilingTestSupport.makeBuffer(width: size, height: size)
        TilingTestSupport.fill(candidate, blue: 100, green: 100, red: 100)
        TilingTestSupport.fill(reference, blue: 100, green: 100, red: 100)

        // A grid whose valid regions meet at x = 32 and y = 32.
        let grid = TileGrid(
            tileSize: 32, contextMargin: 0,
            tiles: [
                TileRegion(validRect: CGRect(x: 0, y: 0, width: 32, height: 32), paddedInputRect: .zero),
                TileRegion(validRect: CGRect(x: 32, y: 0, width: 32, height: 32), paddedInputRect: .zero),
                TileRegion(validRect: CGRect(x: 0, y: 32, width: 32, height: 32), paddedInputRect: .zero),
                TileRegion(validRect: CGRect(x: 32, y: 32, width: 32, height: 32), paddedInputRect: .zero),
            ])

        // Corrupt a single column exactly on the vertical seam.
        CVPixelBufferLockBaseAddress(candidate, [])
        if let view = PixelView(candidate) {
            var pixel: [UInt8] = [180, 180, 180, 255]
            var strip = view.sub(x: 32, y: 0, width: 1, height: size).vImage
            vImageBufferFill_ARGB8888(&strip, &pixel, vImage_Flags(kvImageNoFlags))
        }
        CVPixelBufferUnlockBaseAddress(candidate, [])

        let profile = try ContextMargin.seamProfile(
            candidate: candidate, oracle: reference, grid: grid, boundaryRadius: 4)

        #expect(profile.interior.meanAbs255 == 0, "nothing outside the seam band was touched")
        #expect(profile.boundary.meanAbs255 > 0)
        #expect(profile.ratio == .infinity, "a clean interior and a dirty boundary is an infinite ratio")
        #expect(profile.boundary.sampleCount < profile.overall.sampleCount)
    }

    @Test("a uniform error is not a seam — the ratio stays near one")
    func uniformErrorIsNotASeam() throws {
        let candidate = try TilingTestSupport.makeBuffer(width: 64, height: 64)
        let reference = try TilingTestSupport.makeBuffer(width: 64, height: 64)
        TilingTestSupport.fill(candidate, blue: 105, green: 105, red: 105)
        TilingTestSupport.fill(reference, blue: 100, green: 100, red: 100)

        let grid = TileGrid(
            tileSize: 32, contextMargin: 0,
            tiles: [
                TileRegion(validRect: CGRect(x: 0, y: 0, width: 32, height: 64), paddedInputRect: .zero),
                TileRegion(validRect: CGRect(x: 32, y: 0, width: 32, height: 64), paddedInputRect: .zero),
            ])

        let profile = try ContextMargin.seamProfile(
            candidate: candidate, oracle: reference, grid: grid, boundaryRadius: 4)

        #expect(abs(profile.ratio - 1.0) < 1e-6, "an even error must not look like a seam")
        #expect(abs(profile.overall.meanAbs255 - 5.0) < 1e-6)
    }

    @Test("the ring profile localises error to the distance it actually sits at")
    func ringProfileLocalisesError() throws {
        let size = 32
        let candidate = try TilingTestSupport.makeBuffer(width: size, height: size)
        let reference = try TilingTestSupport.makeBuffer(width: size, height: size)
        TilingTestSupport.fill(candidate, blue: 100, green: 100, red: 100)
        TilingTestSupport.fill(reference, blue: 100, green: 100, red: 100)

        // Corrupt only the outermost frame, which is ring 0.
        CVPixelBufferLockBaseAddress(candidate, [])
        if let view = PixelView(candidate) {
            var pixel: [UInt8] = [150, 150, 150, 255]
            for rect in [
                (0, 0, size, 1), (0, size - 1, size, 1), (0, 0, 1, size), (size - 1, 0, 1, size),
            ] {
                var strip = view.sub(x: rect.0, y: rect.1, width: rect.2, height: rect.3).vImage
                vImageBufferFill_ARGB8888(&strip, &pixel, vImage_Flags(kvImageNoFlags))
            }
        }
        CVPixelBufferUnlockBaseAddress(candidate, [])

        let (values, width, height) = try ContextMargin.absoluteDifference(candidate, reference)
        let profile = ContextMargin.ringProfile(values: values, width: width, height: height)

        #expect(abs(profile.ringMean[0] - 50.0) < 1e-6, "ring 0 is the corrupted frame")
        #expect(profile.ringMean[1] == 0.0, "ring 1 is untouched")
        #expect(profile.ringMean.dropFirst().allSatisfy { $0 == 0.0 })
        #expect(profile.coreMean[1] == 0.0, "everything inside ring 1 is clean")
        #expect(profile.coreMean[0] > 0.0, "the core measured from ring 0 includes the corruption")
    }

    @Test("the derivation takes the first crossing and doubles it")
    func derivationDoublesTheCrossing() {
        let probes = [
            ContextMargin.Probe(margin: 0, coreMeanAbs255: 4.0, ringMeanAbs255: 8.0, ringMaxAbs255: 20),
            ContextMargin.Probe(margin: 4, coreMeanAbs255: 1.2, ringMeanAbs255: 2.5, ringMaxAbs255: 9),
            ContextMargin.Probe(margin: 8, coreMeanAbs255: 0.4, ringMeanAbs255: 0.6, ringMaxAbs255: 5),
            ContextMargin.Probe(margin: 12, coreMeanAbs255: 0.3, ringMeanAbs255: 0.3, ringMaxAbs255: 4),
        ]
        let derived = ContextMargin.derive(from: probes)

        #expect(derived?.crossing == 8, "the first margin whose ring is under 1/255")
        #expect(derived?.margin == 16, "safety factor x2")
    }

    @Test("a model whose error never decays yields no margin rather than a made-up one")
    func derivationRefusesWhenErrorNeverDecays() {
        let probes = (0..<4).map {
            ContextMargin.Probe(
                margin: $0 * 4, coreMeanAbs255: 6.0, ringMeanAbs255: 6.0, ringMaxAbs255: 20)
        }
        #expect(ContextMargin.derive(from: probes) == nil)
    }
}
