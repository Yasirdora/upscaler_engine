import CoreVideo
import Foundation
import Testing

@testable import UpscaleCore

/// Tests for the band-spooled destination surface.
///
/// Model-free and backend-free by design: a destination surface is pixels and file offsets, so
/// these drive it directly with known byte patterns. No `.mlpackage`, no `InferenceBackend`, no
/// image decode.
///
/// The load-bearing test is `spooledBytesMatchTheWholeImageSurface`: the same writes through the
/// whole-image `CVPixelBuffer` destination and through the spool must produce identical bytes. That
/// is the host-side half of the output-equivalence evidence — it cannot substitute for the Level A
/// comparison against the reference oracles, but a failure here would mean the spooled path is
/// wrong before any model is involved.
struct OutputSurfaceTests {

    // MARK: - Helpers

    /// A deterministic, position-dependent byte so a misplaced row or column cannot pass.
    static func pattern(x: Int, y: Int, channel: Int) -> UInt8 {
        let termX: Int = x &* 7
        let termY: Int = y &* 31
        let termC: Int = channel &* 13
        let sum: Int = (termX &+ termY &+ termC) & 0xFF
        return UInt8(truncatingIfNeeded: sum)
    }

    /// Fill an absolute rect in a view with `pattern`.
    static func fill(_ view: PixelView, x: Int, y: Int, width: Int, height: Int) {
        let sub = view.sub(x: x, y: y, width: width, height: height)
        for row in 0..<height {
            let rowBase = sub.base.advanced(by: row * sub.rowBytes)
            for column in 0..<width {
                let pixel = rowBase.advanced(by: column * PixelView.bytesPerPixel)
                for channel in 0..<PixelView.bytesPerPixel {
                    pixel.advanced(by: channel).storeBytes(
                        of: pattern(x: x + column, y: y + row, channel: channel), as: UInt8.self)
                }
            }
        }
    }

    /// The whole image as a tight BGRA raster, built directly from `pattern`.
    static func expectedRaster(width: Int, height: Int) -> [UInt8] {
        var bytes = [UInt8](repeating: 0, count: width * height * PixelView.bytesPerPixel)
        for y in 0..<height {
            for x in 0..<width {
                for channel in 0..<PixelView.bytesPerPixel {
                    bytes[(y * width + x) * PixelView.bytesPerPixel + channel] =
                        pattern(x: x, y: y, channel: channel)
                }
            }
        }
        return bytes
    }

    static func temporarySpoolURL() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("spool-\(UUID().uuidString).bgra")
    }

    // MARK: - Window sizing is geometry, not a tunable

    @Test("centerValidRegion keeps exactly one band open")
    func validRegionWindowIsOneBand() {
        let grid = TilePlanner(scaleFactor: 4).plan(
            imageSize: CGSize(width: 4032, height: 3024), tileSize: 512, contextMargin: 52)

        let rows = BandSpoolSurface.openRowCount(for: grid, scaleFactor: 4, method: .centerValidRegion)

        // core = 512 - 2*52 = 408; 408 * 4 = 1632.
        #expect(grid.validCoreSize == 408)
        #expect(rows == 1632)
    }

    @Test("the feathers need a margin's reach above and below their band")
    func featherWindowAddsTwoMargins() {
        let grid = TilePlanner(scaleFactor: 4).plan(
            imageSize: CGSize(width: 4032, height: 3024), tileSize: 512, contextMargin: 52)

        // (408 + 2*52) * 4 = 2048.
        #expect(
            BandSpoolSurface.openRowCount(for: grid, scaleFactor: 4, method: .linearFeather) == 2048)
        #expect(
            BandSpoolSurface.openRowCount(for: grid, scaleFactor: 4, method: .cosineFeather) == 2048)
    }

    @Test("hardCrop is the most expensive to spool, and is the naive control")
    func hardCropWindowIsWidest() {
        let grid = TilePlanner(scaleFactor: 4).plan(
            imageSize: CGSize(width: 4032, height: 3024), tileSize: 512, contextMargin: 52)

        // (512 + 408) * 4 = 3680.
        #expect(BandSpoolSurface.openRowCount(for: grid, scaleFactor: 4, method: .hardCrop) == 3680)
    }

    @Test("the window is a fraction of the whole-image surface it replaces")
    func windowIsMateriallySmallerThanTheSurface() {
        let grid = TilePlanner(scaleFactor: 4).plan(
            imageSize: CGSize(width: 4032, height: 3024), tileSize: 512, contextMargin: 52)
        let outputWidth = 4032 * 4
        let outputHeight = 3024 * 4

        let rows = BandSpoolSurface.openRowCount(for: grid, scaleFactor: 4, method: .centerValidRegion)
        let windowBytes = rows * outputWidth * PixelView.bytesPerPixel
        let surfaceBytes = outputHeight * outputWidth * PixelView.bytesPerPixel

        // The whole point of the band spool: residency stops scaling with output image size.
        #expect(surfaceBytes == 780_337_152)  // 744.2 MiB, the measured whole-image figure
        #expect(windowBytes < surfaceBytes / 7)
    }

    // MARK: - The spool is a faithful raster

    @Test("a full image written band by band spools byte-exact")
    func spoolIsByteExact() throws {
        let width = 64
        let height = 40
        let url = Self.temporarySpoolURL()
        defer { try? FileManager.default.removeItem(at: url) }

        let surface = try BandSpoolSurface(
            width: width, height: height, windowRows: 8, spoolURL: url)

        // Eight-row bands, closed as they complete — the raster-order contract the engine keeps.
        for band in stride(from: 0, to: height, by: 8) {
            try surface.withOpenWindow { view in
                Self.fill(view, x: 0, y: band, width: width, height: 8)
            }
            try surface.closeRows(below: band + 8)
        }
        try surface.finish()

        let spooled = try Data(contentsOf: url)
        #expect(spooled.count == width * height * PixelView.bytesPerPixel)
        #expect(Array(spooled) == Self.expectedRaster(width: width, height: height))
    }

    @Test("absolute coordinates survive a window slide")
    func absoluteCoordinatesSurviveSliding() throws {
        let width = 16
        let height = 24
        let url = Self.temporarySpoolURL()
        defer { try? FileManager.default.removeItem(at: url) }

        let surface = try BandSpoolSurface(
            width: width, height: height, windowRows: 8, spoolURL: url)

        // Write row 20 while the window sits at 16 — the view must resolve it to buffer row 4, not
        // row 20. Getting this wrong is the failure mode PixelView.originY exists to prevent.
        for band in stride(from: 0, to: height, by: 8) {
            try surface.withOpenWindow { view in
                #expect(view.originY == band)
                Self.fill(view, x: 0, y: band, width: width, height: 8)
            }
            try surface.closeRows(below: band + 8)
        }
        try surface.finish()

        let spooled = Array(try Data(contentsOf: url))
        let row20 = (20 * width) * PixelView.bytesPerPixel
        #expect(spooled[row20] == Self.pattern(x: 0, y: 20, channel: 0))
        #expect(spooled[row20 + 1] == Self.pattern(x: 0, y: 20, channel: 1))
    }

    // MARK: - Equivalence with the surface it replaces

    @Test("spooled bytes match the whole-image surface for identical writes")
    func spooledBytesMatchTheWholeImageSurface() throws {
        let width = 48
        let height = 32
        let url = Self.temporarySpoolURL()
        defer { try? FileManager.default.removeItem(at: url) }

        // Same writes, two destinations.
        let buffer = try SequentialTilingEngine.allocate(width: width, height: height)
        let inMemory = PixelBufferSurface(buffer)
        try inMemory.withOpenWindow { view in
            for band in stride(from: 0, to: height, by: 8) {
                Self.fill(view, x: 0, y: band, width: width, height: 8)
            }
        }

        let spool = try BandSpoolSurface(width: width, height: height, windowRows: 8, spoolURL: url)
        for band in stride(from: 0, to: height, by: 8) {
            try spool.withOpenWindow { view in
                Self.fill(view, x: 0, y: band, width: width, height: 8)
            }
            try spool.closeRows(below: band + 8)
        }
        try spool.finish()

        // Read the CVPixelBuffer back row by row — its rowBytes may be padded, the spool's is not.
        var fromBuffer = [UInt8]()
        fromBuffer.reserveCapacity(width * height * PixelView.bytesPerPixel)
        try inMemory.withOpenWindow { view in
            for y in 0..<height {
                let row = view.base.advanced(by: y * view.rowBytes)
                for byte in 0..<(width * PixelView.bytesPerPixel) {
                    fromBuffer.append(row.advanced(by: byte).load(as: UInt8.self))
                }
            }
        }

        #expect(Array(try Data(contentsOf: url)) == fromBuffer)
    }

    // MARK: - The contract the engine must keep

    @Test("closing is monotonic — a backward close is a no-op, never a rewind")
    func closingIsMonotonic() throws {
        let url = Self.temporarySpoolURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let surface = try BandSpoolSurface(width: 8, height: 24, windowRows: 8, spoolURL: url)

        try surface.withOpenWindow { view in Self.fill(view, x: 0, y: 0, width: 8, height: 8) }
        try surface.closeRows(below: 8)
        #expect(surface.openRows.lowerBound == 8)

        try surface.closeRows(below: 4)  // backwards
        #expect(surface.openRows.lowerBound == 8, "a backward close must not rewind the window")

        try surface.closeRows(below: 8)  // equal
        #expect(surface.openRows.lowerBound == 8, "an equal close must not re-spool")
    }

    @Test("closing more than the window ever held is a loud failure")
    func closingBeyondTheWindowThrows() throws {
        let url = Self.temporarySpoolURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let surface = try BandSpoolSurface(width: 8, height: 64, windowRows: 8, spoolURL: url)

        // A geometry bug would corrupt the spool in a way no pixel comparison could localise, so
        // it must throw rather than write garbage.
        #expect(throws: TilingError.self) {
            try surface.closeRows(below: 32)
        }
    }

    @Test("finish flushes the tail without an explicit close")
    func finishFlushesTheTail() throws {
        let width = 8
        let height = 12  // not a multiple of the window
        let url = Self.temporarySpoolURL()
        defer { try? FileManager.default.removeItem(at: url) }

        let surface = try BandSpoolSurface(width: width, height: height, windowRows: 8, spoolURL: url)
        try surface.withOpenWindow { view in Self.fill(view, x: 0, y: 0, width: width, height: 8) }
        try surface.closeRows(below: 8)
        try surface.withOpenWindow { view in Self.fill(view, x: 0, y: 8, width: width, height: 4) }
        try surface.finish()

        let spooled = try Data(contentsOf: url)
        #expect(spooled.count == width * height * PixelView.bytesPerPixel)
        #expect(Array(spooled) == Self.expectedRaster(width: width, height: height))
    }

    // MARK: - End to end: the two destinations must agree

    /// A `CVPixelBuffer` as a tightly packed raster, so it can be compared with a spool file whose
    /// rows carry no padding.
    static func tightBytes(of buffer: CVPixelBuffer) -> [UInt8] {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return [] }

        var bytes = [UInt8]()
        bytes.reserveCapacity(width * height * PixelView.bytesPerPixel)
        for y in 0..<height {
            let row = base.advanced(by: y * rowBytes)
            for byte in 0..<(width * PixelView.bytesPerPixel) {
                bytes.append(row.advanced(by: byte).load(as: UInt8.self))
            }
        }
        return bytes
    }

    /// The claim the band spool lives or dies on: spooling changed no pixels.
    ///
    /// Every stitch method, through both destinations, on an image tall enough that the window
    /// genuinely slides — at tile 64, margin 8, scale 4 the window is 192 rows against a 280-row
    /// output, so the spool advances twice. This is the host-side half of the evidence; the Level A
    /// comparison against the reference oracles is the other half, and is not this.
    @Test(
        "the spooled path is byte-identical to the in-memory path",
        arguments: StitchMethod.allCases
    )
    func spooledPathMatchesTheInMemoryPath(method: StitchMethod) async throws {
        let image = try TilingTestSupport.gradientImage(width: 100, height: 70)
        let engine = SequentialTilingEngine(contextMargin: 8)
        let url = Self.temporarySpoolURL()
        defer { try? FileManager.default.removeItem(at: url) }

        let inMemory = try await engine.upscale(
            image, model: TilingTestSupport.descriptor, backend: NearestNeighbourBackend(scaleFactor: 4),
            method: method, progress: nil)
        let spooled = try await engine.upscale(
            image, model: TilingTestSupport.descriptor, backend: NearestNeighbourBackend(scaleFactor: 4),
            method: method, spoolingTo: url, progress: nil)

        #expect(spooled.width == 400 && spooled.height == 280)
        #expect(spooled.bytesPerRow == 400 * PixelView.bytesPerPixel, "the spool raster is tight")

        let onDisk = Array(try Data(contentsOf: url))
        #expect(onDisk.count == spooled.byteCount)
        #expect(
            onDisk == Self.tightBytes(of: inMemory),
            "\(method.rawValue) spooled differently than it buffered")
    }

    @Test("a spooled context-free model is still a lossless rearrangement")
    func spooledTilingIsLossless() async throws {
        let image = try TilingTestSupport.gradientImage(width: 100, height: 70)
        let engine = SequentialTilingEngine(contextMargin: 8)
        let url = Self.temporarySpoolURL()
        defer { try? FileManager.default.removeItem(at: url) }

        _ = try await engine.upscale(
            image, model: TilingTestSupport.descriptor, backend: NearestNeighbourBackend(scaleFactor: 4),
            method: .centerValidRegion, spoolingTo: url, progress: nil)
        let untiled = try TilingTestSupport.nearestNeighbour(image, scaleFactor: 4)

        // Same claim TilingTests makes for the in-memory path: a model with no receptive field
        // means tiling — and now spooling — cannot have changed a pixel.
        #expect(Array(try Data(contentsOf: url)) == Self.tightBytes(of: untiled))
    }

    @Test(
        "images that are not tile multiples spool exactly too",
        arguments: [
            CGSize(width: 64, height: 64), CGSize(width: 65, height: 63), CGSize(width: 30, height: 20),
        ]
    )
    func awkwardSizesSpoolExactly(size: CGSize) async throws {
        let image = try TilingTestSupport.gradientImage(width: Int(size.width), height: Int(size.height))
        let engine = SequentialTilingEngine(contextMargin: 8)
        let url = Self.temporarySpoolURL()
        defer { try? FileManager.default.removeItem(at: url) }

        let inMemory = try await engine.upscale(
            image, model: TilingTestSupport.descriptor, backend: NearestNeighbourBackend(scaleFactor: 4),
            method: .centerValidRegion, progress: nil)
        let spooled = try await engine.upscale(
            image, model: TilingTestSupport.descriptor, backend: NearestNeighbourBackend(scaleFactor: 4),
            method: .centerValidRegion, spoolingTo: url, progress: nil)

        #expect(spooled.byteCount == Int(size.width) * 4 * Int(size.height) * 4 * PixelView.bytesPerPixel)
        #expect(Array(try Data(contentsOf: url)) == Self.tightBytes(of: inMemory))
    }

    @Test("the whole-image surface keeps every row open and spools nothing")
    func pixelBufferSurfaceIsAlwaysOpen() throws {
        let buffer = try SequentialTilingEngine.allocate(width: 16, height: 16)
        let surface = PixelBufferSurface(buffer)

        #expect(surface.openRows == 0..<16)
        try surface.closeRows(below: 8)
        #expect(surface.openRows == 0..<16, "closing must not narrow a whole-image destination")
        try surface.finish()
    }
}
