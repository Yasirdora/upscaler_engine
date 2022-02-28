import CoreVideo
import Foundation

/// Where a stitcher writes its tiles.
///
/// A single whole-image `CVPixelBuffer` makes peak memory scale with **output image size**: 744.2
/// MiB at 12 MP→4×, 2,976.8 MiB at 48 MP→4×, against a measured floor-device (iPhone 12 Pro)
/// ceiling of 3.07–3.12 GiB. This protocol is the seam that lets the same certified stitchers write
/// somewhere bounded instead.
///
/// **Absolute coordinates throughout.** A surface may hold only a sliding window of rows, but every
/// method takes and returns absolute output-image coordinates. `PixelView.originY` does the
/// translation, which is why the four stitchers are unchanged by it.
///
/// **The contract a caller must keep:** rows are closed in ascending order, and a row is closed
/// only once no future tile can write to it. Writing to a closed row is a programmer error — it is
/// not checked on the hot path, and it cannot happen under the row-major grid the planner emits.
protocol DestinationSurface: AnyObject {
    /// Output image width in pixels.
    var width: Int { get }
    /// Output image height in pixels.
    var height: Int { get }
    /// The absolute rows currently writable.
    var openRows: Range<Int> { get }

    /// Run `body` against the open window, addressed in absolute image coordinates.
    ///
    /// The view's `width`/`height` are the **image's**, not the window's: the stitchers derive
    /// their clip bounds from them (`Stitching.geometry`), and those bounds are absolute. Only
    /// `openRows` are backed by storage — addressing a row outside them is a programmer error
    /// (see the contract below), not something the view's extent can be used to enforce.
    func withOpenWindow<T>(_ body: (PixelView) throws -> T) throws -> T

    /// Declare that no future write will touch any row below `row`. The surface may spool them.
    ///
    /// Monotonic: a `row` at or below the current window top is a no-op, never a rewind.
    func closeRows(below row: Int) throws

    /// Close every remaining row. After this the destination holds the complete image.
    func finish() throws
}

// MARK: - Window sizing

extension DestinationSurface {
    /// How many rows must stay open while a tile row is being written, for a given stitch method.
    ///
    /// This is the residency bound of the whole destination-surface design, and it is **geometry,
    /// not a tunable**. The planner steps by `core = tileSize - 2·margin` in input space
    /// (`TilePlanner.plan`), so:
    ///
    /// - `centerValidRegion` (the contract default) writes only `validRect`, and valid rects are
    ///   disjoint and raster-ordered — one band, `core · scale` rows, is all that is ever open.
    /// - The feathers blend into already-written neighbours, reaching `margin · scale` rows *above*
    ///   their own band and the same below, so they need `(core + 2·margin) · scale`.
    /// - `hardCrop` writes the tile's whole output rect, which the window-slide can place up to
    ///   `margin` above the band; `(tileSize + core) · scale` bounds it. It is the naive control
    ///   (`StitchMethod` doc comment), and it is the most expensive to spool — recorded, not
    ///   optimised.
    ///
    /// Worked, at `tileSize` 512, `margin` 52, scale 4 — so `core` 408:
    ///
    /// | Method | Rows open | 12MP→4× (16128 px wide) | 48MP→4× (32000 px wide) |
    /// |---|---|---|---|
    /// | `centerValidRegion` | 1632 | ≈ 100 MiB | ≈ 199 MiB |
    /// | feathers | 2048 | ≈ 126 MiB | ≈ 250 MiB |
    /// | `hardCrop` | 3680 | ≈ 226 MiB | ≈ 449 MiB |
    ///
    /// against the 744.2 MiB and ~2,930 MiB whole-image surfaces they replace. **Residency scales
    /// with image width, not image area** — that is the whole of the win, and it is also the honest
    /// limit: this design does not make the destination free, it makes it one-dimensional.
    ///
    /// This is independent of the spool **write chunk** (`BandSpoolSurface.writeChunkBytes`) — how
    /// much is handed to one `write(2)` — which is a tunable and has nothing to do with how much
    /// must stay open.
    static func openRowCount(for grid: TileGrid, scaleFactor: Int, method: StitchMethod) -> Int {
        let core = grid.validCoreSize
        let margin = grid.contextMargin
        switch method {
        case .centerValidRegion:
            return core * scaleFactor
        case .linearFeather, .cosineFeather:
            return (core + 2 * margin) * scaleFactor
        case .hardCrop:
            return (grid.tileSize + core) * scaleFactor
        }
    }
}

// MARK: - Whole-image surface

/// The whole-image destination: one `CVPixelBuffer`, every row open forever.
///
/// Kept, not deprecated. It is the control against which the spooled path is shown to change no
/// pixels, and it remains the right destination whenever the image is small enough that a band buys
/// nothing.
final class PixelBufferSurface: DestinationSurface, @unchecked Sendable {
    let buffer: CVPixelBuffer
    let width: Int
    let height: Int

    init(_ buffer: CVPixelBuffer) {
        self.buffer = buffer
        self.width = CVPixelBufferGetWidth(buffer)
        self.height = CVPixelBufferGetHeight(buffer)
    }

    var openRows: Range<Int> { 0..<height }

    func withOpenWindow<T>(_ body: (PixelView) throws -> T) throws -> T {
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let view = PixelView(buffer) else {
            throw TilingError.stitchFailed("could not lock the output buffer")
        }
        return try body(view)
    }

    /// No-op: nothing is spooled, so nothing closes.
    func closeRows(below row: Int) throws {}

    func finish() throws {}
}

// MARK: - Band-spooled surface

/// The destination for large images: a sliding window of rows over a spool file.
///
/// Whole-surface `mmap` gives *no* footprint relief — measured at 746.4 MiB against a 746.0 MiB
/// anonymous control — because dirty mapped pages count exactly like anonymous dirty pages while
/// resident, and `madvise(DONTNEED)` was not honoured as a naive design would assume. So this never
/// maps the image. It holds `openRowCount` rows, writes rows out with plain `write(2)` as they
/// close, and never grows: **residency is bounded by construction, provable without assuming kernel
/// behaviour.**
///
/// The cost is named rather than hidden: one extra full-image write pass at disk speed, and one
/// sequential read pass when the image is encoded. Read-ahead pages on the way back are clean, and
/// clean pages are free under `phys_footprint`, which is the metric the system actually enforces.
///
/// The spool file is a **tightly packed** BGRA raster — `width · 4` bytes per row, no padding — so
/// read-back is a plain sequential scan with no stride arithmetic.
final class BandSpoolSurface: DestinationSurface, @unchecked Sendable {
    let width: Int
    let height: Int
    /// Where the completed raster lands. The caller owns this file's lifetime.
    let spoolURL: URL

    /// Bytes per row in both the window and the spool file. Tight, by design.
    let rowBytes: Int
    /// How many rows the window holds.
    let windowRows: Int

    private let lock = NSLock()
    private let window: UnsafeMutableRawPointer
    private let handle: FileHandle
    /// Absolute image row that the window's first row represents.
    private var windowTop: Int
    /// Rows already written to the spool. Invariant: `spooledRows == windowTop`.
    private var spooledRows: Int
    private var finished: Bool

    /// Largest slice handed to one `write(2)`: 32 MiB, a tunable that bounds syscall size, not
    /// residency (see `openRowCount`).
    static let writeChunkBytes = 32 * 1024 * 1024

    init(width: Int, height: Int, windowRows: Int, spoolURL: URL) throws {
        guard width > 0, height > 0 else { throw TilingError.emptyImage }
        let rows = max(1, min(windowRows, height))
        self.width = width
        self.height = height
        self.spoolURL = spoolURL
        self.rowBytes = width * PixelView.bytesPerPixel
        self.windowRows = rows
        self.windowTop = 0
        self.spooledRows = 0
        self.finished = false

        let byteCount = rows * width * PixelView.bytesPerPixel
        self.window = UnsafeMutableRawPointer.allocate(
            byteCount: byteCount, alignment: MemoryLayout<UInt32>.alignment)
        // Deterministic contents. The pooled `CVPixelBuffer` path this replaces reused storage and
        // could hand a stitcher stale pixels; a zeroed window means any gap a stitch method leaves
        // is reproducible rather than whatever ran last.
        self.window.initializeMemory(as: UInt8.self, repeating: 0, count: byteCount)

        let manager = FileManager.default
        guard manager.createFile(atPath: spoolURL.path, contents: nil) else {
            self.window.deallocate()
            throw TilingError.allocationFailed("could not create spool file at \(spoolURL.path)")
        }
        do {
            self.handle = try FileHandle(forWritingTo: spoolURL)
        } catch {
            self.window.deallocate()
            throw TilingError.allocationFailed("could not open spool file: \(error)")
        }
    }

    deinit {
        try? handle.close()
        window.deallocate()
    }

    var openRows: Range<Int> {
        lock.withLock { windowTop..<min(windowTop + windowRows, height) }
    }

    func withOpenWindow<T>(_ body: (PixelView) throws -> T) throws -> T {
        try lock.withLock {
            // `height` is the image's, not the window's: the stitchers clip against the view's
            // extent in absolute coordinates, so a window-sized height would clip away every row
            // below the first band. `originY` keeps the pointer arithmetic honest; the backed
            // rows remain `[windowTop, windowTop + windowRows)`.
            let view = PixelView(
                base: window, width: width, height: height,
                rowBytes: rowBytes, originY: windowTop)
            return try body(view)
        }
    }

    func closeRows(below row: Int) throws {
        try lock.withLock {
            guard !finished else { return }
            let target = min(row, height)
            guard target > windowTop else { return }  // monotonic: never rewind
            try spool(upTo: target)
        }
    }

    func finish() throws {
        try lock.withLock {
            guard !finished else { return }
            try spool(upTo: height)
            try handle.synchronize()
            try handle.close()
            finished = true
        }
    }

    // MARK: - Spooling

    /// Write rows `[windowTop, target)` to the file, then slide the window down to `target`.
    ///
    /// Caller holds `lock`.
    private func spool(upTo target: Int) throws {
        let count = target - windowTop
        guard count > 0 else { return }
        guard count <= windowRows else {
            // The engine asked to close more than the window ever held, which means a tile wrote
            // outside the open range. Loud rather than silent: this is a geometry bug, and the
            // spooled file would be wrong in a way no pixel comparison could localise.
            throw TilingError.stitchFailed(
                "closeRows(below: \(target)) exceeds the \(windowRows)-row window at top \(windowTop)"
            )
        }

        try writeRows(from: window, count: count)
        spooledRows += count

        // Slide the remainder to the front and zero the rows that scroll in.
        let remaining = windowRows - count
        if remaining > 0 {
            window.copyMemory(
                from: window.advanced(by: count * rowBytes), byteCount: remaining * rowBytes)
        }
        window.advanced(by: remaining * rowBytes)
            .initializeMemory(as: UInt8.self, repeating: 0, count: count * rowBytes)
        windowTop = target
    }

    /// Hand the rows to the file in bounded slices, so one `write` never sees the whole window.
    private func writeRows(from base: UnsafeMutableRawPointer, count: Int) throws {
        let rowsPerChunk = max(1, Self.writeChunkBytes / max(rowBytes, 1))
        var written = 0
        while written < count {
            let rows = min(rowsPerChunk, count - written)
            let slice = Data(
                bytesNoCopy: base.advanced(by: written * rowBytes),
                count: rows * rowBytes, deallocator: .none)
            do {
                try handle.write(contentsOf: slice)
            } catch {
                throw TilingError.stitchFailed("spool write failed at row \(spooledRows + written): \(error)")
            }
            written += rows
        }
    }
}
