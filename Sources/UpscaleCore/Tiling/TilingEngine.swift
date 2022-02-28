import Accelerate
import CoreGraphics
import CoreVideo
import Foundation

/// Typed tiling failures. Nothing is swallowed; every message names what a caller can act on.
public enum TilingError: Error, Equatable {
    case emptyImage
    case unsupportedPixelFormat(String)
    case tileSizeMismatch(expected: Int, got: Int)
    case allocationFailed(String)
    case paddingFailed(String)
    case stitchFailed(String)
}

extension TilingError: CustomStringConvertible {
    public var description: String {
        switch self {
        case .emptyImage: return "the image has no pixels"
        case .unsupportedPixelFormat(let format):
            return "unsupported pixel format '\(format)' — 32BGRA required"
        case .tileSizeMismatch(let expected, let got):
            return "tile size mismatch: the model is fixed-shape at \(expected), asked for \(got)"
        case .allocationFailed(let reason): return "allocation failed: \(reason)"
        case .paddingFailed(let reason): return "context padding failed: \(reason)"
        case .stitchFailed(let reason): return "stitch failed: \(reason)"
        }
    }
}

/// A completed upscale that lives in a file rather than in memory.
///
/// What the band-spooled path returns instead of a whole-image `CVPixelBuffer`. The raster is
/// **tightly packed** BGRA (`bytesPerRow == width * 4`, no padding), written top row first, so a
/// consumer encodes it with a sequential read and no stride arithmetic. Read-ahead pages on that
/// pass are clean, and clean pages do not count against `phys_footprint` — which is the whole
/// reason the image is allowed to be this large.
///
/// The caller owns the file: the engine creates and fills it, and never deletes it.
public struct SpooledImage: Sendable, Equatable {
    public let url: URL
    public let width: Int
    public let height: Int
    public let bytesPerRow: Int

    public init(url: URL, width: Int, height: Int, bytesPerRow: Int) {
        self.url = url
        self.width = width
        self.height = height
        self.bytesPerRow = bytesPerRow
    }

    /// Total bytes the spool file should hold once `finish()` has run.
    public var byteCount: Int { bytesPerRow * height }
}

/// One rung's model artifact and a backend already loaded for it.
///
/// The engine cannot build this itself: `ModelDescriptor.identifier` and `packageURL` name a
/// *converted artifact*, and the 384 and 512 artifacts are different files from the 256 one. Which
/// artifact belongs to which rung is the caller's knowledge, so the caller supplies it and the
/// engine only checks that what came back has the geometry it asked for.
public struct ShapedBackend: Sendable {
    public let model: ModelDescriptor
    public let backend: any InferenceBackend

    public init(model: ModelDescriptor, backend: any InferenceBackend) {
        self.model = model
        self.backend = backend
    }
}

/// Where the engine gets a backend for each rung of the tile-shape ladder.
///
/// **Additive, and deliberately not part of the tiling contract.** `TilingEngine` and
/// `InferenceBackend` stay at version 1; this is a new seam next to them, following the band-spool
/// destination's precedent (an additive *method*, deliberately not a protocol requirement, because
/// adding one would be a contract change).
///
/// **Called at most once per rung per image.** The engine caches what it is handed for the whole
/// image and never asks twice: `CoreMLBackend.load` recompiles on every call, so a per-tile — or
/// even per-step-down — reload would pay a compilation spike exactly when the system is short. The
/// compiled-model cache's identifier + geometry key is what lets two shapes be resident at once
/// without evicting each other; this protocol is what asks for the second one.
public protocol TileShapeBackendProviding: Sendable {
    /// The model and a loaded backend for `shape`. May be expensive; it is called once per rung.
    func backend(for shape: TileShape) async throws -> ShapedBackend
}

/// What a laddered run did — the typed engine outcome, at image scope.
///
/// Presentation belongs to a UI layer: this carries shapes, counts and the ladder's own `Outcome`
/// values, never user-facing prose.
public struct LadderedRun: Sendable, Equatable {
    /// One contiguous run of tiles at one shape. A step down ends a pass and opens the next.
    ///
    /// `tilesPredicted` is smaller than `tilesPlanned` for every pass but the last: a pass that was
    /// abandoned stopped part-way through its own grid, and the tiles it had already produced for
    /// the abandoned *row* are re-run by the next pass (a recompute bounded at one row per step).
    public struct Pass: Sendable, Equatable {
        public let shape: TileShape
        /// Input-space image row this pass's grid starts at. 0 for the first pass.
        public let resumedAtImageRow: Int
        public let tilesPredicted: Int
        public let tilesPlanned: Int
    }

    public let image: SpooledImage
    /// The ladder after `endImage()`, for the caller to carry into the next image.
    public let ladder: TileShapeLadder
    /// The shape the last tile ran at.
    public let shape: TileShape
    /// Every ladder decision this image took, in order: `beginImage`'s first, then one per trigger.
    public let outcomes: [TileShapeLadder.Outcome]
    public let passes: [Pass]
    /// Model calls actually made, retries included.
    public let tilesPredicted: Int
    /// Tiles whose work a step down discarded — bounded at one row per step.
    public let tilesRecomputed: Int

    public var stepDowns: Int { passes.count - 1 }

    /// The job completed in a degraded state — the floor rung held under a trigger. Certification
    /// spec §6's 48 MP rule is satisfied by completing, which this type's existence proves
    /// happened.
    public var isDegraded: Bool { outcomes.contains { $0.isDegraded } }
}

/// The tiling contract, version 1.
///
/// The band-spooled destination is an **additive** method on `SequentialTilingEngine`, deliberately
/// not a protocol requirement: adding one here would be a contract change, and the spool is a
/// memory change.
public protocol TilingEngine: Sendable {
    func upscale(
        _ image: CVPixelBuffer,
        model: ModelDescriptor,
        backend: InferenceBackend,
        method: StitchMethod,
        progress: (@Sendable (Double) -> Void)?
    ) async throws -> CVPixelBuffer
}

/// Sequential tiling engine, v0.
///
/// Sequential on purpose: the budget's concurrency term is **pinned at 1**
/// (`MemoryBudget.concurrency`). A correct seam certification is worth more than an uncertified
/// fast one.
///
/// Unpinning is not free arithmetic either: certification spec §6's budget formula multiplies the
/// per-shape term by concurrency, the measured per-shape figures are *whole-process* peaks, and the
/// fixed and marginal parts cannot be separated from the data as it stands. Whoever raises the term
/// owes that separation first.
///
/// Allocation policy: **one output buffer per image**, taken from a `CVPixelBufferPool` so repeated
/// images of the same size reuse storage, plus **one input tile buffer reused for every tile**. The
/// per-tile output buffer that `InferenceBackend.predict` returns is allocated inside the version 1
/// inference contract, and is a copy this engine cannot remove without changing that contract.
///
/// That pool is **the engine's only cache** — the only thing it holds across images purely because
/// the next image might want it again — so it is the whole of what certification spec §6's "caches
/// purged, model retained" names on this side. See `purgeCaches()`.
///
/// Concurrency: a lock-guarded final class rather than an actor, matching `CoreMLBackend`.
/// `CVPixelBuffer` is `@_nonSendable`, so an actor would impose region-transfer semantics on every
/// caller of `upscale` and make the returned buffer unusable to them.
public final class SequentialTilingEngine: TilingEngine, @unchecked Sendable {
    /// Context margin in input pixels, derived by measurement (`ContextMargin`), never copied.
    public let contextMargin: Int

    private let lock = NSLock()
    private var pool: CVPixelBufferPool?
    private var poolDimensions: (width: Int, height: Int)?
    private var allocationCount = 0
    private var purgeCount = 0

    public init(contextMargin: Int) {
        self.contextMargin = contextMargin
    }

    /// Number of output buffers this engine has allocated. Test instrumentation for the
    /// single-allocation-per-image requirement; not part of the tiling contract.
    var outputAllocations: Int { lock.withLock { allocationCount } }

    /// Times `purgeCaches()` has run, whether or not it found anything to drop.
    ///
    /// **A call count, never a memory figure.** It says the engine reacted to a pressure event; it
    /// says nothing about bytes, because the byte effect is `unavailable` until device runs measure
    /// it (see `purgeCaches()`). Test instrumentation, like `outputAllocations`; not part of the
    /// tiling contract.
    var cachePurges: Int { lock.withLock { purgeCount } }

    /// Identity of the pooled cache the engine is holding, or `nil` when it holds none.
    ///
    /// The observable that separates "the pool was dropped" from "the pool was kept and quietly
    /// reused across images". **Meaningful only while a pool is alive:** comparing an identity taken
    /// before a purge with one taken after is not evidence of anything, because CoreVideo recycles
    /// the freed address (measured — see `purgeCaches()`). Internal for the same reason
    /// `outputAllocations` is — instrumentation, not contract.
    var pooledCacheIdentity: ObjectIdentifier? {
        lock.withLock { pool.map { ObjectIdentifier($0) } }
    }

    public func upscale(
        _ image: CVPixelBuffer,
        model: ModelDescriptor,
        backend: InferenceBackend,
        method: StitchMethod,
        progress: (@Sendable (Double) -> Void)?
    ) async throws -> CVPixelBuffer {
        let (grid, width, height) = try plan(image, model: model)
        let scale = model.scaleFactor
        let output = try allocateOutput(width: width * scale, height: height * scale)
        try await run(
            image, grid: grid, model: model, backend: backend, method: method,
            destination: PixelBufferSurface(output), progress: progress)
        return output
    }

    /// Upscale one region of an image — the preview path.
    ///
    /// **The model is passed in, never chosen here.** Scale, tile geometry and margin must be the
    /// ones the *whole image* would use. When a caller selects the model from the input's size, a
    /// preview crop is small: a 12 MP photograph might route to a ×2 model while a 512×512 crop of
    /// it, judged on its own, would route to a ×4 one. A preview rendered by a different model at a
    /// different scale than the final render is not a preview.
    ///
    /// Tiles are read from the **full image** at their absolute coordinates, so the region's edges
    /// draw real neighbouring pixels rather than reflections. Combined with the grid alignment
    /// `TilePlanner.plan(…croppingTo:)` preserves, the result is bit-identical to the corresponding
    /// region of a whole-image run — which is the acceptance criterion, and is asserted end to end.
    ///
    /// - Parameters:
    ///   - image: the FULL source image, not a pre-cropped one.
    ///   - inputRect: the region of interest, in input pixels.
    ///   - model: the descriptor the whole image would use.
    ///   - backend: a loaded inference backend.
    ///   - method: stitch method; the contract default is `centerValidRegion`.
    ///   - progress: fraction of tiles completed, as elsewhere.
    /// - Returns: the upscaled region, exactly `inputRect` × `scaleFactor`.
    /// - Throws: `TilingError.emptyImage` when the region lies wholly outside the image or is
    ///   smaller than a pixel, `TilingError.unsupportedPixelFormat` for a non-32BGRA source, and
    ///   whatever the backend throws from `predict`.
    public func upscale(
        _ image: CVPixelBuffer,
        in inputRect: CGRect,
        model: ModelDescriptor,
        backend: InferenceBackend,
        method: StitchMethod,
        progress: (@Sendable (Double) -> Void)?
    ) async throws -> CVPixelBuffer {
        let (_, width, height) = try plan(image, model: model)
        let clamped = inputRect.intersection(
            CGRect(x: 0, y: 0, width: CGFloat(width), height: CGFloat(height)))
        guard !clamped.isNull, clamped.width >= 1, clamped.height >= 1 else {
            throw TilingError.emptyImage
        }

        let planner = TilePlanner(scaleFactor: model.scaleFactor)
        let cropped = planner.plan(
            imageSize: CGSize(width: width, height: height), tileSize: model.inputTileSize,
            contextMargin: contextMargin, croppingTo: clamped)
        guard !cropped.grid.tiles.isEmpty else { throw TilingError.emptyImage }

        // Allocated to `covered`, not to the request: kept tiles overhang the crop, and sizing to
        // the request would put the stitcher out of bounds. Trimmed afterwards, so the certified
        // stitcher runs exactly as it does for a whole image.
        let canvas = try allocateOutput(
            width: Int(cropped.covered.width), height: Int(cropped.covered.height))

        // Both rects arrive absolute, so every tile is shifted into the canvas's space TOGETHER —
        // `paddedInputRect` in input pixels, `validRect` in output pixels, by the same amount.
        // Shifting only one is what an earlier draft did, and it placed every tile 192 px out.
        // The source is still read from the ORIGINAL image at absolute coordinates, which is what
        // gives the region real neighbouring context instead of a reflection at its edge.
        let dx = cropped.originInFullOutput.x, dy = cropped.originInFullOutput.y
        let scale = CGFloat(model.scaleFactor)
        let shifted = TileGrid(
            tileSize: cropped.grid.tileSize, contextMargin: cropped.grid.contextMargin,
            tiles: cropped.grid.tiles.map {
                TileRegion(
                    validRect: $0.validRect.offsetBy(dx: -dx, dy: -dy),
                    paddedInputRect: $0.paddedInputRect.offsetBy(dx: -dx / scale, dy: -dy / scale))
            })
        try await run(
            image, grid: shifted, model: model, backend: backend, method: method,
            destination: PixelBufferSurface(canvas), progress: progress,
            sourceOffset: CGPoint(x: dx / scale, y: dy / scale))
        return try Self.crop(canvas, to: cropped.requested)
    }

    /// Upscale into a **spool file** instead of a whole-image buffer.
    ///
    /// The destination-surface path. Peak memory stops scaling with output image size: the engine
    /// keeps only the rows a tile row can still write (`DestinationSurface.openRowCount`) and
    /// spools the rest, so a 48 MP job costs roughly a fifth of what the whole-image buffer cost.
    /// See `OutputSurface.swift` for the measured window sizes and the honest limit — residency
    /// scales with image *width*, not area.
    ///
    /// Additive on purpose: `upscale(_:model:backend:method:progress:)` above is untouched, so the
    /// CLI and the existing tiling tests keep the certified in-memory path as their control, and
    /// the two paths can be compared.
    ///
    /// The caller owns `url`: this creates and fills it, and never deletes it.
    public func upscale(
        _ image: CVPixelBuffer,
        model: ModelDescriptor,
        backend: InferenceBackend,
        method: StitchMethod,
        spoolingTo url: URL,
        progress: (@Sendable (Double) -> Void)?
    ) async throws -> SpooledImage {
        let (grid, width, height) = try plan(image, model: model)
        let scale = model.scaleFactor
        let outputWidth = width * scale
        let outputHeight = height * scale

        let surface = try BandSpoolSurface(
            width: outputWidth, height: outputHeight,
            windowRows: BandSpoolSurface.openRowCount(
                for: grid, scaleFactor: scale, method: method),
            spoolURL: url)
        try await run(
            image, grid: grid, model: model, backend: backend, method: method,
            destination: surface, progress: progress)
        try surface.finish()

        return SpooledImage(
            url: url, width: outputWidth, height: outputHeight, bytesPerRow: surface.rowBytes)
    }

    // MARK: - The laddered retry path

    /// Upscale into a spool file **with the tile-shape ladder live**.
    ///
    /// This is the path certification spec §5.5 certifies ("forced-failure tests must demonstrate
    /// 512→384→256 retry **and successful completion**") and §10's row requires ("Memory pressure
    /// mid-job → adaptive tile fallback → retry → continue; whole-job failure is a defect").
    ///
    /// ## What a step down does
    ///
    /// At every tile boundary the engine consumes whatever the pressure latch holds
    /// (`MemoryPressureMonitor.respond(ladder:plan:)`) and, if a tile's own prediction fails, hands
    /// the ladder a `.tileFailure` trigger. Either way, when the ladder moves a rung the engine:
    ///
    /// 1. **abandons the tile it was about to run** at the current shape — never a tile in flight,
    ///    which is why `finishesCurrentTile` is invariantly true;
    /// 2. **re-plans from the top of that tile's row** through
    ///    `TilePlanning.plan(imageSize:tileSize:contextMargin:resumingAtImageRow:)`;
    /// 3. **re-runs the tiles to its left in that row** at the smaller shape.
    ///
    /// Row, not tile, is the granularity: a smaller core moves column boundaries as well as row
    /// boundaries, so "the rest of this row at a new tile size" names no well-defined set of tiles.
    /// The cost is bounded and is compute, not correctness — at most one row of already computed
    /// tiles is recomputed per step down, and `LadderedRun.tilesRecomputed` reports it.
    ///
    /// **Rows already assembled are never re-gridded.** They may already be spooled and closed
    /// (`BandSpoolSurface.closeRows`), which is exactly why a step *up* is barred mid-image while a
    /// step down is legal.
    ///
    /// ## Why the fixed `windowRows` is safe across a step down
    ///
    /// `BandSpoolSurface.windowRows` is `let`, sized once from the **initial** grid, so a mid-image
    /// step down frees no destination memory. For correctness the direction is the benign one, and
    /// it is benign for a reason rather than by luck. `DestinationSurface.openRowCount` is monotone
    /// in tile size for every stitch method — `core · scale`, `(core + 2·margin) · scale` and
    /// `(tileSize + core) · scale` all shrink as `tileSize` shrinks with `margin` fixed — so a
    /// lower rung needs *fewer* open rows than the window already holds. A step **up** mid-image
    /// would need more, which is one more reason it is barred.
    ///
    /// Placement holds too. Let `v` be the resumed row's output origin. The window top is wherever
    /// the previous row's close left it, which is the *old* shape's `lowestWrittenRow` for that
    /// row; the resumed row's `lowestWrittenRow` at the new shape is never above it. For
    /// `centerValidRegion` both are `v`; for the feathers both are `v − margin · scale`; for
    /// `hardCrop` it is the window origin, and `windowOrigin` is monotone — the unclamped origin is
    /// `v + validExtent/2 − tileSize/2`, whose shape-dependence cancels to `v − margin` whenever
    /// the row is a full core, is larger when the row is short, and both clamps preserve the
    /// ordering because `imageExtent − tileSize` grows as `tileSize` shrinks. So no write lands
    /// above the window, and `BandSpoolSurface.closeRows` throws loudly if that reasoning is ever
    /// wrong.
    ///
    /// ## Two shapes live
    ///
    /// `live` holds one `ShapedBackend` and one input tile buffer per rung the image touched, for
    /// the whole image. Nothing is released on a step down: there is deliberately no
    /// teardown/reload path, because `CoreMLBackend.load` recompiles on every call and dumping what
    /// you hold under pressure can cost more than it frees. The engine-side cost is exact and small
    /// — one BGRA tile buffer per rung, 1 048 576 B at 512, 589 824 B at 384, 262 144 B at 256, so
    /// 1 900 544 B if an image touches all three. The *model-side* cost of a second resident shape
    /// is **unavailable**: the measured table is whole-process single-shape peaks, and its fixed
    /// and marginal parts cannot be separated. Nothing is retained past the image — `live` is a
    /// local.
    ///
    /// ## Cost
    ///
    /// No polling, no timers, no re-budget: `plan` is computed once by the caller and used only to
    /// report what a move costs. `consumePending()` is one lock acquisition and no allocation, so
    /// the per-tile-boundary consult is a read-and-clear, not per-tile churn. The ladder is asked
    /// once per image (`beginImage`) plus once per trigger. Each rung's model is fetched once. The
    /// only new full-image work a step down causes is re-running one tile row, which is the point.
    ///
    /// The caller owns `url`: this creates and fills it, and never deletes it. The ladder is taken
    /// **by value** and handed back in the result, so a job that throws leaves the caller's ladder
    /// exactly as it was rather than stranded mid-image.
    public func upscale(
        _ image: CVPixelBuffer,
        shapes provider: TileShapeBackendProviding,
        ladder startingLadder: TileShapeLadder,
        budget plan: MemoryBudget.Plan,
        pressure monitor: MemoryPressureMonitor?,
        method: StitchMethod,
        spoolingTo url: URL,
        progress: (@Sendable (Double) -> Void)?
    ) async throws -> LadderedRun {
        let width = CVPixelBufferGetWidth(image)
        let height = CVPixelBufferGetHeight(image)
        guard width > 0, height > 0 else { throw TilingError.emptyImage }
        let format = CVPixelBufferGetPixelFormatType(image)
        guard format == kCVPixelFormatType_32BGRA else {
            throw TilingError.unsupportedPixelFormat(Self.fourCharacterCode(format))
        }

        var ladder = startingLadder
        var outcomes = [ladder.beginImage(plan)]
        var live: [TileShape: LiveShape] = [:]
        var shape = ladder.shape
        var current = try await Self.acquire(shape, from: provider, into: &live)

        // The scale factor is the *initial* rung's and is never re-read: it fixes the output
        // geometry, and the destination is already allocated against it. A rung whose artifact
        // disagrees is caught per tile by the predicted-width guard in `render`, by name.
        let scale = current.model.scaleFactor
        let planner = TilePlanner(scaleFactor: scale)
        let imageSize = CGSize(width: width, height: height)
        var grid = planner.plan(
            imageSize: imageSize, tileSize: shape.inputTileSize, contextMargin: contextMargin)
        guard !grid.tiles.isEmpty else { throw TilingError.emptyImage }

        let outputHeight = height * scale
        let surface = try BandSpoolSurface(
            width: width * scale, height: outputHeight,
            windowRows: BandSpoolSurface.openRowCount(
                for: grid, scaleFactor: scale, method: method),
            spoolURL: url)
        let stitcher = Stitcher.makeSurfaceStitcher(method)

        var passes: [LadderedRun.Pass] = []
        var resumeRow = 0
        var predicted = 0
        var recomputed = 0
        var predictedThisPass = 0
        var index = 0

        while index < grid.tiles.count {
            try Task.checkCancellation()

            // The tile boundary. `respond` is `nil` at almost every one of them, and when it is the
            // ladder is not touched at all — no trigger, no dwell reset, no arithmetic.
            var abandon = false
            if let monitor, let response = monitor.respond(ladder: &ladder, plan: plan) {
                if let outcome = response.outcome { outcomes.append(outcome) }
                // Certification spec §6's "caches purged, model retained", actuated on the same
                // event, at the same boundary, through the same seam — not a second one. It is
                // additional to the rung drop rather than instead of it, and it runs at the floor
                // rung too, where `steppedDown` is false and there is nothing left to drop.
                if response.purgesCaches { purgeCaches() }
                // `steppedDown` is false at the floor rung, which is intended and not a gap: 256 is
                // the floor, so the job continues degraded rather than stalling.
                abandon = response.steppedDown
            }

            if !abandon {
                do {
                    try await Self.render(
                        grid.tiles[index], of: image, grid: grid, into: surface,
                        backend: current.backend, stitcher: stitcher, inputTile: current.tile,
                        tileSize: shape.inputTileSize, scaleFactor: scale)
                    predicted += 1
                    predictedThisPass += 1
                } catch is CancellationError {
                    // Cancellation is not a ladder trigger. Structured concurrency only — a
                    // cancelled job stops, it does not step down and retry (certification spec
                    // §10).
                    throw CancellationError()
                } catch {
                    let outcome = ladder.respond(
                        to: .tileFailure(String(describing: error)), plan: plan)
                    outcomes.append(outcome)
                    // At the floor rung there is no lower shape to retry at, so the error is the
                    // caller's. Bounded by construction: at most one step down per rung, two in all.
                    guard outcome.movedRung else { throw error }
                    abandon = true
                }
            }

            if abandon {
                passes.append(
                    LadderedRun.Pass(
                        shape: shape, resumedAtImageRow: resumeRow,
                        tilesPredicted: predictedThisPass, tilesPlanned: grid.tiles.count))
                recomputed += index % Self.columnCount(of: grid)
                resumeRow = Int(grid.tiles[index].validRect.minY.rounded()) / scale
                shape = ladder.shape
                current = try await Self.acquire(shape, from: provider, into: &live)
                grid = planner.plan(
                    imageSize: imageSize, tileSize: shape.inputTileSize,
                    contextMargin: contextMargin, resumingAtImageRow: resumeRow)
                // The runtime half of the `windowRows` argument in this method's documentation, and
                // the only check the retry path adds. `DestinationSurface` documents a write to a
                // closed row as an unchecked programmer error — true and affordable under the
                // row-major grid the planner emits, but a *re-grid* is the one thing that could
                // violate it, and the failure would be a write past the window rather than an
                // error. Checked once per step down (at most twice per image), never per tile.
                let open = surface.openRows
                guard
                    Self.resumedRowIsWritable(
                        grid, open: open, outputHeight: outputHeight, scaleFactor: scale,
                        method: method)
                else {
                    throw TilingError.stitchFailed(
                        "a step down to \(shape.inputTileSize) cannot resume at image row "
                            + "\(resumeRow): the destination has rows \(open) open")
                }
                predictedThisPass = 0
                index = 0
                continue
            }

            let region = grid.tiles[index]
            let next = index + 1
            if next < grid.tiles.count, grid.tiles[next].validRect.minY > region.validRect.minY {
                try surface.closeRows(
                    below: Self.lowestWrittenRow(
                        grid.tiles[next], grid: grid, scaleFactor: scale, method: method))
            }
            // Progress spans the rows this pass owns, so it is monotone within a pass and ends at
            // exactly 1.0. A step down replays a row, so it steps **back** to the resumed row's
            // fraction — honest rather than smoothed; presentation is the UI layer's problem.
            let base = Double(resumeRow * scale) / Double(outputHeight)
            progress?(base + (1 - base) * Double(next) / Double(grid.tiles.count))
            index = next
        }

        try Task.checkCancellation()
        passes.append(
            LadderedRun.Pass(
                shape: shape, resumedAtImageRow: resumeRow, tilesPredicted: predictedThisPass,
                tilesPlanned: grid.tiles.count))
        try surface.finish()
        ladder.endImage()

        return LadderedRun(
            image: SpooledImage(
                url: url, width: width * scale, height: outputHeight,
                bytesPerRow: surface.rowBytes),
            ladder: ladder, shape: shape, outcomes: outcomes, passes: passes,
            tilesPredicted: predicted, tilesRecomputed: recomputed)
    }

    /// One rung's live resources, held for the whole image and released with it.
    private struct LiveShape {
        let model: ModelDescriptor
        let backend: any InferenceBackend
        let tile: CVPixelBuffer
    }

    /// The rung's backend, fetched **once** and kept. Never a reload, never a second compile.
    private static func acquire(
        _ shape: TileShape, from provider: TileShapeBackendProviding,
        into live: inout [TileShape: LiveShape]
    ) async throws -> LiveShape {
        if let existing = live[shape] { return existing }
        let supplied = try await provider.backend(for: shape)
        guard supplied.model.inputTileSize == shape.inputTileSize else {
            throw TilingError.tileSizeMismatch(
                expected: shape.inputTileSize, got: supplied.model.inputTileSize)
        }
        let entry = LiveShape(
            model: supplied.model, backend: supplied.backend,
            tile: try allocate(width: shape.inputTileSize, height: shape.inputTileSize))
        live[shape] = entry
        return entry
    }

    /// Whether a re-gridded remainder's first row still lands inside what the destination has open.
    ///
    /// **Both ends matter, and neither is checked anywhere else.** `DestinationSurface` documents a
    /// write outside the open rows as an unchecked programmer error, which is affordable because it
    /// "cannot happen under the row-major grid the planner emits" — a re-grid is precisely the case
    /// that assumption did not cover. A remainder starting *above* the window would write into rows
    /// already spooled; one starting *below* it would write past the window buffer's end. Under
    /// mutation testing both mis-resumes segfault without this check, so it converts memory
    /// corruption into a typed error.
    ///
    /// The first row is sufficient. Every later row of the pass is reached only through
    /// `closeRows`, which throws when a slide exceeds the window, and no lower rung needs a wider
    /// window than the initial grid sized (`DestinationSurface.openRowCount` is monotone in tile
    /// size at a fixed margin).
    ///
    /// `false` for an empty remainder too: a step down that planned nothing would otherwise finish
    /// a truncated image silently, and certification spec §6 rules that out absolutely ("never
    /// crash, never corrupt output"). It is unreachable by construction — the resume row is an
    /// existing tile's row origin — and folded in here rather than left as a second branch nothing
    /// can reach.
    static func resumedRowIsWritable(
        _ grid: TileGrid, open: Range<Int>, outputHeight: Int, scaleFactor: Int,
        method: StitchMethod
    ) -> Bool {
        guard let first = grid.tiles.first else { return false }
        let lowest = lowestWrittenRow(
            first, grid: grid, scaleFactor: scaleFactor, method: method)
        // Clamped to the image because the stitchers clip their writes to it (`Stitching.geometry`),
        // which is what lets a `hardCrop` window hang past the last row of a sub-tile image.
        let highest = min(
            outputHeight,
            highestWrittenRow(first, grid: grid, scaleFactor: scaleFactor, method: method))
        return lowest >= open.lowerBound && highest <= open.upperBound
    }

    /// Tiles per row in a row-major grid — the planner emits a constant column count, so the first
    /// row's length answers for every row, and `index % columns` is a tile's place within its row.
    static func columnCount(of grid: TileGrid) -> Int {
        guard let first = grid.tiles.first else { return 1 }
        return max(1, grid.tiles.prefix { $0.validRect.minY == first.validRect.minY }.count)
    }

    // MARK: - Shared tiling core

    /// Validate the image and plan the grid. Shared by both destinations so neither can drift.
    private func plan(
        _ image: CVPixelBuffer, model: ModelDescriptor
    ) throws -> (grid: TileGrid, width: Int, height: Int) {
        let width = CVPixelBufferGetWidth(image)
        let height = CVPixelBufferGetHeight(image)
        guard width > 0, height > 0 else { throw TilingError.emptyImage }

        let format = CVPixelBufferGetPixelFormatType(image)
        guard format == kCVPixelFormatType_32BGRA else {
            throw TilingError.unsupportedPixelFormat(Self.fourCharacterCode(format))
        }

        let planner = TilePlanner(scaleFactor: model.scaleFactor)
        let grid = planner.plan(
            imageSize: CGSize(width: width, height: height),
            tileSize: model.inputTileSize,
            contextMargin: contextMargin
        )
        guard !grid.tiles.isEmpty else { throw TilingError.emptyImage }
        return (grid, width, height)
    }

    /// The tiling loop itself, against whichever destination it is handed.
    ///
    /// Rows are closed at tile-row boundaries, and only down to the lowest row the *next* tile row
    /// can still touch (`lowestWrittenRow`). For a whole-image destination `closeRows` is a no-op,
    /// so this is exactly the whole-image loop with one extra call that does nothing.
    private func run(
        _ image: CVPixelBuffer,
        grid: TileGrid,
        model: ModelDescriptor,
        backend: InferenceBackend,
        method: StitchMethod,
        destination: DestinationSurface,
        progress: (@Sendable (Double) -> Void)?,
        /// Added back to each tile's input rect before reading. The preview path shifts its grid
        /// into a small canvas's coordinates, but the SOURCE is still the full image — so reads
        /// must be un-shifted. Zero for every whole-image caller, whose grid is already absolute.
        sourceOffset: CGPoint = .zero
    ) async throws {
        let tileSize = model.inputTileSize
        let scale = model.scaleFactor
        let inputTile = try Self.allocate(width: tileSize, height: tileSize)
        let stitcher = Stitcher.makeSurfaceStitcher(method)

        for (index, region) in grid.tiles.enumerated() {
            try Task.checkCancellation()
            try await Self.render(
                region, of: image, grid: grid, into: destination, backend: backend,
                stitcher: stitcher, inputTile: inputTile, tileSize: tileSize, scaleFactor: scale,
                sourceOffset: sourceOffset)

            // A tile row is complete when the next tile starts a lower row. Every tile in a row
            // shares its y-geometry (`TilePlanner.windowOrigin` depends only on the row), so one
            // lookup answers for the whole row.
            let next = index + 1
            if next < grid.tiles.count, grid.tiles[next].validRect.minY > region.validRect.minY {
                try destination.closeRows(
                    below: Self.lowestWrittenRow(
                        grid.tiles[next], grid: grid, scaleFactor: scale, method: method))
            }
            progress?(Double(index + 1) / Double(grid.tiles.count))
        }
        try Task.checkCancellation()
    }

    /// Pad, infer, check the shape, stitch — one tile, against whichever destination it is handed.
    ///
    /// Extracted so the fixed-shape loop above and the laddered loop run **the same** per-tile body:
    /// two copies of the predicted-width guard is exactly the kind of drift the certification could
    /// not localise later.
    private static func render(
        _ region: TileRegion,
        of image: CVPixelBuffer,
        grid: TileGrid,
        into destination: DestinationSurface,
        backend: InferenceBackend,
        stitcher: SurfaceStitching,
        inputTile: CVPixelBuffer,
        tileSize: Int,
        scaleFactor scale: Int,
        sourceOffset: CGPoint = .zero
    ) async throws {
        try fillPaddedTile(
            source: image,
            window: IntRect(
                region.paddedInputRect.offsetBy(dx: sourceOffset.x, dy: sourceOffset.y)),
            into: inputTile)
        let predicted = try await backend.predict(inputTile)
        guard CVPixelBufferGetWidth(predicted) == tileSize * scale else {
            throw TilingError.tileSizeMismatch(
                expected: tileSize * scale, got: CVPixelBufferGetWidth(predicted))
        }
        try stitcher.write(
            tile: predicted, region: region, grid: grid, scaleFactor: scale, into: destination)
    }

    /// The lowest output row a tile may write, per stitch method.
    ///
    /// This is the mirror of `DestinationSurface.openRowCount`: that says how many rows must stay
    /// open, this says where the open range starts. They must agree, and
    /// `BandSpoolSurface.closeRows` throws loudly if they ever do not.
    static func lowestWrittenRow(
        _ region: TileRegion, grid: TileGrid, scaleFactor: Int, method: StitchMethod
    ) -> Int {
        let validY = Int(region.validRect.minY.rounded())
        switch method {
        case .centerValidRegion:
            return validY
        case .linearFeather, .cosineFeather:
            // The feathers reach a margin's depth above their own band, into rows already written.
            return max(0, validY - grid.contextMargin * scaleFactor)
        case .hardCrop:
            // Writes the tile's whole output rect, which the window-slide can place above the band.
            return max(0, Int(region.paddedInputRect.minY.rounded()) * scaleFactor)
        }
    }

    /// One past the last output row a tile may write — the exclusive mirror of `lowestWrittenRow`,
    /// derived from the same per-method reach. Used only by `resumedRowIsWritable`, once per step
    /// down; the tiling loop itself never needs it, because a window only ever slides downward.
    static func highestWrittenRow(
        _ region: TileRegion, grid: TileGrid, scaleFactor: Int, method: StitchMethod
    ) -> Int {
        let validMaxY = Int(region.validRect.maxY.rounded())
        switch method {
        case .centerValidRegion:
            return validMaxY
        case .linearFeather, .cosineFeather:
            return validMaxY + grid.contextMargin * scaleFactor
        case .hardCrop:
            return Int(region.paddedInputRect.maxY.rounded()) * scaleFactor
        }
    }

    // MARK: - Cache purge under memory pressure (certification spec §6)

    /// Drop the pooled output cache — certification spec §6's *"caches purged, model retained"*.
    ///
    /// ## What is purged, exactly
    ///
    /// The `CVPixelBufferPool` and nothing else. It exists so that repeated images of the same size
    /// reuse one whole-image output buffer, which means that between images it holds a recycled
    /// buffer the size of the last output — 744.2 MiB at 12 MP, 2 976.8 MiB at 48 MP — for no
    /// reason other than that the *next* image might be the same size. That is the definition of a
    /// cache, and it is the only one this engine has.
    ///
    /// Two steps, because they cover different cases and both are one call:
    ///
    /// 1. `CVPixelBufferPoolFlush(_:.excessBuffers)` frees the pool's unused buffers **now**,
    ///    regardless of their age and regardless of who else holds the pool. This is Apple's
    ///    documented purge primitive, and it is kept as defense-in-depth rather than because step 2 is
    ///    known to be insufficient — see the measurement below.
    /// 2. Clearing `pool`/`poolDimensions` means the next image builds a fresh pool sized for
    ///    itself, instead of the engine silently re-warming the cache the system just asked it to give
    ///    up.
    ///
    /// **Measured on the host (macOS 14, arm64), and it corrected a wrong assumption:** a
    /// `CVPixelBuffer` vended from a pool does **not** keep that pool alive. Drop the last
    /// reference while a vended buffer is still in use and the pool is deallocated immediately —
    /// the next pool created with the same attributes lands at the freed address, which two
    /// simultaneously-live pools never do (they are not interned; that was measured too). So step 2
    /// alone is what ends the cache's lifetime here, and step 1 is insurance against an
    /// undocumented CoreVideo behaviour changing under us. The same measurement is why no test
    /// asserts pool *identity* across a purge: the allocator recycles addresses, so identity is
    /// evidence in neither direction.
    ///
    /// ## What is not purged, and must never be
    ///
    /// No model, no backend, no compiled artifact — nothing in this method names one.
    /// `CoreMLBackend.load` calls `MLModel.compileModel` on **every** call, so a teardown/reload
    /// would pay a compilation spike exactly when the system is short of memory. No teardown/reload
    /// path exists, and `MemoryPressureResponse.retainsModel` is invariantly `true`.
    ///
    /// Nor the laddered run's live per-rung input tile buffers (`LiveShape`): those are the current
    /// image's working set rather than a cache, they total 1 900 544 B if an image touches all
    /// three rungs, and freeing them would force an immediate reallocation of the same bytes.
    ///
    /// ## The job is never abandoned
    ///
    /// This is a *policy change*, which is what Apple's memory guidance recommends, taken at a tile
    /// boundary and **additional** to the rung drop the same event triggers — never instead of it.
    /// It runs at the floor rung too, where there is no rung left to drop.
    ///
    /// ## Memory effect: `unavailable`
    ///
    /// Nothing here claims the purge lowers footprint, and nothing here reports bytes. Apple
    /// cautions that dumping caches can cost more than it frees, because the compressor may have to
    /// decompress what is dropped first (WWDC18 session 416, "iOS Memory Deep Dive"). The
    /// hypothesis is that a pool of dirty IOSurface-backed buffers is the other case, where release
    /// genuinely returns footprint; it is **unproven**, and only device measurement can settle it.
    ///
    /// Cheap and idempotent: one lock acquisition, no allocation, no I/O, nothing scheduled. Safe
    /// when nothing is held, and safe both between images and at a tile boundary.
    ///
    /// Additive on the concrete engine rather than on the `TilingEngine` protocol, following the
    /// band spool's precedent: the tiling contract is v1 and a memory change does not move it.
    ///
    /// - Returns: whether a pooled cache was actually held and dropped.
    @discardableResult
    public func purgeCaches() -> Bool {
        lock.withLock {
            purgeCount += 1
            guard let held = pool else { return false }
            CVPixelBufferPoolFlush(held, .excessBuffers)
            pool = nil
            poolDimensions = nil
            return true
        }
    }

    // MARK: - Allocation

    /// Copy a sub-rect out of a buffer, row by row.
    ///
    /// A fresh allocation rather than a view: `CVPixelBuffer` has no sub-rect type, and handing
    /// back the oversized canvas with a rect attached would put the trim in every caller instead
    /// of in one place. Rows are copied with the stride read from each buffer — a preview canvas
    /// and its trim have different widths and therefore different padding, and flat indexing would
    /// read the padding as pixels.
    static func crop(_ source: CVPixelBuffer, to rect: CGRect) throws -> CVPixelBuffer {
        let x = Int(rect.origin.x), y = Int(rect.origin.y)
        let width = Int(rect.width), height = Int(rect.height)
        guard width > 0, height > 0,
            x >= 0, y >= 0,
            x + width <= CVPixelBufferGetWidth(source),
            y + height <= CVPixelBufferGetHeight(source)
        else { throw TilingError.emptyImage }

        var out: CVPixelBuffer?
        let attributes: [CFString: Any] = [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary]
        guard
            CVPixelBufferCreate(
                kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA,
                attributes as CFDictionary, &out) == kCVReturnSuccess,
            let destination = out
        else { throw TilingError.allocationFailed("preview crop \(width)x\(height)") }

        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(destination, [])
        defer {
            CVPixelBufferUnlockBaseAddress(destination, [])
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
        }
        guard let from = CVPixelBufferGetBaseAddress(source),
            let to = CVPixelBufferGetBaseAddress(destination)
        else { throw TilingError.allocationFailed("preview crop base address") }

        let sourceStride = CVPixelBufferGetBytesPerRow(source)
        let destinationStride = CVPixelBufferGetBytesPerRow(destination)
        for row in 0..<height {
            memcpy(
                to.advanced(by: row * destinationStride),
                from.advanced(by: (y + row) * sourceStride + x * 4),
                width * 4)
        }
        return destination
    }

    private func allocateOutput(width: Int, height: Int) throws -> CVPixelBuffer {
        try lock.withLock {
            if poolDimensions == nil || poolDimensions?.width != width || poolDimensions?.height != height {
                pool = try Self.makePool(width: width, height: height)
                poolDimensions = (width, height)
            }
            guard let pool else { throw TilingError.allocationFailed("no pixel buffer pool") }
            var buffer: CVPixelBuffer?
            let status = CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &buffer)
            guard status == kCVReturnSuccess, let buffer else {
                throw TilingError.allocationFailed("CVPixelBufferPoolCreatePixelBuffer returned \(status)")
            }
            allocationCount += 1
            return buffer
        }
    }

    /// IOSurface-backed so the buffer can be handed to Core ML, Metal or the display without a
    /// copy (CVPixelBuffer.h — `kCVPixelBufferIOSurfacePropertiesKey`).
    private static func makePool(width: Int, height: Int) throws -> CVPixelBufferPool {
        let attributes: [CFString: Any] = [
            kCVPixelBufferPixelFormatTypeKey: Int(kCVPixelFormatType_32BGRA),
            kCVPixelBufferWidthKey: width,
            kCVPixelBufferHeightKey: height,
            kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary,
        ]
        var pool: CVPixelBufferPool?
        let status = CVPixelBufferPoolCreate(
            kCFAllocatorDefault, nil, attributes as CFDictionary, &pool)
        guard status == kCVReturnSuccess, let pool else {
            throw TilingError.allocationFailed("CVPixelBufferPoolCreate returned \(status)")
        }
        return pool
    }

    static func allocate(width: Int, height: Int) throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        let attributes: [CFString: Any] = [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA,
            attributes as CFDictionary, &buffer)
        guard status == kCVReturnSuccess, let buffer else {
            throw TilingError.allocationFailed("CVPixelBufferCreate(\(width)x\(height)) returned \(status)")
        }
        return buffer
    }

    // MARK: - Context padding

    /// Fill a fixed-shape tile from `window`, reflecting where the window hangs outside the image.
    ///
    /// Reflection is the specified padding. Where the image is thinner than the pad — only possible
    /// for images smaller than one tile — one reflection cannot fill the strip, and the outermost
    /// line is replicated for the remainder. That is a loop over *lines*, each line a single vImage
    /// call; no pixel is touched by Swift.
    static func fillPaddedTile(source: CVPixelBuffer, window: IntRect, into tile: CVPixelBuffer) throws {
        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(tile, [])
        defer {
            CVPixelBufferUnlockBaseAddress(tile, [])
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
        }
        guard let sourceView = PixelView(source), let tileView = PixelView(tile) else {
            throw TilingError.paddingFailed("could not lock the source or tile buffer")
        }

        let bounds = IntRect(x: 0, y: 0, width: sourceView.width, height: sourceView.height)
        let real = window.intersection(bounds)
        guard !real.isEmpty else {
            throw TilingError.paddingFailed("window \(window) lies entirely outside the image")
        }

        // 1. The real content, copied once into its place in the tile.
        let insetX = real.x - window.x
        let insetY = real.y - window.y
        var sourceBuffer = sourceView.sub(
            x: real.x, y: real.y, width: real.width, height: real.height
        ).vImage
        var destination = tileView.sub(
            x: insetX, y: insetY, width: real.width, height: real.height
        ).vImage
        var error = vImageCopyBuffer(
            &sourceBuffer, &destination, PixelView.bytesPerPixel, vImage_Flags(kvImageNoFlags))
        guard error == kvImageNoError else {
            throw TilingError.paddingFailed("vImageCopyBuffer failed with \(error)")
        }

        // 2. Left and right strips, over the rows that have real content.
        try reflectHorizontally(
            tile: tileView, rows: insetY..<(insetY + real.height),
            realColumns: insetX..<(insetX + real.width))
        // 3. Top and bottom strips, over the full width, so the corners inherit reflected content.
        try reflectVertically(
            tile: tileView, realRows: insetY..<(insetY + real.height))

        error = kvImageNoError
    }

    private static func reflectHorizontally(
        tile: PixelView, rows: Range<Int>, realColumns: Range<Int>
    ) throws {
        let height = rows.count
        let realWidth = realColumns.count
        guard height > 0, realWidth > 0 else { return }

        // Left: mirror the innermost `count` columns outward, then replicate if the strip is wider
        // than the image (only reachable for images narrower than one tile).
        var remaining = realColumns.lowerBound
        while remaining > 0 {
            let count = min(remaining, realWidth)
            var source = tile.sub(
                x: remaining, y: rows.lowerBound, width: count, height: height
            ).vImage
            var destination = tile.sub(
                x: remaining - count, y: rows.lowerBound, width: count, height: height
            ).vImage
            let error = vImageHorizontalReflect_ARGB8888(
                &source, &destination, vImage_Flags(kvImageNoFlags))
            guard error == kvImageNoError else {
                throw TilingError.paddingFailed("vImageHorizontalReflect failed with \(error)")
            }
            remaining -= count
        }

        var filled = realColumns.upperBound
        while filled < tile.width {
            let count = min(tile.width - filled, realWidth)
            var source = tile.sub(
                x: filled - count, y: rows.lowerBound, width: count, height: height
            ).vImage
            var destination = tile.sub(
                x: filled, y: rows.lowerBound, width: count, height: height
            ).vImage
            let error = vImageHorizontalReflect_ARGB8888(
                &source, &destination, vImage_Flags(kvImageNoFlags))
            guard error == kvImageNoError else {
                throw TilingError.paddingFailed("vImageHorizontalReflect failed with \(error)")
            }
            filled += count
        }
    }

    private static func reflectVertically(tile: PixelView, realRows: Range<Int>) throws {
        let realHeight = realRows.count
        guard realHeight > 0 else { return }

        var remaining = realRows.lowerBound
        while remaining > 0 {
            let count = min(remaining, realHeight)
            var source = tile.sub(x: 0, y: remaining, width: tile.width, height: count).vImage
            var destination = tile.sub(
                x: 0, y: remaining - count, width: tile.width, height: count
            ).vImage
            let error = vImageVerticalReflect_ARGB8888(
                &source, &destination, vImage_Flags(kvImageNoFlags))
            guard error == kvImageNoError else {
                throw TilingError.paddingFailed("vImageVerticalReflect failed with \(error)")
            }
            remaining -= count
        }

        var filled = realRows.upperBound
        while filled < tile.height {
            let count = min(tile.height - filled, realHeight)
            var source = tile.sub(x: 0, y: filled - count, width: tile.width, height: count).vImage
            var destination = tile.sub(x: 0, y: filled, width: tile.width, height: count).vImage
            let error = vImageVerticalReflect_ARGB8888(
                &source, &destination, vImage_Flags(kvImageNoFlags))
            guard error == kvImageNoError else {
                throw TilingError.paddingFailed("vImageVerticalReflect failed with \(error)")
            }
            filled += count
        }
    }

    /// Core Video formats are sometimes four-character codes ('BGRA') and sometimes plain numbers
    /// (`kCVPixelFormatType_32ARGB` is 32). Rendering the latter as "????" tells a caller nothing,
    /// so unprintable codes fall back to their numeric value.
    static func fourCharacterCode(_ code: OSType) -> String {
        let bytes = [24, 16, 8, 0].map { UInt8((code >> $0) & 0xFF) }
        guard bytes.allSatisfy({ (32...126).contains($0) }) else { return String(code) }
        return String(bytes.map { Character(UnicodeScalar($0)) })
    }
}
