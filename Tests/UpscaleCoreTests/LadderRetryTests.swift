import CoreGraphics
import CoreVideo
import Foundation
import Testing

@testable import UpscaleCore

/// The engine's ladder retry path.
///
/// This file is certification spec §5.5's acceptance test: *"forced-failure tests must demonstrate
/// 512→384→256 retry **and successful completion**"*, and §10's row for memory pressure mid-job
/// (*"adaptive tile fallback → retry → continue; whole-job failure is a defect"*).
///
/// **Model-free.** Both forcing seams are injected the way the tiling tests inject cancellation —
/// from inside a synthetic prediction, so there is no timing race and no device: a scripted
/// prediction index either throws (`.tileFailure`) or posts a level into `ManualPressureSource`
/// (`.pressureWarning`), which the engine consumes at the next tile boundary. No `.mlpackage`, no
/// Core ML, no sleeps.
///
/// **What is asserted, and what deliberately is not.** Every count, pass record and row boundary
/// below is a literal written out here and re-derivable from the geometry stated in
/// `LadderFixture`; none of it is recomputed by the engine. Byte-exactness against the untiled
/// result is asserted **only because the test backend has no receptive field** — nearest-neighbour
/// scaling makes tiling a lossless rearrangement at *any* shape, so a laddered assembly must
/// reproduce it exactly, and a hole, a stale row or a misplaced band shows up as a mismatched byte.
/// It is a property of the oracle, **not a claim that a laddered assembly is pixel-identical with a
/// real model**: cross-shape divergence with a real model is ≤ 0.007 dB — quality-equivalent and
/// non-zero.
struct LadderRetryTests {

    // MARK: - The window-sizing reasoning the retry path depends on

    /// `BandSpoolSurface.windowRows` is `let`, sized once from the initial grid. A mid-image step
    /// down is safe only if every lower rung needs **fewer** open rows than the window already
    /// holds. That is geometry, not luck: all three formulas shrink with `tileSize` at a fixed
    /// margin. Pinned at literals so a change to `openRowCount` that broke the ordering fails here
    /// rather than as a corrupted spool three tests later.
    @Test("every lower rung needs fewer open rows than the rung above it", arguments: StitchMethod.allCases)
    func openRowCountIsMonotoneDownTheLadder(method: StitchMethod) {
        let rows = TileShape.allCases.sorted().map { shape in
            BandSpoolSurface.openRowCount(
                for: TileGrid(
                    tileSize: shape.inputTileSize,
                    contextMargin: LadderFixture.margin, tiles: []),
                scaleFactor: LadderFixture.scale, method: method)
        }

        // core = tileSize - 104; scale 2. centerValidRegion: core*2. Feathers: (core + 104)*2.
        // hardCrop: (tileSize + core)*2. Ascending by shape: 256, 384, 512.
        let expected: [Int]
        switch method {
        case .centerValidRegion: expected = [304, 560, 816]
        case .linearFeather, .cosineFeather: expected = [512, 768, 1024]
        case .hardCrop: expected = [816, 1328, 1840]
        }
        #expect(rows == expected)
        #expect(rows[0] < rows[1] && rows[1] < rows[2], "a step down must never need a wider window")
    }

    /// The runtime half of the same reasoning. A resumed row must never need a row the spool has
    /// already closed — and because `DestinationSurface` documents a write to a closed row as
    /// *unchecked on the hot path*, a re-grid that got this wrong would corrupt the window rather
    /// than fail. Both sides of the boundary are pinned here, at hand-computed rows.
    @Test("a resumed row is refused when it would need a row the destination has closed")
    func resumedRowMustLieInsideTheOpenWindow() {
        // The 384 remainder resumed at input row 408: its first valid row is output row 816.
        let remainder = TilePlanner(scaleFactor: LadderFixture.scale).plan(
            imageSize: LadderFixture.imageSize, tileSize: 384, contextMargin: LadderFixture.margin,
            resumingAtImageRow: 408)
        #expect(remainder.tiles[0].validRect.minY == 816)

        // centerValidRegion writes only its own band: rows 816 up to 816 + 280*2 = 1376.
        #expect(Self.writable(remainder, open: 816..<1376, method: .centerValidRegion))
        // One row too high — the destination has already closed row 816.
        #expect(!Self.writable(remainder, open: 817..<1376, method: .centerValidRegion))
        // One row too short — the band's last row would land past the window buffer's end. This is
        // the mis-resume that segfaults without the check, not merely one that writes a wrong pixel.
        #expect(!Self.writable(remainder, open: 816..<1375, method: .centerValidRegion))

        // A feather reaches a margin beyond its band on both sides: 816 - 104 and 1376 + 104.
        #expect(Self.writable(remainder, open: 712..<1480, method: .linearFeather))
        #expect(!Self.writable(remainder, open: 713..<1480, method: .linearFeather))
        #expect(!Self.writable(remainder, open: 712..<1479, method: .linearFeather))

        // An empty remainder is refused too — finishing a truncated image silently is the one
        // outcome certification spec §6 rules out absolutely.
        #expect(
            !SequentialTilingEngine.resumedRowIsWritable(
                TileGrid(tileSize: 384, contextMargin: LadderFixture.margin, tiles: []),
                open: 0..<LadderFixture.outputSide, outputHeight: LadderFixture.outputSide,
                scaleFactor: LadderFixture.scale, method: .centerValidRegion))
    }

    /// `resumedRowIsWritable` against the fixture's output height, so the cases above read as rows.
    static func writable(_ grid: TileGrid, open: Range<Int>, method: StitchMethod) -> Bool {
        SequentialTilingEngine.resumedRowIsWritable(
            grid, open: open, outputHeight: 2000, scaleFactor: LadderFixture.scale, method: method)
    }

    // MARK: - §5.5: the full descent, and completion

    /// **The acceptance test.** Two forced prediction failures drive 512 → 384 → 256, the abandoned
    /// row is re-run at each smaller shape, and the image completes.
    ///
    /// Run against all four stitch methods because each one sizes and places the spool window
    /// differently (`openRowCount` / `lowestWrittenRow`); a step down that wrote outside the window
    /// would make `BandSpoolSurface.closeRows` throw rather than silently corrupt the file.
    @Test(
        "a forced failure demonstrates the 512 → 384 → 256 retry and completes the image",
        arguments: StitchMethod.allCases)
    func forcedFailureDescendsTheLadderAndCompletes(method: StitchMethod) async throws {
        let url = LadderFixture.temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let backends = LadderBackends(scaleFactor: LadderFixture.scale)
        // Prediction 1 is tile 1 of the 512 grid (row 0, column 1); prediction 3 is tile 1 of the
        // 384 grid. Both are mid-row, so each step down re-runs the tile to its left.
        backends.failAt = [1, 3]
        let engine = SequentialTilingEngine(contextMargin: LadderFixture.margin)

        let image = try TilingTestSupport.stripedImage(
            width: LadderFixture.side, height: LadderFixture.side)
        let run = try await engine.upscale(
            image, shapes: backends, ladder: TileShapeLadder(startingAt: .n512),
            budget: LadderFixture.plan(method: method), pressure: nil, method: method,
            spoolingTo: url, progress: nil)

        // 700 / (512-104) = 2x2; 700 / (384-104) = 3x3; 700 / (256-104) = 5x5. Hand-counted.
        #expect(
            run.passes == [
                LadderedRun.Pass(shape: .n512, resumedAtImageRow: 0, tilesPredicted: 1, tilesPlanned: 4),
                LadderedRun.Pass(shape: .n384, resumedAtImageRow: 0, tilesPredicted: 1, tilesPlanned: 9),
                LadderedRun.Pass(shape: .n256, resumedAtImageRow: 0, tilesPredicted: 25, tilesPlanned: 25),
            ])
        #expect(run.shape == .n256)
        #expect(run.stepDowns == 2)
        #expect(run.tilesPredicted == 27, "1 at 512 + 1 at 384 + the whole 25-tile 256 grid")
        #expect(run.tilesRecomputed == 2, "one tile to the left of each abandoned tile")
        #expect(backends.predictionCount == 29, "the two failed calls are attempts, not tiles")
        #expect(!run.isDegraded, "the floor rung was never held under a trigger")

        // The ladder's own account of the descent — its typed outcome.
        #expect(run.outcomes.map(\.policy) == [.plannedRung, .stepDownMidImage, .stepDownMidImage])
        #expect(run.outcomes.map(\.shape) == [.n512, .n384, .n256])
        #expect(run.outcomes.map(\.previousShape) == [.n512, .n512, .n384])
        #expect(run.ladder.shape == .n256, "the caller carries the rung into the next image")

        try LadderFixture.expectCompleteAndExact(run, of: image, at: url)
    }

    /// The same descent driven by the **pressure** seam rather than the failure seam — requirement
    /// 5's "drop a ladder rung for *subsequent* tiles" arriving at a tile boundary.
    @Test("two memory-pressure warnings walk the ladder down to the floor and the image completes")
    func pressureWarningsDescendTheLadderAndComplete() async throws {
        let url = LadderFixture.temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let backends = LadderBackends(scaleFactor: LadderFixture.scale)
        backends.pressureAt = [0: .warning, 1: .warning]
        let monitor = MemoryPressureMonitor(source: backends.source)
        monitor.start()
        defer { monitor.stop() }
        let engine = SequentialTilingEngine(contextMargin: LadderFixture.margin)

        let image = try TilingTestSupport.stripedImage(
            width: LadderFixture.side, height: LadderFixture.side)
        let run = try await engine.upscale(
            image, shapes: backends, ladder: TileShapeLadder(startingAt: .n512),
            budget: LadderFixture.plan(method: .centerValidRegion), pressure: monitor,
            method: .centerValidRegion, spoolingTo: url, progress: nil)

        // The warning posted inside tile 0's prediction is latched and acted on at the boundary
        // *after* that tile — the rung drops for subsequent tiles, never mid-tile.
        #expect(
            run.passes == [
                LadderedRun.Pass(shape: .n512, resumedAtImageRow: 0, tilesPredicted: 1, tilesPlanned: 4),
                LadderedRun.Pass(shape: .n384, resumedAtImageRow: 0, tilesPredicted: 1, tilesPlanned: 9),
                LadderedRun.Pass(shape: .n256, resumedAtImageRow: 0, tilesPredicted: 25, tilesPlanned: 25),
            ])
        #expect(
            run.outcomes.map(\.reason) == [
                .plan(.largestThatFits(.n512)), .pressureWarning, .pressureWarning,
            ])
        #expect(run.tilesPredicted == 27)
        #expect(backends.predictionCount == 27, "no prediction failed, so attempts equal tiles")
        #expect(run.shape == .n256)

        try LadderFixture.expectCompleteAndExact(run, of: image, at: url)
    }

    /// Each rung's backend is asked for **once**, however many tiles run at it: no per-tile
    /// recompilation, and `CoreMLBackend.load` recompiles on every call.
    @Test("each rung's backend is fetched exactly once, and both shapes stay live")
    func eachRungIsFetchedOnce() async throws {
        let url = LadderFixture.temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let backends = LadderBackends(scaleFactor: LadderFixture.scale)
        backends.failAt = [0, 1]
        let engine = SequentialTilingEngine(contextMargin: LadderFixture.margin)

        let image = try TilingTestSupport.stripedImage(
            width: LadderFixture.side, height: LadderFixture.side)
        let run = try await engine.upscale(
            image, shapes: backends, ladder: TileShapeLadder(startingAt: .n512),
            budget: LadderFixture.plan(method: .centerValidRegion), pressure: nil,
            method: .centerValidRegion, spoolingTo: url, progress: nil)

        #expect(backends.requestedShapes == [.n512, .n384, .n256])
        #expect(run.tilesPredicted == 25, "both 512 and 384 failed on their first tile")
        // The abandoned rungs were never re-fetched, which is only possible if they stayed live.
        #expect(Set(backends.requestedShapes).count == backends.requestedShapes.count)
    }

    // MARK: - A step down in the final row

    @Test("a step down in the final tile row re-plans that row and finishes below it")
    func stepDownInTheFinalRow() async throws {
        let url = LadderFixture.temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let backends = LadderBackends(scaleFactor: LadderFixture.scale)
        // The 512 grid is 2x2, so prediction 3 is the last tile of the last row.
        backends.failAt = [3]
        let engine = SequentialTilingEngine(contextMargin: LadderFixture.margin)

        let image = try TilingTestSupport.stripedImage(
            width: LadderFixture.side, height: LadderFixture.side)
        let run = try await engine.upscale(
            image, shapes: backends, ladder: TileShapeLadder(startingAt: .n512),
            budget: LadderFixture.plan(method: .centerValidRegion), pressure: nil,
            method: .centerValidRegion, spoolingTo: url, progress: nil)

        // Resume at input row 408 — the top of the 512 grid's second row. At 384 the remainder is
        // rows 408 (height 280) and 688 (height 12), three columns each: 6 tiles.
        #expect(
            run.passes == [
                LadderedRun.Pass(shape: .n512, resumedAtImageRow: 0, tilesPredicted: 3, tilesPlanned: 4),
                LadderedRun.Pass(shape: .n384, resumedAtImageRow: 408, tilesPredicted: 6, tilesPlanned: 6),
            ])
        #expect(run.shape == .n384)
        #expect(run.tilesRecomputed == 1, "one tile of the final row had already been computed")
        #expect(run.stepDowns == 1)

        try LadderFixture.expectCompleteAndExact(run, of: image, at: url)
    }

    /// Every output row must end up owned by exactly one tile of exactly one pass, and that pass
    /// must cover the row's full width. Computed from the passes the engine reported, against the
    /// row boundaries written out here — not read back off the engine.
    @Test("every output row is written exactly once across a shape change")
    func everyOutputRowHasOneFinalOwner() async throws {
        let url = LadderFixture.temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let backends = LadderBackends(scaleFactor: LadderFixture.scale)
        backends.failAt = [3]
        let engine = SequentialTilingEngine(contextMargin: LadderFixture.margin)

        let image = try TilingTestSupport.stripedImage(
            width: LadderFixture.side, height: LadderFixture.side)
        let run = try await engine.upscale(
            image, shapes: backends, ladder: TileShapeLadder(startingAt: .n512),
            budget: LadderFixture.plan(method: .centerValidRegion), pressure: nil,
            method: .centerValidRegion, spoolingTo: url, progress: nil)

        let (owner, width) = LadderFixture.finalRowOwnership(run.passes)

        // Pass 0 ran the whole first 512 row (input rows 0..407 -> output 0..815) plus one tile of
        // the second, which pass 1 then re-ran in full. So the seam sits at output row 408*2 = 816.
        for y in 0..<816 {
            #expect(owner[y] == 0, "output row \(y) should still belong to the 512 pass")
        }
        for y in 816..<LadderFixture.outputSide {
            #expect(owner[y] == 1, "output row \(y) should have been re-planned at 384")
        }
        #expect(
            width.allSatisfy { $0 == LadderFixture.outputSide },
            "the owning pass must cover every row's full width, with no gap and no double claim")
    }

    // MARK: - The floor rung

    /// At 256 there is no lower rung. The job must **continue, degraded** — certification spec §6:
    /// complete or degrade gracefully, never crash; never fail the whole job at 80%.
    @Test("pressure at the floor rung holds the rung, reports degradation, and still completes")
    func floorRungHoldsAndCompletes() async throws {
        let url = LadderFixture.temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let backends = LadderBackends(scaleFactor: LadderFixture.scale)
        backends.pressureAt = [0: .critical, 4: .warning]
        let monitor = MemoryPressureMonitor(source: backends.source)
        monitor.start()
        defer { monitor.stop() }
        let engine = SequentialTilingEngine(contextMargin: LadderFixture.margin)

        let image = try TilingTestSupport.stripedImage(
            width: LadderFixture.side, height: LadderFixture.side)
        let run = try await engine.upscale(
            image, shapes: backends,
            ladder: try LadderFixture.steppedDown(to: .floorRung, method: .centerValidRegion),
            budget: LadderFixture.plan(method: .centerValidRegion), pressure: monitor,
            method: .centerValidRegion, spoolingTo: url, progress: nil)

        // A ladder walked down to the floor by an earlier image may not climb back out of it: dwell
        // is 0 of 3. A *fresh* ladder would have adopted the plan's 512 instead, which is why this
        // job carries one in rather than constructing one at the floor.
        #expect(run.outcomes[0].policy == .stepUpDeclined)
        #expect(run.outcomes[0].reason == .dwellNotMet(images: 0, required: 3))
        // Both pressure events held the rung, so neither re-gridded anything.
        #expect(run.outcomes.map(\.policy) == [.stepUpDeclined, .floorRungHeld, .floorRungHeld])
        #expect(run.passes.count == 1)
        #expect(
            run.passes[0]
                == LadderedRun.Pass(
                    shape: .n256, resumedAtImageRow: 0, tilesPredicted: 25, tilesPlanned: 25))
        #expect(run.isDegraded, "the floor rung held under a trigger — that is the degraded case")
        #expect(run.tilesRecomputed == 0)
        #expect(run.shape == .n256)

        try LadderFixture.expectCompleteAndExact(run, of: image, at: url)
    }

    /// The other half of the floor: a *failure* there has no smaller shape to retry at, so it is
    /// the caller's error rather than an endless retry loop.
    @Test("a prediction failure at the floor rung propagates instead of retrying forever")
    func failureAtTheFloorRungPropagates() async throws {
        let url = LadderFixture.temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let backends = LadderBackends(scaleFactor: LadderFixture.scale)
        backends.failAt = [0]
        let engine = SequentialTilingEngine(contextMargin: LadderFixture.margin)

        let image = try TilingTestSupport.stripedImage(
            width: LadderFixture.side, height: LadderFixture.side)
        // Carried in at the floor rather than constructed there: a fresh ladder adopts the plan's
        // 512 and the failure would have a rung below it to retry at.
        let atTheFloor = try LadderFixture.steppedDown(to: .floorRung, method: .centerValidRegion)
        await #expect(throws: InferenceError.predictionFailed("forced failure at prediction 0")) {
            _ = try await engine.upscale(
                image, shapes: backends, ladder: atTheFloor,
                budget: LadderFixture.plan(method: .centerValidRegion), pressure: nil,
                method: .centerValidRegion, spoolingTo: url, progress: nil)
        }
        #expect(backends.predictionCount == 1, "one attempt, no retry — 256 is the floor")
    }

    // MARK: - Cancellation

    @Test("cancellation is honoured at a tile boundary during a retry")
    func cancellationDuringARetry() async throws {
        let url = LadderFixture.temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let backends = LadderBackends(scaleFactor: LadderFixture.scale)
        backends.failAt = [0]  // step 512 -> 384
        backends.cancelAt = 1  // the first tile of the 384 pass cancels its own task
        let engine = SequentialTilingEngine(contextMargin: LadderFixture.margin)

        let image = try TilingTestSupport.stripedImage(
            width: LadderFixture.side, height: LadderFixture.side)
        await #expect(throws: CancellationError.self) {
            _ = try await engine.upscale(
                image, shapes: backends, ladder: TileShapeLadder(startingAt: .n512),
                budget: LadderFixture.plan(method: .centerValidRegion), pressure: nil,
                method: .centerValidRegion, spoolingTo: url, progress: nil)
        }
        #expect(backends.predictionCount == 2, "it stops at the next boundary, not at the grid's end")
        #expect(backends.requestedShapes == [.n512, .n384], "the retry had begun when it was cancelled")
    }

    // MARK: - Progress

    @Test("progress ends at exactly 1.0 and replays the resumed row after a step down")
    func progressReplaysTheResumedRow() async throws {
        let url = LadderFixture.temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let backends = LadderBackends(scaleFactor: LadderFixture.scale)
        backends.failAt = [3]
        let engine = SequentialTilingEngine(contextMargin: LadderFixture.margin)
        let reported = Reported()

        let image = try TilingTestSupport.stripedImage(
            width: LadderFixture.side, height: LadderFixture.side)
        _ = try await engine.upscale(
            image, shapes: backends, ladder: TileShapeLadder(startingAt: .n512),
            budget: LadderFixture.plan(method: .centerValidRegion), pressure: nil,
            method: .centerValidRegion, spoolingTo: url, progress: { reported.append($0) })

        let values = reported.values
        #expect(values.count == 9, "one report per rendered tile: 3 at 512, 6 at 384")
        #expect(values.last == 1.0)
        // The 384 pass owns output rows 816..1399, so its first report is 816/1400 + 1/6 of the
        // remaining 584/1400 — strictly below the 3/4 the abandoned 512 pass had reached.
        #expect(values[2] == 0.75)
        #expect(values[3] < values[2], "a replayed row reports honestly rather than freezing")
        #expect(abs(values[3] - (816.0 / 1400.0 + (584.0 / 1400.0) / 6.0)) < 1e-12)
        #expect(zip(values.dropFirst(3), values.dropFirst(4)).allSatisfy { $0 < $1 })
    }
}

// MARK: - Fixture

/// The one geometry every test above is written against, stated once so the literals are
/// re-derivable: a 700x700 image at scale 2 with a 52 px margin.
///
///     core = tileSize - 104        columns = ceil(700 / core)      rows = ceil(700 / core)
///     512 -> core 408, 2 x 2 = 4   384 -> core 280, 3 x 3 = 9      256 -> core 152, 5 x 5 = 25
///
/// Scale 2 rather than 4 keeps the spool at 7.84 MB and the predictions cheap, without touching the
/// quantity under test: the ladder moves *tile shape*, and the scale factor is a multiplier the
/// tiling suite already pins.
enum LadderFixture {
    static let scale = 2
    static let margin = 52
    static let side = 700
    static let outputSide = 1400

    static var imageSize: CGSize { CGSize(width: side, height: side) }

    static func descriptor(for shape: TileShape) -> ModelDescriptor {
        ModelDescriptor(
            identifier: "SyntheticLadderRung\(shape.rawValue)", scaleFactor: scale, contextMargin: 52,
            packageURL: URL(fileURLWithPath: "/dev/null"), inputTileSize: shape.inputTileSize,
            precision: .float32)
    }

    /// A budget whose availability is injected, so the plan is arithmetic and not a host reading. 4
    /// GiB available at the standard ρ = 0.5 is a 2 GiB allowance, which every rung clears — so the
    /// plan names 512 and the ladder's starting rung, not the budget, is what each test varies.
    static func plan(
        method: StitchMethod, availableBytes: Int = 4 * 1024 * 1024 * 1024
    ) -> MemoryBudget.Plan {
        MemoryBudget(
            available: MemoryBudget.Figure(
                bytes: availableBytes, provenance: .measured, source: "test literal")
        ).plan(
            for: ImageWorkload(
                inputWidth: side, inputHeight: side, scaleFactor: scale, contextMargin: margin,
                method: method, destination: .bandSpool))
    }

    /// A ladder the caller carries in **already sitting at `shape`**, having run an image and been
    /// walked down to it — the state the engine is in on the second image of a job that met trouble
    /// on the first.
    ///
    /// `TileShapeLadder(startingAt: .floorRung)` does not produce a run at the floor against this
    /// fixture's plan: the plan names 512 and a fresh ladder adopts it on the first `beginImage`.
    /// The floor-rung tests below need a ladder that will genuinely *stay* at the floor, and this
    /// is the state in which one does — not fresh, so the plan's 512 is a climb, and carrying zero
    /// dwell, so the climb is refused.
    ///
    /// Walked down on `.tileFailure` so the pressure tests are not set up by the pressure path.
    static func steppedDown(to shape: TileShape, method: StitchMethod) throws -> TileShapeLadder {
        let plan = plan(method: method)
        var ladder = TileShapeLadder()
        _ = ladder.beginImage(plan)
        while ladder.shape > shape {
            _ = ladder.respond(to: .tileFailure("fixture: walking down to \(shape.rawValue)"), plan: plan)
        }
        ladder.endImage()
        try #require(ladder.shape == shape)
        try #require(ladder.dwellCount == 0, "a step down spends the dwell")
        return ladder
    }

    static func temporaryURL() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("ladder-\(UUID().uuidString).bgra")
    }

    /// A `CVPixelBuffer` as a tightly packed BGRA raster — the spool file's layout.
    static func tightRaster(_ buffer: CVPixelBuffer) -> [UInt8] {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let view = PixelView(buffer) else { return [] }
        let rowBytes = view.width * PixelView.bytesPerPixel
        var bytes = [UInt8](repeating: 0, count: rowBytes * view.height)
        for y in 0..<view.height {
            let row = view.base.advanced(by: y * view.rowBytes)
            for byte in 0..<rowBytes {
                bytes[y * rowBytes + byte] = row.advanced(by: byte).load(as: UInt8.self)
            }
        }
        return bytes
    }

    /// "Successful completion", checked three ways: the spool is the right size, it is on disk, and
    /// — because the test backend has no receptive field — its bytes are the untiled result
    /// exactly.
    ///
    /// The last check is what a hole, a stale band or a row written by the wrong grid would fail.
    /// It is **not** a pixel-identity claim about laddered assembly with a real model.
    static func expectCompleteAndExact(
        _ run: LadderedRun, of image: CVPixelBuffer, at url: URL
    ) throws {
        #expect(run.image.width == outputSide && run.image.height == outputSide)
        #expect(run.image.bytesPerRow == outputSide * PixelView.bytesPerPixel)
        #expect(run.image.byteCount == 7_840_000)
        #expect(run.image.url == url)

        let spooled = try Data(contentsOf: url)
        #expect(spooled.count == 7_840_000, "a short file is an unfinished image")
        let untiled = try TilingTestSupport.nearestNeighbour(image, scaleFactor: scale)
        #expect(Array(spooled) == tightRaster(untiled))
    }

    /// Which pass finally owns each output row, and how much of that row it covered.
    ///
    /// Rebuilt from the passes the engine reported, through the resume planner (tested separately)
    /// — the claim under test here is the engine's *sequencing*, not the planner's geometry. Later
    /// passes overwrite earlier ones, so a row's width count resets when its owner changes: what
    /// must total the image width is the **final** owner's coverage.
    static func finalRowOwnership(_ passes: [LadderedRun.Pass]) -> (owner: [Int], width: [Int]) {
        var owner = [Int](repeating: -1, count: outputSide)
        var width = [Int](repeating: 0, count: outputSide)
        let planner = TilePlanner(scaleFactor: scale)

        for (passIndex, pass) in passes.enumerated() {
            let grid = planner.plan(
                imageSize: imageSize, tileSize: pass.shape.inputTileSize, contextMargin: margin,
                resumingAtImageRow: pass.resumedAtImageRow)
            #expect(grid.tiles.count == pass.tilesPlanned, "the pass's own grid size")
            for tile in grid.tiles.prefix(pass.tilesPredicted) {
                for y in Int(tile.validRect.minY)..<Int(tile.validRect.maxY) {
                    if owner[y] != passIndex {
                        owner[y] = passIndex
                        width[y] = 0
                    }
                    width[y] += Int(tile.validRect.width)
                }
            }
        }
        return (owner, width)
    }
}

// MARK: - Test doubles

/// One rung's backend: nearest-neighbour scaling, with the forcing seam delegated to its owner so
/// every rung shares one prediction counter and one script.
final class LadderRungBackend: InferenceBackend, @unchecked Sendable {
    private let owner: LadderBackends
    private let scaleFactor: Int

    init(owner: LadderBackends, scaleFactor: Int) {
        self.owner = owner
        self.scaleFactor = scaleFactor
    }

    func load(_ model: ModelDescriptor, context: InferenceContext) async throws {}

    func predict(_ tile: CVPixelBuffer) async throws -> CVPixelBuffer {
        try owner.step()
        return try TilingTestSupport.nearestNeighbour(tile, scaleFactor: scaleFactor)
    }
}

/// The ladder's backends, and the forced-failure / forced-pressure seam certification spec §5.5
/// needs.
///
/// A model with **no receptive field** at every rung, so a laddered assembly is still a lossless
/// rearrangement and any sequencing defect is a byte difference. The script is keyed on the
/// **global** prediction index, which is what makes the tests deterministic: the engine's tile
/// boundaries are the only thing that decides when a latched event is acted on.
final class LadderBackends: TileShapeBackendProviding, @unchecked Sendable {
    let scaleFactor: Int
    /// Where scripted pressure levels are posted, exactly as the kernel's handler would post them.
    let source = ManualPressureSource()

    private let lock = NSLock()
    private var requested: [TileShape] = []
    private var calls = 0
    private var failures: Set<Int> = []
    private var events: [Int: MemoryPressureLevel] = [:]
    private var cancelIndex: Int?

    init(scaleFactor: Int) { self.scaleFactor = scaleFactor }

    /// Global prediction indices that throw instead of predicting.
    var failAt: Set<Int> {
        get { lock.withLock { failures } }
        set { lock.withLock { failures = newValue } }
    }

    /// Global prediction indices that post a pressure level *before* predicting, so the engine sees
    /// it at the following tile boundary.
    var pressureAt: [Int: MemoryPressureLevel] {
        get { lock.withLock { events } }
        set { lock.withLock { events = newValue } }
    }

    /// The prediction that cancels its own surrounding task — the tiling suite's cancellation
    /// pattern.
    var cancelAt: Int? {
        get { lock.withLock { cancelIndex } }
        set { lock.withLock { cancelIndex = newValue } }
    }

    var predictionCount: Int { lock.withLock { calls } }
    /// Every rung asked for, in order. Duplicates would mean the engine reloaded a shape.
    var requestedShapes: [TileShape] { lock.withLock { requested } }

    func backend(for shape: TileShape) async throws -> ShapedBackend {
        lock.withLock { requested.append(shape) }
        return ShapedBackend(
            model: LadderFixture.descriptor(for: shape),
            backend: LadderRungBackend(owner: self, scaleFactor: scaleFactor))
    }

    /// Count this prediction and run whatever the script says about it.
    func step() throws {
        let (index, level, cancels, fails) = lock.withLock {
            () -> (Int, MemoryPressureLevel?, Bool, Bool) in
            defer { calls += 1 }
            return (calls, events[calls], cancelIndex == calls, failures.contains(calls))
        }
        if let level { source.send(level) }
        if cancels { withUnsafeCurrentTask { $0?.cancel() } }
        if fails { throw InferenceError.predictionFailed("forced failure at prediction \(index)") }
    }
}
