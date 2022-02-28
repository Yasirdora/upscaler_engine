import CoreVideo
import Foundation
import Testing

@testable import UpscaleCore

/// The cache purge under memory pressure — certification spec §6's *"memory warnings handled
/// gracefully (caches purged, model retained)"*.
///
/// **Model-free, device-free, and free of real system pressure.** Pressure is posted by hand into
/// `ManualPressureSource` from inside a synthetic prediction (the seam `LadderRetryTests` already
/// uses), so a warning always lands at a known tile boundary: no sleeps, no polling, no kernel.
///
/// **What these tests prove:**
///
/// 1. the engine's only cache — the output `CVPixelBufferPool` — is genuinely a cache (two images
///    of one size share one pool) and is genuinely released by a purge;
/// 2. an image still completes, byte-exactly, after a purge lands in the middle of its run;
/// 3. the model is retained across it — each rung's backend is fetched once and never re-fetched;
/// 4. nothing purges without a pressure event.
///
/// **What they deliberately do not prove: that the purge lowers footprint.** No assertion here
/// reads a byte total, because there is none to read. Apple cautions that dumping caches can cost
/// more than it frees; the hypothesis that a pool of dirty buffers is the opposite case is
/// unproven, and only device runs can settle it. The memory effect of the purge is
/// **`unavailable`**.
///
/// Every expectation below is a literal written out here — geometry from `LadderFixture` (700×700
/// at scale 2, margin 52: 512 → 2×2, 384 → 3×3, 256 → 5×5) — never a value recomputed by the code
/// under test.
struct CachePurgeTests {

    // MARK: - The decision layer: which levels purge

    /// `purgesCaches` is a lookup on the level, and `retainsModel` is unaffected by it. Both halves
    /// of certification spec §6's clause are asserted together because §6 asks for them in one
    /// breath.
    @Test("both pressure levels purge, relief does not, and the model is retained at every level")
    func purgeIsDecidedByTheLevelAlone() throws {
        var fixture = try MemoryPressureTests.midImage(at: .n512)

        let warning = MemoryPressureResponse.respond(
            to: .warning, ladder: &fixture.ladder, plan: fixture.plan)
        #expect(warning.purgesCaches)
        #expect(!warning.checkpoints, "a warning purges and drops a rung; it does not checkpoint")
        #expect(warning.retainsModel)
        #expect(warning.finishesCurrentTile, "a purge abandons no work in flight")

        let critical = MemoryPressureResponse.respond(
            to: .critical, ladder: &fixture.ladder, plan: fixture.plan)
        #expect(critical.purgesCaches)
        #expect(critical.checkpoints)
        #expect(critical.retainsModel)

        // Relief is the absence of an event. Dropping a warm pool because the kernel said the
        // process is fine again would be pointless churn.
        let normal = MemoryPressureResponse.respond(
            to: .normal, ladder: &fixture.ladder, plan: fixture.plan)
        #expect(!normal.purgesCaches)
        #expect(normal.outcome == nil, "the ladder is not consulted at .normal either")
        #expect(normal.retainsModel)
    }

    // MARK: - The actuator: the pool is really released

    /// Purging an engine that has never allocated is a no-op that still counts as a reaction.
    @Test("purging an engine holding no pool reports nothing dropped, and is still counted")
    func purgingAnEmptyEngineIsSafe() {
        let engine = SequentialTilingEngine(contextMargin: LadderFixture.margin)
        #expect(engine.pooledCacheIdentity == nil)
        #expect(engine.purgeCaches() == false, "nothing was held, so nothing was dropped")
        #expect(engine.cachePurges == 1, "the call is counted whether or not it found a cache")
    }

    /// **The release proof**, and the reuse it is the negation of.
    ///
    /// Two same-size images share **one** pool: that is what makes it a cache, and it is asserted
    /// first so the purge has something to be the removal of. The purge then leaves the engine
    /// holding no pool at all, and a third image builds one from scratch.
    ///
    /// **Why the proof is "the engine holds nothing" and not "the new pool is a different object".**
    /// The first draft of this test compared pool identities across the purge and failed, which
    /// turned out to be the test's defect rather than the engine's. Measured on the host: a
    /// `CVPixelBuffer` vended from a pool does *not* keep its pool alive, so dropping the engine's
    /// reference deallocates the pool immediately, and CoreVideo then hands the freed address to
    /// the next pool. Two simultaneously-live pools never share an address (pools are not
    /// interned), so identity across a free is evidence in neither direction — and it is
    /// allocator-dependent, which a deterministic suite cannot assert on. `pooledCacheIdentity ==
    /// nil` is the deterministic statement, and since the engine is the only holder it is exactly
    /// the statement that the cache's lifetime ended.
    @Test("the pool is reused across images, released by a purge, and rebuilt after one")
    func purgeReleasesThePoolAndTheNextImageRebuildsIt() async throws {
        let image = try TilingTestSupport.stripedImage(
            width: LadderFixture.side, height: LadderFixture.side)
        let oracle = LadderFixture.tightRaster(
            try TilingTestSupport.nearestNeighbour(image, scaleFactor: LadderFixture.scale))
        let engine = SequentialTilingEngine(contextMargin: LadderFixture.margin)
        let backend = NearestNeighbourBackend(scaleFactor: LadderFixture.scale)
        let model = LadderFixture.descriptor(for: .n256)

        _ = try await engine.upscale(
            image, model: model, backend: backend, method: .centerValidRegion, progress: nil)
        let warmed = try #require(engine.pooledCacheIdentity, "one image warms the pool")
        #expect(engine.outputAllocations == 1)
        #expect(backend.predictionCount == 25, "700/(256-104) = 5 columns × 5 rows")

        // The cache being a cache: a second image of the same size takes a second buffer from the
        // *same* pool. Both identities are read while that pool is alive, so this comparison is
        // sound where one spanning a free would not be.
        _ = try await engine.upscale(
            image, model: model, backend: backend, method: .centerValidRegion, progress: nil)
        #expect(engine.pooledCacheIdentity == warmed, "same size, same pool — that is the caching")
        #expect(engine.outputAllocations == 2)

        #expect(engine.purgeCaches() == true, "the pool was held, so it was dropped")
        #expect(engine.pooledCacheIdentity == nil, "the engine's only reference to it is gone")
        #expect(engine.cachePurges == 1)

        // Idempotent: a second purge finds nothing and says so, rather than throwing or re-warming.
        #expect(engine.purgeCaches() == false)
        #expect(engine.cachePurges == 2)
        #expect(engine.pooledCacheIdentity == nil, "a purge never rebuilds what it dropped")

        let third = try await engine.upscale(
            image, model: model, backend: backend, method: .centerValidRegion, progress: nil)
        #expect(engine.pooledCacheIdentity != nil, "the next image builds its own pool")
        #expect(engine.outputAllocations == 3)
        #expect(LadderFixture.tightRaster(third) == oracle, "the purge must not cost a pixel")
    }

    // MARK: - The seam: a purge mid-run, and the image still completes

    /// **The integration proof.** One engine holds a warm pool from a previous image; a warning
    /// arrives at a tile boundary of the *next* job; the pool is purged, the rung drops, and the
    /// image completes byte-exactly with every rung's backend fetched exactly once.
    ///
    /// This is certification spec §6's clause end to end: caches purged **and** model retained, on
    /// one event.
    @Test("a warning mid-run purges the cache, keeps the model, and the image still completes")
    func aWarningMidRunPurgesAndTheImageCompletes() async throws {
        let url = LadderFixture.temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let image = try TilingTestSupport.stripedImage(
            width: LadderFixture.side, height: LadderFixture.side)
        let engine = SequentialTilingEngine(contextMargin: LadderFixture.margin)

        // Warm the cache first, through the pooled path, so the purge has something real to drop:
        // the laddered path spools to a file and never touches the pool by itself.
        _ = try await engine.upscale(
            image, model: LadderFixture.descriptor(for: .n256),
            backend: NearestNeighbourBackend(scaleFactor: LadderFixture.scale),
            method: .centerValidRegion, progress: nil)
        #expect(engine.pooledCacheIdentity != nil, "the fixture is vacuous without a warm pool")
        #expect(engine.cachePurges == 0, "an ordinary image purges nothing")

        let backends = LadderBackends(scaleFactor: LadderFixture.scale)
        // Posted inside tile 0's prediction, so it is latched and acted on at the boundary *after*
        // that tile — the drop is for subsequent tiles, and so is the purge.
        backends.pressureAt = [0: .warning]
        let monitor = MemoryPressureMonitor(source: backends.source)
        monitor.start()
        defer { monitor.stop() }

        let run = try await engine.upscale(
            image, shapes: backends, ladder: TileShapeLadder(startingAt: .n512),
            budget: LadderFixture.plan(method: .centerValidRegion), pressure: monitor,
            method: .centerValidRegion, spoolingTo: url, progress: nil)

        // Caches purged: one event, one purge, and the pool is gone.
        #expect(engine.cachePurges == 1)
        #expect(engine.pooledCacheIdentity == nil)
        #expect(engine.outputAllocations == 1, "the laddered path allocates no pooled output")

        // Model retained: two rungs ran, each backend was asked for once, and none was re-fetched.
        #expect(backends.requestedShapes == [.n512, .n384])
        #expect(
            Set(backends.requestedShapes).count == backends.requestedShapes.count,
            "a duplicate request would be a reload — and a recompile under pressure")

        // Job continued: one tile at 512, then the whole 3×3 grid at 384, and the image is exact.
        #expect(
            run.passes == [
                LadderedRun.Pass(shape: .n512, resumedAtImageRow: 0, tilesPredicted: 1, tilesPlanned: 4),
                LadderedRun.Pass(shape: .n384, resumedAtImageRow: 0, tilesPredicted: 9, tilesPlanned: 9),
            ])
        #expect(run.outcomes.map(\.reason) == [.plan(.largestThatFits(.n512)), .pressureWarning])
        #expect(run.shape == .n384)
        #expect(run.tilesPredicted == 10)
        #expect(backends.predictionCount == 10, "no prediction failed, so attempts equal tiles")
        try LadderFixture.expectCompleteAndExact(run, of: image, at: url)
    }

    /// The floor rung has no lower shape to step to, so `steppedDown` is false there — but §6's
    /// clause is not conditional on a rung moving. The purge happens anyway, and the job continues
    /// degraded rather than stalling.
    @Test("a critical event at the floor rung purges even though it cannot step down")
    func theFloorRungPurgesWithoutSteppingDown() async throws {
        let url = LadderFixture.temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let image = try TilingTestSupport.stripedImage(
            width: LadderFixture.side, height: LadderFixture.side)
        let engine = SequentialTilingEngine(contextMargin: LadderFixture.margin)

        _ = try await engine.upscale(
            image, model: LadderFixture.descriptor(for: .n256),
            backend: NearestNeighbourBackend(scaleFactor: LadderFixture.scale),
            method: .centerValidRegion, progress: nil)
        #expect(engine.pooledCacheIdentity != nil, "the fixture is vacuous without a warm pool")

        let backends = LadderBackends(scaleFactor: LadderFixture.scale)
        backends.pressureAt = [0: .critical]
        let monitor = MemoryPressureMonitor(source: backends.source)
        monitor.start()
        defer { monitor.stop() }

        // Carried in at the floor, not constructed there: a fresh ladder adopts the plan's 512 on
        // its first image, and the critical event would then have a rung to drop to — which is the
        // one thing this test needs it not to have.
        let run = try await engine.upscale(
            image, shapes: backends,
            ladder: try LadderFixture.steppedDown(to: .floorRung, method: .centerValidRegion),
            budget: LadderFixture.plan(method: .centerValidRegion), pressure: monitor,
            method: .centerValidRegion, spoolingTo: url, progress: nil)

        #expect(engine.pooledCacheIdentity == nil, "the floor rung purges like any other rung")
        #expect(engine.cachePurges == 1)
        #expect(run.outcomes.map(\.policy) == [.stepUpDeclined, .floorRungHeld])
        #expect(run.isDegraded, "the floor rung held under a trigger — the degraded case")
        #expect(run.shape == .n256)
        #expect(run.passes.count == 1, "nothing was abandoned, so nothing was re-planned")
        #expect(run.tilesPredicted == 25)
        try LadderFixture.expectCompleteAndExact(run, of: image, at: url)
    }

    // MARK: - Nothing purges without an event

    /// Relief is not an event to react to. A `.normal` transition reaches the engine through the
    /// same seam and must leave the warm pool exactly where it is — dropping a cache because the
    /// kernel said the process is fine again is churn. This is the one level whose branch nothing
    /// else here exercises end to end.
    @Test("a relief transition mid-run purges nothing and leaves the warm pool alone")
    func reliefMidRunPurgesNothing() async throws {
        let url = LadderFixture.temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let image = try TilingTestSupport.stripedImage(
            width: LadderFixture.side, height: LadderFixture.side)
        let engine = SequentialTilingEngine(contextMargin: LadderFixture.margin)

        _ = try await engine.upscale(
            image, model: LadderFixture.descriptor(for: .n256),
            backend: NearestNeighbourBackend(scaleFactor: LadderFixture.scale),
            method: .centerValidRegion, progress: nil)
        let warmed = try #require(engine.pooledCacheIdentity)

        let backends = LadderBackends(scaleFactor: LadderFixture.scale)
        backends.pressureAt = [0: .normal]
        let monitor = MemoryPressureMonitor(source: backends.source)
        monitor.start()
        defer { monitor.stop() }

        let run = try await engine.upscale(
            image, shapes: backends, ladder: TileShapeLadder(startingAt: .n512),
            budget: LadderFixture.plan(method: .centerValidRegion), pressure: monitor,
            method: .centerValidRegion, spoolingTo: url, progress: nil)

        #expect(engine.cachePurges == 0)
        #expect(engine.pooledCacheIdentity == warmed, "the pool the engine had is the pool it kept")
        #expect(run.outcomes.count == 1, "relief is not a ladder transition either")
        #expect(run.shape == .n512, "and it moves no rung")
        #expect(run.tilesPredicted == 4)
        try LadderFixture.expectCompleteAndExact(run, of: image, at: url)
    }

    /// A purge is a reaction, not a routine: an ordinary laddered run — with a live monitor and no
    /// event, and again with no monitor at all — must not touch the cache once.
    @Test("an untroubled run never purges, with or without a monitor attached")
    func noEventMeansNoPurge() async throws {
        for attachMonitor in [true, false] {
            let url = LadderFixture.temporaryURL()
            defer { try? FileManager.default.removeItem(at: url) }
            let image = try TilingTestSupport.stripedImage(
                width: LadderFixture.side, height: LadderFixture.side)
            let engine = SequentialTilingEngine(contextMargin: LadderFixture.margin)

            _ = try await engine.upscale(
                image, model: LadderFixture.descriptor(for: .n256),
                backend: NearestNeighbourBackend(scaleFactor: LadderFixture.scale),
                method: .centerValidRegion, progress: nil)
            let warmed = try #require(engine.pooledCacheIdentity)

            let backends = LadderBackends(scaleFactor: LadderFixture.scale)
            let monitor: MemoryPressureMonitor? =
                attachMonitor ? MemoryPressureMonitor(source: backends.source) : nil
            monitor?.start()
            defer { monitor?.stop() }

            let run = try await engine.upscale(
                image, shapes: backends, ladder: TileShapeLadder(startingAt: .n512),
                budget: LadderFixture.plan(method: .centerValidRegion), pressure: monitor,
                method: .centerValidRegion, spoolingTo: url, progress: nil)

            #expect(engine.cachePurges == 0, "monitor attached: \(attachMonitor)")
            #expect(engine.pooledCacheIdentity == warmed, "the warm pool survives an untroubled run")
            #expect(run.passes.count == 1)
            #expect(run.tilesPredicted == 4, "the 512 grid is 2×2 and nothing interrupted it")
            try LadderFixture.expectCompleteAndExact(run, of: image, at: url)
        }
    }
}
