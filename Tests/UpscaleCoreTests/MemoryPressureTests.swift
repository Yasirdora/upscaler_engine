import Dispatch
import Foundation
import Testing

@testable import UpscaleCore

/// Tests for the event-driven pressure response.
///
/// **Model-free, device-free, and free of real system pressure.** The pressure source is injected
/// (`MemoryPressureSource`), so `warn` and `critical` are driven by hand and delivered
/// synchronously on the test thread: no sleeps, no polling, no waiting on the kernel, nothing
/// timing-dependent.
///
/// Every expectation is a literal a reader can check by hand. The per-rung totals come from
/// `MemoryTests`, where their arithmetic is derived line by line, and the deltas below are
/// subtractions of those figures:
///
/// | Workload | 256 | 384 | 512 |
/// |---|---|---|---|
/// | B — 12MP (4032×3024) | 326 755 123 | 334 829 158 | 434 129 306 |
/// | A — 1080p (1920×1080) | 265 732 915 | **256 505 446** | 338 504 090 |
///
/// At the 2 GiB fixture budget and ρ = 0.5 the allowance is 1 073 741 824 B.
struct MemoryPressureTests {

    // MARK: - Fixtures

    /// A mid-image ladder at `shape`, with the roomy plan already taken — the state a pressure
    /// event actually interrupts.
    ///
    /// `beginImage(_:)` adopts the plan's rung on a fresh ladder, so opening the image can leave
    /// the ladder **above** `shape` — at this fixture's 2 GiB budget the plan names 512, whatever
    /// rung the ladder was constructed at. It is walked back down to `shape` on triggers, which is
    /// the only way a running engine ever gets below the plan, rather than the tests being relaxed
    /// about which rung they interrupt.
    ///
    /// The walk-down trigger is `.tileFailure`, so no test below is set up by the pressure
    /// transition it is asserting on.
    static func midImage(
        at shape: TileShape, workload: ImageWorkload = MemoryTests.workloadB,
        available: Int = 2_147_483_648
    ) throws -> (ladder: TileShapeLadder, plan: MemoryBudget.Plan) {
        let plan = try MemoryTests.budget(available: available).plan(for: workload)
        var ladder = TileShapeLadder(startingAt: shape)
        _ = ladder.beginImage(plan)
        while ladder.shape > shape {
            _ = ladder.respond(
                to: .tileFailure("fixture: opening the image at \(shape.rawValue)"), plan: plan)
        }
        try #require(
            ladder.shape == shape,
            "the plan endorses \(plan.shape.rawValue): a ladder is only ever driven down, never up")
        try #require(ladder.isMidImage, "the pressure event has to land inside an image")
        return (ladder, plan)
    }

    // MARK: - Reading the dispatch event mask

    @Test("a dispatch event mask reads worst-first, and an undefined mask is not a level")
    func eventMaskReadsWorstFirst() {
        #expect(MemoryPressureLevel(.critical) == .critical)
        #expect(MemoryPressureLevel(.warning) == .warning)
        #expect(MemoryPressureLevel(.normal) == .normal)

        // The kernel may set more than one bit. Escalating is the safe direction: answering a
        // critical event with a warning's response would skip the checkpoint.
        #expect(MemoryPressureLevel([.warning, .critical]) == .critical)
        #expect(MemoryPressureLevel([.normal, .warning]) == .warning)
        #expect(MemoryPressureLevel(DispatchMemoryPressureSource.eventMask) == .critical)

        // A mask naming none of the three is not a level and must not be invented into one.
        #expect(MemoryPressureLevel([]) == nil)
    }

    @Test("the subscription covers all three transitions, so relief is reported and never polled")
    func eventMaskCoversEveryTransition() {
        let mask = DispatchMemoryPressureSource.eventMask
        #expect(mask.contains(.normal))
        #expect(mask.contains(.warning))
        #expect(mask.contains(.critical))
    }

    @Test("levels order by severity, which is what the latch collapses on")
    func levelsOrderBySeverity() {
        #expect(MemoryPressureLevel.allCases == [.normal, .warning, .critical])
        #expect(MemoryPressureLevel.normal < MemoryPressureLevel.warning)
        #expect(MemoryPressureLevel.warning < MemoryPressureLevel.critical)
    }

    // MARK: - The pressure-to-ladder mapping

    @Test("each level maps onto exactly its ladder trigger")
    func levelsMapOntoLadderTriggers() {
        #expect(MemoryPressureLevel.warning.ladderTrigger == .pressureWarning)
        #expect(MemoryPressureLevel.critical.ladderTrigger == .pressureCritical)
        // No upward trigger exists, and none may be invented: a step up is between images only, on
        // dwell plus margin (`TileShapeLadder`).
        #expect(MemoryPressureLevel.normal.ladderTrigger == nil)
    }

    // MARK: - warn: drop a rung for subsequent tiles, stop starting work at the current shape

    @Test("warn steps down one rung mid-image and does not checkpoint")
    func warnStepsDownOneRung() throws {
        var (ladder, plan) = try Self.midImage(at: .n512)
        let response = MemoryPressureResponse.respond(to: .warning, ladder: &ladder, plan: plan)

        let outcome = try #require(response.outcome)
        #expect(outcome.previousShape == .n512)
        #expect(outcome.shape == .n384, "one rung, never two")
        #expect(outcome.policy == .stepDownMidImage)
        #expect(outcome.reason == .pressureWarning)
        // 334 829 158 − 434 129 306, read off the plan's own assessments.
        #expect(outcome.projectedDeltaBytes == -99_300_148)

        #expect(response.subsequentTileShape == .n384)
        #expect(response.steppedDown, "new work no longer starts at 512")
        #expect(response.checkpoints == false, "checkpointing is the critical clause only")
        #expect(response.isDegraded == false)
        #expect(ladder.shape == .n384)
    }

    @Test("a step down under pressure can cost more memory, and the response says so")
    func aStepDownCanCostMoreMemory() throws {
        // The measured per-shape table is non-monotonic: at 1080p the 384 rung's total sits below
        // the 256 floor's, so this step down raises the projected total. 265 732 915 − 256 505 446
        // = 9 227 469.
        var (ladder, plan) = try Self.midImage(at: .n384, workload: MemoryTests.workloadA)
        let response = MemoryPressureResponse.respond(to: .warning, ladder: &ladder, plan: plan)

        let outcome = try #require(response.outcome)
        #expect(outcome.shape == .n256)
        #expect(outcome.projectedDeltaBytes == 9_227_469)
        #expect(outcome.stepsDownButCostsMore, "surfaced, never assumed away")
    }

    // MARK: - critical: finish the current tile, then checkpoint

    @Test("critical steps down one rung and checkpoints")
    func criticalCheckpoints() throws {
        var (ladder, plan) = try Self.midImage(at: .n384)
        let response = MemoryPressureResponse.respond(to: .critical, ladder: &ladder, plan: plan)

        let outcome = try #require(response.outcome)
        #expect(outcome.shape == .n256, "critical does not skip two rungs")
        #expect(outcome.reason == .pressureCritical)
        // 326 755 123 − 334 829 158.
        #expect(outcome.projectedDeltaBytes == -8_074_035)

        #expect(response.checkpoints)
        #expect(response.finishesCurrentTile, "the current tile is finished, never abandoned")
        #expect(response.steppedDown)
    }

    // MARK: - The invariants that are acceptance criteria

    @Test("no level abandons the current tile, and no level frees the model")
    func noLevelTearsDownTheModelOrTheTile() throws {
        for level in MemoryPressureLevel.allCases {
            var (ladder, plan) = try Self.midImage(at: .n512)
            let response = MemoryPressureResponse.respond(to: level, ladder: &ladder, plan: plan)
            #expect(response.finishesCurrentTile, "\(level) must finish the tile in flight")
            #expect(response.retainsModel, "\(level) must not free and reload the model")
        }
    }

    // MARK: - The floor rung

    @Test("at the floor rung a warn holds the rung and degrades gracefully rather than stalling")
    func floorRungHoldsAndDegrades() throws {
        // available 660 000 000 → allowance 330 000 000: 256 fits at 326 755 123, 384 does not at
        // 334 829 158. The plan therefore selects the floor rung.
        var (ladder, plan) = try Self.midImage(at: .n256, available: 660_000_000)
        #expect(plan.selection == .largestThatFits(.n256))

        let response = MemoryPressureResponse.respond(to: .warning, ladder: &ladder, plan: plan)
        let outcome = try #require(response.outcome)
        #expect(outcome.policy == .floorRungHeld)
        #expect(outcome.shape == .n256)
        #expect(outcome.reason == .pressureWarning)

        #expect(response.isDegraded, "certification spec §6: complete or degrade gracefully, never crash")
        #expect(response.steppedDown == false, "there is no rung below 256 to stop work at")
        #expect(response.subsequentTileShape == .n256, "the job continues at the floor rung")
        #expect(ladder.shape == .n256)
    }

    // MARK: - normal: not a ladder transition at all

    @Test("normal never consults the ladder, so pressure relief cannot spend sustained headroom")
    func normalDoesNotTouchTheLadder() throws {
        let plan = try MemoryTests.budget(available: 2_147_483_648).plan(for: MemoryTests.workloadB)
        // Dwell only accrues at a rung that has one above it clearing the margin, and only on a
        // ladder that has to climb at all — so the sustained headroom this test is about can only
        // be built on a stepped-down ladder, not a fresh one.
        var ladder = try TileShapeLadderTests.steppedDown(to: .n256, plan: plan)
        TileShapeLadderTests.runImages(&ladder, count: 3, plan: plan)
        #expect(ladder.dwellCount == 3)

        let response = MemoryPressureResponse.respond(to: .normal, ladder: &ladder, plan: plan)
        #expect(response.outcome == nil)
        #expect(response.subsequentTileShape == nil)
        #expect(response.checkpoints == false)
        #expect(ladder.dwellCount == 3, "the ladder was not consulted, so no dwell was spent")

        // The dwell survived intact, so the climb it earned still happens.
        // 1 073 741 824 − 334 829 158 = 738 912 666 of headroom at the candidate rung.
        let climb = ladder.beginImage(plan)
        #expect(climb.policy == .stepUpBetweenImages)
        #expect(climb.shape == .n384)
        #expect(climb.reason == .headroomSustained(images: 3, headroomBytes: 738_912_666))
    }

    @Test("a warn at the same point does spend the dwell — the difference normal must not make")
    func warnSpendsTheDwell() throws {
        let plan = try MemoryTests.budget(available: 2_147_483_648).plan(for: MemoryTests.workloadB)
        // The same stepped-down ladder and the same three images as the `.normal` case above, so
        // the only difference between the two tests is the level. Built on a *fresh* ladder this
        // would have reached the warn with a dwell of 0 and proved nothing by spending it.
        var ladder = try TileShapeLadderTests.steppedDown(to: .n256, plan: plan)
        TileShapeLadderTests.runImages(&ladder, count: 3, plan: plan)
        #expect(ladder.dwellCount == 3, "the dwell the warn is about to spend")

        _ = MemoryPressureResponse.respond(to: .warning, ladder: &ladder, plan: plan)
        #expect(ladder.dwellCount == 0)

        let declined = ladder.beginImage(plan)
        #expect(declined.policy == .stepUpDeclined)
        #expect(declined.reason == .dwellNotMet(images: 0, required: 3))
    }

    // MARK: - The latch between the kernel's queue and the tile loop

    @Test("the monitor starts at normal and has nothing pending")
    func monitorStartsQuiet() {
        let monitor = MemoryPressureMonitor(source: ManualPressureSource())
        #expect(monitor.level == .normal)
        #expect(monitor.consumePending() == nil)
    }

    @Test("the latch keeps the worst level seen since the last consume, not the latest")
    func latchKeepsTheWorstLevel() {
        let source = ManualPressureSource()
        let monitor = MemoryPressureMonitor(source: source)
        monitor.start()

        // A warn that is over by the time the tile ends still happened.
        source.send(.warning)
        source.send(.normal)
        #expect(monitor.level == .normal, "the latest reading is still readable")
        #expect(monitor.consumePending() == .warning)
        #expect(monitor.consumePending() == nil, "consuming clears it")

        source.send(.warning)
        source.send(.critical)
        source.send(.warning)
        #expect(monitor.consumePending() == .critical)
    }

    @Test("start is idempotent and stop cancels the subscription")
    func startAndStopManageOneSubscription() {
        let source = ManualPressureSource()
        let monitor = MemoryPressureMonitor(source: source)
        monitor.start()
        monitor.start()
        source.send(.warning)
        #expect(monitor.observedEventCount == 1, "one subscription, one recorded event")

        monitor.stop()
        #expect(source.isActive == false)
        #expect(source.cancels == 1)
        source.send(.critical)
        #expect(monitor.observedEventCount == 1, "nothing is delivered after stop")
    }

    @Test("with nothing pending the tile-boundary call does not touch the ladder at all")
    func respondIsANoOpWithoutAnEvent() throws {
        let source = ManualPressureSource()
        let monitor = MemoryPressureMonitor(source: source)
        monitor.start()
        var (ladder, plan) = try Self.midImage(at: .n512)

        #expect(monitor.respond(ladder: &ladder, plan: plan) == nil)
        #expect(ladder.shape == .n512)
        #expect(ladder.dwellCount == 0)
    }

    @Test("an event delivered mid-tile is actuated at the next tile-boundary call")
    func respondActuatesTheLatchedEvent() throws {
        let source = ManualPressureSource()
        let monitor = MemoryPressureMonitor(source: source)
        monitor.start()
        var (ladder, plan) = try Self.midImage(at: .n512)

        source.send(.critical)
        let response = try #require(monitor.respond(ladder: &ladder, plan: plan))
        #expect(response.level == .critical)
        #expect(response.checkpoints)
        #expect(ladder.shape == .n384)
        #expect(monitor.respond(ladder: &ladder, plan: plan) == nil, "one event, one response")
        #expect(ladder.shape == .n384, "the rung does not keep falling on its own")
    }

    // MARK: - The shipped source

    @Test("the shipped dispatch source activates and cancels without a real pressure event")
    func dispatchSourceLifecycleIsSafe() {
        // Lifecycle only. A real `warn` cannot be raised from a host test, so this claims nothing
        // about kernel behaviour — it pins that the dispatch source can be created, subscribed and
        // torn down on the host that runs this suite, and that cancelling twice is safe.
        let source = DispatchMemoryPressureSource()
        source.activate { _ in }
        source.cancel()
        source.cancel()

        let monitor = MemoryPressureMonitor()
        monitor.start()
        monitor.stop()
        #expect(monitor.level == .normal)
    }
}

/// A pressure source the test drives by hand — the whole reason `MemoryPressureSource` exists.
///
/// Delivery is synchronous on the calling thread, so every test above is deterministic: no queue
/// hop, no expectation, no sleep.
final class ManualPressureSource: MemoryPressureSource, @unchecked Sendable {
    private let lock = NSLock()
    private var handler: (@Sendable (MemoryPressureLevel) -> Void)?
    private var cancelCount = 0

    func activate(_ handler: @escaping @Sendable (MemoryPressureLevel) -> Void) {
        lock.withLock { self.handler = handler }
    }

    func cancel() {
        lock.withLock {
            handler = nil
            cancelCount += 1
        }
    }

    /// Deliver one level, exactly as the kernel's handler would.
    func send(_ level: MemoryPressureLevel) {
        let handler = lock.withLock { self.handler }
        handler?(level)
    }

    var isActive: Bool { lock.withLock { handler != nil } }
    var cancels: Int { lock.withLock { cancelCount } }
}
