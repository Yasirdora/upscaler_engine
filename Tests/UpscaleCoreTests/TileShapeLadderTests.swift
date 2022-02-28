import Foundation
import Testing

@testable import UpscaleCore

/// Tests for the adaptive tile-shape ladder.
///
/// **Model-free, device-free, syscall-free.** The ladder is a value type over an injected
/// `MemoryBudget.Plan`, so every expectation below is a literal a reader can check by hand — never
/// a value recomputed by the code under test.
///
/// Workload fixtures and the injected budget come from `MemoryTests`, where their arithmetic is
/// derived line by line. The totals those derivations produce, reused here:
///
/// | Workload | 256 | 384 | 512 |
/// |---|---|---|---|
/// | B — 12MP (4032×3024) | 326 755 123 | 334 829 158 | 434 129 306 |
/// | A — 1080p (1920×1080) | 265 732 915 | **256 505 446** | 338 504 090 |
///
/// Workload A's 384 total sitting **below** its 256 total is the measured non-monotonicity the
/// ladder must not assume away.
///
/// At the fixture budget of 2 GiB available and ρ = 0.5 the allowance is 1 073 741 824 B and the
/// step-up margin is 1 − ρ = 0.5 of it → **536 870 912 B** of headroom required to climb.
struct TileShapeLadderTests {

    // MARK: - Fixtures

    /// 2 GiB available → allowance 1 073 741 824.
    static func roomyBudget() throws -> MemoryBudget { try MemoryTests.budget(available: 2_147_483_648) }

    static func plan(_ workload: ImageWorkload, available: Int) throws -> MemoryBudget.Plan {
        try MemoryTests.budget(available: available).plan(for: workload)
    }

    /// Run `count` complete images (begin → end) and hand back the begin outcomes.
    @discardableResult
    static func runImages(
        _ ladder: inout TileShapeLadder, count: Int, plan: MemoryBudget.Plan
    ) -> [TileShapeLadder.Outcome] {
        (0..<count).map { _ in
            let outcome = ladder.beginImage(plan)
            ladder.endImage()
            return outcome
        }
    }

    /// A ladder sitting **between images** at `shape`, having already run one — the state in which
    /// a rung below what the plan endorses has to be *earned* back.
    ///
    /// `TileShapeLadder(startingAt:)` does not hand back a ladder that will still be at `shape`
    /// after its first `beginImage(_:)`: a fresh ladder adopts the plan's rung without consulting
    /// dwell. So a ladder carrying less than the plan endorses is built the only way the engine
    /// ever reaches one — run an image, walk it down the ladder on triggers, close the image.
    ///
    /// The walk-down trigger is `.tileFailure` deliberately: the pressure suites use this fixture,
    /// and setting them up with the very transition they assert on would be circular. Every trigger
    /// steps down identically — `everyTriggerStepsDownOneRung` is what pins that.
    ///
    /// What comes back carries `dwellCount == 0`, the same count a fresh ladder carries, and is
    /// **not** fresh. That difference is the whole subject of `theFirstPlanIsAdoptedWithoutDwell`.
    static func steppedDown(
        to shape: TileShape, plan: MemoryBudget.Plan,
        parameters: TileShapeLadder.Parameters = .defaults()
    ) throws -> TileShapeLadder {
        var ladder = TileShapeLadder(parameters: parameters)
        _ = ladder.beginImage(plan)
        while ladder.shape > shape {
            _ = ladder.respond(to: .tileFailure("fixture: walking down to \(shape.rawValue)"), plan: plan)
        }
        ladder.endImage()
        try #require(
            ladder.shape == shape,
            "the plan endorses \(plan.shape.rawValue): a ladder is only ever driven down, never up")
        try #require(ladder.dwellCount == 0, "a step down spends the dwell")
        try #require(ladder.isMidImage == false, "the fixture hands back a closed image")
        return ladder
    }

    // MARK: - The ladder order

    @Test("the ladder order is the 256 → 384 → 512 sequence, with real ends")
    func ladderOrderIsTheSpecifiedSequence() {
        #expect(TileShape.ladderOrder == [.n256, .n384, .n512])

        #expect(TileShape.n512.nextRungDown == .n384)
        #expect(TileShape.n384.nextRungDown == .n256)
        #expect(TileShape.n256.nextRungDown == nil, "256 is the floor rung")

        #expect(TileShape.n256.nextRungUp == .n384)
        #expect(TileShape.n384.nextRungUp == .n512)
        #expect(TileShape.n512.nextRungUp == nil, "512 is the top rung")
    }

    // MARK: - Parameters: what is measured and what is not

    @Test("the dwell default is a labelled placeholder, not a tuned value")
    func dwellDefaultIsLabelledAsUnmeasured() {
        let parameters = TileShapeLadder.Parameters.defaults()
        #expect(parameters.dwellImages.value == 3)
        #expect(parameters.dwellImages.provenance == .unavailable)
        #expect(parameters.dwellImages.source.contains("PLACEHOLDER"))
        #expect(parameters.dwellImages.source.contains("no device evidence"))
        #expect(parameters.isMeasured == false, "no device run has produced these numbers yet")
        #expect(parameters.provenance == .unavailable, "weakest link: dwell has no figure at all")
    }

    @Test("the step-up margin is derived from ρ and labelled as a derivation")
    func marginIsDerivedFromRho() throws {
        let parameters = TileShapeLadder.Parameters.defaults()
        // 1 − ρ at the standard ρ = 0.5.
        #expect(parameters.stepUpMarginFraction.value == 0.5)
        #expect(parameters.stepUpMarginFraction.provenance == .estimated)
        #expect(parameters.stepUpMarginFraction.source.contains("pending device runs"))

        // The derivation tracks ρ across its whole permitted range, rather than being a pinned 0.5.
        let low = try #require(MemoryBudget.SafetyFraction(0.4))
        let high = try #require(MemoryBudget.SafetyFraction(0.6))
        #expect(low.stepUpMarginFraction == 0.6)
        #expect(MemoryBudget.SafetyFraction.standard.stepUpMarginFraction == 0.5)
        #expect(high.stepUpMarginFraction == 0.4)
        #expect(TileShapeLadder.Parameters.defaults(safetyFraction: high).stepUpMarginFraction.value == 0.4)
    }

    @Test("the required headroom rounds up — a threshold you must clear, not an allowance")
    func requiredHeadroomRoundsUp() {
        let parameters = TileShapeLadder.Parameters.defaults()
        // 400 000 001 × 0.5 = 200 000 000.5 → 200 000 001. Rounding down would let a step up
        // through on exactly half a byte less than the margin.
        #expect(parameters.requiredHeadroomBytes(allowance: 400_000_001) == 200_000_001)
        #expect(parameters.requiredHeadroomBytes(allowance: 400_000_000) == 200_000_000)
        // The fixture allowance: 1 073 741 824 × 0.5 = 536 870 912, exactly.
        #expect(parameters.requiredHeadroomBytes(allowance: 1_073_741_824) == 536_870_912)
    }

    // MARK: - Stepping down (certification spec §10: fallback → retry → continue)

    @Test(
        "every trigger steps down exactly one rung, from either rung above the floor",
        arguments: [
            TileShapeLadder.Trigger.budgetShortfall(bytes: 12_345),
            .pressureWarning, .pressureCritical, .tileFailure("injected"),
        ])
    func everyTriggerStepsDownOneRung(trigger: TileShapeLadder.Trigger) throws {
        let plan = try Self.plan(MemoryTests.workloadB, available: 2_147_483_648)

        var fromTop = TileShapeLadder(startingAt: .n512)
        let first = fromTop.respond(to: trigger, plan: plan)
        #expect(first.shape == .n384)
        #expect(first.previousShape == .n512)
        #expect(first.policy == .stepDownMidImage)
        #expect(fromTop.shape == .n384)

        // And again, to the floor — one rung per trigger, never two.
        let second = fromTop.respond(to: trigger, plan: plan)
        #expect(second.shape == .n256)
        #expect(second.previousShape == .n384)
        #expect(second.policy == .stepDownMidImage)
        #expect(fromTop.shape == .n256)
    }

    @Test("a step down carries the trigger's own reason, payload intact")
    func stepDownReportsWhy() throws {
        let plan = try Self.plan(MemoryTests.workloadB, available: 2_147_483_648)
        var ladder = TileShapeLadder(startingAt: .n512)

        let shortfall = ladder.respond(to: .budgetShortfall(bytes: 34_129_306), plan: plan)
        #expect(shortfall.reason == .budgetShortfall(bytes: 34_129_306))

        let failure = ladder.respond(to: .tileFailure("tile 47 refused"), plan: plan)
        #expect(failure.reason == .tileFailure("tile 47 refused"))
        #expect(failure.isDegraded == false, "256 was reached, but no further step was refused yet")
    }

    @Test("the typed outcome names the shape, the policy and the projected cost of the move")
    func outcomeContents() throws {
        let plan = try Self.plan(MemoryTests.workloadB, available: 2_147_483_648)
        var ladder = TileShapeLadder(startingAt: .n512)
        let outcome = ladder.respond(to: .pressureWarning, plan: plan)

        #expect(outcome.shape == .n384)
        #expect(outcome.previousShape == .n512)
        #expect(outcome.policy == .stepDownMidImage)
        #expect(outcome.reason == .pressureWarning)
        #expect(outcome.movedRung)
        #expect(outcome.isDegraded == false)
        // 334 829 158 − 434 129 306 = −99 300 148: at 12MP this step down really is cheaper.
        #expect(outcome.projectedDeltaBytes == -99_300_148)
        #expect(outcome.stepsDownButCostsMore == false)
        #expect(outcome.description == "512 → 384 [stepDownMidImage] — pressureWarning")
    }

    // MARK: - The floor rung

    @Test("the floor rung refuses to descend, forever, and says the job is degraded")
    func floorRungIsNeverBreached() throws {
        let plan = try Self.plan(MemoryTests.workloadB, available: 2_147_483_648)
        var ladder = TileShapeLadder(startingAt: .n256)

        for _ in 0..<5 {
            let outcome = ladder.respond(to: .pressureCritical, plan: plan)
            #expect(outcome.shape == .n256)
            #expect(outcome.previousShape == .n256)
            #expect(outcome.policy == .floorRungHeld)
            #expect(outcome.reason == .pressureCritical)
            #expect(outcome.isDegraded, "certification spec §6: degrade gracefully, never crash")
            #expect(outcome.movedRung == false)
            #expect(outcome.projectedDeltaBytes == 0)
            #expect(ladder.shape == .n256)
        }
        #expect(ladder.shape == TileShape.floorRung)
    }

    @Test("a plan where nothing fits runs at the floor rung, degraded, rather than failing")
    func noRungFitsRunsDegradedAtTheFloor() throws {
        // 400 MB available → allowance 200 000 000; the floor rung costs 326 755 123 → short by
        // 126 755 123 (the literal MemoryTests derives).
        let plan = try Self.plan(MemoryTests.workloadB, available: 400_000_000)
        var ladder = TileShapeLadder(startingAt: .n512)
        let outcome = ladder.beginImage(plan)

        #expect(outcome.shape == .n256)
        #expect(outcome.previousShape == .n512)
        #expect(outcome.policy == .floorRungHeld)
        #expect(outcome.reason == .plan(.noRungFits(floor: .n256, shortfallBytes: 126_755_123)))
        #expect(outcome.isDegraded)
        // 326 755 123 − 434 129 306 = −107 374 183.
        #expect(outcome.projectedDeltaBytes == -107_374_183)
    }

    // MARK: - The first plan is adopted, never climbed to

    /// The first `beginImage(_:)` takes whatever rung the budget endorses, *without consulting
    /// dwell*: the first image runs at the largest fitting shape, and a fresh ladder starts where
    /// the budget says rather than spending three images at the floor.
    ///
    /// Both halves of the rule are pinned here, because the second is what stops the first from
    /// being over-applied:
    ///
    /// 1. a **fresh** ladder adopts the plan — right shape, `.plannedRung`, `.plan(selection)`, the
    ///    projected delta of the move, and no dwell consulted;
    /// 2. a ladder that has *run* an image and been reset to zero dwell by a step down is **not**
    ///    fresh and must still earn its way up, one rung at a time.
    ///
    /// That is why the implementation's flag is `hasPlannedAnImage` and not `dwellCount == 0`: the
    /// two are indistinguishable at the moment of the second call, and collapsing them would let
    /// every stepped-down ladder jump straight back to the plan's rung, which is the dwell gate
    /// deleted rather than bypassed once. An edit that collapses them fails part 2.
    @Test("a fresh ladder adopts the plan's rung without dwell — and only a fresh one does")
    func theFirstPlanIsAdoptedWithoutDwell() throws {
        let plan = try Self.plan(MemoryTests.workloadB, available: 2_147_483_648)
        #expect(plan.selection == .largestThatFits(.n512))

        // 1. The fresh ladder. Two rungs in one call, which `climb(_:)` structurally cannot do.
        var fresh = TileShapeLadder(startingAt: .floorRung)
        #expect(fresh.dwellCount == 0)
        let adopted = fresh.beginImage(plan)

        #expect(adopted.shape == .n512, "the first image runs at the largest shape that fits")
        #expect(adopted.previousShape == .n256)
        #expect(adopted.policy == .plannedRung, "the plan was taken as-is, not climbed to")
        #expect(adopted.reason == .plan(.largestThatFits(.n512)))
        // 434 129 306 − 326 755 123 = 107 374 183: adopting the plan's rung costs this much more
        // than the floor it left, read off the plan's own assessments like every other move.
        #expect(adopted.projectedDeltaBytes == 107_374_183)
        #expect(adopted.movedRung)
        #expect(fresh.shape == .n512)
        #expect(fresh.isMidImage)
        fresh.endImage()

        // Dwell was not consulted, pinned by making it unsatisfiable: a first plan that asked the
        // dwell gate anything at all could never leave the floor with this ladder.
        let unmeetableDwell = TileShapeLadder.Parameters(
            dwellImages: TileShapeLadder.Tunable(
                value: Int.max, provenance: .unavailable,
                source: "test — a dwell no run could ever satisfy"),
            stepUpMarginFraction: TileShapeLadder.Parameters.defaults().stepUpMarginFraction)
        var barred = TileShapeLadder(parameters: unmeetableDwell, startingAt: .floorRung)
        let stillAdopted = barred.beginImage(plan)
        #expect(stillAdopted.shape == .n512)
        #expect(stillAdopted.policy == .plannedRung)
        #expect(stillAdopted.reason == .plan(.largestThatFits(.n512)))

        // 2. The boundary. This ladder has run an image and been walked back to the floor, so it
        // carries exactly the `dwellCount == 0` a fresh ladder carries — and is not fresh.
        var used = try Self.steppedDown(to: .n256, plan: plan)
        #expect(used.dwellCount == 0, "the same zero a fresh ladder starts with")

        let refused = used.beginImage(plan)
        #expect(refused.shape == .n256, "not fresh: the plan's 512 has to be earned")
        #expect(refused.previousShape == .n256)
        #expect(refused.policy == .stepUpDeclined)
        #expect(refused.reason == .dwellNotMet(images: 0, required: 3))
        #expect(refused.movedRung == false)
        #expect(refused.projectedDeltaBytes == 0)
        #expect(used.shape == .n256)
    }

    // MARK: - Stepping up: dwell and margin, both required

    @Test("a step up is refused until dwell is satisfied, and then takes exactly one rung")
    func stepUpNeedsDwellAndClimbsOneRung() throws {
        // Plan selects 512 (largest that fits at a 1 073 741 824 allowance), so every image below
        // is one the budget would happily have run at 512 — the ladder still makes it earn the climb.
        let plan = try Self.plan(MemoryTests.workloadB, available: 2_147_483_648)
        #expect(plan.selection == .largestThatFits(.n512))

        // The scenario the dwell gate actually governs: a ladder that has been **stepped down** and
        // must climb back. A *fresh* ladder never earns its way anywhere — it adopts the plan's
        // rung on the first image and skips this gate entirely
        // (`theFirstPlanIsAdoptedWithoutDwell`), so it cannot be the subject here.
        var ladder = try Self.steppedDown(to: .n256, plan: plan)
        let declined = Self.runImages(&ladder, count: 3, plan: plan)
        #expect(declined.map(\.policy) == [.stepUpDeclined, .stepUpDeclined, .stepUpDeclined])
        #expect(
            declined.map(\.reason) == [
                .dwellNotMet(images: 0, required: 3), .dwellNotMet(images: 1, required: 3),
                .dwellNotMet(images: 2, required: 3),
            ])
        #expect(declined.allSatisfy { $0.shape == .n256 })
        #expect(ladder.dwellCount == 3)

        // Fourth image: dwell 3 ≥ 3, and 384's headroom 1 073 741 824 − 334 829 158 = 738 912 666
        // clears the 536 870 912 margin.
        let climb = ladder.beginImage(plan)
        #expect(climb.shape == .n384, "one rung at a time — not straight to the plan's 512")
        #expect(climb.previousShape == .n256)
        #expect(climb.policy == .stepUpBetweenImages)
        #expect(climb.reason == .headroomSustained(images: 3, headroomBytes: 738_912_666))
        // 334 829 158 − 326 755 123 = 8 074 035: climbing costs more, which is the point of a margin.
        #expect(climb.projectedDeltaBytes == 8_074_035)
        #expect(ladder.dwellCount == 0, "the dwell is spent, not kept")

        // The climb image itself accrues toward the *next* rung: it runs at 384 with 512's headroom
        // (639 612 518 ≥ 536 870 912) clear throughout, which is exactly what dwell measures.
        ladder.endImage()
        #expect(ladder.dwellCount == 1)

        // So two more complete images, not three, buy the second rung.
        Self.runImages(&ladder, count: 2, plan: plan)
        #expect(ladder.dwellCount == 3)
        let top = ladder.beginImage(plan)
        #expect(top.shape == .n512)
        #expect(top.policy == .stepUpBetweenImages)
        #expect(top.reason == .headroomSustained(images: 3, headroomBytes: 639_612_518))
        #expect(top.projectedDeltaBytes == 99_300_148)
        ladder.endImage()

        // At the top rung the plan and the ladder agree and nothing moves.
        let held = ladder.beginImage(plan)
        #expect(held.policy == .plannedRung)
        #expect(held.reason == .plan(.largestThatFits(.n512)))
        #expect(held.movedRung == false)
    }

    @Test("dwell alone is not enough — the margin must clear at the candidate rung")
    func stepUpNeedsTheMarginToo() throws {
        // 1 320 000 000 available → allowance 660 000 000. 512 fits (434 129 306), so the plan
        // selects it. But 384's headroom is 660 000 000 − 334 829 158 = 325 170 842, short of the
        // 0.5 × 660 000 000 = 330 000 000 margin — so the ladder declines to climb into it.
        let plan = try Self.plan(MemoryTests.workloadB, available: 1_320_000_000)
        #expect(plan.selection == .largestThatFits(.n512))

        var ladder = try Self.steppedDown(to: .n256, plan: plan)
        Self.runImages(&ladder, count: 6, plan: plan)
        #expect(ladder.dwellCount == 0, "an image whose margin fails earns no dwell credit")

        let outcome = ladder.beginImage(plan)
        #expect(outcome.shape == .n256)
        #expect(outcome.policy == .stepUpDeclined)
        #expect(outcome.reason == .dwellNotMet(images: 0, required: 3))
    }

    @Test("with dwell forced, a failing margin is what declines the climb, and says by how much")
    func marginNotMetIsReportedWithBothSides() throws {
        // Same budget as above, but dwell is set to 0 so only the margin can decline the climb.
        let parameters = TileShapeLadder.Parameters(
            dwellImages: TileShapeLadder.Tunable(
                value: 0, provenance: .unavailable, source: "test — dwell removed from the test"),
            stepUpMarginFraction: TileShapeLadder.Parameters.defaults().stepUpMarginFraction)
        let plan = try Self.plan(MemoryTests.workloadB, available: 1_320_000_000)

        var ladder = try Self.steppedDown(to: .n256, plan: plan, parameters: parameters)
        let outcome = ladder.beginImage(plan)
        #expect(outcome.shape == .n256)
        #expect(outcome.policy == .stepUpDeclined)
        #expect(
            outcome.reason
                == .marginNotMet(headroomBytes: 325_170_842, requiredBytes: 330_000_000))
    }

    @Test("unknown availability can never justify a climb, however many images pass")
    func unknownAvailabilityNeverClimbs() {
        // No allowance exists, so there is nothing to take a margin of: "256 when unknown".
        let plan = MemoryTests.unknownBudget.plan(for: MemoryTests.workloadB)
        #expect(plan.selection == .floorRungAvailabilityUnknown(.n256))

        var ladder = TileShapeLadder(startingAt: .n512)
        let first = ladder.beginImage(plan)
        #expect(first.shape == .n256)
        #expect(first.policy == .stepDownToFitBudget)
        #expect(first.reason == .plan(.floorRungAvailabilityUnknown(.n256)))
        #expect(first.projectedDeltaBytes == -107_374_183)
        ladder.endImage()

        Self.runImages(&ladder, count: 10, plan: plan)
        #expect(ladder.shape == .n256)
        #expect(ladder.dwellCount == 0)
    }

    // MARK: - Step up cannot happen mid-image (the planner cannot re-grid mid-assembly)

    @Test("mid-image, a climb-worthy plan moves nothing — the grid is already emitted")
    func noStepUpWhileAnImageIsInFlight() throws {
        let plan = try Self.plan(MemoryTests.workloadB, available: 2_147_483_648)
        var ladder = try Self.steppedDown(to: .n256, plan: plan)
        Self.runImages(&ladder, count: 3, plan: plan)

        let opened = ladder.beginImage(plan)
        #expect(opened.policy == .stepUpBetweenImages)
        #expect(ladder.isMidImage)

        // The same plan, re-offered mid-assembly. It is fully climb-worthy and changes nothing.
        let refused = ladder.beginImage(plan)
        #expect(refused.policy == .stepUpDeclined)
        #expect(refused.reason == .midImageAssembly)
        #expect(refused.shape == .n384)
        #expect(refused.previousShape == .n384)
        #expect(refused.movedRung == false)
        #expect(ladder.shape == .n384)
        #expect(ladder.isMidImage, "declining a mis-sequenced call must not close the image")
    }

    @Test("mid-image the ladder can only ever descend — a step up would re-grid assembled work")
    func midImageRespondOnlyDescends() throws {
        let plan = try Self.plan(MemoryTests.workloadB, available: 2_147_483_648)
        var ladder = try Self.steppedDown(to: .n384, plan: plan)
        let opened = ladder.beginImage(plan)
        #expect(opened.policy == .stepUpDeclined, "not fresh, and dwell is 0 of 3 — the image opens at 384")
        #expect(ladder.shape == .n384)
        #expect(ladder.isMidImage)

        // The plan says 512 fits with 639 612 518 B of headroom. A mid-image trigger still goes down.
        let outcome = ladder.respond(to: .pressureWarning, plan: plan)
        #expect(outcome.shape == .n256)
        #expect(outcome.policy == .stepDownMidImage)
        #expect(ladder.shape == .n256)
        #expect(
            ladder.isMidImage,
            "the image is still in flight: the tile is retried and the remainder re-gridded")
    }

    @Test("a trigger spends the dwell — an image that needed a step down was not sustained")
    func aTriggerResetsTheDwell() throws {
        let plan = try Self.plan(MemoryTests.workloadB, available: 2_147_483_648)
        var ladder = try Self.steppedDown(to: .n256, plan: plan)
        Self.runImages(&ladder, count: 2, plan: plan)
        #expect(ladder.dwellCount == 2)

        _ = ladder.beginImage(plan)
        _ = ladder.respond(to: .pressureWarning, plan: plan)
        #expect(ladder.dwellCount == 0)
        ladder.endImage()
        #expect(ladder.dwellCount == 0, "the interrupted image earns no credit either")
    }

    // MARK: - The non-monotonic case

    @Test("at 1080p a step down from 384 to 256 costs MORE memory, and the outcome says so")
    func steppingDownCanCostMore() throws {
        // Workload A totals: 256 → 265 732 915, 384 → 256 505 446, 512 → 338 504 090. 384 measures
        // below 256 on device, so the ladder's last step is uphill.
        let plan = try Self.plan(MemoryTests.workloadA, available: 2_147_483_648)
        #expect(plan.assessments.map(\.totalBytes) == [265_732_915, 256_505_446, 338_504_090])

        var ladder = TileShapeLadder(startingAt: .n384)
        let uphill = ladder.respond(to: .pressureCritical, plan: plan)
        #expect(uphill.shape == .n256, "the order is 512→384→256; the ladder does not reorder it")
        // 265 732 915 − 256 505 446 = +9 227 469.
        #expect(uphill.projectedDeltaBytes == 9_227_469)
        #expect(
            uphill.stepsDownButCostsMore,
            "a ladder that assumed descending shape means descending memory would report false here")
    }

    @Test("the same ladder's other step down really is cheaper — the ordering is read, not assumed")
    func theOtherStepDownIsCheaper() throws {
        let plan = try Self.plan(MemoryTests.workloadA, available: 2_147_483_648)
        var ladder = TileShapeLadder(startingAt: .n512)
        let downhill = ladder.respond(to: .pressureCritical, plan: plan)
        // 256 505 446 − 338 504 090 = −81 998 644.
        #expect(downhill.projectedDeltaBytes == -81_998_644)
        #expect(downhill.stepsDownButCostsMore == false)
    }

    @Test("the delta is read from the plan's own assessments, all three rungs of it")
    func deltaComesFromThePlansAssessments() throws {
        let plan = try Self.plan(MemoryTests.workloadA, available: 2_147_483_648)
        #expect(TileShapeLadder.projectedDelta(plan, from: .n512, to: .n256) == -72_771_175)
        #expect(TileShapeLadder.projectedDelta(plan, from: .n256, to: .n512) == 72_771_175)
        #expect(TileShapeLadder.projectedDelta(plan, from: .n384, to: .n384) == 0)
    }
}
