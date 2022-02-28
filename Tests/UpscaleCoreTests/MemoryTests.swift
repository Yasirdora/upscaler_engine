import CoreGraphics
import Foundation
import Testing

@testable import UpscaleCore

/// Tests for the proactive memory budget.
///
/// **Model-free, device-free, syscall-free by construction.** `MemoryBudget` never calls
/// `os_proc_available_memory()`; availability is injected as a `Figure`. That is what lets every
/// assertion below be a literal computed by hand rather than a value the code under test produced.
///
/// The literals are derived once, here, so a reader can check them without running anything:
///
/// - 1 MiB = 1 048 576 B.
/// - Measured per-shape peaks (iPhone 12 Pro, `cpuAndNeuralEngine`): 227.7 MiB → 238 760 755 B ·
///   203.9 MiB → 213 804 646 B · 267.1 MiB → 280 074 650 B.
/// - Workload B (12MP, 4032×3024, ×4): output 16128×12096, row = 16128 × 4 = 64 512 B.
///   `centerValidRegion` keeps `core × scale` rows, core = N − 2 × 52: 256 → 152 × 4 = 608 rows → 39
///   223 296 B · 384 → 1120 → 72 253 440 B · 512 → 1632 → 105 283 584 B.
/// - Workload B input surface: 4032 × 3024 × 4 = 48 771 072 B (46.5 MiB).
/// - Totals at concurrency 1 (peak + input + destination): 256 → 326 755 123 · 384 → 334 829 158 ·
///   512 → 434 129 306 B.
struct MemoryTests {

    // MARK: - Fixtures (descriptions only — no arithmetic lives here)

    /// Workload B, the primary 12MP case (certification spec §7.1).
    static let workloadB = ImageWorkload(
        inputWidth: 4032, inputHeight: 3024, scaleFactor: 4, contextMargin: 52,
        method: .centerValidRegion, destination: .bandSpool)

    /// Workload A, 1080p.
    static let workloadA = ImageWorkload(
        inputWidth: 1920, inputHeight: 1080, scaleFactor: 4, contextMargin: 52,
        method: .centerValidRegion, destination: .bandSpool)

    /// Workload C, the 48MP survival case.
    static let workloadC = ImageWorkload(
        inputWidth: 8064, inputHeight: 6048, scaleFactor: 4, contextMargin: 52,
        method: .centerValidRegion, destination: .bandSpool)

    static func budget(available: Int, rho: Double = 0.5) throws -> MemoryBudget {
        let fraction = try #require(MemoryBudget.SafetyFraction(rho))
        return MemoryBudget(
            available: MemoryBudget.Figure(
                bytes: available, provenance: .measured, source: "test injection"),
            safetyFraction: fraction)
    }

    static let unknownBudget = MemoryBudget(
        available: .unavailable(source: "test injection — availability withheld"))

    // MARK: - ρ, the safety fraction

    @Test("ρ is 0.5, named, and is the default")
    func rhoDefaultsToOneHalf() {
        #expect(MemoryBudget.SafetyFraction.standard.value == 0.5)
        #expect(MemoryBudget.SafetyFraction.permittedRange == 0.4...0.6)

        let budget = MemoryBudget(
            available: MemoryBudget.Figure(bytes: 1, provenance: .measured, source: "test"))
        #expect(budget.safetyFraction == MemoryBudget.SafetyFraction.standard)
    }

    @Test(
        "the permitted range is enforced, not clamped",
        arguments: [
            (0.39, false), (0.4, true), (0.5, true), (0.6, true), (0.61, false), (0.0, false),
            (1.0, false),
        ])
    func safetyFractionRefusesValuesOutsideThePermittedRange(value: Double, accepted: Bool) {
        #expect((MemoryBudget.SafetyFraction(value) != nil) == accepted)
    }

    @Test("the allowance is ρ × available, floored")
    func allowanceAppliesTheSafetyFraction() throws {
        // 2 GiB = 2 147 483 648 B.
        // × 0.4 = 858 993 459.2 → 858 993 459 · × 0.5 = 1 073 741 824 · × 0.6 = 1 288 490 188.8
        //   → 1 288 490 188.
        #expect(try Self.budget(available: 2_147_483_648, rho: 0.4).allowance?.bytes == 858_993_459)
        #expect(try Self.budget(available: 2_147_483_648, rho: 0.5).allowance?.bytes == 1_073_741_824)
        #expect(try Self.budget(available: 2_147_483_648, rho: 0.6).allowance?.bytes == 1_288_490_188)
    }

    @Test("an unavailable availability yields no allowance at all — not a zero one")
    func allowanceIsAbsentWhenAvailabilityIs() {
        #expect(Self.unknownBudget.allowance == nil)
    }

    // MARK: - The measured per-shape table

    @Test(
        "the per-shape table is the measured floor-device medians, converted exactly",
        arguments: [
            (TileShape.n256, 238_760_755), (TileShape.n384, 213_804_646),
            (TileShape.n512, 280_074_650),
        ])
    func measuredPeaksMatchTheMeasuredTable(shape: TileShape, bytes: Int) {
        let figure = MemoryBudget.measuredPeak(at: shape)
        #expect(figure.bytes == bytes)
        #expect(figure.provenance == .measured)
        #expect(figure.source.contains("iPhone 12 Pro"))
    }

    @Test("the measured table is not monotonic in shape — 384 sits below 256")
    func measuredTableIsNotMonotonic() {
        // Addendum 3's finding, pinned so nobody "fixes" it into an N² curve: the 512 peak is
        // 1.17× the 256 peak where area scaling predicts 4×, and 384 measures below both.
        #expect(MemoryBudget.measuredPeak(at: .n384).bytes < MemoryBudget.measuredPeak(at: .n256).bytes)
        #expect(MemoryBudget.measuredPeak(at: .n256).bytes < MemoryBudget.measuredPeak(at: .n512).bytes)
    }

    // MARK: - The destination term (the band spool's cost model)

    @Test(
        "the band-spool destination costs one open window, not one image",
        arguments: [
            (TileShape.n256, 39_223_296), (TileShape.n384, 72_253_440), (TileShape.n512, 105_283_584),
        ])
    func bandSpoolDestinationIsOneWindow(shape: TileShape, bytes: Int) {
        #expect(Self.workloadB.destinationBytes(at: shape) == bytes)
    }

    @Test("the whole-image destination is still the 744.2 MiB the spool replaced")
    func wholeImageDestinationIsTheFigureRequirementTwoReplaced() {
        let whole = ImageWorkload(
            inputWidth: 4032, inputHeight: 3024, scaleFactor: 4, contextMargin: 52,
            method: .centerValidRegion, destination: .wholeImage)
        // 16128 × 12096 × 4 = 780 337 152 B = 744.2 MiB — the same literal OutputSurfaceTests pins.
        #expect(whole.destinationBytes(at: .n512) == 780_337_152)
        #expect(whole.destinationBytes(at: .n256) == 780_337_152, "shape cannot shrink a whole image")
    }

    @Test("the stitch method changes the window, exactly as openRowCount says")
    func destinationFollowsTheStitchMethod() {
        let feather = ImageWorkload(
            inputWidth: 4032, inputHeight: 3024, scaleFactor: 4, contextMargin: 52,
            method: .linearFeather, destination: .bandSpool)
        let hard = ImageWorkload(
            inputWidth: 4032, inputHeight: 3024, scaleFactor: 4, contextMargin: 52,
            method: .hardCrop, destination: .bandSpool)

        // feathers: (408 + 2×52) × 4 = 2048 rows × 64 512 B = 132 120 576 B
        #expect(feather.destinationBytes(at: .n512) == 132_120_576)
        // hardCrop: (512 + 408) × 4 = 3680 rows × 64 512 B = 237 404 160 B
        #expect(hard.destinationBytes(at: .n512) == 237_404_160)
    }

    @Test("a window taller than the image is clamped to the image, as BandSpoolSurface clamps it")
    func destinationIsClampedToTheImage() {
        let tiny = ImageWorkload(
            inputWidth: 100, inputHeight: 70, scaleFactor: 4, contextMargin: 52,
            method: .centerValidRegion, destination: .bandSpool)
        // Output 400×280; the 608-row window does not fit, so 280 rows × 1600 B = 448 000 B.
        #expect(tiny.destinationBytes(at: .n256) == 448_000)
    }

    @Test("the budget clamps the margin the way the shipped planner does")
    func plannedMarginAgreesWithTheShippedPlanner() {
        // A margin of 200 at tile 256 would leave no core, so TilePlanner clamps to (256−1)/2 = 127.
        let grid = TilePlanner(scaleFactor: 4).plan(
            imageSize: CGSize(width: 4032, height: 3024), tileSize: 256, contextMargin: 200)
        #expect(grid.contextMargin == 127)
        #expect(ImageWorkload.plannedMargin(tileSize: 256, requested: 200) == 127)

        // core = 256 − 254 = 2 → 8 rows × 64 512 B = 516 096 B. Budgeting the *requested* 200 would
        // have produced a negative core and a nonsense window.
        let clamped = ImageWorkload(
            inputWidth: 4032, inputHeight: 3024, scaleFactor: 4, contextMargin: 200,
            method: .centerValidRegion, destination: .bandSpool)
        #expect(clamped.destinationBytes(at: .n256) == 516_096)
    }

    // MARK: - Assessment arithmetic

    @Test("concurrency is pinned at 1")
    func concurrencyIsPinnedAtOne() throws {
        #expect(MemoryBudget.concurrency == 1)
        let assessment = try Self.budget(available: 2_147_483_648).assess(Self.workloadB, at: .n512)
        #expect(assessment.model.bytes == 280_074_650, "the pinned term must not scale the peak")
    }

    @Test("the input surface is its own term — the source image is live for the whole job")
    func inputSurfaceIsCarriedAsItsOwnTerm() throws {
        // 4032 × 3024 × 4 = 48 771 072 B = 46.5 MiB.
        #expect(Self.workloadB.inputSurfaceBytes == 48_771_072)
        // 8064 × 6048 × 4 = 195 084 288 B = 186.0 MiB — too large to leave out of a 48MP budget.
        #expect(Self.workloadC.inputSurfaceBytes == 195_084_288)

        let assessment = try Self.budget(available: 2_147_483_648).assess(Self.workloadB, at: .n512)
        #expect(assessment.input.bytes == 48_771_072)
        #expect(assessment.input.provenance == .estimated)
    }

    @Test(
        "total = measured peak × concurrency + input surface + destination window",
        arguments: [
            (TileShape.n256, 326_755_123), (TileShape.n384, 334_829_158), (TileShape.n512, 434_129_306),
        ])
    func totalIsTheSumOfTheThreeTerms(shape: TileShape, total: Int) throws {
        let assessment = try Self.budget(available: 2_147_483_648).assess(Self.workloadB, at: shape)
        #expect(assessment.totalBytes == total)
    }

    @Test("a workload that fits reports its headroom")
    func aWorkloadThatFits() throws {
        // 2 GiB available → allowance 1 073 741 824; 512 costs 434 129 306.
        let assessment = try Self.budget(available: 2_147_483_648).assess(Self.workloadB, at: .n512)
        #expect(assessment.fit == .fits)
        #expect(assessment.headroomBytes == 639_612_518)
    }

    @Test("a workload that does not fit reports a negative headroom, not a zero")
    func aWorkloadThatDoesNotFit() throws {
        // 800 MB available → allowance 400 000 000; 512 costs 434 129 306 → short by 34 129 306.
        let assessment = try Self.budget(available: 800_000_000).assess(Self.workloadB, at: .n512)
        #expect(assessment.fit == .doesNotFit)
        #expect(assessment.headroomBytes == -34_129_306)
    }

    @Test("the fit boundary is inclusive — exactly the allowance fits")
    func exactlyTheAllowanceFits() throws {
        // 868 258 612 × 0.5 = 434 129 306, exactly the 512 total.
        let onTheLine = try Self.budget(available: 868_258_612).assess(Self.workloadB, at: .n512)
        #expect(onTheLine.fit == .fits)
        #expect(onTheLine.headroomBytes == 0)

        // Two bytes less → allowance 434 129 305, one byte short.
        let justUnder = try Self.budget(available: 868_258_610).assess(Self.workloadB, at: .n512)
        #expect(justUnder.fit == .doesNotFit)
        #expect(justUnder.headroomBytes == -1)
    }

    // MARK: - Shape selection at each of the three rungs

    @Test("with room to spare the plan takes the largest rung")
    func selectsFiveTwelveWhenItFits() throws {
        let plan = try Self.budget(available: 2_147_483_648).plan(for: Self.workloadB)
        #expect(plan.selection == .largestThatFits(.n512))
        #expect(plan.shape == .n512)
        #expect(plan.fit == .fits)
        #expect(plan.headroomBytes == 639_612_518)
    }

    @Test("with 512 out of reach the plan takes 384")
    func selectsThreeEightyFour() throws {
        // allowance 400 000 000: 512 needs 434 129 306, 384 needs 334 829 158.
        let plan = try Self.budget(available: 800_000_000).plan(for: Self.workloadB)
        #expect(plan.selection == .largestThatFits(.n384))
        #expect(plan.headroomBytes == 65_170_842)
    }

    @Test("with only the floor rung affordable the plan takes 256")
    func selectsTwoFiftySix() throws {
        // allowance 330 000 000: 384 needs 334 829 158, 256 needs 326 755 123.
        let plan = try Self.budget(available: 660_000_000).plan(for: Self.workloadB)
        #expect(plan.selection == .largestThatFits(.n256))
        #expect(plan.headroomBytes == 3_244_877)
    }

    @Test("when no rung fits the plan says so, with the floor rung's shortfall")
    func noRungFits() throws {
        // allowance 200 000 000; the floor rung costs 326 755 123 → short by 126 755 123.
        let plan = try Self.budget(available: 400_000_000).plan(for: Self.workloadB)
        #expect(plan.selection == .noRungFits(floor: .n256, shortfallBytes: 126_755_123))
        #expect(plan.shape == .n256)
        #expect(plan.fit == .doesNotFit)
        #expect(plan.headroomBytes == -126_755_123)
    }

    @Test("unknown availability selects the floor rung and says the fit is unknown")
    func unknownAvailabilitySelectsTheFloorRung() {
        let plan = Self.unknownBudget.plan(for: Self.workloadB)
        #expect(plan.selection == .floorRungAvailabilityUnknown(.n256))
        #expect(TileShape.floorRung == .n256)
        #expect(plan.fit == .unknown, "unknown is not the same answer as no")
        #expect(plan.headroomBytes == nil)
    }

    @Test("a plan costs every rung, ascending, so nothing has to be re-costed")
    func planCoversEveryRung() throws {
        let plan = try Self.budget(available: 2_147_483_648).plan(for: Self.workloadB)
        #expect(plan.assessments.map(\.shape) == [.n256, .n384, .n512])
        #expect(plan.assessments.map(\.totalBytes) == [326_755_123, 334_829_158, 434_129_306])
        #expect(plan.selected.shape == .n512)
    }

    // MARK: - Provenance

    @Test("the model term is measured, the destination term is estimated, the assessment is both")
    func provenanceLabelsAreCarriedAndCombined() throws {
        let assessment = try Self.budget(available: 2_147_483_648).assess(Self.workloadB, at: .n512)
        #expect(assessment.model.provenance == .measured)
        #expect(assessment.destination.provenance == .estimated)
        #expect(assessment.allowance?.provenance == .measured)
        // Weakest link: a measurement plus an estimate is an estimate.
        #expect(assessment.provenance == .estimated)
    }

    @Test("an unavailable input poisons the whole assessment's provenance")
    func unavailableProvenanceDominates() {
        let assessment = Self.unknownBudget.assess(Self.workloadB, at: .n512)
        #expect(assessment.provenance == .unavailable)
        #expect(assessment.fit == .unknown)
    }

    @Test(
        "weakest is the ordering measured < estimated < unavailable",
        arguments: [
            ([MemoryBudget.Provenance.measured, .measured], MemoryBudget.Provenance.measured),
            ([.measured, .estimated], .estimated),
            ([.estimated, .unavailable], .unavailable),
            ([.measured, .unavailable], .unavailable),
            ([], .unavailable),
        ])
    func weakestProvenanceWins(values: [MemoryBudget.Provenance], expected: MemoryBudget.Provenance) {
        #expect(MemoryBudget.Provenance.weakest(values) == expected)
    }

    @Test("every figure renders its provenance and its citation")
    func figuresDescribeThemselves() {
        let figure = MemoryBudget.measuredPeak(at: .n512)
        #expect(figure.description.hasPrefix("267.1 MiB [measured]"))
        #expect(figure.mebibytes > 267.0 && figure.mebibytes < 267.2)
    }

    // MARK: - The injection seam

    @Test("os_proc_available_memory is read outside the budget, and labelled by platform")
    func availableMemoryIsInjectedNeverCalledFromTheBudget() {
        let figure = AvailableMemory.now()
        #if os(iOS) || os(tvOS) || os(watchOS) || os(visionOS)
            // On a device the API answers, unless it returns 0 (not an app / limit exceeded).
            #expect(figure.provenance == .measured || figure.provenance == .unavailable)
            #expect(figure.provenance != .measured || figure.bytes > 0)
        #else
            // macOS is absent from Apple's availability list, and this is the host that runs this
            // suite — which is exactly why the budget takes the value injected.
            #expect(figure.provenance == .unavailable)
            #expect(figure.bytes == 0)
        #endif
    }

    // MARK: - The claims the band spool and the ladder rest on

    @Test("48MP survives band-spooled where the whole-image surface cannot")
    func fortyEightMegapixelSurvivesOnlyOnTheSpooledPath() throws {
        // 3 GiB available → allowance 1 610 612 736, against a floor device measured at a
        // 3.07–3.12 GiB ceiling.
        let budget = try Self.budget(available: 3_221_225_472)

        // Band-spooled at 512: 280 074 650 + 195 084 288 + 1632 × 129 024 (= 210 567 168)
        // = 685 726 106 B, comfortably inside the allowance.
        let spooled = budget.plan(for: Self.workloadC)
        #expect(spooled.assessments.map(\.totalBytes).last == 685_726_106)
        #expect(spooled.selection == .largestThatFits(.n512))

        // Whole-image: the surface alone is 32256 × 24192 × 4 = 3 121 348 608 B, so even the floor
        // rung costs 3 555 193 651 and is short by 1 944 580 915 B. No tile-size policy can close
        // that — which is the whole argument for the band-spooled destination.
        let whole = ImageWorkload(
            inputWidth: 8064, inputHeight: 6048, scaleFactor: 4, contextMargin: 52,
            method: .centerValidRegion, destination: .wholeImage)
        let wholePlan = budget.plan(for: whole)
        #expect(wholePlan.selected.totalBytes == 3_555_193_651)
        #expect(wholePlan.selection == .noRungFits(floor: .n256, shortfallBytes: 1_944_580_915))
    }

    @Test("at 1080p the cheapest rung is 384, not the floor rung")
    func theFloorRungIsNotAlwaysTheCheapest() throws {
        // A consequence of the measured table's non-monotonicity, recorded rather than papered over.
        // Input surface 1920 × 1080 × 4 = 8 294 400; rows × 30 720 B/row:
        // 256 → 238 760 755 + 8 294 400 + 18 677 760 = 265 732 915
        // 384 → 213 804 646 + 8 294 400 + 34 406 400 = 256 505 446   ← cheaper than the floor rung
        // 512 → 280 074 650 + 8 294 400 + 50 135 040 = 338 504 090
        let plan = try Self.budget(available: 2_147_483_648).plan(for: Self.workloadA)
        #expect(plan.assessments.map(\.totalBytes) == [265_732_915, 256_505_446, 338_504_090])
        #expect(plan.selection == .largestThatFits(.n512), "the largest shape that fits")
    }

    // MARK: - The engine seam

    @Test("the engine supplies its own context margin to the workload")
    func engineWorkloadCarriesTheEnginesMargin() {
        let engine = SequentialTilingEngine(contextMargin: 52)
        let workload = engine.workload(
            imageWidth: 4032, imageHeight: 3024,
            model: ModelDescriptor(
                identifier: "test", scaleFactor: 4, contextMargin: 76,
                packageURL: URL(fileURLWithPath: "/dev/null"), inputTileSize: 512,
                precision: .float16),
            method: .centerValidRegion, destination: .bandSpool)

        #expect(workload.contextMargin == 52)
        #expect(workload.scaleFactor == 4)
        #expect(workload.outputWidth == 16128)
        #expect(workload.outputHeight == 12096)
        #expect(workload.outputRowBytes == 64512)
        // The descriptor's own 512 must not leak in: the shape is what the budget solves for.
        #expect(workload.destinationBytes(at: .n256) == 39_223_296)
    }
}
