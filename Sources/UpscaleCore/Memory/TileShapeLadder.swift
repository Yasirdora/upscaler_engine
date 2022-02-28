import Foundation

// MARK: - Ladder order

extension TileShape {
    /// The rungs in ladder order, ascending: 256 → 384 → 512.
    ///
    /// The certification spec (§5 item 5, §10) names the sequence explicitly ("512→384→256 retry"),
    /// so the order is fixed by the specification and is not this policy's to choose.
    ///
    /// **It is a shape ordering, never a cost ordering.** The measured per-shape table is
    /// non-monotonic — 384 costs 213.8 MiB against 256's 238.8 MiB (measured on an iPhone 12 Pro) —
    /// so at 1080p the 384 rung's *total* lands below the 256 floor's. A step down this ladder can
    /// therefore raise the projected total. `TileShapeLadder.Outcome.projectedDeltaBytes` reads
    /// that off the plan's own assessments rather than assuming it away: assuming that a smaller
    /// shape always means less memory would reintroduce the N² cost model the measurements
    /// falsified.
    static var ladderOrder: [TileShape] { allCases.sorted() }

    /// The next rung down, or `nil` at the floor rung.
    var nextRungDown: TileShape? {
        let order = Self.ladderOrder
        guard let index = order.firstIndex(of: self), index > 0 else { return nil }
        return order[index - 1]
    }

    /// The next rung up, or `nil` at the top rung.
    var nextRungUp: TileShape? {
        let order = Self.ladderOrder
        guard let index = order.firstIndex(of: self), index + 1 < order.count else { return nil }
        return order[index + 1]
    }
}

// MARK: - The ladder

/// The adaptive tile-shape ladder.
///
/// The decision layer between the memory budget (which states a fact: does this workload fit, at
/// which shape, with how much headroom) and the engine (which needs one shape and a way to survive
/// when that shape stops being affordable).
///
/// ## Step down mid-image, step up between images
///
/// A step **down** happens mid-job; a step **up** happens only between images, because the planner
/// cannot re-grid mid-assembly. The two rules are consistent once "step down" is defined precisely:
///
/// - A step down means **abandon the current tile at the current shape and retry that tile at the
///   smaller shape.** The planner then re-grids **for the remainder of the image, from that tile
///   forward.** Tiles already assembled are never re-gridded.
/// - *"Mid-job"* therefore means **mid-image, not mid-tile.**
///
/// The asymmetry between the two directions follows from one fact: **already-stitched output is
/// never revisited.** A step down only ever touches the not-yet-assembled remainder, so it is legal
/// mid-image. A step up would have to re-grid *backwards* over work already written into the
/// destination — the rows are gone, spooled and closed (`BandSpoolSurface.closeRows`) — so it is
/// only ever legal at an image boundary, where there is no assembled work to contradict.
///
/// This is the behaviour the certification spec requires: *"Adaptive tile fallback (512→384→256) →
/// **retry** → continue; whole-job failure is a defect"* (§10), and *"forced-failure tests must
/// demonstrate 512→384→256 **retry** and successful completion"* (§5 item 5).
///
/// ## How the engine executes a step down
///
/// 1. `TilePlanning` has a resume entry point that plans the remainder of an image from a given
///    row, alongside the whole-image `plan(imageSize:tileSize:contextMargin:)`, which is unchanged.
/// 2. Planning the remainder as its own sub-image and translating the rects was rejected: it
///    changes pixels. `TilePlanner.windowOrigin` slides the fixed window inside the extent it is given
///    and clamps to it, so the resumed row's window would take **no context above it at all**, where
///    an interior row of a full-image plan sees the full context margin of real rows.
/// 3. `SequentialTilingEngine.run` holds one `ModelDescriptor`, one fixed-shape `InferenceBackend`
///    and one input tile buffer allocated at `model.inputTileSize`, so a retry at a smaller shape
///    needs a second compiled shape live at once; the compiled-model cache key permits that.
/// 4. `BandSpoolSurface.windowRows` is fixed at construction from the *initial* grid, so a
///    mid-image step down does **not** shrink the destination window. It relieves only the per-shape
///    model term, and combined with the non-monotonic table that relief can be negative: at 1080p,
///    stepping 384 → 256 raises the projected total by 9 227 469 B and frees no destination memory at
///    all. `Outcome.projectedDeltaBytes` surfaces this rather than assuming it away.
///
/// An image assembled from two tile shapes is not pixel-identical to a single-shape run: measured
/// divergence across shapes is ≤ 0.007 dB. Ladder-path equivalence is therefore a Level A/B quality
/// claim under the regime framework of certification spec §11, not a pixel-identity claim.
///
/// This type decides **which rung**, **why**, and **when the decision is allowed to be taken** —
/// nothing else.
///
/// ## What this deliberately does not do
///
/// No `DispatchSourceMemoryPressure`, no signal source, no notification handling: `MemoryPressure`
/// *actuates* the transitions defined here by mapping its events onto `Trigger`. This type is pure,
/// `Sendable`, allocation-light and injectable, so the pressure monitor and the engine both test
/// against it without a device, a model or a pressure event.
///
/// ## Cost
///
/// Nothing here polls, sleeps or subscribes. **Ladder decisions are amortized per image**:
/// `beginImage(_:)` and `endImage()` run once per image, and `respond(to:plan:)` runs once per
/// *trigger* — a pressure event or a failed tile — never once per tile. It re-uses the
/// `MemoryBudget.Plan` computed at image start rather than re-planning, so a step down costs no
/// syscall, no allocation and no second pass over anything.
///
/// ## State, and why it is a value type
///
/// Dwell is state, so the ladder carries it and the transitions are `mutating` — a struct the
/// caller owns, not a shared actor. A caller that needs to share it across concurrency domains can
/// wrap it without this type changing.
public struct TileShapeLadder: Sendable, Equatable {

    // MARK: Tunables

    /// A ladder tuning number that knows where it came from — `MemoryBudget.Figure`'s idea, applied
    /// to quantities that are not bytes.
    ///
    /// It exists so the dwell default cannot be mistaken for a tuned value: a number without
    /// provenance is a defect, and the honest label for a number no device run has produced yet is
    /// `.unavailable`.
    public struct Tunable<Value: Sendable & Equatable>: Sendable, Equatable {
        public let value: Value
        public let provenance: MemoryBudget.Provenance
        /// Citation, or — where there is nothing to cite yet — the reason there is not.
        public let source: String

        public init(value: Value, provenance: MemoryBudget.Provenance, source: String) {
            self.value = value
            self.provenance = provenance
            self.source = source
        }
    }

    /// The dwell and margin parameters of the ladder, each carrying its own provenance.
    ///
    /// The *approach* is fixed — step up between images only, with the margin derived from ρ —
    /// while the numeric defaults are outputs of device measurement, not inputs to it. Accordingly:
    ///
    /// | Parameter | Default at ρ = 0.5 | Provenance | Status |
    /// |---|---|---|---|
    /// | `stepUpMarginFraction` | 0.5 | `.estimated` | derived from ρ; derivation is a design choice |
    /// | `dwellImages` | 3 | `.unavailable` | **placeholder** — no device evidence exists |
    ///
    /// `dwellImages` is a placeholder in the strict sense: 3 is the number that lets the state
    /// machine run and it is supported by nothing. It is labelled `.unavailable` rather than
    /// `.estimated` because there is no measurement and no derivation behind it, and `isMeasured`
    /// is `false` so a caller can refuse to certify against it.
    public struct Parameters: Sendable, Equatable {
        /// Consecutive completed images that must each end with the next rung up clearing the
        /// margin test before the ladder is allowed to step up — the "sustained" in "sustained
        /// headroom".
        public let dwellImages: Tunable<Int>

        /// Fraction of the allowance that must still be free *at the candidate rung* before a step
        /// up is taken.
        public let stepUpMarginFraction: Tunable<Double>

        public init(dwellImages: Tunable<Int>, stepUpMarginFraction: Tunable<Double>) {
            self.dwellImages = dwellImages
            self.stepUpMarginFraction = stepUpMarginFraction
        }

        /// The defaults, derived from ρ where a derivation exists and labelled where one does not.
        ///
        /// The margin is `MemoryBudget.SafetyFraction.stepUpMarginFraction` — 1 − ρ, the same
        /// proportional reserve inside the allowance that ρ already keeps outside it. At the
        /// standard ρ = 0.5 that is 0.5.
        public static func defaults(
            safetyFraction: MemoryBudget.SafetyFraction = .standard
        ) -> Parameters {
            Parameters(
                dwellImages: Tunable(
                    value: 3, provenance: .unavailable,
                    source: "PLACEHOLDER — no device evidence; the certification spec §6 protocol "
                        + "runs produce this number"),
                stepUpMarginFraction: Tunable(
                    value: safetyFraction.stepUpMarginFraction, provenance: .estimated,
                    source: "1 − ρ, ρ = \(safetyFraction.value) (margin derived from ρ). The "
                        + "derivation is a design choice, not a measurement — pending "
                        + "device runs"))
        }

        /// Weakest link across the tunables. Today `.unavailable`, because dwell is a placeholder.
        public var provenance: MemoryBudget.Provenance {
            MemoryBudget.Provenance.weakest([dwellImages.provenance, stepUpMarginFraction.provenance])
        }

        /// `true` only when every tunable has been measured on a device. **Today: `false`.**
        public var isMeasured: Bool { provenance == .measured }

        /// Bytes of headroom the candidate rung must clear, given the plan's allowance.
        ///
        /// Rounded **up**: a threshold you must clear rounds up, an allowance you may occupy rounds
        /// down (`MemoryBudget.allowance`). Both round toward not running out of memory.
        public func requiredHeadroomBytes(allowance: Int) -> Int {
            Int((Double(allowance) * stepUpMarginFraction.value).rounded(.up))
        }
    }

    // MARK: What moved the ladder

    /// The events that can push the ladder down. **Requirement 5 maps its pressure source onto
    /// these; this type never subscribes to anything.**
    ///
    /// One trigger steps down exactly one rung. A `critical` event does not skip two: nothing
    /// measured supports a bigger jump, and inventing one would be a desk guess dressed as policy.
    public enum Trigger: Sendable, Equatable {
        /// The budget says the current rung does not fit, short by this many bytes.
        case budgetShortfall(bytes: Int)
        /// A `warn`-level pressure event (`DispatchSourceMemoryPressure.warning`).
        case pressureWarning
        /// A `critical`-level pressure event.
        case pressureCritical
        /// A tile failed at the current rung — the injected seam certification spec §5 item 5 tests
        /// through.
        case tileFailure(String)

        var reason: Reason {
            switch self {
            case .budgetShortfall(let bytes): return .budgetShortfall(bytes: bytes)
            case .pressureWarning: return .pressureWarning
            case .pressureCritical: return .pressureCritical
            case .tileFailure(let detail): return .tileFailure(detail)
            }
        }
    }

    /// Which policy produced an outcome.
    public enum Policy: String, Sendable, CaseIterable {
        /// The budget's plan was taken as-is: the largest shape that fits, or 256 when unknown.
        case plannedRung
        /// Between images: the plan capped the rung below the one the ladder was carrying.
        case stepDownToFitBudget
        /// Mid-image: abandon the current tile, retry it at the smaller shape, and re-grid the
        /// **remainder** of the image from that tile forward (certification spec §10: "fallback →
        /// retry → continue"). Assembled tiles are never revisited.
        case stepDownMidImage
        /// At the floor rung with a trigger still outstanding, or with no rung fitting at all. The
        /// job continues, degraded — certification spec §6: complete or degrade gracefully, never
        /// crash.
        case floorRungHeld
        /// Between images: stepped up one rung on sustained headroom.
        case stepUpBetweenImages
        /// A step up was considered and declined. Never a failure.
        case stepUpDeclined
    }

    /// Why the policy applied.
    public enum Reason: Sendable, Equatable {
        /// The budget's own selection, carried verbatim so the reason keeps its shortfall figure.
        case plan(MemoryBudget.Selection)
        case budgetShortfall(bytes: Int)
        case pressureWarning
        case pressureCritical
        case tileFailure(String)
        /// Enough consecutive images cleared the margin; `headroomBytes` is the candidate rung's.
        case headroomSustained(images: Int, headroomBytes: Int)
        case dwellNotMet(images: Int, required: Int)
        /// `headroomBytes` is `nil` when there is no allowance to test against at all.
        case marginNotMet(headroomBytes: Int?, requiredBytes: Int)
        /// An image is in flight: the planner cannot re-grid mid-assembly, so no step up.
        case midImageAssembly
    }

    /// The typed engine outcome: the shape applied, the policy that applied it, and the reason.
    ///
    /// Presentation is left to the UI layer — the engine stays headless, so this carries numbers
    /// and cases, never user-facing prose.
    public struct Outcome: Sendable, Equatable, CustomStringConvertible {
        /// The shape the engine runs at after this decision.
        public let shape: TileShape
        /// The shape it was running at before. Equal to `shape` when nothing moved.
        public let previousShape: TileShape
        public let policy: Policy
        public let reason: Reason

        /// Projected total at `shape` minus the projected total at `previousShape`, read off the
        /// plan's assessments. `nil` when the plan did not cost one of the two rungs.
        ///
        /// **Positive means the move costs *more* memory, not less** — and for a step *down* that
        /// is a real, measured possibility, not a defensive nicety. See `TileShape.ladderOrder`.
        public let projectedDeltaBytes: Int?

        public init(
            shape: TileShape, previousShape: TileShape, policy: Policy, reason: Reason,
            projectedDeltaBytes: Int?
        ) {
            self.shape = shape
            self.previousShape = previousShape
            self.policy = policy
            self.reason = reason
            self.projectedDeltaBytes = projectedDeltaBytes
        }

        /// The job continues in a degraded state. Certification spec §6's 48 MP rule is satisfied
        /// by continuing; this is what a UI layer surfaces.
        public var isDegraded: Bool { policy == .floorRungHeld }

        public var movedRung: Bool { shape != previousShape }

        /// A step **down** the ladder that the plan says will cost more memory than the rung it
        /// left. Non-monotonicity made visible instead of assumed away.
        public var stepsDownButCostsMore: Bool {
            shape < previousShape && (projectedDeltaBytes ?? 0) > 0
        }

        public var description: String {
            let movement =
                movedRung
                ? "\(previousShape.rawValue) → \(shape.rawValue)" : "held at \(shape.rawValue)"
            return "\(movement) [\(policy.rawValue)] — \(reason)"
        }
    }

    /// The step-up margin test for one candidate rung, with both sides of the comparison kept so a
    /// declined step can say by how much it was declined.
    struct MarginCheck: Sendable, Equatable {
        let candidate: TileShape
        let headroomBytes: Int
        let requiredBytes: Int

        var isMet: Bool { headroomBytes >= requiredBytes }
    }

    // MARK: State

    public let parameters: Parameters

    /// The rung the engine is running at.
    public private(set) var shape: TileShape

    /// Consecutive completed images that ended with the next rung up clearing the margin test.
    public private(set) var dwellCount: Int

    /// `true` between `beginImage(_:)` and `endImage()`. While it is set the planner has already
    /// gridded the image, so no step up may be taken.
    public private(set) var isMidImage: Bool

    /// Whether the image in flight is still on course to earn a dwell credit. Cleared by any
    /// downward trigger: an image that needed a step down did not have sustained headroom.
    private var imageEarnsDwellCredit: Bool

    /// Whether `beginImage(_:)` has ever run. Exists only to let the **first** plan be adopted
    /// without dwell; every later upward move is a climb and is gated as before. Deliberately not
    /// `dwellCount == 0`: a ladder that has run images and been reset to zero dwell by a step down
    /// is not a fresh ladder, and must still earn its way up.
    private var hasPlannedAnImage = false

    /// Start at the floor rung by default — "256 when unknown" is also the safe place to start when
    /// nothing has been planned yet. The first `beginImage(_:)` raises it to whatever the budget's
    /// plan endorses, without dwell: dwell gates *climbing*, not the first plan.
    public init(parameters: Parameters = .defaults(), startingAt shape: TileShape = .floorRung) {
        self.parameters = parameters
        self.shape = shape
        self.dwellCount = 0
        self.isMidImage = false
        self.imageEarnsDwellCredit = false
    }

    // MARK: Transitions — between images

    /// Take the rung for the image about to be planned. **Once per image**, before the planner runs.
    ///
    /// This is the only entry point that may raise the rung, because it is the only one that runs
    /// while the planner can still act on a new shape. Everything it can do:
    ///
    /// - the plan says no rung fits → hold the floor rung and report a degraded outcome;
    /// - the plan caps below the carried rung → step down to the plan;
    /// - the plan is above the carried rung → step up **one** rung, and only if dwell *and* margin
    ///   are both satisfied; otherwise decline and say which one failed;
    /// - otherwise → run at the planned rung.
    ///
    /// Called while an image is already in flight it changes nothing and says why. That is a
    /// caller error, and it is answered rather than trapped so a mis-sequenced caller degrades to
    /// "no step up" instead of crashing a job at 80 %.
    public mutating func beginImage(_ plan: MemoryBudget.Plan) -> Outcome {
        let previous = shape
        guard !isMidImage else {
            return Outcome(
                shape: previous, previousShape: previous, policy: .stepUpDeclined,
                reason: .midImageAssembly, projectedDeltaBytes: 0)
        }

        let outcome: Outcome
        if case .noRungFits = plan.selection {
            shape = plan.shape
            outcome = Outcome(
                shape: shape, previousShape: previous, policy: .floorRungHeld,
                reason: .plan(plan.selection),
                projectedDeltaBytes: Self.projectedDelta(plan, from: previous, to: shape))
        } else if plan.shape < previous {
            shape = plan.shape
            outcome = Outcome(
                shape: shape, previousShape: previous, policy: .stepDownToFitBudget,
                reason: .plan(plan.selection),
                projectedDeltaBytes: Self.projectedDelta(plan, from: previous, to: shape))
        } else if plan.shape > previous, !hasPlannedAnImage {
            // The first plan is adopted, not climbed to: a fresh ladder starts where the budget
            // says rather than spending three images at the floor. Routing this through `climb(_:)`
            // would leave a new ladder unable to leave the floor rung until three images had earned
            // dwell, whatever the budget selected — a 10-run warm series would spend its first
            // three runs at 256 regardless of headroom, measuring the floor rung rather than the
            // ladder.
            //
            // `.plannedRung` rather than a new case: its own documentation already names exactly
            // this — "the budget's plan was taken as-is". Dwell still gates every *subsequent*
            // climb; it never gated the first plan.
            shape = plan.shape
            outcome = Outcome(
                shape: shape, previousShape: previous, policy: .plannedRung,
                reason: .plan(plan.selection),
                projectedDeltaBytes: Self.projectedDelta(plan, from: previous, to: shape))
        } else if plan.shape > previous {
            outcome = climb(plan)
        } else {
            outcome = Outcome(
                shape: previous, previousShape: previous, policy: .plannedRung,
                reason: .plan(plan.selection), projectedDeltaBytes: 0)
        }

        isMidImage = true
        hasPlannedAnImage = true
        imageEarnsDwellCredit = marginCheck(above: shape, plan: plan)?.isMet ?? false
        return outcome
    }

    /// Close the image. Credits a dwell image, or resets the count to zero.
    ///
    /// Dwell counts *completed* images that never needed a step down and whose next rung up cleared
    /// the margin — "sustained headroom" measured over images, which is the only cadence at which a
    /// step up can be acted on anyway.
    ///
    /// A climb spends the count (`climb(_:)` resets it), but the climbing image still earns its
    /// credit here: it ran at the new rung with the rung *above that* clear throughout, which is
    /// evidence toward the next step and not evidence that was already spent. So a climbing image
    /// counts as the **first** of the `dwellImages` required at its new rung, not a fourth image's
    /// worth of extra wait.
    public mutating func endImage() {
        guard isMidImage else { return }
        isMidImage = false
        dwellCount = imageEarnsDwellCredit ? dwellCount + 1 : 0
        imageEarnsDwellCredit = false
    }

    // MARK: Transitions — down, at any time

    /// Step down one rung in response to a trigger, and continue. **Never fails the job.**
    ///
    /// Called once per *trigger*, never per tile. It takes no new reading and does not re-budget:
    /// `plan` is the one computed at image start, used only to report what the move costs.
    ///
    /// **Mid-image** this is the step down described on the type: the current tile is abandoned,
    /// retried at the returned shape, and the image is re-gridded from that tile forward. Tiles
    /// already assembled keep the geometry they were written with — the ladder never asks for work
    /// already spooled to be revisited. **Between images** it simply lowers the rung the next image
    /// is planned at.
    ///
    /// **There is no path through this method that raises the rung.** That is what makes "step up
    /// only between images" structural rather than a convention: raising the rung would imply
    /// re-gridding backwards over assembled output, and only `beginImage(_:)` — which runs when
    /// there is no assembled output to contradict — can do it.
    ///
    /// At the floor rung it returns a `.floorRungHeld` outcome and keeps going: 256 is the floor,
    /// and below it certification spec §6's 48 MP rule applies — degrade gracefully, never crash.
    public mutating func respond(to trigger: Trigger, plan: MemoryBudget.Plan) -> Outcome {
        let previous = shape
        dwellCount = 0
        imageEarnsDwellCredit = false

        guard let next = previous.nextRungDown else {
            return Outcome(
                shape: previous, previousShape: previous, policy: .floorRungHeld,
                reason: trigger.reason, projectedDeltaBytes: 0)
        }
        shape = next
        return Outcome(
            shape: next, previousShape: previous, policy: .stepDownMidImage,
            reason: trigger.reason,
            projectedDeltaBytes: Self.projectedDelta(plan, from: previous, to: next))
    }

    // MARK: Internals

    /// One rung up, if dwell and margin both allow it.
    private mutating func climb(_ plan: MemoryBudget.Plan) -> Outcome {
        let previous = shape
        let required = parameters.dwellImages.value
        guard dwellCount >= required else {
            return Outcome(
                shape: previous, previousShape: previous, policy: .stepUpDeclined,
                reason: .dwellNotMet(images: dwellCount, required: required),
                projectedDeltaBytes: 0)
        }

        let check = marginCheck(above: previous, plan: plan)
        guard let check, check.isMet else {
            return Outcome(
                shape: previous, previousShape: previous, policy: .stepUpDeclined,
                reason: .marginNotMet(
                    headroomBytes: check?.headroomBytes, requiredBytes: check?.requiredBytes ?? 0),
                projectedDeltaBytes: 0)
        }

        let dwelt = dwellCount
        shape = check.candidate
        dwellCount = 0
        return Outcome(
            shape: shape, previousShape: previous, policy: .stepUpBetweenImages,
            reason: .headroomSustained(images: dwelt, headroomBytes: check.headroomBytes),
            projectedDeltaBytes: Self.projectedDelta(plan, from: previous, to: shape))
    }

    /// The margin test for the rung above `shape`. `nil` at the top rung, and `nil` when the plan
    /// has no allowance — an unknown availability can never justify climbing.
    func marginCheck(above shape: TileShape, plan: MemoryBudget.Plan) -> MarginCheck? {
        guard let candidate = shape.nextRungUp,
            let allowance = plan.allowance,
            let assessment = plan.assessments.first(where: { $0.shape == candidate }),
            let headroom = assessment.headroomBytes
        else { return nil }
        return MarginCheck(
            candidate: candidate, headroomBytes: headroom,
            requiredBytes: parameters.requiredHeadroomBytes(allowance: allowance.bytes))
    }

    /// What moving from one rung to another does to the projected total, per the plan's own
    /// assessments. The ladder reads the ordering instead of assuming it.
    static func projectedDelta(
        _ plan: MemoryBudget.Plan, from: TileShape, to: TileShape
    ) -> Int? {
        guard let before = plan.assessments.first(where: { $0.shape == from }),
            let after = plan.assessments.first(where: { $0.shape == to })
        else { return nil }
        return after.totalBytes - before.totalBytes
    }
}
