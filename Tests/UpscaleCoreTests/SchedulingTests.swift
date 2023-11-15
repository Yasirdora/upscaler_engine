import Foundation
import Testing

@testable import UpscaleCore

/// The job and progress surface.
///
/// **Model-free, device-free, clock-free.** Nothing here loads a model, allocates a pixel buffer or
/// reads a clock: `JobTracker.report(_:at:)` takes its instant as an argument, so every ETA below
/// is arithmetic on numbers written into the test. No sleeps, no timing tolerance, no flakes.
///
/// Every expectation is a literal derived by hand in the comment above it. Where a value is the
/// result of a division it is compared with `close(_:_:)` at 1e-12 — the derivation is the rational
/// number stated in the comment, and the tolerance is there because a decimal literal is not always
/// the nearest `Double` to it, never to relax what is being asserted.
enum SchedulingFixture {

    /// Hand-derived expectation, at the tolerance that binary floating point makes necessary.
    static func close(_ actual: Double, _ expected: Double) -> Bool {
        abs(actual - expected) <= 1e-12
    }

    /// A pinned identity: nothing below asserts against a UUID the runtime happened to generate.
    static let id = Scheduling.JobID(
        UUID(uuidString: "00000000-0000-4000-8000-000000000001") ?? UUID())

    /// A tracker parked in `running(.inferring)` — the state the tile loop reports into, and the
    /// only state with a fraction to show.
    static func inferring() throws -> Scheduling.JobTracker {
        let tracker = Scheduling.JobTracker(id: id)
        try tracker.begin()
        try tracker.enter(.preparingModel)
        try tracker.enter(.inferring)
        return tracker
    }
}

// MARK: - Monotone progress

/// The rule on backwards progress, tested against the sequence the engine actually emits.
///
/// `SequentialTilingEngine` reports `base + (1 - base) · next / count`, and on a ladder step-down
/// it re-plans from the top of the current tile row, so `base` drops to that row's fraction and
/// `next` restarts at 1. That is deliberate engine behaviour. `MonotonicProgress` is the
/// presentation rule laid over it.
struct MonotonicProgressTests {

    @Test("an untroubled run is passed through unchanged")
    func passThrough() {
        // No revision, so floor = 0 and rawOrigin = 0: the mapping is the identity.
        var progress = Scheduling.MonotonicProgress()
        #expect(progress.record(0.125) == 0.125)
        #expect(progress.record(0.5) == 0.5)
        #expect(progress.record(1) == 1)
        #expect(progress.revisions == 0)
    }

    /// The marquee case, with the engine's own arithmetic written out.
    ///
    /// Geometry: a 4-row × 2-column grid (8 tiles), scale 2, output height 800, so each tile row is
    /// 200 output rows. Pass 1 reports `next/8` for next = 1…5. The ladder then steps down inside
    /// row 2, which re-plans from that row's start — input row 200, so `base = 200·2/800 = 0.5` —
    /// into a 2-row × 3-column grid of 6 tiles reporting `0.5 + 0.5 · next/6`.
    ///
    /// The raw signal therefore goes **0.625 → 0.5833…**: backwards, correctly, because the engine
    /// has just thrown away half a row of finished work and has more tiles to run than it thought.
    ///
    /// The displayed sequence, derived by hand: the drop is absorbed (floor = 5/8, rawOrigin = 7/12,
    /// span = 5/12), and every later raw maps to `5/8 + 3/8 · (raw − 7/12) / (5/12)`:
    ///
    /// | raw | (raw − 7/12) ÷ (5/12) | display |
    /// |---|---|---|
    /// | 8/12 | 1/5 | 5/8 + 3/40 = 0.7 |
    /// | 9/12 | 2/5 | 5/8 + 6/40 = 0.775 |
    /// | 10/12 | 3/5 | 5/8 + 9/40 = 0.85 |
    /// | 11/12 | 4/5 | 5/8 + 12/40 = 0.925 |
    /// | 12/12 | 1 | 1.0 |
    @Test("a ladder step-down is absorbed: the bar holds for one tile, then climbs to exactly 1")
    func ladderStepDownIsAbsorbed() {
        let raw: [Double] = [
            0.125, 0.25, 0.375, 0.5, 0.625,
            0.5833333333333333, 0.6666666666666666, 0.75, 0.8333333333333333, 0.9166666666666667, 1,
        ]
        let expected: [Double] = [
            0.125, 0.25, 0.375, 0.5, 0.625,
            0.625, 0.7, 0.775, 0.85, 0.925, 1,
        ]

        var progress = Scheduling.MonotonicProgress()
        let shown = raw.map { progress.record($0) }

        #expect(shown.count == expected.count)
        for (index, value) in shown.enumerated() {
            #expect(
                SchedulingFixture.close(value, expected[index]),
                "tile \(index): showed \(value), expected \(expected[index])")
        }
        #expect(progress.revisions == 1)

        // The property that separates this rule from `max(raw, shown)`. A clamp would hold 0.625
        // for the whole replayed row — six tiles of a frozen bar, which on a 4 602-tile plan is a
        // long silence indistinguishable from a hang. Here the bar resumes on the very next tile.
        #expect(shown[5] == shown[4], "the drop itself must not move the bar")
        #expect(shown[6] > shown[5], "the tile after the drop must move it again")
        #expect(progress.value == 1, "and the pass must still end at exactly 1")
    }

    /// The full ladder, 512 → 384 → 256: certification spec §5.5's two step-downs, at the surface.
    ///
    /// Rebase 1: floor 1/2, rawOrigin 1/4, span 3/4. raw 1/2 maps to 1/2 + 1/2 · (1/4 ÷ 3/4) = 2/3.
    /// Rebase 2: floor 2/3, rawOrigin 1/8, span 7/8. raw 1 maps to 2/3 + 1/3 · 1 = 1.
    @Test("two step-downs in one image still end at exactly 1, and count as two revisions")
    func twoStepDowns() {
        var progress = Scheduling.MonotonicProgress()
        #expect(progress.record(0.5) == 0.5)
        #expect(progress.record(0.25) == 0.5)
        #expect(SchedulingFixture.close(progress.record(0.5), 0.6666666666666666))
        #expect(SchedulingFixture.close(progress.record(0.125), 0.6666666666666666))
        #expect(progress.record(1) == 1)
        #expect(progress.revisions == 2)
    }

    @Test("the displayed value never decreases, whatever the engine reports")
    func neverDecreases() {
        // Deliberately adversarial: rises, collapses, oscillates, ends low.
        let raw: [Double] = [0.4, 0.9, 0.1, 0.15, 0.05, 0.2, 0.9, 0.3, 0.31]
        var progress = Scheduling.MonotonicProgress()
        var previous = 0.0
        for value in raw {
            let shown = progress.record(value)
            #expect(shown >= previous, "\(value) drove the bar from \(previous) to \(shown)")
            #expect(shown <= 1)
            previous = shown
        }
        // Three decreases in that sequence, counted by hand: 0.9→0.1, 0.15→0.05 and 0.9→0.3.
        #expect(progress.revisions == 3)
    }

    @Test("a non-finite report is ignored outright, and does not poison the mapping")
    func nonFiniteIsIgnored() {
        var progress = Scheduling.MonotonicProgress()
        #expect(progress.record(0.5) == 0.5)
        #expect(progress.record(.nan) == 0.5)
        #expect(progress.record(.infinity) == 0.5)
        #expect(progress.record(-.infinity) == 0.5)
        // Neither a revision nor a new origin: the identity mapping is still in force.
        #expect(progress.revisions == 0)
        #expect(progress.record(0.75) == 0.75)
    }

    @Test("out-of-range reports are clamped, and a clamped decrease is still a revision")
    func clamping() {
        var progress = Scheduling.MonotonicProgress()
        #expect(progress.record(2) == 1)
        #expect(progress.record(0.5) == 1, "1.0 is the floor now; the bar holds")
        #expect(progress.revisions == 1)

        var second = Scheduling.MonotonicProgress()
        #expect(second.record(0.5) == 0.5)
        #expect(second.record(-1) == 0.5, "−1 clamps to 0, which is a decrease")
        #expect(second.revisions == 1)
    }

    /// A revision that lands on exactly 1.0 leaves nothing to map. The guard exists so the mapping
    /// cannot divide by a zero span; without it the next report would be a NaN on the bar.
    @Test("a revision at exactly 1.0 holds the floor instead of dividing by a zero span")
    func revisionAtFullSpan() {
        var progress = Scheduling.MonotonicProgress()
        #expect(progress.record(1) == 1)
        #expect(progress.record(0.5) == 1)
        #expect(progress.record(1) == 1)
        #expect(progress.value == 1)
    }
}

// MARK: - The state machine

/// Every legal transition, and every rejection.
struct JobStateMachineTests {

    // MARK: Legal

    @Test("a new job is queued, with no bar and no estimate")
    func queued() {
        let snapshot = Scheduling.JobTracker(id: SchedulingFixture.id).snapshot
        #expect(snapshot.id == SchedulingFixture.id)
        #expect(snapshot.state == .queued)
        #expect(snapshot.progress == nil)
        #expect(snapshot.estimate == nil)
        #expect(snapshot.revisions == 0)
        #expect(snapshot.phase == nil)
        #expect(snapshot.isTerminal == false)
    }

    @Test("the four phases walk forward, and only the tile phase shows a bar")
    func phasesWalkForward() throws {
        let tracker = Scheduling.JobTracker(id: SchedulingFixture.id)

        #expect(try tracker.begin().state == .preparing(.preparingInput))
        #expect(tracker.snapshot.progress == nil, "decode has no honest fraction")

        // The phase that exists because `MLModel.compileModel` is seconds of silence on a cache
        // miss (`CoreMLBackend.load`).
        #expect(try tracker.enter(.preparingModel).state == .preparing(.preparingModel))
        #expect(tracker.snapshot.progress == nil)

        #expect(try tracker.enter(.inferring).state == .running(.inferring))
        #expect(tracker.snapshot.progress == 0, "the bar appears at zero when tiles start")

        #expect(try tracker.enter(.encoding).state == .running(.encoding))
        #expect(
            tracker.snapshot.progress == 1,
            "every tile really is done; the phase label, not the bar, says the file is still being written")

        #expect(try tracker.finish().state == .finished)
        #expect(tracker.snapshot.progress == 1)
        #expect(tracker.snapshot.isTerminal)
    }

    @Test("a phase may be skipped forward — a warm run has no model to prepare")
    func phaseMaySkipForward() throws {
        let tracker = Scheduling.JobTracker(id: SchedulingFixture.id)
        try tracker.begin()
        #expect(try tracker.enter(.inferring).state == .running(.inferring))
        #expect(try tracker.enter(.encoding).state == .running(.encoding))
    }

    @Test("a queued job that is cancelled stops at once, with no cancelling state to render")
    func cancellingAQueuedJobIsImmediate() {
        let tracker = Scheduling.JobTracker(id: SchedulingFixture.id)
        let snapshot = tracker.requestCancellation()
        #expect(snapshot.state == .failed(.cancelled))
        #expect(snapshot.progress == nil)
    }

    @Test(
        "cancellation mid-phase keeps the phase it interrupted",
        arguments: [Scheduling.JobPhase.preparingInput, .preparingModel, .inferring, .encoding])
    func cancellationKeepsThePhase(phase: Scheduling.JobPhase) throws {
        let tracker = Scheduling.JobTracker(id: SchedulingFixture.id)
        try tracker.begin()
        if phase != .preparingInput { try tracker.enter(phase) }

        let snapshot = tracker.requestCancellation()
        #expect(snapshot.state == .cancelling(phase))
        #expect(snapshot.phase == phase)
        #expect(snapshot.isTerminal == false)
        #expect(snapshot.estimate == nil, "a stopping job makes no claim about time remaining")
    }

    @Test("a cancelled job ends as failed(.cancelled), which is terminal but not an error")
    func cancellationIsTerminalButNotAnError() throws {
        let tracker = try SchedulingFixture.inferring()
        tracker.requestCancellation()
        let snapshot = tracker.fail(.cancelled)
        #expect(snapshot.state == .failed(.cancelled))
        #expect(snapshot.isTerminal)
        #expect(snapshot.progress == nil, "a finished-with-a-verdict job shows no bar")

        guard case .failed(let error) = snapshot.state else {
            Issue.record("expected a failed state")
            return
        }
        #expect(error.isCancellation)
        #expect(error.message == "Upscale cancelled.")
    }

    /// The engine's cancellation guard is at the *top* of the tile loop, so the tile already in
    /// flight when the user taps Cancel still finishes and still reports. That report is real
    /// completed work and the bar takes it.
    @Test("the in-flight tile still counts after cancellation is requested")
    func theInFlightTileStillCounts() throws {
        let tracker = try SchedulingFixture.inferring()
        tracker.report(0.5, at: 0)
        tracker.requestCancellation()

        let snapshot = tracker.report(0.75, at: 1)
        #expect(snapshot.state == .cancelling(.inferring))
        #expect(snapshot.progress == 0.75)
    }

    @Test("a job that finishes before the stop takes effect keeps its output")
    func cancellingMayStillFinish() throws {
        let tracker = try SchedulingFixture.inferring()
        try tracker.enter(.encoding)
        tracker.requestCancellation()
        #expect(try tracker.finish().state == .finished)
    }

    @Test(
        "a job can fail from any state that has not reached a verdict",
        arguments: [
            Scheduling.JobPhase.preparingInput, .preparingModel, .inferring, .encoding,
        ])
    func failsFromAnyLiveState(phase: Scheduling.JobPhase) throws {
        let tracker = Scheduling.JobTracker(id: SchedulingFixture.id)
        try tracker.begin()
        if phase != .preparingInput { try tracker.enter(phase) }

        let error = Scheduling.JobError.unexpected(stage: phase, detail: "forced")
        #expect(tracker.fail(error).state == .failed(error))
    }

    // MARK: Rejected

    @Test("a job cannot begin twice")
    func beginIsOnce() throws {
        let tracker = Scheduling.JobTracker(id: SchedulingFixture.id)
        try tracker.begin()
        #expect(throws: Scheduling.TransitionError.notQueued(.preparing(.preparingInput))) {
            try tracker.begin()
        }
    }

    @Test("nothing transitions after a verdict")
    func terminalRejectsEverything() throws {
        let tracker = try SchedulingFixture.inferring()
        try tracker.enter(.encoding)
        try tracker.finish()

        #expect(throws: Scheduling.TransitionError.alreadyTerminal(.finished)) { try tracker.begin() }
        #expect(throws: Scheduling.TransitionError.alreadyTerminal(.finished)) {
            try tracker.enter(.encoding)
        }
        #expect(throws: Scheduling.TransitionError.alreadyTerminal(.finished)) { try tracker.finish() }
    }

    @Test("a phase never repeats and never goes backwards")
    func phasesOnlyMoveForward() throws {
        let tracker = try SchedulingFixture.inferring()
        #expect(
            throws: Scheduling.TransitionError.phaseDidNotAdvance(from: .inferring, to: .inferring)
        ) { try tracker.enter(.inferring) }
        #expect(
            throws: Scheduling.TransitionError.phaseDidNotAdvance(
                from: .inferring, to: .preparingModel)
        ) { try tracker.enter(.preparingModel) }
        #expect(tracker.snapshot.state == .running(.inferring), "a rejected transition changes nothing")
    }

    @Test("a queued job has no phase to advance")
    func queuedHasNoPhase() {
        let tracker = Scheduling.JobTracker(id: SchedulingFixture.id)
        #expect(throws: Scheduling.TransitionError.noPhaseToAdvance(.queued)) {
            try tracker.enter(.inferring)
        }
    }

    @Test("a stopping job does not announce new work")
    func cancellingCannotAdvance() throws {
        let tracker = try SchedulingFixture.inferring()
        tracker.requestCancellation()
        #expect(throws: Scheduling.TransitionError.noPhaseToAdvance(.cancelling(.inferring))) {
            try tracker.enter(.encoding)
        }
    }

    /// The encode step writes the file. A job that "finished" before it ran claims a result that
    /// does not exist on disk — the one rejection here that would be a *user-visible* lie.
    @Test(
        "a job cannot finish before its output is written",
        arguments: [
            Scheduling.JobState.queued, .preparing(.preparingModel), .running(.inferring),
        ])
    func finishRequiresTheEncode(state: Scheduling.JobState) throws {
        let tracker = Scheduling.JobTracker(id: SchedulingFixture.id)
        switch state {
        case .preparing(let phase), .running(let phase):
            try tracker.begin()
            if phase != .preparingInput { try tracker.enter(phase) }
        default:
            break
        }
        #expect(tracker.snapshot.state == state)
        #expect(throws: Scheduling.TransitionError.outputNotWritten(state)) { try tracker.finish() }
    }

    @Test("a report outside the tile phase is ignored, not recorded")
    func reportsOutsideInferringAreIgnored() throws {
        let tracker = Scheduling.JobTracker(id: SchedulingFixture.id)
        try tracker.begin()
        #expect(tracker.report(0.5, at: 0).progress == nil)

        // If the value had been recorded silently, the bar would jump to 0.5 the moment tiles start.
        #expect(try tracker.enter(.inferring).state == .running(.inferring))
        #expect(tracker.snapshot.progress == 0)

        try tracker.enter(.encoding)
        tracker.report(0.25, at: 1)
        #expect(tracker.snapshot.progress == 1, "an encoding job is past its tiles")
    }

    @Test("cancellation is idempotent, and a verdict absorbs it")
    func cancellationIsIdempotent() throws {
        let tracker = try SchedulingFixture.inferring()
        #expect(tracker.requestCancellation().state == .cancelling(.inferring))
        #expect(tracker.requestCancellation().state == .cancelling(.inferring))

        let finished = Scheduling.JobTracker(id: SchedulingFixture.id)
        try finished.begin()
        try finished.enter(.encoding)
        try finished.finish()
        #expect(finished.requestCancellation().state == .finished)
    }

    @Test("the first verdict wins")
    func firstVerdictWins() throws {
        let tracker = try SchedulingFixture.inferring()
        let first = Scheduling.JobError.unexpected(stage: .inferring, detail: "the real cause")
        #expect(tracker.fail(first).state == .failed(first))
        #expect(tracker.fail(.cancelled).state == .failed(first))
        #expect(tracker.report(0.9, at: 9).state == .failed(first))
    }
}

// MARK: - The estimate

/// The ETA, and every case in which it refuses to answer.
struct JobEstimateTests {

    /// Eight tiles, four seconds apart, the bar reaching 1/4. Derived by hand:
    /// climbed = 8/32 − 1/32 = 7/32, over 32 − 4 = 28 s, so the rate is 7/(32·28) = 1/128 per
    /// second. Remaining = (1 − 1/4) ÷ 1/128 = 0.75 · 128 = **96 s**.
    @Test("an estimate arrives once the window is full, and says what it is made of")
    func estimateAfterAFullWindow() throws {
        let tracker = try SchedulingFixture.inferring()

        for tile in 1...7 {
            let snapshot = tracker.report(Double(tile) * 0.03125, at: Double(tile) * 4)
            #expect(snapshot.estimate == nil, "seven samples is not the window")
        }

        let snapshot = tracker.report(8 * 0.03125, at: 32)
        #expect(snapshot.progress == 0.25)
        #expect(
            snapshot.estimate
                == Scheduling.Estimate(
                    secondsRemaining: 96, sampleCount: 8, sampleSeconds: 28))
    }

    /// The window slides: adding a ninth tile drops the first. Samples 2…9 span 36 − 8 = 28 s and
    /// the same 7/32 of bar, so the rate is unchanged at 1/128 and the remaining bar is
    /// 1 − 9/32 = 23/32, giving 23/32 · 128 = **92 s**.
    @Test("the window slides, so the estimate tracks the recent rate rather than the whole job")
    func windowSlides() throws {
        let tracker = try SchedulingFixture.inferring()
        for tile in 1...9 { tracker.report(Double(tile) * 0.03125, at: Double(tile) * 4) }

        #expect(
            tracker.snapshot.estimate
                == Scheduling.Estimate(
                    secondsRemaining: 92, sampleCount: 8, sampleSeconds: 28))
    }

    /// The rule: a step-down withdraws the estimate rather than adjusting it. Both the tile count
    /// and the per-tile cost have just changed, so every measured interval describes work that is
    /// no longer the work remaining.
    @Test("a ladder step-down withdraws the estimate until eight fresh tiles have been measured")
    func stepDownWithdrawsTheEstimate() throws {
        let tracker = try SchedulingFixture.inferring()
        for tile in 1...8 { tracker.report(Double(tile) * 0.03125, at: Double(tile) * 4) }
        #expect(tracker.snapshot.estimate != nil)

        let stepDown = tracker.report(0.125, at: 36)
        #expect(stepDown.estimate == nil, "the plan changed; the measured intervals are stale")
        #expect(stepDown.revisions == 1)
        #expect(stepDown.progress == 0.25, "the bar holds where it was")

        // Seven fresh tiles are still not a window.
        for tile in 1...7 {
            let snapshot = tracker.report(0.125 + Double(tile) * 0.03125, at: 36 + Double(tile) * 4)
            #expect(snapshot.estimate == nil)
        }
        #expect(tracker.report(0.125 + 8 * 0.03125, at: 68).estimate != nil)
    }

    @Test("a phase change clears the window")
    func phaseChangeClearsTheWindow() throws {
        let tracker = Scheduling.JobTracker(id: SchedulingFixture.id)
        try tracker.begin()
        try tracker.enter(.inferring)
        for tile in 1...8 { tracker.report(Double(tile) * 0.03125, at: Double(tile) * 4) }
        #expect(tracker.snapshot.estimate != nil)

        try tracker.enter(.encoding)
        #expect(tracker.snapshot.estimate == nil, "the encode is not measured by a tile counter")
    }

    @Test("a stopping job offers no estimate, even with a full window")
    func cancellingOffersNoEstimate() throws {
        let tracker = try SchedulingFixture.inferring()
        for tile in 1...8 { tracker.report(Double(tile) * 0.03125, at: Double(tile) * 4) }
        #expect(tracker.snapshot.estimate != nil)

        tracker.requestCancellation()
        #expect(tracker.snapshot.estimate == nil)
    }

    @Test("a bar that has not moved cannot be extrapolated")
    func aStalledBarGivesNoEstimate() throws {
        let tracker = try SchedulingFixture.inferring()
        for tile in 1...10 { tracker.report(0.25, at: Double(tile) * 4) }
        #expect(tracker.snapshot.progress == 0.25)
        #expect(tracker.snapshot.estimate == nil)
    }

    @Test("two tiles reported at the same instant are not a rate")
    func zeroElapsedGivesNoEstimate() throws {
        let tracker = try SchedulingFixture.inferring()
        for tile in 1...8 { tracker.report(Double(tile) * 0.03125, at: 4) }
        #expect(tracker.snapshot.estimate == nil)
    }

    /// The seam to `TilingEngine.upscale(…, progress:)`, whose signature is unchanged. A scripted
    /// clock stands in for the caller's monotonic one.
    @Test("the progress handler adapts the engine's bare Double without the engine changing")
    func progressHandlerAdaptsTheEngineCallback() throws {
        let tracker = try SchedulingFixture.inferring()
        let ticks = Ticks()
        let handler = tracker.progressHandler(clock: { ticks.next() })

        for tile in 1...8 { handler(Double(tile) * 0.03125) }

        #expect(tracker.snapshot.progress == 0.25)
        #expect(
            tracker.snapshot.estimate
                == Scheduling.Estimate(
                    secondsRemaining: 96, sampleCount: 8, sampleSeconds: 28))
    }

    /// A clock that advances four seconds per call, starting at four — the fixture's tile cadence.
    final class Ticks: @unchecked Sendable {
        private let lock = NSLock()
        private var count = 0

        func next() -> Double {
            lock.withLock {
                count += 1
                return Double(count) * 4
            }
        }
    }
}

// MARK: - Refusals

/// Decode's named pre-processing refusals, and the words shown for them.
///
/// Pinned as literals because this is copy: a change to any sentence below is a change to what a
/// user reads, and it should have to be made deliberately.
struct JobRefusalMessageTests {

    @Test("every named refusal has a sentence and a way forward")
    func namedRefusals() {
        let cases: [(Scheduling.JobError, String, String?)] = [
            (
                .unreadable(origin: "photo.jpg"), "This file couldn't be opened.",
                "Check that photo.jpg is still there and that you have permission to open it."
            ),
            (
                .unrecognisedFormat(origin: "photo.jpg"),
                "This file isn't in a format that can be read as an image.",
                "Supported formats are JPEG, PNG, HEIF and TIFF."
            ),
            (
                .corruptOrTruncated(format: "JPEG", origin: "photo.jpg"),
                "This JPEG file is incomplete or damaged.",
                "The image data ends before the image does. Try re-exporting or downloading it again."
            ),
            (
                .animatedNotSupported(format: "GIF"),
                "Animated images aren't supported yet (GIF).",
                "Export a single frame and upscale that."
            ),
            (
                .videoNotSupported(format: "MP4"), "Video isn't supported (MP4).",
                "This upscales photos. Export a still frame first."
            ),
            (
                .formatNotSupported(format: "BMP"), "BMP isn't a supported image format.",
                "Supported formats are JPEG, PNG, HEIF and TIFF."
            ),
            (
                .exceedsMemoryFeasibility(pixels: 200_000_000, limit: 134_000_000),
                "This image is too large to open on this device (200 MP).",
                "This device can open images up to about 134 MP."
            ),
        ]

        for (error, message, suggestion) in cases {
            #expect(error.message == message)
            #expect(error.suggestion == suggestion)
            #expect(error.isCancellation == false)
        }
    }

    /// Whole megapixels, rounded to nearest. False precision in a refusal is worse than none.
    @Test(
        "a pixel count reads as whole megapixels",
        arguments: [
            (11_499_999, 11), (11_500_000, 12), (12_000_000, 12), (48_000_000, 48), (0, 0),
        ])
    func megapixelRounding(pixels: Int, expected: Int) {
        #expect(Scheduling.JobError.megapixels(pixels) == expected)
    }

    @Test("a cancellation is a notice, not an error, and offers no advice")
    func cancellationCopy() {
        let error = Scheduling.JobError.cancelled
        #expect(error.message == "Upscale cancelled.")
        #expect(error.suggestion == nil)
        #expect(error.isCancellation)
    }

    @Test(
        "an unexpected failure still says which part of the job broke",
        arguments: [
            (Scheduling.JobPhase.preparingInput, "Something went wrong while preparing this image."),
            (.preparingModel, "Something went wrong while preparing the upscaler."),
            (.inferring, "The upscale stopped before it finished."),
            (.encoding, "The upscaled image couldn't be saved."),
        ])
    func unexpectedFailureNamesItsStage(phase: Scheduling.JobPhase, message: String) {
        let error = Scheduling.JobError.unexpected(stage: phase, detail: "CVPixelBufferCreate: -6662")
        #expect(error.message == message)
        #expect(error.suggestion == "Try again.")
        #expect(error.isCancellation == false)
    }

    /// The detail the user is spared is exactly what a bug report needs, so it lives in the log
    /// line and nowhere else.
    @Test("the log line carries the detail the message withholds")
    func descriptionCarriesTheDetail() {
        let error = Scheduling.JobError.unexpected(stage: .encoding, detail: "CGImageDestination nil")
        #expect(
            error.description
                == "The upscaled image couldn't be saved. [encoding — CGImageDestination nil]")
        #expect(Scheduling.JobError.cancelled.description == "Upscale cancelled.")
    }

    @Test("no message names the product: copy is about the user's file")
    func noBrandNameInCopy() {
        let all: [Scheduling.JobError] = [
            .unreadable(origin: "a.jpg"), .unrecognisedFormat(origin: "a.jpg"),
            .corruptOrTruncated(format: "PNG", origin: "a.png"),
            .animatedNotSupported(format: "GIF"), .videoNotSupported(format: "MOV"),
            .formatNotSupported(format: "BMP"),
            .exceedsMemoryFeasibility(pixels: 1, limit: 1), .cancelled,
            .unexpected(stage: .inferring, detail: "x"),
        ]
        for error in all {
            let text = error.message + (error.suggestion ?? "")
            #expect(!text.lowercased().contains("upscaler"))
        }
    }
}

// MARK: - The rest of the surface

/// The small pieces the four suites above only reach in passing: the phase ordering the transition
/// rules are built on, the diagnostics a rejected transition produces, the derivation behind the
/// estimate, and the one seam that reads a clock.
struct JobSurfaceDetailTests {

    @Test("the phases are ordered as the pipeline runs them, and only the first two are preparation")
    func phaseOrder() {
        #expect(
            Scheduling.JobPhase.allCases == [.preparingInput, .preparingModel, .inferring, .encoding])
        #expect(Scheduling.JobPhase.preparingInput < Scheduling.JobPhase.preparingModel)
        #expect(Scheduling.JobPhase.preparingModel < Scheduling.JobPhase.inferring)
        #expect(Scheduling.JobPhase.inferring < Scheduling.JobPhase.encoding)
        #expect(Scheduling.JobPhase.preparingInput.isPreparation)
        #expect(Scheduling.JobPhase.preparingModel.isPreparation)
        #expect(Scheduling.JobPhase.inferring.isPreparation == false)
        #expect(Scheduling.JobPhase.encoding.isPreparation == false)
    }

    @Test("a rejected transition says what it refused, in words a log can carry")
    func transitionDiagnostics() {
        #expect(
            Scheduling.TransitionError.notQueued(.running(.inferring)).description
                == "a job can only begin once: this one is running")
        #expect(
            Scheduling.TransitionError.noPhaseToAdvance(.queued).description
                == "a queued job has no phase to advance")
        #expect(
            Scheduling.TransitionError.phaseDidNotAdvance(from: .encoding, to: .inferring).description
                == "phases only move forward: encoding cannot be followed by inferring")
        #expect(
            Scheduling.TransitionError.outputNotWritten(.preparing(.preparingModel)).description
                == "a preparing job has not written its output yet")
        #expect(
            Scheduling.TransitionError.alreadyTerminal(.failed(.cancelled)).description
                == "the job is already failed")
    }

    /// The window is empty until it is full, holds exactly eight, and reports the oldest and newest
    /// of those eight — the ring's wrap is the part a sliding estimate depends on.
    @Test("the rate window fills, then slides")
    func rateWindowSlides() {
        var window = Scheduling.RateWindow()
        #expect(window.span == nil)

        for tile in 1...8 { window.record(value: Double(tile) * 0.03125, at: Double(tile) * 4) }
        #expect(window.span?.oldest == Scheduling.RateWindow.Sample(value: 0.03125, time: 4))
        #expect(window.span?.newest == Scheduling.RateWindow.Sample(value: 0.25, time: 32))

        window.record(value: 0.28125, at: 36)
        #expect(window.span?.oldest == Scheduling.RateWindow.Sample(value: 0.0625, time: 8))
        #expect(window.span?.newest == Scheduling.RateWindow.Sample(value: 0.28125, time: 36))

        window.reset()
        #expect(window.span == nil)
        #expect(window.estimate == nil)
    }

    /// A bar creeping by a subnormal amount over a span long enough to underflow the rate: the
    /// division reads `+infinity` seconds, which is not an estimate anyone can show.
    @Test("an underflowed rate is refused rather than reported as infinite seconds")
    func infiniteEstimateIsRefused() {
        var window = Scheduling.RateWindow()
        window.record(value: 0, at: 0)
        for tile in 2...7 { window.record(value: 0, at: Double(tile)) }
        window.record(value: .leastNonzeroMagnitude, at: 1e300)
        #expect(window.estimate == nil)
    }

    @Test("a job identity is stable and readable")
    func identity() {
        let uuid = UUID(uuidString: "00000000-0000-4000-8000-000000000001")
        #expect(uuid != nil)
        let id = Scheduling.JobID(uuid ?? UUID())
        #expect(id.description == "00000000-0000-4000-8000-000000000001")
        #expect(id == SchedulingFixture.id)
        #expect(Scheduling.JobTracker(id: id).id == id)
    }

    /// The one ambient-time read in the surface. Asserted as a property — monotone and positive —
    /// never against a value, so nothing here depends on how long the test took.
    @Test("the default clock is positive and never runs backwards")
    func defaultClock() {
        let first = Scheduling.monotonicSeconds()
        let second = Scheduling.monotonicSeconds()
        #expect(first > 0)
        #expect(second >= first)
    }
}
