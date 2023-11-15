import Foundation

/// Scheduling — the job and progress surface an app renders.
///
/// A whole-image job on a phone takes minutes, not seconds, so a spinner is not enough. This file
/// is the state machine that makes a long job legible: what the job is doing, how far it has got,
/// whether that number can be trusted, and what to say when it refuses. Progress is phased (a bare
/// `Double` cannot say "preparing model" during compilation) and monotonic (the raw tile fraction
/// steps backwards after a ladder step-down, deliberately; the displayed value does not).
///
/// ## What this is not
///
/// Not the app, and not the engine. It holds no `CVPixelBuffer`, starts no `Task`, reads no clock
/// except through one named seam (`monotonicSeconds`), and imports no UI framework — the boundary
/// CI enforces. It is a *state machine over reported events*: the caller drives the pipeline
/// (decode, colour, orientation, alpha, inference, recombine, metadata, encode) and tells this type
/// what happened.
///
/// It also does not own cancellation. Cancellation is `Task.isCancelled` with no custom tokens, so
/// `requestCancellation()` records the *intent* for the UI to render; the actual stop is the
/// caller's `Task`, honoured by the engine at a tile boundary (certification spec §10).
///
/// ## Boundary note — why the refusals are re-declared here
///
/// `UpscaleCore` may not import `ImagePipeline`. The pre-processing refusals are
/// `ImagePipeline.Decode.DecodeError` cases, so this target cannot see the type it must map.
/// `JobError` therefore re-declares that vocabulary case-for-case, with the same names and the same
/// payloads, and the one-switch adaptation the app writes is spelled out on `JobError` for copying.
/// The alternative — a dependency edge from the engine to the pipeline — would be a contract
/// change.
public enum Scheduling {}

// MARK: - Identity

extension Scheduling {

    /// A job's identity, stable for its whole life and meaningful in a log.
    ///
    /// A wrapper rather than a bare `UUID` so a job identity cannot be passed where a model or
    /// device identity is wanted, and so tests can pin one (`JobID(UUID(uuidString:))`) instead of
    /// asserting against whatever the runtime generated.
    public struct JobID: Sendable, Hashable, CustomStringConvertible {
        public let rawValue: UUID

        public init(_ rawValue: UUID = UUID()) {
            self.rawValue = rawValue
        }

        public var description: String { rawValue.uuidString }
    }
}

// MARK: - Phase

extension Scheduling {

    /// What the job is doing right now — the distinction a spinner cannot draw.
    ///
    /// The four cases are the four spans of the pipeline's call sequence, not an invented taxonomy:
    ///
    /// | Phase | Pipeline steps | Why the user waits |
    /// |---|---|---|
    /// | `preparingInput` | decode, colour plan, orientation, alpha split | **Every named refusal fires here** — none fire mid-job. |
    /// | `preparingModel` | model load | `CoreMLBackend.load` compiles on a cache miss (`MLModel.compileModel`). Seconds of silence, and the reason this phase is separate at all. |
    /// | `inferring` | the tile loop | The only phase with an honest fraction. |
    /// | `encoding` | alpha scale, recombine, metadata, `Encode.write` | |
    ///
    /// `Comparable` in that order, and the ordering is load-bearing: `JobTracker.enter(_:)` accepts
    /// a phase only if it is strictly later, which is what stops a UI oscillating between
    /// "Preparing" and "Upscaling".
    public enum JobPhase: Int, Sendable, CaseIterable, Comparable {
        case preparingInput = 0
        case preparingModel = 1
        case inferring = 2
        case encoding = 3

        /// True for the phases that run before the first tile — the phases with no fraction to show.
        public var isPreparation: Bool { self < .inferring }

        public static func < (lhs: JobPhase, rhs: JobPhase) -> Bool { lhs.rawValue < rhs.rawValue }
    }
}

// MARK: - Errors, and the words for them

extension Scheduling {

    /// Why a job ended other than by finishing — with the sentence to show for it.
    ///
    /// The first seven cases are decode's named pre-processing refusals plus its feasibility limit,
    /// re-declared here for the boundary reason in this file's header. The app owns one exhaustive
    /// switch, which is the whole adaptation:
    ///
    /// ```swift
    /// func jobError(_ error: Decode.DecodeError) -> Scheduling.JobError {
    ///     switch error {
    ///     case .unreadable(let origin):                   .unreadable(origin: origin)
    ///     case .unrecognisedFormat(let origin):           .unrecognisedFormat(origin: origin)
    ///     case .corruptOrTruncated(let format, let origin):
    ///         .corruptOrTruncated(format: format, origin: origin)
    ///     case .animatedNotSupported(let format):         .animatedNotSupported(format: format)
    ///     case .videoNotSupported(let format):            .videoNotSupported(format: format)
    ///     case .formatNotSupported(let format):           .formatNotSupported(format: format)
    ///     case .exceedsMemoryFeasibility(let pixels, let limit):
    ///         .exceedsMemoryFeasibility(pixels: pixels, limit: limit)
    ///     case .colorModelNotSupported, .bufferAllocationFailed, .drawingContextUnavailable:
    ///         .unexpected(stage: .preparingInput, detail: String(describing: error))
    ///     }
    /// }
    /// ```
    ///
    /// The compiler checks that switch for exhaustiveness, so a case added to `DecodeError` becomes
    /// a build error in the app rather than a silent "Something went wrong" — which is the property
    /// that mattered, and the one a stringly-typed adapter would have thrown away.
    ///
    /// ## No product name appears in any message
    ///
    /// Deliberate. Engine-level copy names no product: every sentence below is about the user's
    /// file, not about the app.
    public enum JobError: Error, Equatable, Sendable {
        case unreadable(origin: String)
        case unrecognisedFormat(origin: String)
        case corruptOrTruncated(format: String, origin: String)
        case animatedNotSupported(format: String)
        case videoNotSupported(format: String)
        case formatNotSupported(format: String)
        case exceedsMemoryFeasibility(pixels: Int, limit: Int)
        case modelRequiresNetworkOnFirstRun
        case modelAssetCorrupted
        /// The user asked to stop. Terminal, but **not a failure** — see `isCancellation`.
        case cancelled
        /// Anything else, tagged with the phase it broke in so the message can still say where.
        case unexpected(stage: JobPhase, detail: String)

        /// True only for `.cancelled`. A cancelled job is terminal as `.failed(.cancelled)` — the
        /// six states this WO specifies have no `cancelled` terminal — so the shell needs one bit
        /// to tell "show a neutral notice" from "show an error". This is that bit.
        public var isCancellation: Bool { self == .cancelled }

        /// One sentence, for the user, about their file. No jargon, no type names, no file paths
        /// beyond the origin the caller supplied.
        public var message: String {
            switch self {
            case .unreadable:
                return "This file couldn't be opened."
            case .unrecognisedFormat:
                return "This file isn't in a format that can be read as an image."
            case .corruptOrTruncated(let format, _):
                return "This \(format) file is incomplete or damaged."
            case .animatedNotSupported(let format):
                return "Animated images aren't supported yet (\(format))."
            case .videoNotSupported(let format):
                return "Video isn't supported (\(format))."
            case .formatNotSupported(let format):
                return "\(format) isn't a supported image format."
            case .exceedsMemoryFeasibility(let pixels, _):
                return "This image is too large to open on this device (\(Self.megapixels(pixels)) MP)."
            case .modelRequiresNetworkOnFirstRun:
                return
                    "This is the first time running this model. An internet connection is required once to securely unlock it."
            case .modelAssetCorrupted:
                return "The model asset is corrupted and cannot be loaded."
            case .cancelled:
                return "Upscale cancelled."
            case .unexpected(let stage, _):
                switch stage {
                case .preparingInput: return "Something went wrong while preparing this image."
                case .preparingModel: return "Something went wrong while preparing the upscaler."
                case .inferring: return "The upscale stopped before it finished."
                case .encoding: return "The upscaled image couldn't be saved."
                }
            }
        }

        /// What the user can do about it, or `nil` when there is nothing honest to suggest.
        ///
        /// `nil` is a real answer here and not a gap: `.cancelled` needs no advice, and inventing
        /// "try again" for a deliberate stop would be noise.
        public var suggestion: String? {
            switch self {
            case .unreadable(let origin):
                return "Check that \(origin) is still there and that you have permission to open it."
            case .unrecognisedFormat:
                return "Supported formats are JPEG, PNG, HEIF and TIFF."
            case .corruptOrTruncated:
                return "The image data ends before the image does. Try re-exporting or downloading it again."
            case .animatedNotSupported:
                return "Export a single frame and upscale that."
            case .videoNotSupported:
                return "This upscales photos. Export a still frame first."
            case .formatNotSupported:
                return "Supported formats are JPEG, PNG, HEIF and TIFF."
            case .exceedsMemoryFeasibility(_, let limit):
                return "This device can open images up to about \(Self.megapixels(limit)) MP."
            case .modelRequiresNetworkOnFirstRun:
                return "Connect to the internet and try again."
            case .modelAssetCorrupted:
                return "Reinstall the app to restore the model."
            case .cancelled:
                return nil
            case .unexpected:
                return "Try again."
            }
        }

        /// Pixels to whole megapixels, rounded to nearest — 12 000 000 reads as `12`, and
        /// 11 500 000 reads as `12` rather than `11`. Integer arithmetic on purpose: a megapixel
        /// figure with a decimal point in a refusal message is false precision.
        static func megapixels(_ pixels: Int) -> Int {
            (pixels + 500_000) / 1_000_000
        }
    }
}

extension Scheduling.JobError: CustomStringConvertible {
    /// The log line. Identical to `message` except for `.unexpected`, where the detail the user is
    /// spared is exactly what a bug report needs.
    public var description: String {
        switch self {
        case .unexpected(let stage, let detail):
            return "\(message) [\(stage) — \(detail)]"
        default:
            return message
        }
    }
}

// MARK: - State

extension Scheduling {

    /// The renderable state of a job, in the six cases this WO specifies.
    ///
    /// The phase is carried *inside* the states where a phase is meaningful rather than beside
    /// them, so `(state, phase)` cannot disagree. `JobTracker` derives the state from the phase —
    /// preparation phases produce `.preparing`, the rest produce `.running` — so a pairing like
    /// "preparing, encoding" is not reachable through the API.
    ///
    /// **Cancellation is terminal as `.failed(.cancelled)`**, not as a seventh state. The six
    /// states are the specified set; `JobError.isCancellation` is how the shell tells a stop from a
    /// fault without one.
    public enum JobState: Sendable, Equatable {
        case queued
        case preparing(JobPhase)
        case running(JobPhase)
        /// The user asked to stop; the engine has not stopped yet. Carries the phase it interrupted
        /// so the UI can say "finishing the current tile" rather than freezing.
        case cancelling(JobPhase)
        case finished
        case failed(JobError)

        /// The phase this state is in, or `nil` before work starts and after it ends.
        public var phase: JobPhase? {
            switch self {
            case .queued, .finished, .failed: return nil
            case .preparing(let phase), .running(let phase), .cancelling(let phase): return phase
            }
        }

        /// A verdict has been reached. Terminal states reject `begin`, `enter` and `finish`, and
        /// absorb `requestCancellation` and `fail` (see `JobTracker`).
        public var isTerminal: Bool {
            switch self {
            case .finished, .failed: return true
            case .queued, .preparing, .running, .cancelling: return false
            }
        }

        /// The state's own name, without its payload — for transition diagnostics and logs.
        public var name: String {
            switch self {
            case .queued: return "queued"
            case .preparing: return "preparing"
            case .running: return "running"
            case .cancelling: return "cancelling"
            case .finished: return "finished"
            case .failed: return "failed"
            }
        }
    }

    /// A transition the caller asked for that the machine will not make.
    ///
    /// Thrown, not ignored, and typed rather than a `Bool`: these are caller bugs — a screen that
    /// announces encoding while the job is still queued is a bug whether or not anyone notices the
    /// wrong label. The exceptions are documented on the two methods that cannot throw
    /// (`report`, `fail`) and the one that must be idempotent (`requestCancellation`).
    public enum TransitionError: Error, Equatable, CustomStringConvertible {
        /// `begin()` on a job that has already started.
        case notQueued(JobState)
        /// `enter(_:)` on a job that is queued, cancelling, or over.
        case noPhaseToAdvance(JobState)
        /// `enter(_:)` with a phase that is not strictly later than the current one.
        case phaseDidNotAdvance(from: JobPhase, to: JobPhase)
        /// `finish()` before the output file has been written.
        case outputNotWritten(JobState)
        /// Any transition after a verdict.
        case alreadyTerminal(JobState)

        public var description: String {
            switch self {
            case .notQueued(let state):
                return "a job can only begin once: this one is \(state.name)"
            case .noPhaseToAdvance(let state):
                return "a \(state.name) job has no phase to advance"
            case .phaseDidNotAdvance(let from, let to):
                return "phases only move forward: \(from) cannot be followed by \(to)"
            case .outputNotWritten(let state):
                return "a \(state.name) job has not written its output yet"
            case .alreadyTerminal(let state):
                return "the job is already \(state.name)"
            }
        }
    }
}

// MARK: - Monotone progress, and the ladder step-down

extension Scheduling {

    /// A non-decreasing view of a signal that legitimately decreases.
    ///
    /// ## The problem, stated exactly
    ///
    /// `SequentialTilingEngine` reports `base + (1 - base) · next / count` where `base` is the
    /// resumed row's fraction of the image. Its own comment is the honest description: *"A step
    /// down replays a row, so it steps **back** to the resumed row's fraction — honest rather than
    /// smoothed; presentation is the UI layer's problem."* That is deliberate and correct: when the
    /// adaptive ladder drops 512 → 384 → 256 under memory pressure it re-plans from the top of the
    /// current tile row and re-runs the tiles to its left, so the engine's estimate of "how far
    /// through the plan" really did just get worse. **This type is the UI layer holding up its end.**
    ///
    /// The engine's signature is frozen to this WO, so the *only* in-flight evidence of a revision
    /// is that the number went down. That is what `record(_:)` watches for.
    ///
    /// ## The rule, and why this one
    ///
    /// **Ratchet, then re-map the remainder.** On a decrease the displayed value does not move; the
    /// value it had becomes a floor, the incoming raw value becomes the new origin, and every later
    /// raw value is mapped linearly from that origin into what remains:
    ///
    /// ```
    /// display = floor + (1 - floor) · (raw - rawOrigin) / (1 - rawOrigin)
    /// ```
    ///
    /// So the bar **never rewinds and never stalls** — it resumes climbing on the very next tile,
    /// more slowly. Deceleration is the truth: the engine has just added a row of recompute to a
    /// job that is already minutes long.
    ///
    /// Three alternatives were considered and rejected:
    ///
    /// 1. **Pass it through raw.** That fails the bar: "a progress bar that silently runs backwards is
    ///    worse than none."
    /// 2. **`max(raw, shown)` — clamp and move on.** Non-decreasing, but the bar *freezes* for the
    ///    whole replayed row, which on a 4 602-tile 48 MP plan is a long, silent stop that is
    ///    indistinguishable from a hang. This is "clamping and calling it done", which is rejected —
    ///    and it is worse than it looks: the freeze is longest exactly when the machine is least
    ///    healthy.
    /// 3. **Count completed tiles instead of position.** Genuinely monotone in the numerator, but a
    ///    step down revises the *denominator* upward, so `done / total` still falls. Every "fraction
    ///    of an estimate" runs backwards when the estimate is revised; the revision is real
    ///    information and no arithmetic removes it.
    ///
    /// ## What is not hidden
    ///
    /// The revision is reported, not smoothed away: `revisions` counts them, and `JobTracker`
    /// **withdraws the ETA** on every one (a revised plan invalidates every measured tile interval).
    /// The bar stays honest by staying monotone; the estimate stays honest by admitting it no longer
    /// knows. Those are different jobs and this file does not confuse them.
    public struct MonotonicProgress: Sendable, Equatable {
        /// What the user is shown. Non-decreasing across the life of the value, in `0...1`.
        public private(set) var value: Double = 0
        /// How many times the underlying signal revised itself downwards. Zero for a job that never
        /// stepped down, which is the overwhelming majority of them.
        public private(set) var revisions: Int = 0

        private var lastRaw: Double?
        private var floor: Double = 0
        private var rawOrigin: Double = 0

        public init() {}

        /// Fold one raw callback value in, and return what to show.
        ///
        /// Non-finite input is ignored outright rather than clamped: a `NaN` reaching `min`/`max`
        /// propagates silently and would poison every later comparison, and there is no reading of
        /// "not a number" that belongs on a progress bar.
        @discardableResult
        public mutating func record(_ raw: Double) -> Double {
            guard raw.isFinite else { return value }
            let clamped = min(max(raw, 0), 1)

            if let last = lastRaw, clamped < last {
                revisions += 1
                floor = value
                rawOrigin = clamped
                lastRaw = clamped
                return value
            }

            lastRaw = clamped
            let span = 1 - rawOrigin
            // `span == 0` needs the revision to have landed on exactly 1.0, which means the pass it
            // revised was already complete. Nothing remains to map, so the floor is the answer.
            guard span > 0 else { return value }
            value = min(floor + (1 - floor) * (clamped - rawOrigin) / span, 1)
            return value
        }
    }
}

// MARK: - The estimate

extension Scheduling {

    /// A time remaining, together with the evidence for it.
    ///
    /// ## What it is derived from — the whole derivation
    ///
    /// The rate at which **the displayed bar** has moved over the last `windowSize` reported tiles:
    ///
    /// ```
    /// rate            = (value.newest - value.oldest) / (time.newest - time.oldest)
    /// secondsRemaining = (1 - value.newest) / rate
    /// ```
    ///
    /// Deriving it from the displayed value rather than the raw signal is what makes it consistent
    /// with the bar the user is watching — after a revision the bar moves in compressed units, and
    /// an estimate built on raw tiles would contradict it. It is, exactly, *"when will this bar
    /// reach the end if it keeps moving as it has been"*.
    ///
    /// ## How wrong it can be
    ///
    /// - **It is not an upper bound, and it lags in the one direction that matters.** Thermal
    ///   throttling on a multi-minute job is expected, not exceptional (certification spec §10:
    ///   "Thermal throttling → reduce tile concurrency; ETA adapts"). This estimate only ever sees
    ///   throttling *after* it has slowed some tiles down. It will read too optimistic while the
    ///   device is heating and correct itself afterwards.
    /// - **It excludes both preparations and the encode.** It is offered only during
    ///   `JobPhase.inferring`. `MLModel.compileModel` is not observable through a tile counter, and
    ///   `Encode.write` reports nothing at all, so neither is in the number.
    /// - **It is withdrawn, not adjusted, on a ladder step-down.** After a revision both the tile
    ///   count and the per-tile cost have changed; the window is cleared and no estimate is offered
    ///   until `windowSize` fresh tiles have been measured.
    /// - **`windowSize` is a smoothing choice, not a measurement.** It is named here so a device
    ///   run can replace it with a measured one. Eight tiles is short enough to track throttling and
    ///   long enough that one slow tile does not halve the estimate; that is a judgement, and it is
    ///   stated rather than hidden.
    ///
    /// `sampleCount` and `sampleSeconds` travel with the value so a caller can see the evidence and
    /// decide for itself — a UI is free to round "about 4 minutes" from a 12-second sample, or to
    /// show nothing at all.
    public struct Estimate: Sendable, Equatable {
        public let secondsRemaining: Double
        /// Tiles the estimate is derived from. Always `Estimate.windowSize` as produced here.
        public let sampleCount: Int
        /// Wall-clock seconds those tiles took.
        public let sampleSeconds: Double

        public init(secondsRemaining: Double, sampleCount: Int, sampleSeconds: Double) {
            self.secondsRemaining = secondsRemaining
            self.sampleCount = sampleCount
            self.sampleSeconds = sampleSeconds
        }

        /// Reported tiles in the sliding window, and therefore the minimum sample before any
        /// estimate is offered at all.
        public static let windowSize = 8
    }

    /// The sliding window behind `Estimate`. Internal: it is the derivation, not the surface.
    ///
    /// A ring rather than a growing array — one job is thousands of tiles, and an estimate that
    /// allocates per tile would be a per-tile cost added by the *reporting* layer.
    struct RateWindow: Sendable, Equatable {
        private var samples: [Sample] = []
        private var next = 0

        struct Sample: Sendable, Equatable {
            let value: Double
            let time: Double
        }

        /// Drop every sample. Called on a phase change and on a progress revision, both of which
        /// make the measured intervals describe work that is no longer the work remaining.
        mutating func reset() {
            samples.removeAll(keepingCapacity: true)
            next = 0
        }

        mutating func record(value: Double, at time: Double) {
            let sample = Sample(value: value, time: time)
            if samples.count < Scheduling.Estimate.windowSize {
                samples.append(sample)
            } else {
                samples[next] = sample
                next = (next + 1) % Scheduling.Estimate.windowSize
            }
        }

        /// The oldest and newest samples once the window is full, or `nil` while it is filling.
        var span: (oldest: Sample, newest: Sample)? {
            guard samples.count == Scheduling.Estimate.windowSize else { return nil }
            let newestIndex = (next + Scheduling.Estimate.windowSize - 1) % Scheduling.Estimate.windowSize
            return (samples[next], samples[newestIndex])
        }

        /// The estimate, or `nil` when there is no honest one: window not yet full, no time
        /// elapsed, or a bar that has not moved (a stalled or complete job cannot be extrapolated).
        var estimate: Scheduling.Estimate? {
            guard let span else { return nil }
            let seconds = span.newest.time - span.oldest.time
            let climbed = span.newest.value - span.oldest.value
            guard seconds > 0, climbed > 0 else { return nil }
            // The rate can underflow to zero when a bar creeps by a subnormal amount over a long
            // span, and the division then reads `+infinity` seconds. Non-negativity needs no guard:
            // the recorded values are `MonotonicProgress` output, which never exceeds 1.
            let remaining = (1 - span.newest.value) / (climbed / seconds)
            guard remaining.isFinite else { return nil }
            return Scheduling.Estimate(
                secondsRemaining: remaining, sampleCount: Scheduling.Estimate.windowSize,
                sampleSeconds: seconds)
        }
    }
}

// MARK: - The snapshot

extension Scheduling {

    /// Everything a screen needs to draw a job, taken atomically.
    ///
    /// A value, not a live object: the UI renders one consistent set of facts rather than reading
    /// four properties that may change between reads.
    public struct JobSnapshot: Sendable, Equatable {
        public let id: JobID
        public let state: JobState
        /// `nil` means **indeterminate — show a spinner, not a bar**, and it is a real answer: a
        /// queued job, a decode and a model compilation have no honest fraction. See `fraction(in:_:)`.
        public let progress: Double?
        /// `nil` means no honest estimate is available *right now* (`Estimate` documents every
        /// reason). It is not an error, and it is not zero.
        public let estimate: Estimate?
        /// Downward revisions of the underlying signal so far — ladder step-downs, in practice.
        /// Surfaced so a UI can explain a slowdown it would otherwise have to hide.
        public let revisions: Int

        public var phase: JobPhase? { state.phase }
        public var isTerminal: Bool { state.isTerminal }

        /// The rule for whether a bar is shown and what it reads.
        ///
        /// A bar exists exactly while tiles are the thing being waited on, plus the states where
        /// every tile is done. `.encoding` reads `1` deliberately: **all the tiles really are
        /// finished**, and the phase label — not the bar — is what tells the user the file is still
        /// being written. A bar that vanished at 97% to become a spinner would be a worse lie than a
        /// full bar next to the word "Saving".
        ///
        /// `.failed` shows nothing: a failed job's screen is an error, not a measurement.
        static func fraction(in state: JobState, _ value: Double) -> Double? {
            switch state {
            case .queued, .failed:
                return nil
            case .preparing(let phase), .running(let phase), .cancelling(let phase):
                guard !phase.isPreparation else { return nil }
                return phase == .encoding ? 1 : value
            case .finished:
                return 1
            }
        }
    }
}

// MARK: - The tracker

extension Scheduling {

    /// The job state machine. One per job; the app holds it, the engine reports into it.
    ///
    /// ## Concurrency
    ///
    /// A lock-guarded final class, matching `CoreMLBackend` and `SequentialTilingEngine`. It is
    /// written from the engine's task through a `@Sendable` progress callback and read from the
    /// main actor, so the lock is load-bearing and `@unchecked Sendable` is honest about why.
    ///
    /// ## The transition table
    ///
    /// | From | Event | To |
    /// |---|---|---|
    /// | `queued` | `begin()` | `preparing(.preparingInput)` |
    /// | `preparing`/`running` | `enter(p)`, p strictly later | `preparing(p)` or `running(p)` |
    /// | `preparing`/`running` | `requestCancellation()` | `cancelling(current phase)` |
    /// | `queued` | `requestCancellation()` | `failed(.cancelled)` — nothing is in flight to wind down |
    /// | `cancelling` | `fail(.cancelled)` | `failed(.cancelled)` |
    /// | `cancelling` | `finish()` | `finished` — see below |
    /// | `running(.encoding)` | `finish()` | `finished` |
    /// | any non-terminal | `fail(e)` | `failed(e)` |
    ///
    /// `cancelling → finished` is legal on purpose. If the work landed before the stop took effect,
    /// the user has their image, and throwing it away to honour a button they pressed a moment too
    /// late would be gratuitous.
    ///
    /// Everything else is rejected — `begin` twice, a phase that repeats or goes backwards, a phase
    /// entered while queued or cancelling, `finish()` before the file is written, and anything at
    /// all after a verdict.
    ///
    /// Three events deliberately do **not** throw:
    ///
    /// - `report(_:at:)` — a callback is not a caller. It cannot throw, and the engine legitimately
    ///   fires once more after cancellation is requested (the in-flight tile finishes; the
    ///   `Task.checkCancellation()` guard is at the *top* of the loop). Values are folded in during
    ///   `inferring` whether running or cancelling, and ignored everywhere else.
    /// - `requestCancellation()` — idempotent. A cancel button can be tapped twice, and the second
    ///   tap is not a bug.
    /// - `fail(_:)` — the first verdict wins, later ones are dropped. Failure paths run in `catch`
    ///   blocks and cleanup, which race the terminal state by construction.
    ///
    /// ## Resume
    ///
    /// Resume-after-interruption is designed to be *implementable* later (certification spec §10).
    /// It is: the whole of this type's mutable state is `JobState`, a `MonotonicProgress` and a
    /// `RateWindow`, all `Equatable` values with no engine handles in them, so persisting a job's
    /// position is a matter of persisting the tile index alongside them, not of redesigning this.
    public final class JobTracker: @unchecked Sendable {
        private struct Storage {
            var state: JobState = .queued
            var progress = MonotonicProgress()
            var window = RateWindow()
        }

        public let id: JobID
        private let lock = NSLock()
        private var storage = Storage()

        public init(id: JobID = JobID()) {
            self.id = id
        }

        /// The current facts, taken under the lock.
        public var snapshot: JobSnapshot {
            lock.withLock { Self.snapshot(id: id, storage) }
        }

        private static func snapshot(id: JobID, _ storage: Storage) -> JobSnapshot {
            let isInferring = storage.state.phase == .inferring
            return JobSnapshot(
                id: id,
                state: storage.state,
                progress: JobSnapshot.fraction(in: storage.state, storage.progress.value),
                // An estimate is a claim about work in flight, so it is offered only while the job
                // is running the tiles it measured: not while cancelling, and not after a verdict.
                estimate: {
                    guard case .running = storage.state, isInferring else { return nil }
                    return storage.window.estimate
                }(),
                revisions: storage.progress.revisions)
        }

        // MARK: Transitions

        /// Start the job: `queued → preparing(.preparingInput)`.
        @discardableResult
        public func begin() throws -> JobSnapshot {
            try lock.withLock {
                guard !storage.state.isTerminal else {
                    throw TransitionError.alreadyTerminal(storage.state)
                }
                guard case .queued = storage.state else {
                    throw TransitionError.notQueued(storage.state)
                }
                storage.state = .preparing(.preparingInput)
                return Self.snapshot(id: id, storage)
            }
        }

        /// Advance to a strictly later phase.
        ///
        /// The resulting state is derived from the phase, never passed in, so `preparing` and
        /// `running` cannot disagree with it. Skipping forward is allowed — a warm run with a
        /// cached `.mlmodelc` has no model preparation worth announcing, and forcing the caller to
        /// announce a phase it is not in would be a worse contract than allowing the skip.
        @discardableResult
        public func enter(_ phase: JobPhase) throws -> JobSnapshot {
            try lock.withLock {
                guard !storage.state.isTerminal else {
                    throw TransitionError.alreadyTerminal(storage.state)
                }
                let current: JobPhase
                switch storage.state {
                case .preparing(let existing), .running(let existing):
                    current = existing
                case .queued, .cancelling, .finished, .failed:
                    throw TransitionError.noPhaseToAdvance(storage.state)
                }
                guard phase > current else {
                    throw TransitionError.phaseDidNotAdvance(from: current, to: phase)
                }
                storage.state = phase.isPreparation ? .preparing(phase) : .running(phase)
                // Intervals measured in one phase say nothing about the next one.
                storage.window.reset()
                return Self.snapshot(id: id, storage)
            }
        }

        /// Fold in one value from the engine's progress callback, at `seconds` on the caller's
        /// monotonic clock. Never throws; see the type's documentation for why.
        @discardableResult
        public func report(_ raw: Double, at seconds: Double) -> JobSnapshot {
            lock.withLock {
                guard storage.state.phase == .inferring else { return Self.snapshot(id: id, storage) }
                let before = storage.progress.revisions
                let value = storage.progress.record(raw)
                if storage.progress.revisions != before {
                    // A ladder step-down: the plan changed, so every interval in the window
                    // describes tiles that are no longer the tiles remaining.
                    storage.window.reset()
                } else {
                    storage.window.record(value: value, at: seconds)
                }
                return Self.snapshot(id: id, storage)
            }
        }

        /// Record that the user asked to stop. Idempotent; see the type's documentation.
        ///
        /// This does not cancel anything — cancellation is `Task.isCancelled` with no custom
        /// tokens. The caller cancels its `Task`; this is what the screen renders while the engine
        /// winds down to the next tile boundary.
        @discardableResult
        public func requestCancellation() -> JobSnapshot {
            lock.withLock {
                switch storage.state {
                case .preparing(let phase), .running(let phase):
                    storage.state = .cancelling(phase)
                case .queued:
                    // Nothing is in flight, so there is nothing to wind down and no honest moment
                    // to show "Cancelling…". A queued job stops at once.
                    storage.state = .failed(.cancelled)
                case .cancelling, .finished, .failed:
                    break
                }
                return Self.snapshot(id: id, storage)
            }
        }

        /// The output is written. Legal from `running(.encoding)` and from `cancelling`.
        @discardableResult
        public func finish() throws -> JobSnapshot {
            try lock.withLock {
                guard !storage.state.isTerminal else {
                    throw TransitionError.alreadyTerminal(storage.state)
                }
                switch storage.state {
                case .running(.encoding), .cancelling:
                    storage.state = .finished
                case .queued, .preparing, .running:
                    throw TransitionError.outputNotWritten(storage.state)
                case .finished, .failed:
                    throw TransitionError.alreadyTerminal(storage.state)
                }
                return Self.snapshot(id: id, storage)
            }
        }

        /// End the job with a verdict. Never throws, and the first verdict wins.
        @discardableResult
        public func fail(_ error: JobError) -> JobSnapshot {
            lock.withLock {
                if !storage.state.isTerminal { storage.state = .failed(error) }
                return Self.snapshot(id: id, storage)
            }
        }

        // MARK: The engine seam

        /// A callback shaped exactly like `TilingEngine.upscale(…, progress:)`'s parameter.
        ///
        /// **This is the whole adaptation.** The engine's signature is frozen to this WO and this
        /// closure is how it is met: hand the result to `upscale`, and the bare `Double` becomes
        /// phase-aware, monotone, and timed, with no edit to the engine and no change to what it
        /// reports.
        ///
        /// The clock is injected rather than read here, which is what keeps every ETA test
        /// deterministic and this file free of ambient time.
        public func progressHandler(
            clock: @escaping @Sendable () -> Double = Scheduling.monotonicSeconds
        ) -> @Sendable (Double) -> Void {
            { [self] raw in _ = report(raw, at: clock()) }
        }
    }

    /// The default clock: seconds since boot, monotonic, and not affected by the wall clock moving.
    ///
    /// The only ambient time read in this file, isolated here so everything else takes its instants
    /// as arguments. It does not advance while the device is asleep — correct for this use, because
    /// the job is not advancing then either.
    public static let monotonicSeconds: @Sendable () -> Double = {
        ProcessInfo.processInfo.systemUptime
    }
}
