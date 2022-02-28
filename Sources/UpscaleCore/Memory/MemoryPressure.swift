import Dispatch
import Foundation

// MARK: - The signal

/// A memory-pressure level as the kernel reports it.
///
/// **The signal source is `DispatchSourceMemoryPressure`, event-driven.** It is the only candidate
/// mechanism that is both headless-legal — `UIApplication.didReceiveMemoryWarningNotification`
/// would put UIKit inside `UpscaleCore` and break the headless-engine boundary — and free of a
/// polling loop. It is also the only one that works on the macOS host, so the tests below are not
/// iOS-only claims.
///
/// Three cases because the dispatch source reports three transitions, and the engine has a rule for
/// two of them. `.normal` is the third and is deliberately inert — see
/// `MemoryPressureResponse.respond(to:ladder:plan:)`.
public enum MemoryPressureLevel: Int, Sendable, CaseIterable, Comparable {
    /// The kernel says the process is back inside its normal working set.
    case normal = 0
    /// `warn` — drop a ladder rung for *subsequent* tiles.
    case warning = 1
    /// `critical` — finish the current tile, then checkpoint.
    case critical = 2

    /// Ordered by severity, which is exactly what `MemoryPressureMonitor`'s latch needs: two events
    /// arriving between one tile boundary and the next collapse to the **worse** of the two, never
    /// to the more recent one.
    public static func < (lhs: MemoryPressureLevel, rhs: MemoryPressureLevel) -> Bool {
        lhs.rawValue < rhs.rawValue
    }

    /// Read one dispatch event mask as one level.
    ///
    /// The mask is an `OptionSet` and more than one bit can be set, so the cases are tested
    /// **worst-first**: a mask carrying both `warning` and `critical` is a critical event.
    /// Escalating on an ambiguous mask is the safe direction — the opposite would answer a critical
    /// event with a warning's response, which does not checkpoint.
    ///
    /// `nil` when the mask names none of the three. That reading is neither escalated nor
    /// de-escalated: the monitor ignores it and keeps the level it was carrying. Inventing a level
    /// from a mask the overlay did not define would be a guess, and the honest answer to "no figure
    /// exists" is to say so.
    public init?(_ event: DispatchSource.MemoryPressureEvent) {
        if event.contains(.critical) {
            self = .critical
        } else if event.contains(.warning) {
            self = .warning
        } else if event.contains(.normal) {
            self = .normal
        } else {
            return nil
        }
    }

    /// The ladder trigger this level actuates — **the whole of the pressure-to-ladder mapping**.
    ///
    /// `nil` at `.normal`, and there must be no upward trigger here. A step **up** happens between
    /// images only, on dwell plus a ρ-derived margin (`TileShapeLadder.beginImage(_:)`), because a
    /// step up would re-grid backwards over assembled output. A pressure-relief event mid-image is
    /// neither dwell nor margin, and `TileShapeLadder.respond(to:plan:)` has no upward path through
    /// it at all.
    public var ladderTrigger: TileShapeLadder.Trigger? {
        switch self {
        case .normal: return nil
        case .warning: return .pressureWarning
        case .critical: return .pressureCritical
        }
    }
}

// MARK: - The response

/// What the engine does about one pressure event, typed.
///
/// **This type actuates; it decides nothing.** Which rung, why, and when belong to
/// `TileShapeLadder`. Everything here is either read off the ladder's own `Outcome` or is a
/// constant of the pressure policy:
///
/// | Policy | Here |
/// |---|---|
/// | `warn`: drop a rung for *subsequent* tiles | `subsequentTileShape` |
/// | `warn`: stop starting new work at the current shape | `steppedDown` |
/// | `critical`: finish the current tile, then checkpoint | `finishesCurrentTile`, `checkpoints` |
/// | never free and reload the model | `retainsModel` — invariantly `true` |
/// | certification spec §6: "caches purged, model retained" | `purgesCaches` **and** `retainsModel` |
///
/// `purgesCaches` here pairs with `SequentialTilingEngine.purgeCaches()` at the other end.
///
/// **What moving a rung costs is not assumed.** `outcome.projectedDeltaBytes` is the plan's own
/// arithmetic and can be **positive for a step down**: the measured per-shape table is
/// non-monotonic, so at 1080p a pressure-driven 384 → 256 raises the projected total by 9 227 469
/// B. The ladder surfaces that instead of assuming it away, and this type carries it through
/// untouched.
public struct MemoryPressureResponse: Sendable, Equatable {
    /// The level that produced this response.
    public let level: MemoryPressureLevel

    /// The ladder's own typed outcome — shape, policy and reason. `nil` at `.normal`, where the
    /// ladder is deliberately never consulted.
    public let outcome: TileShapeLadder.Outcome?

    public init(level: MemoryPressureLevel, outcome: TileShapeLadder.Outcome?) {
        self.level = level
        self.outcome = outcome
    }

    /// The rung subsequent tiles run at, as the ladder now stands.
    ///
    /// `nil` at `.normal`: the ladder was not consulted, so this response has nothing to say about
    /// the engine's shape. At the floor rung it is 256 — the rung held, and subsequent tiles run at
    /// it, which is a different statement from "the rung moved" (`steppedDown`).
    public var subsequentTileShape: TileShape? { outcome?.shape }

    /// Whether new work at the shape the event interrupted has actually stopped.
    ///
    /// True exactly when the rung moved. **At the floor rung it is `false`, and that is intended
    /// rather than a gap:** 256 is the floor, there is no lower rung to move to, and certification
    /// spec §6's 48 MP rule is to degrade gracefully — never stall, never crash. `isDegraded` is
    /// how that case is surfaced to a caller.
    public var steppedDown: Bool { outcome?.movedRung ?? false }

    /// The `critical` clause: checkpoint once the current tile is finished. Never on `warn`, which
    /// drops a rung and keeps going.
    ///
    /// *What* a checkpoint writes is the engine's business, not this type's: `BandSpoolSurface`
    /// already spools completed rows to disk and closes them at tile-row boundaries
    /// (`SequentialTilingEngine.run`), so the checkpoint this flag asks for is a flush of work
    /// already done — never a re-encode, never a re-plan, never a reload.
    public var checkpoints: Bool { level == .critical }

    /// Certification spec §6's other half — *"memory warnings handled gracefully (**caches
    /// purged**, model retained)"*. True at both pressure levels; false at `.normal`.
    ///
    /// **A lookup on the level, deliberately.** *What* a cache is belongs to whoever holds one:
    /// `SequentialTilingEngine.purgeCaches()` is the actuator, and the engine's only cache is the
    /// output `CVPixelBufferPool`. This type says when, never what.
    ///
    /// **False at `.normal` for the same reason the ladder is not consulted there:** relief is the
    /// absence of an event, not an event. Dropping a warm pool because the kernel said the process
    /// is fine again would be pointless churn.
    ///
    /// **Additional to the rung drop, never a replacement for it.** A `warn` still drops a rung for
    /// subsequent tiles and a `critical` still checkpoints. Nothing about a purge abandons the
    /// image, which is why `finishesCurrentTile` is untouched by it — and nothing about it frees a
    /// model, which is why `retainsModel` is untouched too. Certification spec §6 asks for both
    /// halves in one clause and this response carries both.
    ///
    /// **Memory effect: `unavailable`.** No footprint claim is made here, and none is measurable on
    /// the host. Apple cautions that dumping caches can make things *worse* — the compressor may
    /// have to decompress what is dropped before it can be freed (WWDC18 session 416, "iOS Memory
    /// Deep Dive"). The working hypothesis is that a pool of recycled whole-image buffers is the
    /// other case (dirty pages whose release genuinely returns footprint, not compressed pages that
    /// must be faulted back in first) — but that is a **hypothesis, not a measurement.** Only a
    /// device run settles it, and until then there is no figure, and none is invented.
    public var purgesCaches: Bool { level != .normal }

    /// **Invariantly `true`.** Requirement 5 says on `critical` "finish the current tile", and on
    /// `warn` the drop is for *subsequent* tiles — neither level abandons work in flight, and a
    /// pressure event that arrives mid-tile is answered at the next tile boundary.
    ///
    /// A property rather than a sentence in a comment so that a test can assert it at every level.
    public var finishesCurrentTile: Bool { true }

    /// **Invariantly `true`:** no teardown/reload path exists.
    ///
    /// `CoreMLBackend.load` calls `MLModel.compileModel` on **every** call, so freeing the model
    /// and reloading it pays a compilation spike and a fresh allocation spike exactly when the
    /// system is short of memory — the opposite of the intent. Apple's guidance points the same
    /// way: dumping what you hold can make the compressor decompress it first, and the recommended
    /// reaction is a **policy change**, which is what the rung drop is.
    ///
    /// Nothing in this file references `InferenceBackend`, `CoreMLBackend`, or a model at all. That
    /// absence is the structural version of the same claim, and it is greppable.
    public var retainsModel: Bool { true }

    /// The job continues in a degraded state — the floor rung held under pressure. Certification
    /// spec §6's 48 MP rule is satisfied by continuing; presentation belongs to a UI layer.
    public var isDegraded: Bool { outcome?.isDegraded ?? false }

    /// Actuate one level against the ladder. **This is the entire decision layer of the pressure
    /// policy**, and all of it is a lookup.
    ///
    /// At `.normal` the ladder is **not** consulted, and that is load-bearing rather than an
    /// optimisation: `TileShapeLadder.respond(to:plan:)` resets `dwellCount` on every call, so
    /// handing it a relief event would silently spend the sustained headroom an eventual step up
    /// depends on. Pressure relief is not a ladder transition; it is the absence of one.
    ///
    /// It takes the plan computed at image start (`MemoryBudget.plan(for:)`) and re-budgets
    /// nothing: the plan is used only to report what the move costs.
    ///
    /// - Parameters:
    ///   - level: the level to actuate.
    ///   - ladder: the ladder the engine is carrying; stepped down in place when the level
    ///     triggers.
    ///   - plan: the plan computed once at image start.
    /// - Returns: the typed response for this event.
    public static func respond(
        to level: MemoryPressureLevel, ladder: inout TileShapeLadder, plan: MemoryBudget.Plan
    ) -> MemoryPressureResponse {
        guard let trigger = level.ladderTrigger else {
            return MemoryPressureResponse(level: level, outcome: nil)
        }
        return MemoryPressureResponse(level: level, outcome: ladder.respond(to: trigger, plan: plan))
    }
}

// MARK: - Where levels come from

/// The injection seam for pressure events — the reason the pressure tests need no device, no model
/// and no real system pressure.
///
/// `DispatchMemoryPressureSource` is the shipped conformer; a test conforms its own and drives
/// `warn`/`critical` by hand. The handler may be called on any thread, so a conformer's caller must
/// be safe to invoke from a dispatch queue — `MemoryPressureMonitor` is.
public protocol MemoryPressureSource: Sendable {
    /// Begin delivering transitions to `handler`. A second call replaces the first subscription.
    func activate(_ handler: @escaping @Sendable (MemoryPressureLevel) -> Void)

    /// Stop delivering. Idempotent, and safe without a prior `activate(_:)`.
    func cancel()
}

/// The production source: `DISPATCH_SOURCE_TYPE_MEMORYPRESSURE`, event-driven.
///
/// **No polling, no timers, no sleeps, no repeating work of any kind.** The kernel wakes the
/// handler on a transition and nothing runs in between — that is why this mechanism is used rather
/// than `task_info`, which Apple calls "an expensive call", or `os_proc_available_memory()`, which
/// is cheap but is still something the engine would have to ask on a schedule. The engine already
/// reads availability once per image through `AvailableMemory.now()`; this file adds no second
/// reading of anything.
///
/// A lock-guarded `final class` rather than an actor, matching `CoreMLBackend` and
/// `SequentialTilingEngine`: the dispatch source is not `Sendable`, and it must be cancelled
/// exactly once.
public final class DispatchMemoryPressureSource: MemoryPressureSource, @unchecked Sendable {
    /// Every transition the overlay defines. Subscribing to `.normal` as well as the two pressure
    /// levels is what lets relief be *reported* rather than polled for.
    ///
    /// Computed rather than stored: `DispatchSource.MemoryPressureEvent` is not `Sendable`, so a
    /// `static let` of it is rejected under Swift 6 strict concurrency as shared mutable state. An
    /// `OptionSet` of three bits is free to rebuild, and this way nothing is shared at all.
    static var eventMask: DispatchSource.MemoryPressureEvent { [.normal, .warning, .critical] }

    private let queue: DispatchQueue
    private let lock = NSLock()
    private var source: DispatchSourceMemoryPressure?

    /// - Parameter queue: the queue the kernel's handler runs on. `.utility` because nothing here is
    ///   latency-critical: the response is taken at the next tile boundary either way.
    public init(
        queue: DispatchQueue = DispatchQueue(
            label: "com.tbd.upscaler.memory-pressure", qos: .utility)
    ) {
        self.queue = queue
    }

    deinit { cancel() }

    public func activate(_ handler: @escaping @Sendable (MemoryPressureLevel) -> Void) {
        let created = DispatchSource.makeMemoryPressureSource(
            eventMask: Self.eventMask, queue: queue)

        // The handler captures the source it is attached to — a deliberate retain cycle, and the
        // only way to read `data` for the level that fired. It captures no `self`, so this object
        // still deallocates normally; `deinit` cancels, dispatch then releases the handler, and the
        // cycle goes with it. The capture needs no `nonisolated(unsafe)`: the Dispatch overlay marks
        // `DispatchSourceMemoryPressure` `Sendable` (the compiler rejects the annotation as
        // unnecessary), while `DispatchSource.MemoryPressureEvent` is not — see `eventMask`.
        created.setEventHandler {
            guard let level = MemoryPressureLevel(created.data) else { return }
            handler(level)
        }

        // Activate **before** publishing it to `self.source`, so a concurrent `cancel()` can never
        // reach a source that is still inactive. libdispatch treats releasing a suspended object as
        // a client bug, and a source is suspended until it is activated — so the ordering is a
        // crash-avoidance requirement, not a style choice. Events arriving in the gap are handled:
        // the handler is already attached.
        created.activate()

        let previous: DispatchSourceMemoryPressure? = lock.withLock {
            let previous = self.source
            self.source = created
            return previous
        }
        previous?.cancel()
    }

    public func cancel() {
        let current: DispatchSourceMemoryPressure? = lock.withLock {
            let current = self.source
            self.source = nil
            return current
        }
        current?.cancel()
    }
}

// MARK: - The latch between the kernel and the tile loop

/// Holds what the kernel said until the engine reaches a point where it can act on it.
///
/// **Why a latch and not a callback into the ladder.** The policy fixes the decision point: "drop a
/// ladder rung for *subsequent* tiles" and "finish the current tile, then checkpoint" both mean *at
/// the next tile boundary*, never mid-tile. And `TileShapeLadder` is a value type the engine owns —
/// stepping it from a dispatch queue while the tile loop is running would be a data race on that
/// value, not a policy decision. So the handler records, and the loop consumes.
///
/// **`consumePending()` returns the worst level seen since the last consume, not the latest.** A
/// `warn` that arrives mid-tile and is followed by `normal` before the boundary is still a `warn`
/// that happened; dropping it would make the engine's response depend on where inside a tile the
/// kernel happened to fire. That timing-dependence is what this design exists to remove.
///
/// **Cost:** nothing here polls, sleeps, subscribes on a schedule or wakes on a timer. The handler
/// runs only when the kernel posts a transition; `consumePending()` is one lock acquisition and no
/// allocation, so calling it once per tile boundary is a read-and-clear of a stored enum, not
/// per-tile churn.
///
/// A lock-guarded `final class`, matching `CoreMLBackend` and `SequentialTilingEngine`. It is the
/// one part of the pressure path that is not a value type, because it genuinely holds state written
/// by one thread and read by another.
public final class MemoryPressureMonitor: @unchecked Sendable {
    private let source: MemoryPressureSource
    private let lock = NSLock()
    private var latestLevel: MemoryPressureLevel = .normal
    private var pending: MemoryPressureLevel?
    private var isMonitoring = false
    private var eventCount = 0

    /// - Parameter source: where levels come from. Defaults to the dispatch source; tests inject
    ///   their own and drive `warn`/`critical` by hand.
    public init(source: MemoryPressureSource = DispatchMemoryPressureSource()) {
        self.source = source
    }

    deinit { source.cancel() }

    /// Subscribe. Idempotent: a second call while already monitoring does nothing rather than
    /// double-subscribing, so a mis-sequenced caller degrades to one subscription instead of two
    /// handlers stepping the ladder twice for one event.
    public func start() {
        let shouldActivate: Bool = lock.withLock {
            guard !isMonitoring else { return false }
            isMonitoring = true
            return true
        }
        guard shouldActivate else { return }
        source.activate { [weak self] level in self?.record(level) }
    }

    /// Unsubscribe. Leaves `level` and any pending event as they are — what the kernel said before
    /// the engine stopped listening is still true.
    public func stop() {
        let shouldCancel: Bool = lock.withLock {
            guard isMonitoring else { return false }
            isMonitoring = false
            return true
        }
        guard shouldCancel else { return }
        source.cancel()
    }

    /// The most recent level reported. `.normal` before anything has been reported at all — the
    /// honest default, since the process is not under pressure until the kernel says it is.
    public var level: MemoryPressureLevel { lock.withLock { latestLevel } }

    /// Events recorded since construction. Test instrumentation for the idempotence of `start()`,
    /// deliberately not part of the response contract — the role `outputAllocations` plays for
    /// `SequentialTilingEngine`.
    var observedEventCount: Int { lock.withLock { eventCount } }

    /// The worst level reported since the last consume, and clear it. `nil` when nothing has been
    /// reported since — the ordinary case at almost every tile boundary.
    public func consumePending() -> MemoryPressureLevel? {
        lock.withLock {
            let taken = pending
            pending = nil
            return taken
        }
    }

    /// Consume one pending event and actuate it against the ladder — **the call the engine makes at
    /// a tile boundary.**
    ///
    /// `nil` when no event arrived, and in that case the ladder is not touched at all: no trigger,
    /// no dwell reset, no arithmetic. That is what keeps a per-tile-boundary call cheap.
    ///
    /// - Parameters:
    ///   - ladder: the ladder the engine is carrying.
    ///   - plan: the plan computed once at image start.
    /// - Returns: the response, or `nil` when nothing was pending.
    public func respond(
        ladder: inout TileShapeLadder, plan: MemoryBudget.Plan
    ) -> MemoryPressureResponse? {
        guard let level = consumePending() else { return nil }
        return MemoryPressureResponse.respond(to: level, ladder: &ladder, plan: plan)
    }

    private func record(_ level: MemoryPressureLevel) {
        lock.withLock {
            latestLevel = level
            pending = Swift.max(pending ?? level, level)
            eventCount += 1
        }
    }
}
