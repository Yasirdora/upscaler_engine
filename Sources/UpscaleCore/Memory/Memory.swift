/// Memory — memory budgets, footprint estimation, pressure response.
///
/// `MemoryBudget.swift` — the proactive budget certification spec §6 requires. `MemoryBudget` is a
/// value type with injected inputs: the runtime query lives in `AvailableMemory.now()` and is
/// handed in, never called from inside the budget, so every figure is assertable against
/// hand-computed values with no device and no model. The safety fraction ρ = 0.5 is named in
/// `MemoryBudget.SafetyFraction` (permitted range [0.4, 0.6]); the concurrency term is pinned at 1
/// (`MemoryBudget.concurrency`); every figure carries measured / estimated / unavailable
/// provenance.
///
/// `TileShapeLadder.swift` — the ladder policy. Down one rung on a trigger, mid-image, where the
/// current tile is abandoned, retried at the smaller shape, and the image re-gridded from that tile
/// forward — never failing the job; up one rung only between images, because a step up would
/// re-grid backwards over assembled output, and only on dwell *and* a ρ-derived margin. 256 is the
/// floor, below which certification spec §6's 48 MP rule applies and the outcome is typed as
/// degraded. It is a pure value type: it reads the budget's `Plan`, it subscribes to nothing, and
/// it is never consulted per tile. The **dwell default is a labelled placeholder** (`.unavailable`
/// provenance) awaiting device runs; the margin default derives from ρ and is labelled
/// `.estimated`.
///
/// `MemoryPressure.swift` — the pressure response. A `DispatchSourceMemoryPressure` subscription
/// (event-driven, headless-legal, no polling and no timers) whose whole job is to map `warn` and
/// `critical` onto `TileShapeLadder.Trigger` and let the ladder decide the rest. On `warn` the rung
/// drops for *subsequent* tiles; on `critical` the current tile is finished and a checkpoint is
/// asked for; at the floor rung the job continues degraded. **There is no teardown/reload path** —
/// `CoreMLBackend.load` recompiles on every call, so freeing the model would pay a compilation
/// spike exactly when the system is short. Events are latched rather than acted on in the handler:
/// the decision point is the next tile boundary, where
/// `SequentialTilingEngine.upscale(_:shapes:ladder:budget:pressure:method:spoolingTo:progress:)`
/// consumes them.
///
/// Boundary: no UI framework imports; the engine must run headless.
public enum Memory {}
