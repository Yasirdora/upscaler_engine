/// Pipeline — per-job orchestration across inference, tiling, memory and scheduling.
///
/// Placeholder namespace; orchestration currently lives with the caller (see `Scheduling`).
///
/// Boundary: no UI framework imports; the engine must run headless. Interfaces follow the
/// certification spec (`docs/CERTIFICATION.md`); changing one is a reviewed contract change, never
/// a silent edit.
public enum Pipeline {}
