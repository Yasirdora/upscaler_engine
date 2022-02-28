/// Tiling — tile planner, context margins, valid-region stitching.
///
/// The context margin is derived per model by receptive-field testing, never copied (certification
/// spec §5.1).
///
/// Boundary: no UI framework imports; the engine must run headless. Interfaces follow the
/// certification spec (`docs/CERTIFICATION.md`); changing one is a reviewed contract change, never
/// a silent edit.
public enum Tiling {}
