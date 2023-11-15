/// Cache — on-device storage of verified model artifacts.
///
/// Placeholder namespace. A single resident model is a memory-hazard mitigation.
///
/// Boundary: no UI framework imports; the engine must run headless. Interfaces follow the
/// certification spec (`docs/CERTIFICATION.md`); changing one is a reviewed contract change, never
/// a silent edit.
public enum Cache {}
