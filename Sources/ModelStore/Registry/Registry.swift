/// Registry — model descriptors and the catalogue of installed models.
///
/// Placeholder namespace. Descriptors are intended to carry a recommended tile size (a hint, not a
/// fact), minimum and maximum tile sizes, and the model's context margin.
///
/// Boundary: no UI framework imports; the engine must run headless. Interfaces follow the
/// certification spec (`docs/CERTIFICATION.md`); changing one is a reviewed contract change, never
/// a silent edit.
public enum Registry {}
