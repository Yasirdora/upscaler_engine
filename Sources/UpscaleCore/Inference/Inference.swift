/// Inference — backend abstraction for upscaling engines.
///
/// The `InferenceBackend` protocol (Core ML, fixed-shape, fp16, configurable compute units) and its
/// Core ML implementation live alongside this namespace.
///
/// Boundary: no UI framework imports; the engine must run headless. Interfaces follow the
/// certification spec (`docs/CERTIFICATION.md`); changing one is a reviewed contract change, never
/// a silent edit.
public enum Inference {}
