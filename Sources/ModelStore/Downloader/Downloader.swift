/// Downloader — user-initiated model downloads with disclosed sizes.
///
/// Placeholder namespace. Downloadable models are data resources consumed by the shipped engine,
/// never executable modules.
///
/// Boundary: no UI framework imports; the engine must run headless. Interfaces follow the
/// certification spec (`docs/CERTIFICATION.md`); changing one is a reviewed contract change, never
/// a silent edit.
public enum Downloader {}
