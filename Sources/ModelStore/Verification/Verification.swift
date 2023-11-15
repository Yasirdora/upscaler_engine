/// Verification — SHA256 verification of model artifacts.
///
/// Placeholder namespace. Host-side fetch-with-verification lives in `tools/artifacts/fetch.py`;
/// every artifact identity is recorded in `models/<id>/manifest.json`.
///
/// Boundary: no UI framework imports; the engine must run headless. Interfaces follow the
/// certification spec (`docs/CERTIFICATION.md`); changing one is a reviewed contract change, never
/// a silent edit.
public enum Verification {}
