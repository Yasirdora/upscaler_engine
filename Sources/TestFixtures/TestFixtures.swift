/// TestFixtures — certification corpus and versioned reference outputs.
///
/// Placeholder namespace. Corpus images and reference tensors are never committed (see
/// `.gitignore`); `references_manifest.json` pins every input by SHA256, and `tools/reference/`
/// regenerates the references.
///
/// Boundary: no UI framework imports; the engine must run headless. Interfaces follow the
/// certification spec (`docs/CERTIFICATION.md`); changing one is a reviewed contract change, never
/// a silent edit.
public enum TestFixtures {}
