/// Benchmarking — time, memory, energy, thermal and sustained-throughput measurement.
///
/// Placeholder namespace. The methodology is fixed by certification spec §7.3: cold, warm,
/// sustained and throttled reported separately, median of ≥ 10 warm runs, energy labelled measured
/// / estimated / unavailable.
///
/// Boundary: no UI framework imports; the engine must run headless. Interfaces follow the
/// certification spec (`docs/CERTIFICATION.md`); changing one is a reviewed contract change, never
/// a silent edit.
public enum Benchmarking {}
