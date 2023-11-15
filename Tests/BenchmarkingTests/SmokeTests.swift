import Testing

@testable import Benchmarking

/// Smoke test for the `Benchmarking` target.
///
/// The assertion that matters is the `import` above: this file does not compile or link unless the
/// target builds. The body is deliberately trivially true.
@Test("`Benchmarking` builds, links and is importable")
func benchmarkingTargetLinks() {
    #expect(Bool(true))
}
