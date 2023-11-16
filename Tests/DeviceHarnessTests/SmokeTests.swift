import Testing

@testable import DeviceHarness

/// Smoke test for the `DeviceHarness` target.
///
/// The assertion that matters is the `import` above: this file does not compile or link unless the
/// target builds. The body is deliberately trivially true.
@Test("`DeviceHarness` builds, links and is importable")
func deviceHarnessTargetLinks() {
    #expect(Bool(true))
}
