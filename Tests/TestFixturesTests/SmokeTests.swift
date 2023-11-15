import Testing

@testable import TestFixtures

/// Smoke test for the `TestFixtures` target.
///
/// The assertion that matters is the `import` above: this file does not compile or link unless the
/// target builds. The body is deliberately trivially true.
@Test("`TestFixtures` builds, links and is importable")
func testFixturesTargetLinks() {
    #expect(Bool(true))
}
