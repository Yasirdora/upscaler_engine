import Testing

@testable import ModelStore

/// Smoke test for the `ModelStore` target.
///
/// The assertion that matters is the `import` above: this file does not compile or link unless the
/// target builds. The body is deliberately trivially true.
@Test("`ModelStore` builds, links and is importable")
func modelStoreTargetLinks() {
    #expect(Bool(true))
}
