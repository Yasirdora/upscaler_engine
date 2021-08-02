import Testing

@testable import UpscaleCore

/// Smoke test for the `UpscaleCore` target.
///
/// The assertion that matters is the `import` above: this file does not compile or link unless the
/// target builds. The body is deliberately trivially true.
@Test("`UpscaleCore` builds, links and is importable")
func upscaleCoreTargetLinks() {
    #expect(Bool(true))
}
