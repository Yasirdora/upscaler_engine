import Testing

@testable import ImagePipeline

/// Smoke test for the `ImagePipeline` target.
///
/// The assertion that matters is the `import` above: this file does not compile or link unless the
/// target builds. The body is deliberately trivially true.
@Test("`ImagePipeline` builds, links and is importable")
func imagePipelineTargetLinks() {
    #expect(Bool(true))
}
