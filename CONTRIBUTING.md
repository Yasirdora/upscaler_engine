# Contributing

Thanks for your interest. Issues and pull requests are welcome.

## Before you open a pull request

```bash
swift build -Xswiftc -warnings-as-errors
swift test  -Xswiftc -warnings-as-errors
swift format lint --recursive --strict Package.swift Sources Tests Examples/UpscaleImage/Sources
swift build --package-path Examples/UpscaleImage
python -m pytest tools/ -q            # needs pytest and numpy
```

CI runs the same commands, plus a module-boundary check and an artifact-hygiene check
(`.github/workflows/ci.yml`).

## Ground rules

- **Zero warnings.** The build treats warnings as errors.
- **No third-party Swift dependencies.** The engine uses Apple frameworks only.
- **Respect the module boundaries** declared in `Package.swift`. `UpscaleCore` must not import
  `ImagePipeline` or `ModelStore`, and engine modules must not import SwiftUI, UIKit or AppKit.
  SwiftPM does not reliably reject an undeclared import on a warm build, so CI checks it.
- **Tests assert hand-derived literals.** An expected value is written out in the test, with its
  derivation in a comment when it is not obvious — never recomputed with the code under test.
  Tests stay model-free and device-free; anything that needs Core ML or hardware goes through
  `DeviceHarness`.
- **Numbers carry provenance.** A measured figure says where it was measured; an estimate says it
  is one; a missing figure is `unavailable`, never a zero or a guess.
- **Never commit weights, `.mlpackage`s, corpus images or reference tensors.** `.gitignore` and CI
  enforce this. Artifacts are identified by the SHA256s in their manifests.
- **Certification thresholds only tighten.** Changing a definition or threshold in
  `docs/CERTIFICATION.md` needs evidence attached in the pull request.

## Corpus images

If you add images to a local certification corpus, record each image's source and licence in
`Sources/TestFixtures/corpus/README.md`. Only use images you have the right to use, and never
commit them.
