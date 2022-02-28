import Testing

@testable import UpscaleCore

/// The per-model context margins.
///
/// The margin follows the receptive field, and the supported models' fields differ by more than 20×
/// — 34.00, 348.25 and 697.50 input px. Running one model on another's margin under-pads it, and
/// **under-padding throws nothing**; it produces a picture with seams in it, found by a user.
///
/// These literals are guards, not documentation. Changing one requires a new derivation.
@Suite
struct ContextMarginValueTests {
    @Test("each supported model's margin is exactly the value it runs at")
    func marginsAreExact() {
        // x4plus: derived 76, runs at 76.
        #expect(ContextMargin.derivedMargin(for: "RealESRGAN_x4plus") == 76)
        // x4v3: derived 16, adopted 16. A 34 px field needs little.
        #expect(ContextMargin.derivedMargin(for: "realesr-general-x4v3") == 16)
        // x2plus: derived 108, RUNS AT 76 — seam certification at both margins found no visible
        // seam and no decisive difference, and +8.8 s on a 12 MP photo was judged unjustified
        // without a corpus able to prove one.
        #expect(ContextMargin.derivedMargin(for: "RealESRGAN_x2plus") == 76)
    }

    /// **The one model running below its own derivation, and that must stay visible.**
    ///
    /// Asserting that the three margins are *distinct* would be the wrong guard: x2plus and x4plus
    /// share 76 deliberately. So the divergence is recorded in the source and pinned here instead —
    /// if someone later "tidies" x2plus out of `derivedButNotAdopted`, the fact that it is
    /// knowingly under-padded disappears with it.
    @Test("x2plus's unadopted derivation is recorded, and differs from what it runs")
    func unadoptedDerivationIsVisible() throws {
        let derived = try #require(ContextMargin.derivedButNotAdopted["RealESRGAN_x2plus"])
        let running = try #require(ContextMargin.derivedMargin(for: "RealESRGAN_x2plus"))

        #expect(derived == 108)
        #expect(running == 76)
        #expect(running < derived, "x2plus runs BELOW its derivation; that is the carried risk")
    }

    /// Every other model runs at what was derived for it. Only x2plus diverges, and only by
    /// decision.
    @Test("no model other than x2plus diverges from its derivation")
    func onlyX2plusDiverges() {
        #expect(Set(ContextMargin.derivedButNotAdopted.keys) == ["RealESRGAN_x2plus"])
    }

    /// An unknown model returns nil rather than a default. Any default is some other model's
    /// number, which is the defect; the caller is made to decide instead of inheriting it.
    @Test("an underived model has no margin, and no default is invented")
    func unknownModelHasNoMargin() {
        #expect(ContextMargin.derivedMargin(for: "RealESRGAN_x4plus_anime_6B") == nil)
        #expect(ContextMargin.derivedMargin(for: "") == nil)
    }

    /// Keyed on the MODEL, not the shape-suffixed descriptor identity: a margin is a property of
    /// the network and does not change with tile size.
    @Test("the table is keyed on the model, not on a shape-suffixed identifier")
    func keyIsTheModelNotTheShape() {
        #expect(ContextMargin.derivedMargin(for: "RealESRGAN_x2plus") == 76)
        #expect(ContextMargin.derivedMargin(for: "RealESRGAN_x2plus_256x256") == nil)
        #expect(ContextMargin.derivedMargin(for: "RealESRGAN_x4plus_512x512") == nil)
    }
}
