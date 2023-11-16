import Testing

@testable import UpscaleCLI

/// The FLOOR paragraph of the context-margin report.
///
/// The paragraph reports a measured floor for whichever model was run, so it must never carry a
/// constant or an architectural explanation belonging to some other model. What is pinned here is
/// mostly what the paragraph must NOT say.
struct FloorParagraphTests {
    static func text(_ floor: Double = 1.0612) -> String {
        ContextMarginCommand.floorParagraph(floorValue: floor).joined(separator: "\n")
    }

    @Test("the measured floor is reported, and it comes from the argument")
    func floorValueIsTheMeasuredOne() {
        #expect(Self.text(1.0612).contains("1.0612/255"))
        #expect(Self.text(0.5350).contains("0.5350/255"))
    }

    /// An architectural claim about a model this command was never told the architecture of must
    /// never appear.
    @Test("no architecture is asserted as the explanation for the floor")
    func noArchitecturalClaimIsMade() {
        let text = Self.text()

        #expect(!text.contains("The gap is context that does not fit"))
        #expect(!text.contains("receptive field is several hundred pixels"))
        #expect(!text.contains("No margin can supply all of it"))
    }

    /// No model-specific constant or architecture name may be presented as this run's result.
    @Test("no other model's constant or architecture is reported as this run's measurement")
    func noForeignConstantIsReported() {
        let text = Self.text()

        #expect(!text.contains("0.2687/255"))
        #expect(!text.contains("RRDB"))
    }

    @Test("the two-error-term limitation is stated, not implied")
    func theConflationIsDisclosed() {
        let text = Self.text()

        #expect(text.contains("TWO error terms"))
        #expect(text.contains("conversion divergence"))
        #expect(text.contains("COMBINED signal"))
    }

    @Test("the output is marked provisional")
    func provisionalityIsMarked() {
        let text = Self.text()

        #expect(text.contains("provisional"))
        #expect(text.contains("does not yet do so"))
    }

    @Test("the reader is pointed at the evidence rather than asked to trust the claim")
    func evidenceIsCited() {
        #expect(Self.text().contains("tools/margin/derive_context_margin_torch.py"))
    }
}
