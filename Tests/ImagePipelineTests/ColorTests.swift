import CoreGraphics
import Foundation
import Testing

@testable import ImagePipeline

/// The colour policy.
///
/// Fixtures are constructed, not drawn from the corpus: it holds **zero Display P3 images**, so the
/// central claim cannot be exercised from it at all. Expectations are written-out literals rather
/// than values recomputed from the code under test.
struct ColorTests {
    /// A decode of a real file, so the plan is tested against carriage the decoder actually
    /// produces rather than against a hand-built struct that might not be reachable.
    static func decoded(space: CGColorSpace, components: [[CGFloat]]) throws -> Decode.DecodedImage {
        let directory = try DecodeFixtures.makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let image = try DecodeFixtures.quadrantImage(space: space, components: components)
        let url = try DecodeFixtures.write(
            image, to: directory.appendingPathComponent("c.png"), type: .png)
        return try Decode.decode(contentsOf: url)
    }

    // MARK: - The demonstrated defect, stated as an assertion

    /// The one that matters. A P3 source must reach the encoder as P3.
    @Test("a Display P3 source plans to write Display P3, not sRGB")
    func p3IsPreservedIntoTheOutputProfile() throws {
        let decoded = try Self.decoded(
            space: try DecodeFixtures.displayP3(), components: DecodeFixtures.p3Components)

        let plan = try Color.plan(for: decoded.color)

        #expect(plan.provenance == .tagged)
        #expect(plan.outputProfileName == "kCGColorSpaceDisplayP3")
        #expect(
            plan.outputColorSpace()?.name as String? == (CGColorSpace.displayP3 as String),
            "writing sRGB here IS the demonstrated defect — a P3 photograph returning flattened")
        #expect(plan.outputProfile == decoded.color.bufferProfile)
    }

    @Test("an sRGB source plans to write sRGB and is recorded as tagged, not assumed")
    func srgbIsTaggedNotAssumed() throws {
        let decoded = try Self.decoded(
            space: try DecodeFixtures.srgb(), components: DecodeFixtures.sRGBComponents)

        let plan = try Color.plan(for: decoded.color)

        #expect(plan.provenance == .tagged)
        #expect(plan.outputColorSpace()?.name as String? == (CGColorSpace.sRGB as String))
    }

    /// A tagged sRGB file and an untagged one produce the same pixels and the same output
    /// profile — but must NOT produce the same provenance, or the assumption becomes invisible.
    @Test("tagged and assumed sRGB agree on the profile and disagree on the provenance")
    func taggedAndAssumedAreDistinguishable() throws {
        let tagged = try Color.plan(
            for: try Self.decoded(
                space: try DecodeFixtures.srgb(), components: DecodeFixtures.sRGBComponents
            ).color)
        let assumed = try Color.plan(
            for: Decode.ColorCarriage(
                sourceProfile: nil, sourceProfileName: nil,
                bufferProfile: try #require(Color.assumedSRGBProfile()),
                disposition: .preservedSourceSpace))

        #expect(tagged.outputProfile == assumed.outputProfile)
        #expect(tagged.provenance == .tagged)
        #expect(assumed.provenance == .assumedSRGB)
    }

    // MARK: - Untagged input

    @Test("untagged input is assumed sRGB rather than refused")
    func untaggedIsAssumedNotRefused() throws {
        let carriage = Decode.ColorCarriage(
            sourceProfile: nil, sourceProfileName: nil,
            bufferProfile: try #require(Color.assumedSRGBProfile()),
            disposition: .preservedSourceSpace)

        let plan = try Color.plan(for: carriage)

        #expect(plan.provenance == .assumedSRGB)
        #expect(plan.outputColorSpace()?.name as String? == (CGColorSpace.sRGB as String))
    }

    /// The assumption must be written down, not merely acted on.
    @Test("the assumption is recorded in the notes and the output is explicitly tagged")
    func theAssumptionIsRecorded() throws {
        let plan = try Color.plan(
            for: Decode.ColorCarriage(
                sourceProfile: nil, sourceProfileName: nil,
                bufferProfile: try #require(Color.assumedSRGBProfile()),
                disposition: .preservedSourceSpace))

        #expect(plan.notes.contains { $0.contains("no ICC profile") })
        #expect(plan.notes.contains { $0.contains("sRGB assumed") })
        #expect(!plan.outputProfile.isEmpty, "an assumed space is still written, never left absent")
    }

    // MARK: - Normalisation is labelled, not hidden

    /// Decode normalises a space that cannot back an 8-bit bitmap. The output must then be tagged
    /// with what the samples ARE, not with what the file said — mislabelling normalised samples
    /// with the source's profile would be worse than the normalisation.
    @Test("a normalised decode writes sRGB and says the source space was lost")
    func normalisationIsLabelled() throws {
        let carriage = Decode.ColorCarriage(
            sourceProfile: Data([0x01, 0x02, 0x03]),
            sourceProfileName: "Generic CMYK",
            bufferProfile: try #require(Color.assumedSRGBProfile()),
            disposition: .normalisedToSRGB)

        let plan = try Color.plan(for: carriage)

        #expect(plan.provenance == .normalisedFromUnsupportedSpace)
        #expect(plan.outputProfile != Data([0x01, 0x02, 0x03]), "never write the source's profile")
        #expect(plan.notes.contains { $0.contains("normalised to sRGB") })
    }

    // MARK: - §3.1 gamma correctness

    /// The strongest available guarantee is that no conversion exists to get wrong.
    @Test("every plan is conversion-free, so no transfer function can be misapplied")
    func plansAreConversionFree() throws {
        for (space, components) in [
            (try DecodeFixtures.srgb(), DecodeFixtures.sRGBComponents),
            (try DecodeFixtures.displayP3(), DecodeFixtures.p3Components),
        ] {
            let decoded = try Self.decoded(space: space, components: components)
            let plan = try Color.plan(for: decoded.color)
            #expect(plan.isConversionFree)
            #expect(plan.workingProfile == plan.outputProfile)
        }
    }

    @Test("the working space is the buffer's space, never the source's")
    func workingSpaceFollowsTheBuffer() throws {
        let carriage = Decode.ColorCarriage(
            sourceProfile: Data([0xAA]), sourceProfileName: "Something Exotic",
            bufferProfile: try #require(Color.assumedSRGBProfile()),
            disposition: .normalisedToSRGB)

        let plan = try Color.plan(for: carriage)

        #expect(plan.workingProfile == carriage.bufferProfile)
        #expect(plan.workingProfile != Data([0xAA]))
    }

    // MARK: - Failure

    @Test("a buffer profile that cannot be rebuilt is an error, not a silent sRGB fallback")
    func unusableBufferProfileThrows() throws {
        let carriage = Decode.ColorCarriage(
            sourceProfile: nil, sourceProfileName: nil,
            bufferProfile: Data([0x00, 0x01]), disposition: .preservedSourceSpace)

        #expect(throws: Color.ColorError.unusableBufferProfile) {
            _ = try Color.plan(for: carriage)
        }
    }

    @Test("the assumed sRGB profile is real and rebuilds as sRGB")
    func assumedProfileIsRealSRGB() throws {
        let data = try #require(Color.assumedSRGBProfile())

        #expect(!data.isEmpty)
        #expect(
            CGColorSpace(iccData: data as CFData)?.name as String? == (CGColorSpace.sRGB as String))
    }
}
