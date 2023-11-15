import CoreGraphics
import CoreVideo
import Foundation
import ImageIO
import Testing
import UniformTypeIdentifiers

@testable import ImagePipeline

/// The export round-trip (certification spec §3.1, §3.4, §8.4).
///
/// The assertions here are deliberately end-to-end: decode a real file, plan its colour, normalise
/// its metadata, write it, then **reopen the written file** and check what survived. A test that
/// only inspected the dictionary handed to ImageIO would pass while the file on disk lost
/// everything.
struct EncodeTests {
    struct Written {
        let url: URL
        let properties: [String: Any]
        let profileName: String?
        let width: Int
    }

    /// decode → colour plan → metadata normalise → encode → reopen.
    static func roundTrip(
        space: CGColorSpace, format: Encode.Format, orientation: Int = 1,
        extra: [String: Any] = [:], quality: Double? = nil, in directory: URL
    ) throws -> Written {
        let image = try DecodeFixtures.quadrantImage(
            space: space, components: DecodeFixtures.sRGBComponents)
        var sourceProperties: [String: Any] = [
            kCGImagePropertyOrientation as String: orientation,
            kCGImagePropertyGPSDictionary as String: [
                kCGImagePropertyGPSLatitude as String: 51.5
            ],
            kCGImagePropertyExifDictionary as String: [
                kCGImagePropertyExifDateTimeOriginal as String: "2026:08:31 12:00:00"
            ],
            kCGImagePropertyTIFFDictionary as String: [
                kCGImagePropertyTIFFMake as String: "ACME"
            ],
        ]
        sourceProperties.merge(extra) { _, new in new }
        let input = try DecodeFixtures.write(
            image, to: directory.appendingPathComponent("in.png"), type: .png,
            properties: sourceProperties as CFDictionary)

        let decoded = try Decode.decode(contentsOf: input)
        let plan = try Color.plan(for: decoded.color)
        let normalised = try Metadata.normalisedProperties(from: decoded.metadata)
        let out = directory.appendingPathComponent("out.\(format.rawValue)")
        try Encode.write(
            decoded.pixelBuffer, to: out, format: format, plan: plan, properties: normalised,
            quality: quality)

        let source = try #require(CGImageSourceCreateWithURL(out as CFURL, nil))
        let read = try #require(
            CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [String: Any])
        let reopened = try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
        return Written(
            url: out, properties: read,
            profileName: reopened.colorSpace?.name as String?, width: reopened.width)
    }

    // MARK: - §8.4 — all four formats round-trip with metadata intact

    @Test("every §3.1 format round-trips carrying its metadata", arguments: Encode.Format.allCases)
    func formatsRoundTripWithMetadata(format: Encode.Format) throws {
        let directory = try DecodeFixtures.makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let written = try Self.roundTrip(
            space: try DecodeFixtures.srgb(), format: format,
            quality: format.isLossy ? 0.9 : nil, in: directory)

        let exif = try #require(
            written.properties[kCGImagePropertyExifDictionary as String] as? [String: Any])
        #expect(
            exif[kCGImagePropertyExifDateTimeOriginal as String] as? String
                == "2026:08:31 12:00:00",
            "\(format.rawValue) lost the capture timestamp — 13 of 13 fields is today's behaviour")
        let gps = try #require(
            written.properties[kCGImagePropertyGPSDictionary as String] as? [String: Any])
        #expect(gps[kCGImagePropertyGPSLatitude as String] as? Double == 51.5)
    }

    /// The colour defect, asserted at the only place that settles it: the file on disk.
    @Test("a Display P3 source is written as Display P3, not flattened to sRGB")
    func displayP3SurvivesToTheFile() throws {
        let directory = try DecodeFixtures.makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let written = try Self.roundTrip(
            space: try DecodeFixtures.displayP3(), format: .png, in: directory)

        #expect(
            written.profileName == "kCGColorSpaceDisplayP3",
            "writing sRGB here IS the demonstrated defect: measured cost 117/255 on a P3 sweep")
    }

    @Test("an sRGB source is written as sRGB")
    func srgbSurvivesToTheFile() throws {
        let directory = try DecodeFixtures.makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let written = try Self.roundTrip(space: try DecodeFixtures.srgb(), format: .png, in: directory)

        #expect(written.profileName == "kCGColorSpaceSRGB")
    }

    // MARK: - §3.4 — orientation normalised to 1 on export

    /// §3.4 requires orientation normalised to 1 on export. The source here is tagged 6, and the
    /// written file must claim 1 — otherwise a reader rotates an image that is already upright.
    @Test("an exported file asserts orientation 1 even when the source was tagged 6")
    func exportNormalisesOrientation() throws {
        let directory = try DecodeFixtures.makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let written = try Self.roundTrip(
            space: try DecodeFixtures.srgb(), format: .png, orientation: 6, in: directory)

        #expect(written.properties[kCGImagePropertyOrientation as String] as? Int == 1)
    }

    /// §3.4 preserves by default, so a key the encoder has never heard of must survive the file.
    @Test("a key the encoder does not know about survives to disk")
    func unknownKeysSurviveToDisk() throws {
        let directory = try DecodeFixtures.makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let written = try Self.roundTrip(
            space: try DecodeFixtures.srgb(), format: .tiff,
            extra: [
                kCGImagePropertyTIFFDictionary as String: [
                    kCGImagePropertyTIFFMake as String: "ACME",
                    kCGImagePropertyTIFFModel as String: "Model-Q",
                ]
            ], in: directory)

        let tiff = try #require(
            written.properties[kCGImagePropertyTIFFDictionary as String] as? [String: Any])
        #expect(tiff[kCGImagePropertyTIFFModel as String] as? String == "Model-Q")
    }

    // MARK: - Quality is refused where it means nothing

    /// Accepting a quality for PNG would imply a control the user does not have.
    @Test("a quality supplied for a lossless format is refused, not ignored")
    func qualityIsRefusedForLosslessFormats() throws {
        let directory = try DecodeFixtures.makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        #expect(throws: Encode.EncodeError.qualityNotApplicable(format: "png")) {
            _ = try Self.roundTrip(
                space: try DecodeFixtures.srgb(), format: .png, quality: 0.5, in: directory)
        }
    }

    @Test("a quality outside 0...1 is refused", arguments: [-0.1, 1.1, 42.0])
    func qualityOutOfRangeIsRefused(quality: Double) throws {
        let directory = try DecodeFixtures.makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        #expect(throws: Encode.EncodeError.qualityOutOfRange(quality)) {
            _ = try Self.roundTrip(
                space: try DecodeFixtures.srgb(), format: .jpeg, quality: quality, in: directory)
        }
    }

    @Test("only the lossy formats carry a quality control")
    func lossyClassification() {
        #expect(Encode.Format.jpeg.isLossy)
        #expect(Encode.Format.heif.isLossy)
        #expect(!Encode.Format.png.isLossy)
        #expect(!Encode.Format.tiff.isLossy)
    }

    @Test("an unwritable destination fails loudly rather than silently producing nothing")
    func unwritableDestinationThrows() throws {
        let directory = try DecodeFixtures.makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let image = try DecodeFixtures.quadrantImage(
            space: try DecodeFixtures.srgb(), components: DecodeFixtures.sRGBComponents)
        let input = try DecodeFixtures.write(
            image, to: directory.appendingPathComponent("in.png"), type: .png)
        let decoded = try Decode.decode(contentsOf: input)
        let plan = try Color.plan(for: decoded.color)

        #expect(throws: (any Error).self) {
            try Encode.write(
                decoded.pixelBuffer,
                to: URL(fileURLWithPath: "/nonexistent-directory-xyz/out.png"),
                format: .png, plan: plan, properties: [:])
        }
    }
}
