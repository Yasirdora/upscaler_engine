import CoreGraphics
import CoreVideo
import Foundation
import ImageIO
import Testing

@testable import ImagePipeline

/// Orientation normalisation and metadata carriage (certification spec §3.4, §8.2).
///
/// Fixtures are constructed: the corpus carries **no EXIF orientation but 1**, so orientation
/// handling cannot be exercised from it at all. Expected values are written-out literals.
struct MetadataTests {
    static func quadrantBuffer() throws -> (CVPixelBuffer, URL, URL) {
        let directory = try DecodeFixtures.makeDirectory()
        let image = try DecodeFixtures.quadrantImage(
            space: try DecodeFixtures.srgb(), components: DecodeFixtures.sRGBComponents)
        let url = try DecodeFixtures.write(
            image, to: directory.appendingPathComponent("q.png"), type: .png)
        return (try Decode.decode(contentsOf: url).pixelBuffer, url, directory)
    }

    /// A CGImage rendered into the engine's own buffer format, so the two sides of the
    /// comparison are read through identical code.
    static func buffer(from image: CGImage) throws -> CVPixelBuffer {
        let out = try Decode.makeBuffer(width: image.width, height: image.height)
        CVPixelBufferLockBaseAddress(out, [])
        defer { CVPixelBufferUnlockBaseAddress(out, []) }
        guard
            let ctx = CGContext(
                data: CVPixelBufferGetBaseAddress(out), width: image.width, height: image.height,
                bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(out),
                space: try DecodeFixtures.srgb(),
                bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
                    | CGBitmapInfo.byteOrder32Little.rawValue)
        else { throw Metadata.MetadataError.drawingContextUnavailable }
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        return out
    }

    /// Four distinct quadrants at an arbitrary size. A square fixture cannot tell a transform
    /// that confuses width with height from a correct one.
    static func quadrants(width: Int, height: Int) throws -> CGImage {
        let space = try DecodeFixtures.srgb()
        guard
            let context = CGContext(
                data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                space: space, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { throw Metadata.MetadataError.drawingContextUnavailable }
        for (index, values) in DecodeFixtures.sRGBComponents.enumerated() {
            guard let color = CGColor(colorSpace: space, components: values + [1]) else {
                throw Metadata.MetadataError.drawingContextUnavailable
            }
            context.setFillColor(color)
            context.fill(
                CGRect(
                    x: (index % 2) * width / 2, y: (index / 2) * height / 2,
                    width: width / 2, height: height / 2))
        }
        guard let image = context.makeImage() else {
            throw Metadata.MetadataError.drawingContextUnavailable
        }
        return image
    }

    // MARK: - The eight orientations

    /// Which stored quadrant must end up top-left after each orientation is applied.
    ///
    /// `sRGBSamples` is indexed 0=TL, 1=TR, 2=BL, 3=BR at `samplePoints`.
    ///
    /// These literals are EXIF's semantics and they are **confirmed against ImageIO** by
    /// `agreesWithImageIO` below, not merely reasoned about — all four quarter-turn transforms
    /// were initially implemented with the wrong handedness, and this table is what disagreed
    /// with them. Kept alongside the ImageIO comparison rather than replaced by it: if a future
    /// ImageIO changed its interpretation, that test would still pass while this one failed,
    /// which is the signal worth having.
    @Test(
        "each EXIF orientation puts the right stored quadrant top-left",
        arguments: [
            (1, 0), (2, 1), (3, 3), (4, 2), (5, 0), (6, 2), (7, 3), (8, 1),
        ])
    func orientationsPlaceTheRightQuadrant(stored: Int, expectedIndex: Int) throws {
        let (buffer, _, directory) = try Self.quadrantBuffer()
        defer { try? FileManager.default.removeItem(at: directory) }

        let out = try Metadata.normalise(
            buffer, orientation: stored, colorSpace: try DecodeFixtures.srgb())

        let corner = DecodeFixtures.pixel(of: out, x: 3, y: 3)
        #expect(
            corner == DecodeFixtures.sRGBSamples[expectedIndex],
            "orientation \(stored): top-left should be stored quadrant \(expectedIndex)")
    }

    /// The quarter turns exchange width and height. A wrong answer here draws a correctly rotated
    /// image into a canvas of the wrong shape, which crops it — and nothing throws.
    @Test(
        "the quarter turns exchange the dimensions and the others do not",
        arguments: [
            (1, false), (2, false), (3, false), (4, false),
            (5, true), (6, true), (7, true), (8, true),
        ])
    func dimensionSwapping(stored: Int, swaps: Bool) throws {
        #expect(try #require(Metadata.Orientation(stored: stored)).swapsDimensions == swaps)
    }

    @Test("a non-square raster really does come back transposed")
    func nonSquareDimensionsAreTransposed() throws {
        let directory = try DecodeFixtures.makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let image = try DecodeFixtures.solidImage(
            space: try DecodeFixtures.srgb(), width: 24, height: 8)
        let url = try DecodeFixtures.write(
            image, to: directory.appendingPathComponent("w.png"), type: .png)
        let buffer = try Decode.decode(contentsOf: url).pixelBuffer

        let rotated = try Metadata.normalise(
            buffer, orientation: 6, colorSpace: try DecodeFixtures.srgb())

        #expect(CVPixelBufferGetWidth(rotated) == 8)
        #expect(CVPixelBufferGetHeight(rotated) == 24)
    }

    /// Identity must not resample. A copy through a context would be a chance to lose colour for
    /// no reason at all.
    @Test("orientation 1 returns the very same buffer, untouched")
    func orientationOneIsIdentity() throws {
        let (buffer, _, directory) = try Self.quadrantBuffer()
        defer { try? FileManager.default.removeItem(at: directory) }

        let out = try Metadata.normalise(
            buffer, orientation: 1, colorSpace: try DecodeFixtures.srgb())

        #expect(out === buffer)
    }

    /// Coercing an unknown value to `.up` would hand back a wrongly-flipped image with no error,
    /// which is the failure mode that produced defect 2.
    @Test("an orientation outside EXIF's 1...8 is refused, never coerced", arguments: [0, 9, -1, 42])
    func undefinedOrientationIsRefused(stored: Int) throws {
        let (buffer, _, directory) = try Self.quadrantBuffer()
        defer { try? FileManager.default.removeItem(at: directory) }

        #expect(throws: Metadata.MetadataError.undefinedOrientation(stored: stored)) {
            _ = try Metadata.normalise(
                buffer, orientation: stored, colorSpace: try DecodeFixtures.srgb())
        }
        #expect(Metadata.Orientation(stored: stored) == nil)
    }

    /// Ground truth, taken from the platform rather than from my own derivation.
    ///
    /// The hand-written mapping above is a claim about EXIF semantics, and a sign error in it is
    /// invisible in a symmetric image. This test removes that reasoning from the trust chain
    /// entirely: it tags a real file with each orientation, asks **ImageIO** to apply it
    /// (`kCGImageSourceCreateThumbnailWithTransform`), and requires our transform to agree pixel
    /// for pixel. If Apple and we disagree about what orientation 6 means, the user sees Apple's
    /// answer everywhere else on the system, so Apple's answer is the correct one.
    @Test(
        "our transform agrees with ImageIO's own orientation handling, for all eight",
        arguments: [1, 2, 3, 4, 5, 6, 7, 8], [true, false])
    func agreesWithImageIO(stored: Int, square: Bool) throws {
        let directory = try DecodeFixtures.makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        // A square fixture cannot distinguish a transform that confuses width with height, which
        // is precisely the bug this test caught. The non-square case is the one that matters.
        let image =
            square
            ? try DecodeFixtures.quadrantImage(
                space: try DecodeFixtures.srgb(), components: DecodeFixtures.sRGBComponents)
            : try Self.quadrants(width: 24, height: 8)
        let url = try DecodeFixtures.write(
            image, to: directory.appendingPathComponent("o.png"), type: .png,
            properties: [kCGImagePropertyOrientation as String: stored] as CFDictionary)

        let ours = try Metadata.normalise(
            try Decode.decode(contentsOf: url).pixelBuffer, orientation: stored,
            colorSpace: try DecodeFixtures.srgb())

        let source = try #require(CGImageSourceCreateWithURL(url as CFURL, nil))
        let theirs = try #require(
            CGImageSourceCreateThumbnailAtIndex(
                source, 0,
                [
                    kCGImageSourceCreateThumbnailFromImageAlways: true,
                    kCGImageSourceCreateThumbnailWithTransform: true,
                    // Large enough that ImageIO never RESIZES; we are comparing
                    // orientation, and a downscale would silently change the pixels.
                    kCGImageSourceThumbnailMaxPixelSize: 64,
                ] as CFDictionary))

        #expect(CVPixelBufferGetWidth(ours) == theirs.width)
        #expect(CVPixelBufferGetHeight(ours) == theirs.height)
        // Corner probes derived from the OUTPUT size, so they stay in bounds whichever way the
        // dimensions were exchanged.
        let w = CVPixelBufferGetWidth(ours), h = CVPixelBufferGetHeight(ours)
        let theirBuf = try Self.buffer(from: theirs)
        for (x, y) in [(1, 1), (w - 2, 1), (1, h - 2), (w - 2, h - 2)] {
            #expect(
                DecodeFixtures.pixel(of: ours, x: x, y: y)
                    == DecodeFixtures.pixel(of: theirBuf, x: x, y: y),
                "orientation \(stored) disagrees with ImageIO at (\(x), \(y))")
        }
    }

    // MARK: - §3.4 metadata policy

    static func carriage(
        orientation: Int, extra: [String: Any] = [:]
    ) throws
        -> Decode.MetadataCarriage
    {
        var properties: [String: Any] = [
            kCGImagePropertyOrientation as String: orientation,
            kCGImagePropertyTIFFDictionary as String: [
                kCGImagePropertyTIFFOrientation as String: orientation,
                kCGImagePropertyTIFFMake as String: "ACME",
            ],
            kCGImagePropertyGPSDictionary as String: [
                kCGImagePropertyGPSLatitude as String: 51.5
            ],
            kCGImagePropertyExifDictionary as String: [
                kCGImagePropertyExifDateTimeOriginal as String: "2026:08:31 12:00:00"
            ],
        ]
        properties.merge(extra) { _, new in new }
        let data = try PropertyListSerialization.data(
            fromPropertyList: properties, format: .binary, options: 0)
        return Decode.MetadataCarriage(
            serializedProperties: data, droppedKeyPaths: [], orientation: orientation)
    }

    /// The bug this pins: normalising the top-level key and leaving the TIFF dictionary's copy
    /// behind means ImageIO re-rotates the file on reopen — defect 2 wearing a different hat.
    @Test("orientation is normalised to 1 in EVERY place it appears, not just the top level")
    func orientationIsNormalisedEverywhere() throws {
        let normalised = try Metadata.normalisedProperties(from: try Self.carriage(orientation: 6))

        #expect(normalised[kCGImagePropertyOrientation as String] as? Int == 1)
        let tiff = try #require(
            normalised[kCGImagePropertyTIFFDictionary as String] as? [String: Any])
        #expect(
            tiff[kCGImagePropertyTIFFOrientation as String] as? Int == 1,
            "a stale TIFF orientation re-rotates the image when the file is reopened")
    }

    /// §3.4 preserves by default, so the implementation copies the dictionary whole. A policy
    /// built from a list of keys to keep silently drops every key nobody thought of.
    @Test("GPS, camera and timestamp fields survive normalisation")
    func metadataIsPreserved() throws {
        let normalised = try Metadata.normalisedProperties(from: try Self.carriage(orientation: 8))

        let gps = try #require(normalised[kCGImagePropertyGPSDictionary as String] as? [String: Any])
        #expect(gps[kCGImagePropertyGPSLatitude as String] as? Double == 51.5)
        let tiff = try #require(
            normalised[kCGImagePropertyTIFFDictionary as String] as? [String: Any])
        #expect(tiff[kCGImagePropertyTIFFMake as String] as? String == "ACME")
        let exif = try #require(
            normalised[kCGImagePropertyExifDictionary as String] as? [String: Any])
        #expect(
            exif[kCGImagePropertyExifDateTimeOriginal as String] as? String
                == "2026:08:31 12:00:00")
    }

    /// The allow-list trap, stated directly: a key the implementation has never heard of must
    /// survive, because §3.4 says preserve.
    @Test("a key the implementation does not know about is still preserved")
    func unknownKeysSurvive() throws {
        let normalised = try Metadata.normalisedProperties(
            from: try Self.carriage(orientation: 1, extra: ["com.example.private": "keep me"]))

        #expect(normalised["com.example.private"] as? String == "keep me")
    }

    @Test("an undefined stored orientation is refused by the property path too")
    func propertiesRefuseUndefinedOrientation() throws {
        #expect(throws: Metadata.MetadataError.undefinedOrientation(stored: 11)) {
            _ = try Metadata.normalisedProperties(from: try Self.carriage(orientation: 11))
        }
    }

    /// Export must always assert orientation 1, including for a file that carried no tag at all —
    /// otherwise a reader is left to guess.
    @Test("orientation 1 is asserted even when the source carried no orientation key")
    func orientationIsAssertedWhenAbsent() throws {
        let data = try PropertyListSerialization.data(
            fromPropertyList: ["com.example.only": 1], format: .binary, options: 0)
        let carriage = Decode.MetadataCarriage(
            serializedProperties: data, droppedKeyPaths: [], orientation: 1)

        let normalised = try Metadata.normalisedProperties(from: carriage)

        #expect(normalised[kCGImagePropertyOrientation as String] as? Int == 1)
    }
}
