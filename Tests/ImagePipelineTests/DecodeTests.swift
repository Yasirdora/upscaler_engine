import CoreGraphics
import CoreVideo
import Foundation
import ImageIO
import Testing
import UniformTypeIdentifiers

@testable import ImagePipeline

/// Tests for decode to the engine contract, carrying colour and metadata.
///
/// Model-free by design: decode is ImageIO and Core Graphics, so these drive it directly with
/// constructed files and compare against **written-out literals** in `DecodeFixtures`. No expected
/// value is produced by the code under test.
///
/// Four failure modes motivate the cases: colour (wide gamut flattened to sRGB), orientation,
/// metadata loss and alpha. Decode must make colour and metadata preservable by the later stages;
/// orientation is pinned here as *recorded, not applied*, so `Metadata` has a baseline to move.
struct DecodeTests {

    // MARK: - §3.1 — the four supported formats reach the engine contract

    @Test(
        "each supported format decodes into the 32BGRA IOSurface engine contract",
        arguments: [
            ("png", UTType.png, "public.png", DecodeFixtures.losslessTolerance),
            ("tif", UTType.tiff, "public.tiff", DecodeFixtures.losslessTolerance),
            ("jpg", UTType.jpeg, "public.jpeg", DecodeFixtures.lossyTolerance),
            ("heic", UTType.heic, "public.heic", DecodeFixtures.lossyTolerance),
        ])
    func supportedFormatsDecodeToTheEngineContract(
        suffix: String, type: UTType, identifier: String, tolerance: Int
    ) throws {
        let directory = try DecodeFixtures.makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let image = try DecodeFixtures.quadrantImage(
            space: try DecodeFixtures.srgb(), components: DecodeFixtures.sRGBComponents)
        let url = try DecodeFixtures.write(
            image, to: directory.appendingPathComponent("fixture.\(suffix)"), type: type)

        let decoded = try Decode.decode(contentsOf: url)

        #expect(decoded.width == 16)
        #expect(decoded.height == 16)
        #expect(CVPixelBufferGetPixelFormatType(decoded.pixelBuffer) == kCVPixelFormatType_32BGRA)
        #expect(CVPixelBufferGetIOSurface(decoded.pixelBuffer) != nil)
        #expect(decoded.source.formatIdentifier == identifier)
        #expect(decoded.source.verdict == .supported)
        #expect(decoded.source.frameCount == 1)
        #expect(decoded.source.hasAlpha == false)

        let samples = DecodeFixtures.samples(of: decoded.pixelBuffer)
        #expect(samples.count == 4)
        #expect(
            DecodeFixtures.maximumDeviation(samples, DecodeFixtures.sRGBSamples) <= tolerance,
            "\(suffix) samples \(samples) exceed tolerance \(tolerance) against \(DecodeFixtures.sRGBSamples)"
        )
    }

    // MARK: - §8.1 — 8-bit and 16-bit decode correctness

    @Test(
        "8-bit and 16-bit sources decode to the same written-out samples",
        arguments: [
            ("png", UTType.png, 8), ("png", UTType.png, 16),
            ("tif", UTType.tiff, 8), ("tif", UTType.tiff, 16),
        ])
    func bitDepthsDecodeCorrectly(suffix: String, type: UTType, depth: Int) throws {
        let directory = try DecodeFixtures.makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let image = try DecodeFixtures.quadrantImage(
            space: try DecodeFixtures.srgb(), components: DecodeFixtures.sRGBComponents,
            bitsPerComponent: depth)
        let url = try DecodeFixtures.write(
            image, to: directory.appendingPathComponent("d\(depth).\(suffix)"), type: type)

        let decoded = try Decode.decode(contentsOf: url)

        // The depth the file stored is recorded; the buffer is always 8-bit, which is the
        // contract. A 16-bit sample of 128*257 must land on exactly 128, not 127 or 129.
        #expect(decoded.source.bitsPerComponent == depth)
        #expect(DecodeFixtures.samples(of: decoded.pixelBuffer) == DecodeFixtures.sRGBSamples)
    }

    // MARK: - §3.1 / §8.1 — Display P3 arrives with its profile identifiable

    @Test("a Display P3 source arrives with its profile identifiable and its samples unconverted")
    func displayP3ProfileAndSamplesSurviveDecode() throws {
        let directory = try DecodeFixtures.makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let image = try DecodeFixtures.quadrantImage(
            space: try DecodeFixtures.displayP3(), components: DecodeFixtures.p3Components)
        let url = try DecodeFixtures.write(
            image, to: directory.appendingPathComponent("p3.png"), type: .png)

        let decoded = try Decode.decode(contentsOf: url)

        // Identifiable: the profile travels as bytes and rebuilds into Display P3 by name.
        #expect(decoded.color.sourceProfileName == "Display P3")
        let profile = try #require(decoded.color.sourceProfile)
        #expect(profile.isEmpty == false)
        let rebuilt = try #require(decoded.color.sourceColorSpace())
        #expect(rebuilt.name as String? == (CGColorSpace.displayP3 as String))
        #expect(rebuilt.isWideGamutRGB)

        // Unconverted: the buffer holds the stored P3 samples, not their sRGB images. The second
        // expectation is the one that fails if decode ever draws through sRGB again.
        #expect(decoded.color.disposition == .preservedSourceSpace)
        #expect(decoded.color.sourceProfile == decoded.color.bufferProfile)
        let samples = DecodeFixtures.samples(of: decoded.pixelBuffer)
        #expect(samples == DecodeFixtures.p3PreservedSamples)
        #expect(samples != DecodeFixtures.p3ConvertedToSRGBSamples)
    }

    @Test("an sRGB source keeps its own profile rather than having one re-asserted")
    func sRGBProfileIsPreservedNotReasserted() throws {
        let directory = try DecodeFixtures.makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let image = try DecodeFixtures.quadrantImage(
            space: try DecodeFixtures.srgb(), components: DecodeFixtures.sRGBComponents)
        let url = try DecodeFixtures.write(
            image, to: directory.appendingPathComponent("s.png"), type: .png)

        let decoded = try Decode.decode(contentsOf: url)

        #expect(decoded.color.sourceProfileName == "sRGB IEC61966-2.1")
        #expect(decoded.color.disposition == .preservedSourceSpace)
        #expect(decoded.color.sourceProfile == decoded.color.bufferProfile)
    }

    // MARK: - §3.4 — EXIF, GPS and copyright travel out of decode

    @Test("EXIF, GPS and copyright are carried forward, and orientation is recorded not applied")
    func metadataIsCarriedForward() throws {
        let directory = try DecodeFixtures.makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        // Deliberately non-square: orientation 6 means "rotate 90°". If decode applied it, the
        // decoded geometry would be 8x24 instead of 24x8.
        let image = try DecodeFixtures.solidImage(
            space: try DecodeFixtures.srgb(), width: 24, height: 8)
        let properties: [CFString: Any] = [
            kCGImagePropertyTIFFDictionary: [
                kCGImagePropertyTIFFMake: "ExampleCam",
                kCGImagePropertyTIFFCopyright: "(c) 2026 Example",
            ],
            kCGImagePropertyExifDictionary: [
                kCGImagePropertyExifDateTimeOriginal: "2026:08:31 09:00:00"
            ],
            kCGImagePropertyGPSDictionary: [
                kCGImagePropertyGPSLatitude: 52.3676,
                kCGImagePropertyGPSLatitudeRef: "N",
            ],
            kCGImagePropertyOrientation: 6,
        ]
        let url = try DecodeFixtures.write(
            image, to: directory.appendingPathComponent("meta.jpg"), type: .jpeg,
            properties: properties as CFDictionary)

        let decoded = try Decode.decode(contentsOf: url)

        #expect(decoded.width == 24)
        #expect(decoded.height == 8)
        #expect(decoded.metadata.orientation == 6)
        #expect(decoded.metadata.droppedKeyPaths.isEmpty)

        let recovered = try decoded.metadata.properties()
        let tiff = try #require(recovered["{TIFF}"] as? [String: Any])
        #expect(tiff["Make"] as? String == "ExampleCam")
        #expect(tiff["Copyright"] as? String == "(c) 2026 Example")
        let exif = try #require(recovered["{Exif}"] as? [String: Any])
        #expect(exif["DateTimeOriginal"] as? String == "2026:08:31 09:00:00")
        let gps = try #require(recovered["{GPS}"] as? [String: Any])
        #expect(gps["LatitudeRef"] as? String == "N")
        #expect(gps["Latitude"] as? Double == 52.3676)
    }

    @Test("a file with no EXIF orientation records 1 rather than nothing")
    func absentOrientationRecordsOne() throws {
        let directory = try DecodeFixtures.makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let image = try DecodeFixtures.quadrantImage(
            space: try DecodeFixtures.srgb(), components: DecodeFixtures.sRGBComponents)
        let url = try DecodeFixtures.write(
            image, to: directory.appendingPathComponent("plain.png"), type: .png)

        #expect(try Decode.decode(contentsOf: url).metadata.orientation == 1)
    }

    // MARK: - §3.1 alpha — recorded here, carried by `Alpha`

    @Test("alpha is recorded, and the ground a transparent pixel lands on is deterministic")
    func alphaIsRecordedAndTheGroundIsDeterministic() throws {
        let directory = try DecodeFixtures.makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = try DecodeFixtures.write(
            try DecodeFixtures.alphaImage(),
            to: directory.appendingPathComponent("alpha.png"), type: .png)

        let decoded = try Decode.decode(contentsOf: url)

        #expect(decoded.source.hasAlpha)
        // Left half opaque red, BGRA — `premultipliedFirst` leaves every opaque pixel
        // bit-identical.
        #expect(DecodeFixtures.pixel(of: decoded.pixelBuffer, x: 3, y: 3) == [0, 0, 255, 255])
        // Right half fully transparent. The buffer is `premultipliedFirst`, so alpha survives
        // decode and a transparent pixel is transparent — not composited onto black and thrown
        // away.
        #expect(DecodeFixtures.pixel(of: decoded.pixelBuffer, x: 12, y: 3) == [0, 0, 0, 0])
    }

    // MARK: - §3.3 — unsupported inputs produce clear, named, pre-processing errors

    @Test(
        "a truncated file fails cleanly with a named error",
        arguments: [
            ("png", UTType.png), ("jpg", UTType.jpeg), ("tif", UTType.tiff), ("heic", UTType.heic),
        ])
    func truncatedFilesFailCleanly(suffix: String, type: UTType) throws {
        let directory = try DecodeFixtures.makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let image = try DecodeFixtures.quadrantImage(
            space: try DecodeFixtures.srgb(), components: DecodeFixtures.sRGBComponents)
        let whole = try DecodeFixtures.write(
            image, to: directory.appendingPathComponent("whole.\(suffix)"), type: type)
        let cut = try DecodeFixtures.truncate(whole, to: 0.5, as: "cut.\(suffix)")

        #expect(throws: Decode.DecodeError.self) { try Decode.decode(contentsOf: cut) }
        // Whichever way ImageIO reports the damage, the user-facing text names the limitation.
        do {
            _ = try Decode.decode(contentsOf: cut)
            Issue.record("a truncated \(suffix) decoded successfully")
        } catch let error as Decode.DecodeError {
            #expect(String(describing: error).contains("corrupt or truncated"))
        }
    }

    @Test("a file that is not an image at all is refused as unrecognised")
    func garbageIsRefused() throws {
        let directory = try DecodeFixtures.makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("garbage.png")
        try Data([0, 1, 2, 3, 4, 5, 6, 7]).write(to: url)

        #expect(throws: Decode.DecodeError.unrecognisedFormat(origin: url.path)) {
            try Decode.decode(contentsOf: url)
        }
    }

    @Test("an empty file is refused, not decoded to nothing")
    func emptyFileIsRefused() throws {
        let directory = try DecodeFixtures.makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("empty.png")
        try Data().write(to: url)

        #expect(throws: Decode.DecodeError.unrecognisedFormat(origin: url.path)) {
            try Decode.decode(contentsOf: url)
        }
    }

    @Test("a missing file is refused before anything else happens")
    func missingFileIsRefused() throws {
        let directory = try DecodeFixtures.makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("absent.png")

        #expect(throws: Decode.DecodeError.unreadable(origin: url.path)) {
            try Decode.decode(contentsOf: url)
        }
    }

    /// An animated GIF is refused for being a GIF, not for being animated. §3.1 does not support
    /// the container at any frame count, so the refusal that holds regardless of content is the one
    /// the identifier is entitled to make.
    @Test("an animated GIF is refused as an unsupported format, not as animated")
    func animatedGIFIsRefused() throws {
        let directory = try DecodeFixtures.makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = try DecodeFixtures.writeAnimated(
            to: directory.appendingPathComponent("anim.gif"), type: .gif,
            container: kCGImagePropertyGIFDictionary, delay: kCGImagePropertyGIFDelayTime)

        #expect(
            throws: Decode.DecodeError.formatNotSupported(format: "com.compuserve.gif")
        ) { try Decode.decode(contentsOf: url) }
    }

    @Test("an APNG is refused as animated even though PNG is a supported container")
    func animatedPNGIsRefused() throws {
        let directory = try DecodeFixtures.makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = try DecodeFixtures.writeAnimated(
            to: directory.appendingPathComponent("anim.png"), type: .png,
            container: kCGImagePropertyPNGDictionary, delay: kCGImagePropertyAPNGDelayTime)

        #expect(throws: Decode.DecodeError.animatedNotSupported(format: "APNG")) {
            try Decode.decode(contentsOf: url)
        }
    }

    // MARK: - §3.3 — container trailers (the truncation ImageIO cannot see)

    /// The defect this check exists for, asserted as the defect rather than as the fix.
    ///
    /// ImageIO reports `statusComplete` for this file and hands back a full-size `CGImage` whose
    /// missing rows are simply blank. This test pins BOTH halves: that ImageIO really does accept
    /// it, and that decode nevertheless refuses it. If a future ImageIO starts rejecting the file
    /// itself, the first expectation fails and tells us the ground moved.
    @Test("a large truncated PNG is refused, though ImageIO decodes it happily")
    func largeTruncatedPNGIsRefused() throws {
        let directory = try DecodeFixtures.makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let whole = try DecodeFixtures.noisyPNG(
            side: 256, to: directory.appendingPathComponent("noise.png"))
        let cut = try DecodeFixtures.truncate(whole, to: 0.1, as: "cut.png")

        let source = CGImageSourceCreateWithURL(cut as CFURL, nil)
        #expect(
            source.map { CGImageSourceCreateImageAtIndex($0, 0, nil) != nil } == true,
            "premise: ImageIO still produces an image from the truncated file")

        #expect(
            throws: Decode.DecodeError.corruptOrTruncated(
                format: "public.png", origin: cut.path)
        ) { try Decode.decode(contentsOf: cut) }
    }

    @Test("an intact PNG passes the trailer check")
    func intactPNGSurvivesTheTrailerCheck() throws {
        let directory = try DecodeFixtures.makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let whole = try DecodeFixtures.noisyPNG(
            side: 64, to: directory.appendingPathComponent("whole.png"))

        let decoded = try Decode.decode(contentsOf: whole)

        #expect(decoded.width == 64)
    }

    /// The tolerant form is deliberate: appended data after `EOI` is common in the wild, and
    /// refusing a valid photograph is worse than missing a truncated one.
    @Test("a JPEG with bytes appended after EOI is still accepted")
    func jpegWithTrailingBytesIsAccepted() throws {
        let directory = try DecodeFixtures.makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("padded.jpg")
        _ = try DecodeFixtures.write(
            try DecodeFixtures.solidImage(space: try DecodeFixtures.srgb(), width: 32, height: 32),
            to: url, type: .jpeg)
        var bytes = try Data(contentsOf: url)
        bytes.append(contentsOf: [0x00, 0x00, 0x00, 0x00])
        try bytes.write(to: url)

        #expect(try Decode.decode(contentsOf: url).width == 32)
    }

    /// The tail travels with the bytes, so both entry points enforce the same rule.
    @Test("the trailer check applies to in-memory data too")
    func truncatedDataIsRefused() throws {
        let directory = try DecodeFixtures.makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let whole = try DecodeFixtures.noisyPNG(
            side: 256, to: directory.appendingPathComponent("noise.png"))
        let cut = try Data(contentsOf: whole).prefix(4096)

        #expect(
            throws: Decode.DecodeError.corruptOrTruncated(
                format: "public.png", origin: Decode.dataOrigin)
        ) { try Decode.decode(cut) }
    }

    /// TIFF and HEIF have no terminating marker, so the check must decline to judge them rather
    /// than invent a rule. They are covered by image creation failing, which was measured.
    @Test("a format with no terminator is not refused by the trailer check")
    func formatsWithoutTerminatorsAreNotJudged() throws {
        #expect(throws: Never.self) {
            try Decode.validateTrailer(
                Data([0x00, 0x01, 0x02]), identifier: "public.tiff", origin: "t.tif")
        }
        #expect(throws: Never.self) {
            try Decode.validateTrailer(
                Data([0x00]), identifier: "public.heic", origin: "h.heic")
        }
    }

    /// An unobtainable tail must mean "cannot check", never "invalid" — otherwise an unreadable
    /// file would be reported as a corrupt one, which names the wrong limitation.
    @Test("an absent tail is not treated as a missing terminator")
    func absentTailIsNotAFailure() throws {
        #expect(throws: Never.self) {
            try Decode.validateTrailer(nil, identifier: "public.png", origin: "p.png")
        }
        #expect(throws: Never.self) {
            try Decode.validateTrailer(Data(), identifier: "public.png", origin: "p.png")
        }
    }

    @Test("a CMYK JPEG is refused by colour model, not silently converted")
    func cmykIsRefused() throws {
        let directory = try DecodeFixtures.makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = try DecodeFixtures.write(
            try DecodeFixtures.cmykImage(), to: directory.appendingPathComponent("cmyk.jpg"),
            type: .jpeg)

        #expect(
            throws: Decode.DecodeError.colorModelNotSupported(
                model: "CMYK", format: "public.jpeg")
        ) { try Decode.decode(contentsOf: url) }
    }

    @Test("an image beyond the feasibility limit is refused before any pixel is decoded")
    func oversizedImageIsRefused() throws {
        let directory = try DecodeFixtures.makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let image = try DecodeFixtures.quadrantImage(
            space: try DecodeFixtures.srgb(), components: DecodeFixtures.sRGBComponents)
        let url = try DecodeFixtures.write(
            image, to: directory.appendingPathComponent("big.png"), type: .png)

        // 16x16 = 256 pixels against a limit of 4.
        #expect(
            throws: Decode.DecodeError.exceedsMemoryFeasibility(pixels: 256, limit: 4)
        ) { try Decode.decode(contentsOf: url, limits: Decode.Limits(maximumPixelCount: 4)) }
    }

    @Test("every §3.3 refusal is reachable from the header alone, with no image decoded")
    func refusalsArePreProcessing() throws {
        let directory = try DecodeFixtures.makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let image = try DecodeFixtures.quadrantImage(
            space: try DecodeFixtures.srgb(), components: DecodeFixtures.sRGBComponents)
        let url = try DecodeFixtures.write(
            image, to: directory.appendingPathComponent("pre.png"), type: .png)
        let source = try #require(CGImageSourceCreateWithURL(url as CFURL, nil))

        // `preflight` cannot create a CGImage — it is handed a CGImageSource and returns header
        // facts — so a refusal it makes is provably pre-processing (§10: never mid-job). This is
        // the structural half of the claim; `oversizedImageIsRefused` is the end-to-end half.
        #expect(
            throws: Decode.DecodeError.exceedsMemoryFeasibility(pixels: 256, limit: 4)
        ) {
            try Decode.preflight(
                source: source, origin: url.path, limits: Decode.Limits(maximumPixelCount: 4))
        }

        // The same header, under a limit that admits it, yields the facts decode goes on to use.
        let (header, _) = try Decode.preflight(
            source: source, origin: url.path, limits: Decode.Limits(maximumPixelCount: 256))
        #expect(
            header
                == Decode.Preflight(
                    identifier: "public.png", verdict: .supported, frameCount: 1,
                    width: 16, height: 16))
    }

    @Test("the transparent-ground clear zeroes the buffer, and only when there is alpha")
    func clearIfNeededZeroesOnlyWhenThereIsAlpha() throws {
        // Driven directly rather than through `decode`, on purpose. End-to-end, a freshly
        // allocated buffer is often already zero, so an end-to-end assertion would pass by
        // allocator luck even with the clear deleted. Dirtying the buffer by hand is the only way
        // to make the guard's absence observable.
        func dirtied() throws -> CVPixelBuffer {
            let buffer = try Decode.makeBuffer(width: 16, height: 16)
            guard let base = CVPixelBufferGetBaseAddress(buffer) else {
                Issue.record("no base address")
                return buffer
            }
            memset(base, 0xAB, CVPixelBufferGetBytesPerRow(buffer) * 16)
            CVPixelBufferUnlockBaseAddress(buffer, [])
            return buffer
        }

        let cleared = try dirtied()
        CVPixelBufferLockBaseAddress(cleared, [])
        Decode.clearIfNeeded(cleared, hasAlpha: true)
        CVPixelBufferUnlockBaseAddress(cleared, [])
        #expect(DecodeFixtures.pixel(of: cleared, x: 7, y: 9) == [0, 0, 0, 0])

        let untouched = try dirtied()
        CVPixelBufferLockBaseAddress(untouched, [])
        Decode.clearIfNeeded(untouched, hasAlpha: false)
        CVPixelBufferUnlockBaseAddress(untouched, [])
        #expect(DecodeFixtures.pixel(of: untouched, x: 7, y: 9) == [0xAB, 0xAB, 0xAB, 0xAB])
    }

    @Test("every named refusal explains itself in words a user could act on")
    func refusalsAreUserComprehensible() {
        let cases: [(Decode.DecodeError, String)] = [
            (.animatedNotSupported(format: "APNG"), "animated images are not supported"),
            (.videoNotSupported(format: "public.mpeg-4"), "video is not supported"),
            (.formatNotSupported(format: "org.webmproject.webp"), "not a supported image format"),
            (
                .colorModelNotSupported(model: "CMYK", format: "public.jpeg"),
                "CMYK colour is not supported"
            ),
            (.exceedsMemoryFeasibility(pixels: 999, limit: 8), "too large to decode"),
        ]
        for (error, fragment) in cases {
            #expect(
                String(describing: error).contains(fragment),
                "\(error) does not name its limitation")
        }
    }

    // MARK: - §3.1 / §3.2 / §3.3 classification

    @Test(
        "the input contract classifies container types the way §3 says it should",
        arguments: [
            ("public.jpeg", Decode.FormatVerdict.supported),
            ("public.png", .supported),
            ("public.tiff", .supported),
            ("public.heic", .supported),
            ("public.heif", .supported),
            ("com.adobe.raw-image", .tolerated),
            ("public.camera-raw-image", .tolerated),
            ("com.compuserve.gif", .unsupported("com.compuserve.gif")),
            ("public.heics", .animated("public.heics")),
            ("public.mpeg-4", .video("public.mpeg-4")),
            ("com.apple.quicktime-movie", .video("com.apple.quicktime-movie")),
            ("org.webmproject.webp", .unsupported("org.webmproject.webp")),
            ("com.adobe.photoshop-image", .unsupported("com.adobe.photoshop-image")),
            ("not.a.real.type", .unsupported("not.a.real.type")),
        ])
    func classificationFollowsTheInputContract(
        identifier: String, expected: Decode.FormatVerdict
    ) {
        #expect(Decode.classify(formatIdentifier: identifier) == expected)
    }

    // MARK: - §3.2 — tolerated but not promised: no crash, values normalised

    @Test("a 32-bit float extended-range TIFF normalises into the 8-bit contract without crashing")
    func extendedRangeFloatInputIsNormalised() throws {
        let directory = try DecodeFixtures.makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = try DecodeFixtures.write(
            try DecodeFixtures.extendedRangeFloatImage(),
            to: directory.appendingPathComponent("hdr.tif"), type: .tiff)

        let decoded = try Decode.decode(contentsOf: url)

        #expect(decoded.source.bitsPerComponent == 32)
        #expect(decoded.source.isFloatingPoint)
        // The source space cannot back an 8-bit bitmap, so decode falls back and records it.
        #expect(decoded.color.disposition == .normalisedToSRGB)
        // The fp16-overflow guard, stated as a value: components of 8.0, 4.0 and 2.0 — far beyond
        // the [0, 1] the engine assumes — arrive clamped at 255, so nothing that reaches the fp16
        // model input can exceed 255.
        #expect(DecodeFixtures.pixel(of: decoded.pixelBuffer, x: 3, y: 3) == [255, 255, 255, 255])
    }

    // MARK: - Feasibility limit derivation

    @Test("the automatic feasibility limit admits §7.1 workload C on the §6 device floor")
    func automaticLimitAdmitsWorkloadCOnTheDeviceFloor() {
        // §6 device floor: A14 / 4 GB. 4 GiB / 8 / 4 bytes-per-pixel = 134,217,728 pixels.
        let floor = Decode.Limits.derived(physicalMemory: 4 * 1024 * 1024 * 1024)
        #expect(floor.maximumPixelCount == 134_217_728)
        // §7.1 workload C is 48 MP and must not be refused by this guard.
        #expect(floor.maximumPixelCount > 48_000_000)
        #expect(Decode.Limits.automatic.maximumPixelCount > 48_000_000)
    }

    // MARK: - The in-memory entry point

    @Test("decoding from data reaches the same contract as decoding from a URL")
    func dataAndURLEntryPointsAgree() throws {
        let directory = try DecodeFixtures.makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let image = try DecodeFixtures.quadrantImage(
            space: try DecodeFixtures.srgb(), components: DecodeFixtures.sRGBComponents)
        let url = try DecodeFixtures.write(
            image, to: directory.appendingPathComponent("both.png"), type: .png)

        let fromData = try Decode.decode(try Data(contentsOf: url))

        #expect(fromData.width == 16)
        #expect(DecodeFixtures.samples(of: fromData.pixelBuffer) == DecodeFixtures.sRGBSamples)
        // Empty bytes are bytes that are not a container, which is the same condition — and so
        // the same error — as the empty file in `emptyFileIsRefused`, not an access failure.
        #expect(throws: Decode.DecodeError.unrecognisedFormat(origin: "<in-memory data>")) {
            try Decode.decode(Data())
        }
    }
}
