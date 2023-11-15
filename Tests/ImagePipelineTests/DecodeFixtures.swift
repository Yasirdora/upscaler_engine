import CoreGraphics
import CoreVideo
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// Fixtures for the decode tests, constructed rather than drawn from the corpus.
///
/// The corpus cannot exercise decode: all 45 images are <= 517 px, **zero are Display P3**, exactly
/// one has alpha, and none carries an EXIF orientation other than 1. So these tests construct their
/// own.
///
/// They are **built at test time into a temporary directory** rather than committed as binaries:
/// nothing binary enters the repository, and the expected pixel values are written out as literals
/// below, so a change in what ImageIO produces fails a test loudly instead of silently redefining
/// the baseline.
///
/// Construction uses `CGImageDestination` (encode). The code under test uses `CGImageSource`
/// (decode). No code is shared between the two, so a fixture cannot flatter the decoder.
enum DecodeFixtures {

    enum FixtureError: Error {
        case couldNotBuild(String)
    }

    // MARK: - Geometry

    static let side = 16
    static let half = 8

    /// Buffer coordinates sampled by every pixel assertion: the centre-ish of each 8x8 quadrant,
    /// 3 px clear of every quadrant boundary so JPEG and HEIC are not sampled inside their own
    /// ringing.
    static let samplePoints = [(3, 3), (11, 3), (3, 11), (11, 11)]

    /// Quadrant components for the sRGB fixtures: (0,128,255), (255,0,64), white, black.
    static let sRGBComponents: [[CGFloat]] = [
        [0, 128.0 / 255.0, 1], [1, 0, 64.0 / 255.0], [1, 1, 1], [0, 0, 0],
    ]

    /// Quadrant components for the Display P3 fixture.
    ///
    /// Chosen **inside** the P3 gamut and away from the sRGB primaries on purpose: primaries clip
    /// to the same 8-bit values in both spaces, so a fixture built from them cannot tell
    /// preservation from conversion. These four do — see `p3PreservedSamples` versus
    /// `p3ConvertedToSRGBSamples`.
    static let p3Components: [[CGFloat]] = [
        [0.75, 0.35, 0.20], [0.20, 0.70, 0.45], [0.35, 0.25, 0.85],
        [128.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0],
    ]

    // MARK: - Expected samples (written out, never recomputed)

    /// BGRA at `samplePoints` for every lossless sRGB fixture, 8-bit and 16-bit alike.
    static let sRGBSamples: [[UInt8]] = [
        [255, 255, 255, 255], [0, 0, 0, 255], [255, 128, 0, 255], [64, 0, 255, 255],
    ]

    /// BGRA at `samplePoints` when the P3 fixture is decoded **into its own space** — the stored
    /// samples, byte for byte.
    static let p3PreservedSamples: [[UInt8]] = [
        [217, 64, 89, 255], [128, 128, 128, 255], [51, 89, 191, 255], [115, 179, 51, 255],
    ]

    /// BGRA at `samplePoints` when the same file is decoded **into sRGB** — what a naive decode
    /// produces, and what this decoder must not produce.
    static let p3ConvertedToSRGBSamples: [[UInt8]] = [
        [225, 63, 94, 255], [128, 128, 128, 255], [37, 81, 206, 255], [109, 182, 0, 255],
    ]

    /// Per-format tolerance. PNG and TIFF are lossless and are held to 0; the measured worst case
    /// on this host is 7 for JPEG and 2 for HEIC, so 10 is the band with headroom.
    static let losslessTolerance = 0
    static let lossyTolerance = 10

    // MARK: - Directory

    static func makeDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("decode-fixtures-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    // MARK: - Images

    static func srgb() throws -> CGColorSpace {
        guard let space = CGColorSpace(name: CGColorSpace.sRGB) else {
            throw FixtureError.couldNotBuild("sRGB colour space")
        }
        return space
    }

    static func displayP3() throws -> CGColorSpace {
        guard let space = CGColorSpace(name: CGColorSpace.displayP3) else {
            throw FixtureError.couldNotBuild("Display P3 colour space")
        }
        return space
    }

    /// A `side` x `side` image of four quadrants filled with exact component values of `space`.
    static func quadrantImage(
        space: CGColorSpace, components: [[CGFloat]], bitsPerComponent: Int = 8
    ) throws -> CGImage {
        let info: UInt32 =
            bitsPerComponent == 16
            ? CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder16Little.rawValue
            : CGImageAlphaInfo.noneSkipLast.rawValue
        guard
            let context = CGContext(
                data: nil, width: side, height: side, bitsPerComponent: bitsPerComponent,
                bytesPerRow: 0, space: space, bitmapInfo: info)
        else {
            throw FixtureError.couldNotBuild("\(bitsPerComponent)-bit quadrant context")
        }
        for (index, values) in components.enumerated() {
            guard let color = CGColor(colorSpace: space, components: values + [1]) else {
                throw FixtureError.couldNotBuild("quadrant colour \(index)")
            }
            context.setFillColor(color)
            context.fill(
                CGRect(x: (index % 2) * half, y: (index / 2) * half, width: half, height: half))
        }
        guard let image = context.makeImage() else {
            throw FixtureError.couldNotBuild("quadrant image")
        }
        return image
    }

    /// A single-colour image of arbitrary dimensions — for tests that care about geometry and
    /// metadata rather than pixels.
    static func solidImage(space: CGColorSpace, width: Int, height: Int) throws -> CGImage {
        guard
            let context = CGContext(
                data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                space: space, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue),
            let color = CGColor(colorSpace: space, components: [0.2, 0.4, 0.6, 1])
        else {
            throw FixtureError.couldNotBuild("solid image \(width)x\(height)")
        }
        context.setFillColor(color)
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        guard let image = context.makeImage() else {
            throw FixtureError.couldNotBuild("solid image")
        }
        return image
    }

    /// An RGBA image whose left half is opaque red and whose right half is fully transparent.
    static func alphaImage() throws -> CGImage {
        guard
            let context = CGContext(
                data: nil, width: side, height: side, bitsPerComponent: 8, bytesPerRow: 0,
                space: try srgb(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else {
            throw FixtureError.couldNotBuild("alpha context")
        }
        context.clear(CGRect(x: 0, y: 0, width: side, height: side))
        context.setFillColor(red: 1, green: 0, blue: 0, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: half, height: side))
        guard let image = context.makeImage() else {
            throw FixtureError.couldNotBuild("alpha image")
        }
        return image
    }

    /// A CMYK image — §3.3's named unsupported colour model.
    static func cmykImage() throws -> CGImage {
        guard let space = CGColorSpace(name: CGColorSpace.genericCMYK),
            let context = CGContext(
                data: nil, width: side, height: side, bitsPerComponent: 8, bytesPerRow: 0,
                space: space, bitmapInfo: CGImageAlphaInfo.none.rawValue),
            let color = CGColor(colorSpace: space, components: [0, 1, 1, 0, 1])
        else {
            throw FixtureError.couldNotBuild("CMYK context")
        }
        context.setFillColor(color)
        context.fill(CGRect(x: 0, y: 0, width: side, height: side))
        guard let image = context.makeImage() else {
            throw FixtureError.couldNotBuild("CMYK image")
        }
        return image
    }

    /// A 32-bit float, extended-range image whose components are far outside [0, 1] — §3.2's
    /// tolerated-not-promised HDR tier, and the input-normalisation guard's test case.
    static func extendedRangeFloatImage() throws -> CGImage {
        let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue)
            .union(.floatComponents).union(.byteOrder32Little)
        guard let space = CGColorSpace(name: CGColorSpace.extendedLinearSRGB),
            let context = CGContext(
                data: nil, width: side, height: side, bitsPerComponent: 32, bytesPerRow: 0,
                space: space, bitmapInfo: info.rawValue),
            let color = CGColor(colorSpace: space, components: [8.0, 4.0, 2.0, 1.0])
        else {
            throw FixtureError.couldNotBuild("extended-range float context")
        }
        context.setFillColor(color)
        context.fill(CGRect(x: 0, y: 0, width: side, height: side))
        guard let image = context.makeImage() else {
            throw FixtureError.couldNotBuild("extended-range float image")
        }
        return image
    }

    // MARK: - Writing

    @discardableResult
    static func write(
        _ image: CGImage, to url: URL, type: UTType, properties: CFDictionary? = nil
    ) throws -> URL {
        guard
            let destination = CGImageDestinationCreateWithURL(
                url as CFURL, type.identifier as CFString, 1, nil)
        else {
            throw FixtureError.couldNotBuild("destination for \(url.lastPathComponent)")
        }
        CGImageDestinationAddImage(destination, image, properties)
        guard CGImageDestinationFinalize(destination) else {
            throw FixtureError.couldNotBuild("finalize \(url.lastPathComponent)")
        }
        return url
    }

    /// A two-frame animated container. `delayKeys` selects GIF or APNG framing.
    static func writeAnimated(
        to url: URL, type: UTType, container: CFString, delay: CFString
    )
        throws -> URL
    {
        let image = try quadrantImage(space: try srgb(), components: sRGBComponents)
        guard
            let destination = CGImageDestinationCreateWithURL(
                url as CFURL, type.identifier as CFString, 2, nil)
        else {
            throw FixtureError.couldNotBuild("animated destination")
        }
        let frame = [container: [delay: 0.1]] as CFDictionary
        CGImageDestinationAddImage(destination, image, frame)
        CGImageDestinationAddImage(destination, image, frame)
        guard CGImageDestinationFinalize(destination) else {
            throw FixtureError.couldNotBuild("finalize animated")
        }
        return url
    }

    /// Copy the first `fraction` of a file's bytes — a file that ends before its image does.
    /// A PNG large enough that ImageIO returns a *partial* image when it is cut, rather than
    /// failing outright. Noise, because a flat fill compresses to a few hundred bytes and lands in
    /// the small-file case that already fails cleanly.
    static func noisyPNG(side: Int, to url: URL) throws -> URL {
        let space = try srgb()
        guard
            let context = CGContext(
                data: nil, width: side, height: side, bitsPerComponent: 8, bytesPerRow: 0,
                space: space, bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)
        else { throw FixtureError.couldNotBuild("noise context") }
        var generator = SystemRandomNumberGenerator()
        for y in 0..<side {
            for x in 0..<side {
                context.setFillColor(
                    red: CGFloat.random(in: 0...1, using: &generator),
                    green: CGFloat.random(in: 0...1, using: &generator),
                    blue: CGFloat.random(in: 0...1, using: &generator), alpha: 1)
                context.fill(CGRect(x: x, y: y, width: 1, height: 1))
            }
        }
        guard let image = context.makeImage() else {
            throw FixtureError.couldNotBuild("noise image")
        }
        return try write(image, to: url, type: .png)
    }

    static func truncate(_ url: URL, to fraction: Double, as name: String) throws -> URL {
        let bytes = try Data(contentsOf: url)
        let cut = bytes.prefix(Int(Double(bytes.count) * fraction))
        let out = url.deletingLastPathComponent().appendingPathComponent(name)
        try cut.write(to: out)
        return out
    }

    // MARK: - Reading a decoded buffer

    /// BGRA bytes at `samplePoints`, indexed through `CVPixelBufferGetBytesPerRow`.
    ///
    /// The stride is load-bearing: a 16-px-wide 32BGRA IOSurface is padded to 64 bytes per row on
    /// this host, and flat `(y * width + x) * 4` indexing silently reads padding instead of
    /// pixels.
    static func samples(of buffer: CVPixelBuffer) -> [[UInt8]] {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return [] }
        let bytes = base.assumingMemoryBound(to: UInt8.self)
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        return samplePoints.map { point in
            let offset = point.1 * stride + point.0 * 4
            return [bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]]
        }
    }

    /// One BGRA pixel at an arbitrary coordinate.
    static func pixel(of buffer: CVPixelBuffer, x: Int, y: Int) -> [UInt8] {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return [] }
        let bytes = base.assumingMemoryBound(to: UInt8.self)
        let offset = y * CVPixelBufferGetBytesPerRow(buffer) + x * 4
        return [bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]]
    }

    /// The largest absolute channel difference between two sample sets.
    static func maximumDeviation(_ lhs: [[UInt8]], _ rhs: [[UInt8]]) -> Int {
        guard lhs.count == rhs.count else { return Int.max }
        var worst = 0
        for (left, right) in zip(lhs, rhs) {
            guard left.count == right.count else { return Int.max }
            for (a, b) in zip(left, right) { worst = max(worst, abs(Int(a) - Int(b))) }
        }
        return worst
    }
}
