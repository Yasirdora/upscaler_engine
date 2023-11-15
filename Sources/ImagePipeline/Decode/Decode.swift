import CoreGraphics
import CoreVideo
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// Decode — image decoding (JPEG, PNG, HEIF, TIFF) to the engine's input contract.
///
/// The engine contract is a 32BGRA, IOSurface-backed `CVPixelBuffer` — the same buffer shape
/// `CoreMLBackend` and `SequentialTilingEngine` refuse to run without.
///
/// **What decode does and does not do.** It decodes, and it *carries* colour and metadata forward
/// on `DecodedImage` so the later stages can preserve them. It does **not** convert colour, apply
/// EXIF orientation, recombine alpha, or encode — those are `Color`, `Metadata`, `Alpha` and
/// `Encode`. Colour *conversion* through Core Graphics is already exact (a null test measures a
/// maximum delta of 0); what has to be protected is that wide gamut is *preserved*. So decode's job
/// is to hand the profile on intact, never to rebuild a conversion that is not broken.
///
/// Boundary: no UI framework imports; the engine must run headless. Interfaces follow the
/// certification spec (`docs/CERTIFICATION.md`); a change to one is a reviewed contract change,
/// never a silent edit.
public enum Decode {

    // MARK: - Format classification (certification spec §3.1 / §3.2 / §3.3)

    /// What the input contract says about a container type, and why.
    ///
    /// The associated strings are the *format identifier as the file actually presented it*, so a
    /// refusal can name the limitation rather than say "unsupported" (§3.3, §10).
    public enum FormatVerdict: Sendable, Equatable {
        /// §3.1 — certified in v1: JPEG, PNG, HEIF, TIFF.
        case supported
        /// §3.2 — accepted without crashing, **no quality promise made**: ProRAW and other camera
        /// raw. The engine certifies SDR only; this tier exists so a raw file does not fail a job,
        /// not so it looks good.
        case tolerated
        /// §3.3 — animated containers.
        case animated(String)
        /// §3.3 — video containers.
        case video(String)
        /// §3.3 / §10 — an image container outside both v1 tiers (WebP, PSD, BMP, …).
        case unsupported(String)
    }

    /// Container types that are animation carriers **by definition** — `heics`/`avis` are literally
    /// image *sequences*.
    ///
    /// Formats that merely *can* animate are not here, because this function sees an identifier and
    /// never a file: claiming "animated" from the identifier alone would be a claim about content
    /// that has not been read. PNG is the worked example — a supported §3.1 container whose
    /// animated variant is caught per file by its APNG delay times, in `decode`. WebP is the same
    /// shape of thing and gets the same treatment: it is refused as an unsupported format, which is
    /// the limitation that is actually true of it under §3.1.
    ///
    /// GIF is deliberately **not** here, by the same rule as WebP. `public.heics` and `public.avis`
    /// are sequence types *by definition*: a single-frame HEICS is a contradiction at the type
    /// level, so naming them animated asserts nothing the identifier cannot know. A single-frame
    /// GIF is ordinary, so the same label would be a content claim made without opening the file.
    /// GIF is `.unsupported` under §3.1 whatever its frame count, which is the stronger statement
    /// and the one that is always true.
    static let animatedIdentifiers: Set<String> = [
        "public.heics", "public.avis",
    ]

    /// Classify a container type against the §3 input contract. Pure, so it is testable without a
    /// file on disk — which matters, because the §3.2 raw tier has no fixture that can be built
    /// synthetically.
    public static func classify(formatIdentifier: String) -> FormatVerdict {
        guard let type = UTType(formatIdentifier) else { return .unsupported(formatIdentifier) }
        if type.conforms(to: .movie) || type.conforms(to: .video)
            || type.conforms(to: .audiovisualContent)
        {
            return .video(formatIdentifier)
        }
        if animatedIdentifiers.contains(formatIdentifier) { return .animated(formatIdentifier) }
        // Raw is checked before the §3.1 families deliberately: several vendor raw types are
        // TIFF-derived and would otherwise be promoted into "supported", which would attach a
        // quality promise §3.2 explicitly withholds.
        if type.conforms(to: .rawImage) { return .tolerated }
        if type.conforms(to: .jpeg) || type.conforms(to: .png) || type.conforms(to: .tiff) {
            return .supported
        }
        if let heif = UTType("public.heif-standard"), type.conforms(to: heif) { return .supported }
        return .unsupported(formatIdentifier)
    }

    // MARK: - Errors (§3.3, §10 — never silent, never mid-job)

    /// Typed decode failures, in the house style of `InferenceError`.
    ///
    /// Every case names the limitation it hit. §10 requires a clear *pre-processing* error for
    /// unsupported input, so the classification, colour-model and feasibility checks all run
    /// before a single pixel is decoded.
    public enum DecodeError: Error, Equatable {
        case unreadable(origin: String)
        case unrecognisedFormat(origin: String)
        case corruptOrTruncated(format: String, origin: String)
        case animatedNotSupported(format: String)
        case videoNotSupported(format: String)
        case formatNotSupported(format: String)
        case colorModelNotSupported(model: String, format: String)
        case exceedsMemoryFeasibility(pixels: Int, limit: Int)
        case bufferAllocationFailed(width: Int, height: Int, status: Int32)
        case drawingContextUnavailable(width: Int, height: Int)
    }

    // MARK: - Feasibility limits (§3.3 "files exceeding device-memory feasibility")

    /// The pre-processing feasibility guard.
    ///
    /// This is **not** a certification threshold and is not subject to the §4 ratchet rule — it is
    /// the point at which decode refuses up front rather than dying mid-job (§10). The automatic
    /// default is derived, not invented: the decoded buffer is 4 bytes per pixel, and decode
    /// declines to claim more than an eighth of physical memory for it. The binding constraint the
    /// divisor is chosen against is that §7.1 workload C (48 MP) must still be admitted on the §6
    /// device floor — a 4 GB A14 yields 134 Mpx, so workload C passes with 2.8x headroom.
    public struct Limits: Sendable, Equatable {
        public var maximumPixelCount: Int

        public init(maximumPixelCount: Int) {
            self.maximumPixelCount = maximumPixelCount
        }

        static let bytesPerPixel = 4
        static let physicalMemoryShareDivisor = 8

        /// The derivation, exposed so it can be tested against the §6 device floor without
        /// depending on the host the tests happen to run on.
        public static func derived(physicalMemory: UInt64) -> Limits {
            let share =
                physicalMemory / UInt64(physicalMemoryShareDivisor) / UInt64(bytesPerPixel)
            return Limits(maximumPixelCount: Int(clamping: share))
        }

        public static var automatic: Limits {
            derived(physicalMemory: ProcessInfo.processInfo.physicalMemory)
        }
    }

    // MARK: - Carriage (what travels out of decode)

    /// How the buffer's pixels relate to the source's own colour space.
    public enum ColorDisposition: String, Sendable, Equatable {
        /// The buffer holds the source's samples **byte for byte** — no conversion happened, so
        /// out-of-gamut colour is intact and `sourceProfile` describes the buffer exactly.
        case preservedSourceSpace
        /// The source space cannot back an 8-bit BGRA bitmap (greyscale, extended-range and float
        /// spaces all fail `CGContext` creation — measured, not assumed), so Core Graphics
        /// normalised into sRGB on the way in. This is the §3.2 input-normalisation path.
        case normalisedToSRGB
    }

    /// Colour carried forward, never consumed.
    ///
    /// Two profiles, because they are two different facts: `sourceProfile` is what the file said,
    /// and `bufferProfile` is what the pixels in `pixelBuffer` are actually in. When `disposition`
    /// is `.preservedSourceSpace` they are the same profile. `Color` reads `sourceProfile` to
    /// decide policy; `Encode` writes it back out. Untagged input is `sourceProfile == nil` —
    /// decode **records** that and takes no position; the untagged-input policy belongs to `Color`.
    public struct ColorCarriage: Sendable, Equatable {
        /// The source's ICC profile, verbatim. `nil` when the file carried none.
        public let sourceProfile: Data?
        /// ImageIO's name for the source profile, e.g. `"Display P3"`. Identification only.
        public let sourceProfileName: String?
        /// The ICC profile the buffer's samples are actually encoded in. Never `nil`: a 32BGRA
        /// buffer is always in *some* space, and pretending otherwise is how the profile got lost.
        public let bufferProfile: Data
        public let disposition: ColorDisposition

        /// Rebuild the source space. `nil` for untagged input or an unusable profile.
        public func sourceColorSpace() -> CGColorSpace? {
            guard let sourceProfile else { return nil }
            return CGColorSpace(iccData: sourceProfile as CFData)
        }

        /// Rebuild the space the buffer's samples are in.
        public func bufferColorSpace() -> CGColorSpace? {
            CGColorSpace(iccData: bufferProfile as CFData)
        }
    }

    /// EXIF/GPS/TIFF/IPTC carried forward, never consumed.
    ///
    /// The ImageIO property dictionary is `[String: Any]`, which Swift 6 cannot send across an
    /// isolation domain, so it travels as a **binary property list**: every value ImageIO returns
    /// is already a plist type, the encoding is lossless for those, and `Encode` gets the
    /// dictionary back with `properties()` and hands it straight to `CGImageDestinationAddImage`,
    /// so metadata survives the round trip instead of being dropped by a `nil` properties argument.
    ///
    /// Anything not plist-representable is dropped **loudly**: its key path lands in
    /// `droppedKeyPaths` rather than vanishing.
    public struct MetadataCarriage: Sendable, Equatable {
        public let serializedProperties: Data
        public let droppedKeyPaths: [String]
        /// The EXIF orientation **as stored**, 1 when the file carried none.
        ///
        /// Recorded, deliberately **not applied**: normalising orientation to 1 belongs to
        /// `Metadata`, and the raster in `pixelBuffer` is in storage order. A consumer that treats
        /// this buffer as display-ready shows rotated photographs.
        public let orientation: Int

        public func properties() throws -> [String: Any] {
            let value = try PropertyListSerialization.propertyList(
                from: serializedProperties, options: [], format: nil)
            return value as? [String: Any] ?? [:]
        }

        public var isEmpty: Bool { droppedKeyPaths.isEmpty && serializedProperties.count <= 8 }
    }

    /// Facts about the file, recorded so later stages and the certification record need not
    /// re-open it.
    public struct SourceFacts: Sendable, Equatable {
        public let formatIdentifier: String
        public let verdict: FormatVerdict
        /// Bits per component **as stored** (8, 16, 32) — the buffer is always 8.
        public let bitsPerComponent: Int
        /// True for float samples, i.e. the §3.2 HDR tier.
        public let isFloatingPoint: Bool
        /// True when the file has an alpha channel.
        ///
        /// Recorded, **not carried**: §3.1's alpha policy (RGB through the model, alpha resampled
        /// separately and recombined) belongs to `Alpha`. This buffer is `premultipliedFirst`, so
        /// transparent pixels composite onto black. Decode's one contribution is that the ground is
        /// *deterministic* — see `clearIfNeeded`.
        public let hasAlpha: Bool
        public let frameCount: Int
    }

    /// A decoded image plus everything the later stages must not have to re-derive.
    ///
    /// `@unchecked Sendable`: `CVPixelBuffer` is a CoreFoundation object with no compiler-visible
    /// Sendable guarantee. The justification is the same one `PixelBufferSurface` states in
    /// `UpscaleCore/Tiling/OutputSurface.swift:99` — the buffer is created inside `decode`, filled
    /// once, and handed off; decode retains no reference and never writes to it again.
    public struct DecodedImage: @unchecked Sendable {
        public let pixelBuffer: CVPixelBuffer
        public let width: Int
        public let height: Int
        public let color: ColorCarriage
        public let metadata: MetadataCarriage
        public let source: SourceFacts
    }

    // MARK: - Entry points

    public static func decode(
        contentsOf url: URL, limits: Limits = .automatic
    ) throws
        -> DecodedImage
    {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
            throw DecodeError.unreadable(origin: url.path)
        }
        return try decode(
            source: source, origin: url.path, limits: limits, bytes: .url(url))
    }

    /// `.unreadable` is reachable here only if CoreFoundation refuses to wrap the bytes at all.
    /// Bytes that exist but are not an image container — empty data included — are
    /// `.unrecognisedFormat`, the same verdict an empty *file* gets, because to a caller they are
    /// the same condition and the same fix.
    public static func decode(_ data: Data, limits: Limits = .automatic) throws -> DecodedImage {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else {
            throw DecodeError.unreadable(origin: dataOrigin)
        }
        return try decode(
            source: source, origin: dataOrigin, limits: limits, bytes: .data(data))
    }

    static let dataOrigin = "<in-memory data>"

    // MARK: - Container trailers

    /// Where the trailing bytes come from, so a trailer check never loads a whole file.
    ///
    /// A 48 MP source is ~100 MB; reading it entirely to look at twelve bytes would undo the
    /// band-spool work the engine exists to do. The URL case seeks instead.
    enum SourceBytes {
        case url(URL)
        case data(Data)

        /// The last `count` bytes, or fewer if the source is shorter. `nil` means the tail could
        /// not be obtained, which is treated as "cannot check", never as "invalid".
        func trailer(_ count: Int) -> Data? {
            switch self {
            case .data(let data):
                return data.suffix(count)
            case .url(let url):
                guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
                defer { try? handle.close() }
                guard let end = try? handle.seekToEnd() else { return nil }
                let start = end > UInt64(count) ? end - UInt64(count) : 0
                guard (try? handle.seek(toOffset: start)) != nil else { return nil }
                return try? handle.readToEnd()
            }
        }
    }

    static let trailerWindow = 64

    /// PNG's terminating chunk: zero length, `IEND`, and its constant CRC.
    static let pngIEND: [UInt8] = [0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82]
    /// JPEG's end-of-image marker.
    static let jpegEOI: [UInt8] = [0xFF, 0xD9]

    /// Refuse a container whose terminator is missing — the truncation `CGImageSourceGetStatus`
    /// cannot see.
    ///
    /// Measured on this host: a 197 KB PNG cut to 200 bytes still reports `statusComplete` AND
    /// still yields a full-size `CGImage`, with the missing rows simply blank. Nothing downstream
    /// can tell that from a photograph of a white wall, so §3.3's "clear error naming the
    /// limitation" was not being met for exactly the case a user is most likely to hit — an
    /// interrupted download or a half-copied file.
    ///
    /// Only PNG and JPEG are checked, and the omissions are deliberate rather than pending. TIFF
    /// is IFD-based and HEIF is ISO-BMFF: neither has a terminating marker to look for, so a
    /// trailer check would be a guess. Both were measured to fail image creation when truncated,
    /// which is the path that already refuses them.
    ///
    /// JPEG searches the tail rather than demanding the final two bytes, because appended trailing
    /// data after `EOI` is common in the wild. **Refusing a valid photograph is worse than missing
    /// a truncated one**, so the tolerant form is chosen knowingly: it still catches truncation,
    /// because a cut file has no `EOI` anywhere near its end.
    static func validateTrailer(_ tail: Data?, identifier: String, origin: String) throws {
        guard let tail, !tail.isEmpty else { return }
        let bytes = [UInt8](tail)
        let intact: Bool
        switch identifier {
        case "public.png":
            intact = bytes.count >= pngIEND.count && Array(bytes.suffix(pngIEND.count)) == pngIEND
        case "public.jpeg":
            intact = bytes.indices.dropLast().contains {
                bytes[$0] == jpegEOI[0] && bytes[$0 + 1] == jpegEOI[1]
            }
        default:
            return
        }
        if !intact {
            throw DecodeError.corruptOrTruncated(format: identifier, origin: origin)
        }
    }

    // MARK: - The decode itself

    /// What the header alone establishes about a file.
    struct Preflight: Equatable {
        let identifier: String
        let verdict: FormatVerdict
        let frameCount: Int
        let width: Int
        let height: Int
    }

    /// Every §3.3 refusal that can be made from the header, made **before** a pixel is decoded.
    ///
    /// Split out of `decode` so §10's "never silent, never mid-job" is a *structural* property
    /// rather than a promise in a comment: this function has no access to a `CGImage` and cannot
    /// create one, so any refusal it makes is provably pre-processing. `decode` calls it before
    /// `CGImageSourceCreateImageAtIndex`, and that is the only ordering the caller has to trust.
    static func preflight(
        source: CGImageSource, origin: String, limits: Limits
    ) throws
        -> (Preflight, [String: Any])
    {
        guard let identifier = CGImageSourceGetType(source) as String? else {
            throw DecodeError.unrecognisedFormat(origin: origin)
        }
        let verdict = classify(formatIdentifier: identifier)
        switch verdict {
        case .video(let format): throw DecodeError.videoNotSupported(format: format)
        case .animated(let format): throw DecodeError.animatedNotSupported(format: format)
        case .unsupported(let format): throw DecodeError.formatNotSupported(format: format)
        case .supported, .tolerated: break
        }

        let frameCount = CGImageSourceGetCount(source)
        guard frameCount > 0,
            let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [String: Any]
        else {
            throw DecodeError.corruptOrTruncated(format: identifier, origin: origin)
        }

        // A PNG is animated only if its frames carry delay times; a multi-page TIFF is not
        // animated, and decodes page 0.
        if let png = properties[kCGImagePropertyPNGDictionary as String] as? [String: Any],
            png[kCGImagePropertyAPNGDelayTime as String] != nil
        {
            throw DecodeError.animatedNotSupported(format: "APNG")
        }

        // §3.3 CMYK. Greyscale is neither listed as supported nor as unsupported; decode tolerates
        // it and records the normalisation rather than inventing a refusal the spec does not make.
        let colorModel = properties[kCGImagePropertyColorModel as String] as? String ?? "RGB"
        guard colorModel == "RGB" || colorModel == "Gray" else {
            throw DecodeError.colorModelNotSupported(model: colorModel, format: identifier)
        }

        let width = properties[kCGImagePropertyPixelWidth as String] as? Int ?? 0
        let height = properties[kCGImagePropertyPixelHeight as String] as? Int ?? 0
        guard width > 0, height > 0 else {
            throw DecodeError.corruptOrTruncated(format: identifier, origin: origin)
        }
        // Measured from the header alone — that is what makes it pre-processing.
        guard width * height <= limits.maximumPixelCount else {
            throw DecodeError.exceedsMemoryFeasibility(
                pixels: width * height, limit: limits.maximumPixelCount)
        }

        return (
            Preflight(
                identifier: identifier, verdict: verdict, frameCount: frameCount,
                width: width, height: height),
            properties
        )
    }

    static func decode(
        source: CGImageSource, origin: String, limits: Limits, bytes: SourceBytes? = nil
    ) throws
        -> DecodedImage
    {
        // §10: every refusal that can be made from the header is made here, before any pixel is
        // touched.
        let (header, properties) = try preflight(
            source: source, origin: origin, limits: limits)

        // Still §10 territory: a missing terminator is a structural fact about the container, so
        // the refusal happens before a pixel is decoded rather than after.
        try validateTrailer(
            bytes?.trailer(trailerWindow), identifier: header.identifier, origin: origin)

        // Truncation is caught here, not by `CGImageSourceGetStatus`: measured on this host, a
        // 50%-truncated PNG, JPEG, TIFF and HEIC all still report `statusComplete`, and only image
        // creation fails. Trusting the status would let a truncated file straight through.
        //
        // The partial-image case is closed by the trailer check above: ImageIO returns a full-size
        // image with blank rows for a large truncated PNG, so image creation succeeding is not
        // evidence the file is whole.
        guard let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw DecodeError.corruptOrTruncated(
                format: header.identifier, origin: origin)
        }
        let identifier = header.identifier
        let verdict = header.verdict
        let frameCount = header.frameCount

        let buffer = try makeBuffer(width: image.width, height: image.height)
        let (context, disposition) = try makeContext(for: image, into: buffer)
        clearIfNeeded(buffer, hasAlpha: image.alphaInfo.carriesAlpha)
        context.draw(
            image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        CVPixelBufferUnlockBaseAddress(buffer, [])

        return DecodedImage(
            pixelBuffer: buffer,
            width: image.width,
            height: image.height,
            color: carriage(for: image, disposition: disposition, properties: properties),
            metadata: carriage(for: properties),
            source: SourceFacts(
                formatIdentifier: identifier,
                verdict: verdict,
                bitsPerComponent: properties[kCGImagePropertyDepth as String] as? Int
                    ?? image.bitsPerComponent,
                isFloatingPoint: properties[kCGImagePropertyIsFloat as String] as? Bool ?? false,
                hasAlpha: properties[kCGImagePropertyHasAlpha as String] as? Bool
                    ?? image.alphaInfo.carriesAlpha,
                frameCount: frameCount))
    }

    // MARK: - Buffer and context

    /// The engine's contract: 32BGRA and IOSurface-backed, so the buffer can reach Core ML, Metal
    /// or the display without a copy (CVPixelBuffer.h — `kCVPixelBufferIOSurfacePropertiesKey`).
    static func makeBuffer(width: Int, height: Int) throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        let attributes: [CFString: Any] = [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA,
            attributes as CFDictionary, &buffer)
        guard status == kCVReturnSuccess, let buffer else {
            throw DecodeError.bufferAllocationFailed(
                width: width, height: height, status: status)
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        return buffer
    }

    /// Choose the destination colour space by **trying the source's own first**.
    ///
    /// This is decode's whole colour claim. Drawing into the source's own space is an identity
    /// pass: the samples land byte-for-byte, so a Display P3 file keeps its out-of-gamut colour
    /// instead of being flattened into sRGB. Whether a space can back an 8-bit BGRA bitmap is
    /// decided by asking Core Graphics rather than by enumerating spaces — measured on this host,
    /// sRGB / Display P3 / Adobe RGB succeed while greyscale, CMYK and every extended-range or
    /// float space fail, which is exactly the set that must normalise. The caller must have locked
    /// `buffer`.
    static func makeContext(
        for image: CGImage, into buffer: CVPixelBuffer
    ) throws
        -> (CGContext, ColorDisposition)
    {
        let bitmapInfo =
            CGImageAlphaInfo.premultipliedFirst.rawValue
            | CGBitmapInfo.byteOrder32Little.rawValue
        func context(_ space: CGColorSpace) -> CGContext? {
            CGContext(
                data: CVPixelBufferGetBaseAddress(buffer),
                width: image.width, height: image.height, bitsPerComponent: 8,
                bytesPerRow: CVPixelBufferGetBytesPerRow(buffer), space: space,
                bitmapInfo: bitmapInfo)
        }
        if let space = image.colorSpace, let made = context(space) {
            return (made, .preservedSourceSpace)
        }
        guard let srgb = CGColorSpace(name: CGColorSpace.sRGB), let made = context(srgb) else {
            CVPixelBufferUnlockBaseAddress(buffer, [])
            throw DecodeError.drawingContextUnavailable(width: image.width, height: image.height)
        }
        return (made, .normalisedToSRGB)
    }

    /// Zero the buffer before drawing an image that has alpha.
    ///
    /// `CVPixelBufferCreate` does not promise zeroed memory, and a `premultipliedFirst` draw
    /// composites source-over whatever is already there. Measured: a transparent pixel drawn over a
    /// buffer pre-filled with `0xAB` reads back `0xAB` — the output would depend on allocator luck.
    /// This does not implement the alpha policy (that is `Alpha`); it makes the ground
    /// deterministic so the policy is testable.
    static func clearIfNeeded(_ buffer: CVPixelBuffer, hasAlpha: Bool) {
        guard hasAlpha, let base = CVPixelBufferGetBaseAddress(buffer) else { return }
        memset(base, 0, CVPixelBufferGetBytesPerRow(buffer) * CVPixelBufferGetHeight(buffer))
    }

    // MARK: - Carriage construction

    static func carriage(
        for image: CGImage, disposition: ColorDisposition, properties: [String: Any]
    ) -> ColorCarriage {
        let sourceProfile = (image.colorSpace?.copyICCData() as Data?)
        let bufferProfile: Data?
        switch disposition {
        case .preservedSourceSpace:
            bufferProfile = sourceProfile
        case .normalisedToSRGB:
            bufferProfile =
                CGColorSpace(name: CGColorSpace.sRGB)?.copyICCData() as Data?
        }
        return ColorCarriage(
            sourceProfile: sourceProfile,
            sourceProfileName: properties[kCGImagePropertyProfileName as String] as? String,
            bufferProfile: bufferProfile ?? Data(),
            disposition: disposition)
    }

    static func carriage(for properties: [String: Any]) -> MetadataCarriage {
        var dropped: [String] = []
        let sanitised = sanitise(properties, keyPath: "", dropped: &dropped)
        let data =
            (try? PropertyListSerialization.data(
                fromPropertyList: sanitised, format: .binary, options: 0)) ?? Data()
        return MetadataCarriage(
            serializedProperties: data,
            droppedKeyPaths: dropped.sorted(),
            orientation: properties[kCGImagePropertyOrientation as String] as? Int ?? 1)
    }

    /// Keep only property-list-representable values, recording the key path of anything dropped.
    ///
    /// Every value ImageIO is documented to return is a plist type, so this is expected to drop
    /// nothing; it exists because "13 of 13 fields absent" is the defect being fixed and a silent
    /// serialisation failure would be the same defect wearing a different hat.
    static func sanitise(_ value: Any, keyPath: String, dropped: inout [String]) -> Any {
        switch value {
        case let dictionary as [String: Any]:
            var out: [String: Any] = [:]
            for (key, element) in dictionary {
                let path = keyPath.isEmpty ? key : "\(keyPath).\(key)"
                guard PropertyListSerialization.propertyList(element, isValidFor: .binary) else {
                    dropped.append(path)
                    continue
                }
                out[key] = sanitise(element, keyPath: path, dropped: &dropped)
            }
            return out
        case let array as [Any]:
            return array.enumerated().compactMap { index, element -> Any? in
                let path = "\(keyPath)[\(index)]"
                guard PropertyListSerialization.propertyList(element, isValidFor: .binary) else {
                    dropped.append(path)
                    return nil
                }
                return sanitise(element, keyPath: path, dropped: &dropped)
            }
        default:
            return value
        }
    }
}

extension Decode.DecodeError: CustomStringConvertible {
    public var description: String {
        switch self {
        case .unreadable(let origin):
            return "could not read an image from \(origin)"
        case .unrecognisedFormat(let origin):
            return "unrecognised image format at \(origin) — the container type could not be identified"
        case .corruptOrTruncated(let format, let origin):
            return "corrupt or truncated \(format) at \(origin) — the image data ends before the image does"
        case .animatedNotSupported(let format):
            return "animated images are not supported in v1 (\(format)) — export a single frame first"
        case .videoNotSupported(let format):
            return "video is not supported in v1 (\(format)) — this is an image upscaler"
        case .formatNotSupported(let format):
            return "\(format) is not a supported image format in v1 (supported: JPEG, PNG, HEIF, TIFF)"
        case .colorModelNotSupported(let model, let format):
            return "\(model) colour is not supported in v1 (\(format)) — convert to RGB first"
        case .exceedsMemoryFeasibility(let pixels, let limit):
            return "image is too large to decode on this device: \(pixels) pixels exceeds the "
                + "\(limit) pixel limit"
        case .bufferAllocationFailed(let width, let height, let status):
            return "could not allocate a \(width)x\(height) 32BGRA input buffer "
                + "(CVPixelBufferCreate returned \(status))"
        case .drawingContextUnavailable(let width, let height):
            return "could not create a \(width)x\(height) drawing context for the decoded image"
        }
    }
}

extension CGImageAlphaInfo {
    /// True when the layout has an alpha channel that carries information.
    var carriesAlpha: Bool {
        switch self {
        case .first, .last, .premultipliedFirst, .premultipliedLast, .alphaOnly: return true
        case .none, .noneSkipFirst, .noneSkipLast: return false
        @unknown default: return false
        }
    }
}
