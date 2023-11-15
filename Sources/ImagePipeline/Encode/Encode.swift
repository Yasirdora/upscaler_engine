import CoreGraphics
import CoreVideo
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// Writing the result out — certification spec §3.1, §3.4, §8.4.
///
/// Decode carries the colour profile and the metadata, `Color` decides which profile to write and
/// `Metadata` normalises the orientation; all of that is wasted if the encoder drops it on the
/// floor — passing `nil` properties to ImageIO and drawing through `CGColorSpaceCreateDeviceRGB()`
/// loses every metadata field and returns a Display P3 photograph as sRGB.
///
/// So the rule here is that **nothing is reconstructed**. The profile comes from `Color.Plan`, the
/// properties come from `Metadata.normalisedProperties`, and this module's only job is to hand both
/// to ImageIO without losing them on the way.
public enum Encode {
    /// The §3.1 export formats.
    public enum Format: String, Sendable, CaseIterable {
        case png, jpeg, heif, tiff

        public var type: UTType {
            switch self {
            case .png: return .png
            case .jpeg: return .jpeg
            case .heif: return UTType("public.heic") ?? .image
            case .tiff: return .tiff
            }
        }

        /// Whether a quality setting means anything. PNG and TIFF ignore it, and silently
        /// accepting one for them would imply a control the user does not have.
        public var isLossy: Bool {
            switch self {
            case .jpeg, .heif: return true
            case .png, .tiff: return false
            }
        }
    }

    public enum EncodeError: Error, Equatable {
        case destinationUnavailable(format: String, path: String)
        case imageUnavailable
        case writeFailed(path: String)
        /// A quality outside 0...1, or supplied for a format that has none.
        case qualityNotApplicable(format: String)
        case qualityOutOfRange(Double)
    }

    /// Write a buffer, embedding the planned profile and the carried metadata.
    ///
    /// `properties` must already have come through `Metadata.normalisedProperties`: this function
    /// does not normalise orientation, because an encoder that silently rewrote metadata would be
    /// a second place where the truth about a file is decided, and one is enough.
    public static func write(
        _ buffer: CVPixelBuffer, to url: URL, format: Format, plan: Color.Plan,
        properties: [String: Any], quality: Double? = nil
    ) throws {
        if let quality {
            guard format.isLossy else {
                throw EncodeError.qualityNotApplicable(format: format.rawValue)
            }
            guard (0.0...1.0).contains(quality) else {
                throw EncodeError.qualityOutOfRange(quality)
            }
        }
        let image = try makeImage(from: buffer, plan: plan)
        guard
            let destination = CGImageDestinationCreateWithURL(
                url as CFURL, format.type.identifier as CFString, 1, nil)
        else {
            throw EncodeError.destinationUnavailable(format: format.rawValue, path: url.path)
        }

        var outgoing = properties
        if let quality, format.isLossy {
            outgoing[kCGImageDestinationLossyCompressionQuality as String] = quality
        }
        // The profile travels as bytes rather than as a CGColorSpace name, so a space ImageIO
        // cannot name by constant — a custom ICC profile from a scanner, say — still round-trips.
        outgoing[kCGImagePropertyProfileName as String] = plan.outputProfileName

        CGImageDestinationAddImage(destination, image, outgoing as CFDictionary)
        guard CGImageDestinationFinalize(destination) else {
            throw EncodeError.writeFailed(path: url.path)
        }
    }

    /// Build the image in the space the plan names.
    ///
    /// `CGColorSpaceCreateDeviceRGB()` is what the CLI uses today and is precisely the defect: it
    /// is not a colour space so much as a refusal to name one, and Core Graphics then tags the
    /// output sRGB. Measured cost on a Display P3 sweep: max per-channel delta 117/255.
    static func makeImage(from buffer: CVPixelBuffer, plan: Color.Plan) throws -> CGImage {
        guard let space = plan.outputColorSpace() else { throw EncodeError.imageUnavailable }
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard
            let context = CGContext(
                data: CVPixelBufferGetBaseAddress(buffer),
                width: CVPixelBufferGetWidth(buffer), height: CVPixelBufferGetHeight(buffer),
                bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                space: space,
                bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
                    | CGBitmapInfo.byteOrder32Little.rawValue),
            let image = context.makeImage()
        else { throw EncodeError.imageUnavailable }
        return image
    }
}
