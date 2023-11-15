import CoreGraphics
import CoreVideo
import Foundation
import ImageIO

/// Orientation normalisation and metadata carriage for export — certification spec §3.4, §8.2.
///
/// Decode records the stored EXIF orientation and deliberately does not apply it, so the raster
/// reaching here is in **storage order**. §8.2's pipeline is decode → normalise → upscale → export,
/// which puts normalisation before the model: the model then sees the image the way a person does,
/// and the exported file needs no orientation tag but the identity one.
public enum Metadata {
    /// The eight EXIF orientations, by their stored values.
    ///
    /// All eight are modelled even though §3.1 names 1/3/6/8 for explicit testing: the four
    /// mirrored values are rare but real, and a switch that silently treats an unhandled value as
    /// `.up` would hand back a wrongly-flipped image with no error.
    public enum Orientation: Int, Sendable, CaseIterable {
        case up = 1
        case upMirrored = 2
        case down = 3
        case downMirrored = 4
        case leftMirrored = 5
        case right = 6
        case rightMirrored = 7
        case left = 8

        /// `nil` for a value EXIF does not define, so an out-of-range tag is refused rather than
        /// coerced into `.up`.
        public init?(stored: Int) { self.init(rawValue: stored) }

        /// The four quarter-turn orientations exchange width and height. Getting this wrong
        /// produces a correctly-rotated image drawn into a canvas of the wrong shape, which
        /// crops it — silently, because nothing throws.
        public var swapsDimensions: Bool {
            switch self {
            case .leftMirrored, .right, .rightMirrored, .left: return true
            case .up, .upMirrored, .down, .downMirrored: return false
            }
        }

        public var isMirrored: Bool {
            switch self {
            case .upMirrored, .downMirrored, .leftMirrored, .rightMirrored: return true
            case .up, .down, .right, .left: return false
            }
        }

        /// The transform taking storage order to display order, for a stored raster of `size`.
        ///
        /// Composed as translate-then-rotate/scale in Core Graphics' bottom-left space. Each case
        /// is written out rather than derived from a shared expression: the derivation is where
        /// sign errors hide, and a wrong sign here is invisible in a symmetric test image.
        public func transform(for size: CGSize) -> CGAffineTransform {
            let w = size.width, h = size.height
            switch self {
            case .up:
                return .identity
            case .upMirrored:
                return CGAffineTransform(translationX: w, y: 0).scaledBy(x: -1, y: 1)
            case .down:
                return CGAffineTransform(translationX: w, y: h).rotated(by: .pi)
            case .downMirrored:
                return CGAffineTransform(translationX: 0, y: h).scaledBy(x: 1, y: -1)
            case .leftMirrored:
                return CGAffineTransform(translationX: h, y: 0).rotated(by: .pi / 2)
                    .translatedBy(x: w, y: 0).scaledBy(x: -1, y: 1)
            case .right:
                return CGAffineTransform(translationX: 0, y: w).rotated(by: -.pi / 2)
            case .rightMirrored:
                return CGAffineTransform(translationX: 0, y: 0).rotated(by: -.pi / 2)
                    .scaledBy(x: -1, y: 1)
            case .left:
                return CGAffineTransform(translationX: h, y: 0).rotated(by: .pi / 2)
            }
        }
    }

    public enum MetadataError: Error, Equatable {
        /// A stored orientation outside EXIF's 1...8. Refused, never coerced.
        case undefinedOrientation(stored: Int)
        case drawingContextUnavailable
    }

    /// Every key under which an orientation can hide.
    ///
    /// Writing only the top-level key leaves the TIFF dictionary's copy behind, and ImageIO
    /// prefers whichever the container puts first — so a file normalised in one place and not the
    /// other gets **re-rotated on reopen**, which is defect 2 wearing a different hat. The set is
    /// named here so the sweep below cannot forget one.
    static let orientationKeys: [String] = [
        kCGImagePropertyOrientation as String,
        kCGImagePropertyTIFFOrientation as String,
    ]

    /// Apply the stored orientation, returning a display-ordered buffer.
    ///
    /// Returns the input unchanged for `.up`, which is not an optimisation but a correctness
    /// property: an identity normalisation must not resample, and a copy through a context would
    /// be a chance to lose colour for no reason.
    public static func normalise(
        _ buffer: CVPixelBuffer, orientation stored: Int, colorSpace: CGColorSpace
    ) throws -> CVPixelBuffer {
        guard let orientation = Orientation(stored: stored) else {
            throw MetadataError.undefinedOrientation(stored: stored)
        }
        if orientation == .up { return buffer }

        let width = CVPixelBufferGetWidth(buffer), height = CVPixelBufferGetHeight(buffer)
        let outWidth = orientation.swapsDimensions ? height : width
        let outHeight = orientation.swapsDimensions ? width : height

        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let bitmapInfo =
            CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        guard
            let sourceContext = CGContext(
                data: CVPixelBufferGetBaseAddress(buffer), width: width, height: height,
                bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                space: colorSpace, bitmapInfo: bitmapInfo),
            let image = sourceContext.makeImage()
        else { throw MetadataError.drawingContextUnavailable }

        let output = try Decode.makeBuffer(width: outWidth, height: outHeight)
        CVPixelBufferLockBaseAddress(output, [])
        defer { CVPixelBufferUnlockBaseAddress(output, []) }
        guard
            let destination = CGContext(
                data: CVPixelBufferGetBaseAddress(output), width: outWidth, height: outHeight,
                bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(output),
                space: colorSpace, bitmapInfo: bitmapInfo)
        else { throw MetadataError.drawingContextUnavailable }
        destination.concatenate(orientation.transform(for: CGSize(width: width, height: height)))
        destination.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return output
    }

    /// The property dictionary for export: everything preserved, orientation forced to 1.
    ///
    /// §3.4 preserves EXIF, ICC, timestamps and GPS by default, so this **allow-lists nothing**.
    /// It copies the dictionary whole and rewrites orientation, because a metadata policy built
    /// from a list of keys to keep silently drops every key nobody thought of — including the
    /// camera and GPS fields §3.4 names explicitly.
    public static func normalisedProperties(
        from carriage: Decode.MetadataCarriage
    ) throws -> [String: Any] {
        guard Orientation(stored: carriage.orientation) != nil else {
            throw MetadataError.undefinedOrientation(stored: carriage.orientation)
        }
        var properties = try carriage.properties()
        for key in orientationKeys where properties[key] != nil {
            properties[key] = Orientation.up.rawValue
        }
        properties[kCGImagePropertyOrientation as String] = Orientation.up.rawValue
        if var tiff = properties[kCGImagePropertyTIFFDictionary as String] as? [String: Any] {
            tiff[kCGImagePropertyTIFFOrientation as String] = Orientation.up.rawValue
            properties[kCGImagePropertyTIFFDictionary as String] = tiff
        }
        return properties
    }
}
