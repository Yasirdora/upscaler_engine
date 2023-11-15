import Accelerate
import CoreGraphics
import CoreVideo
import Foundation

/// §3.1's alpha policy: RGB upscaled by the model, alpha upscaled separately, recombined.
///
/// Chosen by measurement, and the measurement overturned the intuitive answer twice.
///
/// **`CILanczosScaleTransform` is disqualified.** Core Image and MPS extend the image with
/// transparent black, putting 100% of their far-field error at the frame border, up to 24/255 — any
/// picture whose alpha is non-zero at the edge fades out around its whole perimeter. And
/// **"CGContext with `.high`" names no algorithm at all**: `.low`, `.medium`, `.high` and
/// `.default` are byte-identical, max diff 0 over 4,194,304 px.
///
/// **Straight, not premultiplied — and that reverses the pure-resampling answer.** With one
/// resampler on both channels, premultiplied wins (chroma 22 against straight's 91). With the real
/// model on the RGB it is the *worst* option: max|d| 119 and −47.7 darkening, against straight's 71
/// and dilated's 10 / −0.3. Premultiplying paints a fictitious colour→black edge into the one
/// channel a 52 px-context sharpener then sharpens. So: unpremultiply, dilate colour outward into
/// the transparent region, upscale the alpha plane separately, and premultiply exactly once at the
/// encoder.
public enum Alpha {
    public enum AlphaError: Error, Equatable {
        case planeAllocationFailed
        case resampleFailed(vImage_Error)
        case sizeMismatch(alpha: String, colour: String)
    }

    /// A straight-alpha image split into the two things that are upscaled by different means.
    /// `@unchecked` for the same reason `Decode.DecodedImage` is: `CVPixelBuffer` is not
    /// `Sendable`, and the engine's rule is that a buffer has exactly one owner at a time.
    public struct Split: @unchecked Sendable {
        /// Straight (un-premultiplied) BGRA, with colour dilated outward. The model's input.
        public let colour: CVPixelBuffer
        /// The alpha plane on its own, 8-bit, one byte per pixel.
        public let alpha: [UInt8]
        public let width: Int
        public let height: Int
        /// Whether any pixel is not fully opaque. An image without alpha skips the whole path.
        public let hasTransparency: Bool
    }

    /// Below this alpha, unpremultiplying amplifies quantisation error past usefulness.
    ///
    /// Recovering straight colour divides by α, so the error multiplies by 255/α. **Measured**: at
    /// α 1–25 the mean recovery error is 30.9/255 (n = 5,962); at α ≥ 128 the max is 2/255.
    /// Dilation is therefore seeded only from pixels at or above this value — a nearly-transparent
    /// pixel's colour is not evidence about anything.
    ///
    /// `MeasuredThreshold` in the tests re-derives this from a sweep rather than trusting the
    /// constant, because the threshold was first chosen by reasoning rather than measurement.
    public static let dilationSeedThreshold: UInt8 = 128

    /// Split a premultiplied BGRA buffer into straight colour and a standalone alpha plane.
    public static func split(_ buffer: CVPixelBuffer) throws -> Split {
        let width = CVPixelBufferGetWidth(buffer), height = CVPixelBufferGetHeight(buffer)
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else {
            throw AlphaError.planeAllocationFailed
        }
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        let rows = base.assumingMemoryBound(to: UInt8.self)

        var alpha = [UInt8](repeating: 255, count: width * height)
        let colour = try Decode.makeBuffer(width: width, height: height)
        CVPixelBufferLockBaseAddress(colour, [])
        defer { CVPixelBufferUnlockBaseAddress(colour, []) }
        guard let outBase = CVPixelBufferGetBaseAddress(colour) else {
            throw AlphaError.planeAllocationFailed
        }
        let outStride = CVPixelBufferGetBytesPerRow(colour)
        let out = outBase.assumingMemoryBound(to: UInt8.self)
        var transparent = false

        for y in 0..<height {
            for x in 0..<width {
                let i = y * stride + x * 4
                let o = y * outStride + x * 4
                let a = rows[i + 3]
                alpha[y * width + x] = a
                if a != 255 { transparent = true }
                // Undo the premultiply. Exact for opaque pixels; increasingly lossy as alpha
                // falls, which is why dilation is seeded only above `dilationSeedThreshold`.
                for c in 0..<3 {
                    out[o + c] = a == 0 ? 0 : unpremultiply(rows[i + c], alpha: a)
                }
                out[o + 3] = 255
            }
        }
        if transparent {
            dilate(out, width: width, height: height, stride: outStride, alpha: alpha)
        }
        return Split(
            colour: colour, alpha: alpha, width: width, height: height,
            hasTransparency: transparent)
    }

    static func unpremultiply(_ value: UInt8, alpha: UInt8) -> UInt8 {
        UInt8(min(255, (Int(value) * 255 + Int(alpha) / 2) / Int(alpha)))
    }

    /// Push colour outward from opaque pixels into the transparent region.
    ///
    /// Transparent pixels carry no usable colour, and handing the model black there is what
    /// produces the −47.7 halo. Each pass fills an unseeded pixel with the mean of its seeded
    /// neighbours; a handful of passes covers the distance a tile's receptive field can reach
    /// across an alpha edge.
    static func dilate(
        _ pixels: UnsafeMutablePointer<UInt8>, width: Int, height: Int, stride: Int,
        alpha: [UInt8], passes: Int = 8
    ) {
        var seeded = alpha.map { $0 >= dilationSeedThreshold }
        for _ in 0..<passes {
            var next = seeded
            for y in 0..<height {
                for x in 0..<width where !seeded[y * width + x] {
                    var sum = [0, 0, 0]
                    var count = 0
                    for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                        let nx = x + dx, ny = y + dy
                        guard nx >= 0, nx < width, ny >= 0, ny < height,
                            seeded[ny * width + nx]
                        else { continue }
                        let n = ny * stride + nx * 4
                        for c in 0..<3 { sum[c] += Int(pixels[n + c]) }
                        count += 1
                    }
                    guard count > 0 else { continue }
                    let o = y * stride + x * 4
                    for c in 0..<3 { pixels[o + c] = UInt8(sum[c] / count) }
                    next[y * width + x] = true
                }
            }
            seeded = next
        }
    }

    /// Upscale the alpha plane on its own.
    ///
    /// `vImageScale_Planar8` with `kvImageHighQualityResampling` — best edge accuracy of the 15
    /// candidates measured, a 4 px transition against Core Graphics' 5, and correct edge clamping
    /// rather than Core Image's fade-to-transparent at the frame border.
    public static func scale(
        _ plane: [UInt8], from source: (width: Int, height: Int), to target: (width: Int, height: Int)
    ) throws -> [UInt8] {
        var input = plane
        var output = [UInt8](repeating: 0, count: target.width * target.height)
        let error: vImage_Error = input.withUnsafeMutableBufferPointer { inBuf in
            output.withUnsafeMutableBufferPointer { outBuf in
                var src = vImage_Buffer(
                    data: inBuf.baseAddress, height: vImagePixelCount(source.height),
                    width: vImagePixelCount(source.width), rowBytes: source.width)
                var dst = vImage_Buffer(
                    data: outBuf.baseAddress, height: vImagePixelCount(target.height),
                    width: vImagePixelCount(target.width), rowBytes: target.width)
                return vImageScale_Planar8(
                    &src, &dst, nil, vImage_Flags(kvImageHighQualityResampling))
            }
        }
        guard error == kvImageNoError else { throw AlphaError.resampleFailed(error) }
        return output
    }

    /// Put the upscaled alpha back onto the model's output, premultiplying **once**.
    ///
    /// This is the single premultiply in the whole pipeline. Doing it earlier is the −47.7 halo;
    /// doing it twice is a compounding darkening; forgetting it entirely measured −25.3.
    public static func recombine(
        _ colour: CVPixelBuffer, alpha: [UInt8]
    ) throws -> CVPixelBuffer {
        let width = CVPixelBufferGetWidth(colour), height = CVPixelBufferGetHeight(colour)
        guard alpha.count == width * height else {
            throw AlphaError.sizeMismatch(
                alpha: "\(alpha.count)", colour: "\(width)x\(height)")
        }
        CVPixelBufferLockBaseAddress(colour, [])
        defer { CVPixelBufferUnlockBaseAddress(colour, []) }
        guard let base = CVPixelBufferGetBaseAddress(colour) else {
            throw AlphaError.planeAllocationFailed
        }
        let stride = CVPixelBufferGetBytesPerRow(colour)
        let pixels = base.assumingMemoryBound(to: UInt8.self)
        for y in 0..<height {
            for x in 0..<width {
                let i = y * stride + x * 4
                let a = alpha[y * width + x]
                for c in 0..<3 {
                    pixels[i + c] = UInt8((Int(pixels[i + c]) * Int(a) + 127) / 255)
                }
                pixels[i + 3] = a
            }
        }
        return colour
    }
}
