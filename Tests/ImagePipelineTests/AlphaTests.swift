import Accelerate
import CoreGraphics
import CoreVideo
import Foundation
import Testing

@testable import ImagePipeline

/// §3.1's alpha policy and §8.3's fringing check (certification spec).
///
/// The corpus holds exactly ONE image with alpha, so every fixture here is constructed.
struct AlphaTests {
    /// A premultiplied BGRA buffer: left half opaque `colour`, right half fully transparent.
    static func halfTransparent(
        width: Int = 16, height: Int = 8, colour: (b: UInt8, g: UInt8, r: UInt8) = (0, 0, 255)
    ) throws -> CVPixelBuffer {
        let buffer = try Decode.makeBuffer(width: width, height: height)
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        let base = try #require(CVPixelBufferGetBaseAddress(buffer))
            .assumingMemoryBound(to: UInt8.self)
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        for y in 0..<height {
            for x in 0..<width {
                let i = y * stride + x * 4
                let opaque = x < width / 2
                base[i] = opaque ? colour.b : 0
                base[i + 1] = opaque ? colour.g : 0
                base[i + 2] = opaque ? colour.r : 0
                base[i + 3] = opaque ? 255 : 0
            }
        }
        return buffer
    }

    // MARK: - The policy's central claim, asserted

    /// The −47.7 halo, stated as the thing that must not happen.
    ///
    /// The transparent half carries no colour of its own. If it reaches the model as black — which
    /// is what a premultiplied buffer hands over — a 52 px-context sharpener sharpens a fictitious
    /// red→black edge. Dilation is what stops that, so the transparent side must come out carrying
    /// the opaque side's colour, not zero.
    @Test("colour is dilated into the transparent region, so the model never sees a false edge")
    func colourIsDilatedNotLeftBlack() throws {
        let split = try Alpha.split(try Self.halfTransparent())

        #expect(split.hasTransparency)
        let justInside = DecodeFixtures.pixel(of: split.colour, x: 7, y: 3)
        let justOutside = DecodeFixtures.pixel(of: split.colour, x: 8, y: 3)
        #expect(justInside == [0, 0, 255, 255], "the opaque side is unchanged")
        #expect(
            justOutside == [0, 0, 255, 255],
            "the transparent side must carry the dilated colour, never black; black here is the fictitious edge that measured -47.7 darkening"
        )
    }

    /// Alpha must survive as its own plane rather than being folded into the colour.
    @Test("the alpha plane is extracted intact")
    func alphaPlaneIsExtracted() throws {
        let split = try Alpha.split(try Self.halfTransparent())

        #expect(split.alpha.count == 16 * 8)
        #expect(split.alpha[3 * 16 + 3] == 255)
        #expect(split.alpha[3 * 16 + 12] == 0)
    }

    @Test("a fully opaque image reports no transparency and is left alone")
    func opaqueImageIsUntouched() throws {
        let buffer = try Decode.makeBuffer(width: 8, height: 8)
        CVPixelBufferLockBaseAddress(buffer, [])
        let base = try #require(CVPixelBufferGetBaseAddress(buffer))
            .assumingMemoryBound(to: UInt8.self)
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        for y in 0..<8 {
            for x in 0..<8 {
                let i = y * stride + x * 4
                base[i] = 10
                base[i + 1] = 20
                base[i + 2] = 30
                base[i + 3] = 255
            }
        }
        CVPixelBufferUnlockBaseAddress(buffer, [])

        let split = try Alpha.split(buffer)

        #expect(!split.hasTransparency)
        #expect(DecodeFixtures.pixel(of: split.colour, x: 4, y: 4) == [10, 20, 30, 255])
    }

    // MARK: - The α ≥ 128 seeding rule, MEASURED

    /// The threshold was first chosen by reasoning rather than measurement. This is the
    /// measurement.
    ///
    /// Recovering straight colour divides by α, so quantisation error multiplies by 255/α. The
    /// sweep below re-derives the error at every α rather than trusting the constant.
    @Test("unpremultiply error grows as alpha falls, and is negligible at the chosen threshold")
    func unpremultiplyErrorSweep() throws {
        // For each alpha, premultiply a known colour then recover it, and record the error.
        func maxError(atAlpha a: UInt8) -> Int {
            var worst = 0
            for straight in 0...255 {
                let premultiplied = UInt8((straight * Int(a) + 127) / 255)
                let recovered = Int(Alpha.unpremultiply(premultiplied, alpha: a))
                worst = max(worst, abs(recovered - straight))
            }
            return worst
        }

        let atThreshold = maxError(atAlpha: Alpha.dilationSeedThreshold)
        #expect(atThreshold <= 2, "measured max recovery error at alpha 128 is \(atThreshold)/255")

        // And it is genuinely worse below. A low-alpha pixel's colour is not evidence.
        let low = (1...25).map { maxError(atAlpha: UInt8($0)) }.max() ?? 0
        #expect(low > 20, "measured max recovery error at alpha 1-25 is \(low)/255")

        // The threshold is a real boundary, not an arbitrary round number: error is monotonically
        // non-increasing in alpha, so everything at or above it inherits the bound.
        for a in stride(from: Int(Alpha.dilationSeedThreshold), through: 255, by: 1) {
            #expect(maxError(atAlpha: UInt8(a)) <= atThreshold)
        }
    }

    /// Seeding from nearly-transparent pixels would import their unreliable colour.
    @Test("dilation is not seeded from pixels below the threshold")
    func lowAlphaPixelsDoNotSeedDilation() throws {
        let width = 8, height = 4
        let buffer = try Decode.makeBuffer(width: width, height: height)
        CVPixelBufferLockBaseAddress(buffer, [])
        let base = try #require(CVPixelBufferGetBaseAddress(buffer))
            .assumingMemoryBound(to: UInt8.self)
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        for y in 0..<height {
            for x in 0..<width {
                let i = y * stride + x * 4
                // x=0 opaque green; x=1 a nearly-transparent BLUE that must not seed; rest clear.
                let a: UInt8 = x == 0 ? 255 : (x == 1 ? 4 : 0)
                base[i] = x == 1 ? 4 : 0
                base[i + 1] = x == 0 ? 255 : 0
                base[i + 2] = 0
                base[i + 3] = a
            }
        }
        CVPixelBufferUnlockBaseAddress(buffer, [])

        let split = try Alpha.split(buffer)

        let far = DecodeFixtures.pixel(of: split.colour, x: 6, y: 2)
        #expect(far[1] > far[0], "green from the opaque seed must dominate, not blue from alpha 4")
    }

    // MARK: - Resampling

    @Test("the alpha plane upscales to the target geometry")
    func planeScalesToTarget() throws {
        let plane = [UInt8](repeating: 200, count: 4 * 4)

        let scaled = try Alpha.scale(plane, from: (4, 4), to: (16, 16))

        #expect(scaled.count == 256)
        #expect(scaled.allSatisfy { $0 == 200 }, "a constant plane must stay constant")
    }

    /// The Core Image disqualification, stated as a property this implementation must have: an
    /// edge that is opaque at the frame border must stay opaque there. CI and MPS extend with
    /// transparent black and fade the whole perimeter out.
    @Test("an opaque border stays opaque after scaling — no fade at the frame edge")
    func borderDoesNotFadeOut() throws {
        let plane = [UInt8](repeating: 255, count: 8 * 8)

        let scaled = try Alpha.scale(plane, from: (8, 8), to: (32, 32))

        #expect(scaled[0] == 255, "top-left corner faded — the Core Image defect")
        #expect(scaled[31] == 255)
        #expect(scaled[32 * 31] == 255)
        #expect(scaled[32 * 32 - 1] == 255)
    }

    /// `kvImageHighQualityResampling` is the headline of the policy, and without this test
    /// **dropping it killed no test at all** — a mutation found that, on a constant plane and a
    /// solid border, neither of which can distinguish one resampling kernel from another. An
    /// adopted flag with no coverage is a decorative flag.
    ///
    /// The two kernels are separated by their behaviour at a hard step edge. The high-quality one
    /// has negative lobes and rings — measured here as 15 and 240 at the profile's shoulders,
    /// against the default kernel's smoother 8 and 247. Literals are written out: if a future
    /// vImage changes these, the resampler changed and this project needs to know, which is exactly
    /// the signal a tolerance would swallow.
    @Test("the high-quality resampling flag is actually in force, not merely requested")
    func highQualityResamplingIsInForce() throws {
        let width = 16, height = 4
        var plane = [UInt8](repeating: 0, count: width * height)
        for y in 0..<height {
            for x in (width / 2)..<width { plane[y * width + x] = 255 }
        }

        let scaled = try Alpha.scale(plane, from: (width, height), to: (width * 4, height * 4))

        let row = 8, stride = width * 4
        #expect(
            scaled[row * stride + 24] == 15,
            "the high-quality kernel rings here; the default kernel gives 8")
        #expect(
            scaled[row * stride + 39] == 240,
            "and overshoots here; the default kernel gives 247")
        // The step itself is still a step — ringing must not have smeared it.
        #expect(scaled[row * stride + 28] == 0)
        #expect(scaled[row * stride + 34] == 255)
    }

    // MARK: - Recombination premultiplies exactly once

    @Test("recombination premultiplies once: opaque unchanged, transparent zeroed")
    func recombinePremultipliesOnce() throws {
        let split = try Alpha.split(try Self.halfTransparent())
        let alpha = try Alpha.scale(
            split.alpha, from: (split.width, split.height), to: (split.width, split.height))

        let out = try Alpha.recombine(split.colour, alpha: alpha)

        #expect(DecodeFixtures.pixel(of: out, x: 3, y: 3) == [0, 0, 255, 255])
        #expect(
            DecodeFixtures.pixel(of: out, x: 12, y: 3) == [0, 0, 0, 0],
            "alpha 0 must premultiply the dilated colour away, leaving a clean transparent pixel")
    }

    /// A half-transparent pixel is where a double premultiply would show, so the literal is
    /// written out rather than computed.
    @Test("a half-transparent pixel is premultiplied exactly once")
    func halfTransparentIsScaledOnce() throws {
        let buffer = try Decode.makeBuffer(width: 4, height: 1)
        CVPixelBufferLockBaseAddress(buffer, [])
        let base = try #require(CVPixelBufferGetBaseAddress(buffer))
            .assumingMemoryBound(to: UInt8.self)
        for x in 0..<4 {
            base[x * 4] = 0
            base[x * 4 + 1] = 0
            base[x * 4 + 2] = 200
            base[x * 4 + 3] = 255
        }
        CVPixelBufferUnlockBaseAddress(buffer, [])

        let out = try Alpha.recombine(buffer, alpha: [128, 128, 128, 128])

        // 200 * 128 / 255 = 100 (one premultiply). Twice would give 50.
        #expect(DecodeFixtures.pixel(of: out, x: 0, y: 0) == [0, 0, 100, 128])
    }

    @Test("an alpha plane of the wrong size is refused, not silently truncated")
    func sizeMismatchIsRefused() throws {
        let buffer = try Decode.makeBuffer(width: 4, height: 4)

        #expect(throws: Alpha.AlphaError.sizeMismatch(alpha: "5", colour: "4x4")) {
            _ = try Alpha.recombine(buffer, alpha: [UInt8](repeating: 255, count: 5))
        }
    }
}
