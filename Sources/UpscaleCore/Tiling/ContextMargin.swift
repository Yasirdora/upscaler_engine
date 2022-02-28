import Accelerate
import CoreGraphics
import CoreVideo
import Foundation

/// Measurement tools for tiling: how far a tile border contaminates its own output, and whether an
/// assembled image carries seams.
///
/// Both answers are measured against the **full-frame oracle** — the same image put through the
/// PyTorch reference at its native size (`tools/reference/`). That is what makes the seam question
/// answerable at all: if tiling were free, a tile's core would equal the oracle's pixels in the
/// same place, and boundary error would look exactly like interior error.
///
/// Everything is compared on the 8-bit product surface: the tiled result only exists as 8 bits, so
/// the oracle is put through the product's own output stage — clamp to [0,1], round to 8 bits —
/// before comparison. What is measured here is therefore tiling and inference, not the surface,
/// whose own cost is the 58.93 dB ceiling recorded in certification spec §11.
public enum ContextMargin {
    /// Context margin in input pixels, **per model, each derived by measurement**.
    ///
    /// Real-ESRGAN's own tiled inference uses `tile_pad ≈ 10`; that is a reference point and
    /// nothing more.
    ///
    /// **A single global constant is not sound.** The margin a model needs follows its receptive
    /// field, and these three differ by more than 40×:
    ///
    /// | model | field (input px) | margin |
    /// |---|---:|---:|
    /// | `realesr-general-x4v3` | 34.00 | 16 |
    /// | `RealESRGAN_x4plus` | 348.25 | 76 |
    /// | `RealESRGAN_x2plus` | **697.50** | **76** (derived 108) |
    ///
    /// x2plus has the *largest* field of the three — it `pixel_unshuffle`s by 2 before the same
    /// 23-block body as x4plus, so one body pixel spans a 2×2 input block and the same body radius
    /// reaches twice as far. **Its ring-decay derivation gives 108 px, and it runs at 76 anyway:**
    /// seam certification at both margins found no visible seam and no decisive statistical
    /// difference — 108 was mildly better on portraits at tile 256 and indistinguishable at tile
    /// 512, for +8.8 s on a 12 MP photo. **x2plus is therefore the one model running below its
    /// derived margin, knowingly.** The trigger to revisit is a reported seam on a portrait.
    ///
    /// Every value carries certification spec §4's ×2 safety factor, applied by the derivation
    /// itself. The derivations are reproducible with `tools/margin/`.
    static let derivedMargins: [String: Int] = [
        "RealESRGAN_x4plus": 76,
        "RealESRGAN_x2plus": 76,
        "realesr-general-x4v3": 16,
    ]

    /// Margins a derivation produced that the engine **does not run**.
    ///
    /// Recorded in code rather than only in prose so the divergence is a fact the test suite can
    /// hold, not a comment that rots. `RealESRGAN_x2plus`'s ring decay gives 108 px and it runs at
    /// 76 — it is the one model running below its own derivation, and that should be visible to
    /// anyone reading the table.
    static let derivedButNotAdopted: [String: Int] = [
        "RealESRGAN_x2plus": 108
    ]

    /// The derived margin for a model, or `nil` when none has been derived.
    ///
    /// **`nil` rather than a default, deliberately.** Any default is some other model's number,
    /// and that is precisely the defect this replaces: too small produces seams, and nothing
    /// throws — the failure is a picture with lines in it, discovered by a user. A model with no
    /// derived margin must not run until one is measured, so the caller is made to say what it
    /// does about that rather than silently inheriting a wrong value.
    public static func derivedMargin(for modelIdentifier: String) -> Int? {
        derivedMargins[modelIdentifier]
    }

    /// The margin for `RealESRGAN_x4plus`, from before margins were per model.
    ///
    /// Retained only for call sites that are not yet model-aware.
    /// **Do not use it for new work** — it is x4plus's number and correct for nothing else.
    @available(*, deprecated, message: "use derivedMargin(for:) — this is x4plus's number")
    public static let derivedContextMargin = 76

    // MARK: - Results

    /// Level A numbers over a region, in the units certification spec §4 states thresholds in.
    public struct Metrics: Sendable, Equatable {
        public let meanAbs255: Double
        public let maxAbs255: Double
        public let psnr: Double
        public let withinToleranceFraction: Double
        public let sampleCount: Int
    }

    /// One step of the pad-and-probe sweep: what the error looks like when `margin` pixels of the
    /// model's output are discarded on every side.
    public struct Probe: Sendable, Equatable {
        public let margin: Int
        /// Mean absolute error over the retained core.
        public let coreMeanAbs255: Double
        /// Mean absolute error of the single-pixel frame at exactly this distance from the border.
        public let ringMeanAbs255: Double
        public let ringMaxAbs255: Double

        public init(
            margin: Int, coreMeanAbs255: Double, ringMeanAbs255: Double, ringMaxAbs255: Double
        ) {
            self.margin = margin
            self.coreMeanAbs255 = coreMeanAbs255
            self.ringMeanAbs255 = ringMeanAbs255
            self.ringMaxAbs255 = ringMaxAbs255
        }
    }

    /// The seam question, answered per method per image.
    public struct SeamProfile: Sendable {
        public let overall: Metrics
        public let boundary: Metrics
        public let interior: Metrics
        /// The number the no-seams criterion is stated in: boundary error ÷ interior error.
        public var ratio: Double {
            interior.meanAbs255 > 0 ? boundary.meanAbs255 / interior.meanAbs255 : .infinity
        }
    }

    // MARK: - Sampling

    /// Absolute per-channel differences of two BGRA buffers, as one float plane of RGB samples.
    ///
    /// Alpha is dropped: it is a compositing artefact, not image content. One vDSP pass per row.
    public static func absoluteDifference(
        _ candidate: CVPixelBuffer, _ reference: CVPixelBuffer
    ) throws -> (values: [Float], width: Int, height: Int) {
        let width = min(CVPixelBufferGetWidth(candidate), CVPixelBufferGetWidth(reference))
        let height = min(CVPixelBufferGetHeight(candidate), CVPixelBufferGetHeight(reference))
        guard width > 0, height > 0 else { throw TilingError.emptyImage }

        CVPixelBufferLockBaseAddress(candidate, .readOnly)
        CVPixelBufferLockBaseAddress(reference, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(reference, .readOnly)
            CVPixelBufferUnlockBaseAddress(candidate, .readOnly)
        }
        guard let left = PixelView(candidate), let right = PixelView(reference) else {
            throw TilingError.stitchFailed("could not lock buffers for comparison")
        }

        let samples = width * 4
        var a = [Float](repeating: 0, count: samples)
        var b = [Float](repeating: 0, count: samples)
        var row = [Float](repeating: 0, count: samples)
        // Three RGB samples per pixel; the alpha lane is skipped by a strided copy.
        var values = [Float](repeating: 0, count: width * height * 3)

        for y in 0..<height {
            let leftRow = left.base.advanced(by: y * left.rowBytes).assumingMemoryBound(to: UInt8.self)
            let rightRow = right.base.advanced(by: y * right.rowBytes).assumingMemoryBound(to: UInt8.self)
            vDSP_vfltu8(leftRow, 1, &a, 1, vDSP_Length(samples))
            vDSP_vfltu8(rightRow, 1, &b, 1, vDSP_Length(samples))
            vDSP_vsub(b, 1, a, 1, &row, 1, vDSP_Length(samples))
            vDSP_vabs(row, 1, &row, 1, vDSP_Length(samples))
            // Strided gather of B, G, R out of the BGRA row, skipping alpha. `vDSP_vsadd` with a
            // zero addend is the supported strided copy; cblas_scopy is deprecated on this SDK.
            row.withUnsafeBufferPointer { source in
                values.withUnsafeMutableBufferPointer { destination in
                    guard let sourceBase = source.baseAddress,
                        let destinationBase = destination.baseAddress
                    else { return }
                    var zero: Float = 0
                    for channel in 0..<3 {
                        vDSP_vsadd(
                            sourceBase.advanced(by: channel), 4, &zero,
                            destinationBase.advanced(by: y * width * 3 + channel), 3,
                            vDSP_Length(width))
                    }
                }
            }
        }
        return (values, width, height)
    }

    /// Level A metrics over the samples selected by `mask` (nil means every sample).
    public static func metrics(values: [Float], mask: [Float]?, tolerance255: Double = 4.0) -> Metrics {
        let selected: [Float]
        if let mask {
            var product = [Float](repeating: 0, count: values.count)
            vDSP_vmul(values, 1, mask, 1, &product, 1, vDSP_Length(values.count))
            selected = product
        } else {
            selected = values
        }

        var count = Double(values.count)
        var sum: Float = 0
        var maximum: Float = 0
        if let mask {
            var maskSum: Float = 0
            vDSP_sve(mask, 1, &maskSum, vDSP_Length(mask.count))
            count = Double(maskSum)
            // Masked-out samples are zero and cannot raise the maximum.
            vDSP_maxv(selected, 1, &maximum, vDSP_Length(selected.count))
        } else {
            vDSP_maxv(selected, 1, &maximum, vDSP_Length(selected.count))
        }
        vDSP_sve(selected, 1, &sum, vDSP_Length(selected.count))
        guard count > 0 else {
            return Metrics(
                meanAbs255: 0, maxAbs255: 0, psnr: .infinity, withinToleranceFraction: 1, sampleCount: 0)
        }

        var squares = [Float](repeating: 0, count: selected.count)
        vDSP_vsq(selected, 1, &squares, 1, vDSP_Length(selected.count))
        var squareSum: Float = 0
        vDSP_sve(squares, 1, &squareSum, vDSP_Length(squares.count))

        let mean = Double(sum) / count
        let meanSquare = Double(squareSum) / count
        let psnr = meanSquare > 0 ? 10.0 * log10((255.0 * 255.0) / meanSquare) : Double.infinity

        // Fraction of *samples* within tolerance, counted with a threshold-and-sum, not a loop.
        //
        // Certification spec §4 states the criterion inclusively — "max-abs **<= 4/255** on >= 99%
        // of pixels" — so a sample sitting exactly on the tolerance is inside it. `vDSP_vthrsc`
        // classifies on `A[n] >= B`, which puts exact-boundary samples on the wrong side and
        // understated this column by 0.54 pp on real data. Thresholding at the next representable
        // float makes `A >= nextUp(t)` exactly equivalent to `A > t` for every finite float, so
        // what remains is `A <= t`: the spec's criterion, with no epsilon and nothing to tune.
        var threshold = Float(tolerance255).nextUp
        var indicator = [Float](repeating: 0, count: selected.count)
        vDSP_vthrsc(selected, 1, &threshold, [1.0, -1.0], &indicator, 1, vDSP_Length(selected.count))
        // vthrsc emits +1 for samples that exceed the tolerance and -1 for the rest; map to 1/0.
        var negativeHalf: Float = -0.5
        var half: Float = 0.5
        vDSP_vsmsa(indicator, 1, &negativeHalf, &half, &indicator, 1, vDSP_Length(indicator.count))
        if let mask {
            vDSP_vmul(indicator, 1, mask, 1, &indicator, 1, vDSP_Length(indicator.count))
        }
        var within: Float = 0
        vDSP_sve(indicator, 1, &within, vDSP_Length(indicator.count))

        return Metrics(
            meanAbs255: mean, maxAbs255: Double(maximum), psnr: psnr,
            withinToleranceFraction: Double(within) / count, sampleCount: Int(count)
        )
    }

    /// A copy of one region of a buffer. Used by the probe to line an oracle's pixels up with a
    /// tile's output; a single vImage copy, no per-pixel work.
    public static func crop(_ buffer: CVPixelBuffer, to rect: CGRect) throws -> CVPixelBuffer {
        let region = IntRect(rect)
        let bounds = IntRect(
            x: 0, y: 0, width: CVPixelBufferGetWidth(buffer), height: CVPixelBufferGetHeight(buffer))
        let clipped = region.intersection(bounds)
        guard clipped == region, !region.isEmpty else {
            throw TilingError.stitchFailed("crop \(region) does not fit inside \(bounds)")
        }

        let output = try SequentialTilingEngine.allocate(width: region.width, height: region.height)
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        CVPixelBufferLockBaseAddress(output, [])
        defer {
            CVPixelBufferUnlockBaseAddress(output, [])
            CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
        }
        guard let source = PixelView(buffer), let destination = PixelView(output) else {
            throw TilingError.stitchFailed("could not lock buffers for cropping")
        }
        var input = source.sub(
            x: region.x, y: region.y, width: region.width, height: region.height
        ).vImage
        var target = destination.vImage
        let error = vImageCopyBuffer(&input, &target, PixelView.bytesPerPixel, vImage_Flags(kvImageNoFlags))
        guard error == kvImageNoError else {
            throw TilingError.stitchFailed("vImageCopyBuffer failed with \(error)")
        }
        return output
    }

    /// Build the fixed-shape input tile for a window, reflecting where it leaves the image.
    /// Exposed so the derivation tool feeds the model exactly what the engine would.
    public static func paddedTile(
        from image: CVPixelBuffer, window: CGRect, tileSize: Int
    ) throws -> CVPixelBuffer {
        let tile = try SequentialTilingEngine.allocate(width: tileSize, height: tileSize)
        try SequentialTilingEngine.fillPaddedTile(source: image, window: IntRect(window), into: tile)
        return tile
    }

    // MARK: - Pad and probe

    /// Error of the single-pixel frame at each distance from the window border, plus the running
    /// error of everything inside it.
    ///
    /// One inference answers the whole sweep: retaining a core inset by `m` from a fixed window is
    /// exactly what padding that core by `m` would have produced, so the padding sweep the work
    /// order asks for is read off one tile rather than paid for `m` times.
    static func ringProfile(
        values: [Float], width: Int, height: Int
    ) -> (ringMean: [Double], ringMax: [Double], coreMean: [Double]) {
        let maximumDistance = min(width, height) / 2
        guard maximumDistance > 0 else { return ([], [], []) }

        var ringSums = [Double](repeating: 0, count: maximumDistance)
        var ringCounts = [Double](repeating: 0, count: maximumDistance)
        var ringMaxima = [Double](repeating: 0, count: maximumDistance)

        values.withUnsafeBufferPointer { buffer in
            guard let base = buffer.baseAddress else { return }
            let rowSamples = width * 3
            for distance in 0..<maximumDistance {
                let low = distance
                let highRow = height - 1 - distance
                let highColumn = width - 1 - distance
                let span = width - 2 * distance
                guard span > 0 else { continue }

                var sum: Float = 0
                var maximum: Float = 0
                var partial: Float = 0
                var partialMaximum: Float = 0

                // The two horizontal edges of the frame, contiguous in memory.
                for row in [low, highRow] where row != highRow || low != highRow {
                    let start = base.advanced(by: row * rowSamples + low * 3)
                    vDSP_sve(start, 1, &partial, vDSP_Length(span * 3))
                    vDSP_maxv(start, 1, &partialMaximum, vDSP_Length(span * 3))
                    sum += partial
                    maximum = max(maximum, partialMaximum)
                    ringCounts[distance] += Double(span * 3)
                }
                if low == highRow {  // a one-row frame was counted once above
                    ringCounts[distance] = Double(span * 3)
                }

                // The two vertical edges, minus the corners already counted, read with a stride.
                let innerRows = (low + 1)..<highRow
                if !innerRows.isEmpty {
                    for column in [low, highColumn] where column != highColumn || low != highColumn {
                        for channel in 0..<3 {
                            let start = base.advanced(
                                by: (low + 1) * rowSamples + column * 3 + channel)
                            vDSP_sve(start, vDSP_Stride(rowSamples), &partial, vDSP_Length(innerRows.count))
                            vDSP_maxv(
                                start, vDSP_Stride(rowSamples), &partialMaximum, vDSP_Length(innerRows.count))
                            sum += partial
                            maximum = max(maximum, partialMaximum)
                            ringCounts[distance] += Double(innerRows.count)
                        }
                    }
                }
                ringSums[distance] = Double(sum)
                ringMaxima[distance] = Double(maximum)
            }
        }

        let ringMean = zip(ringSums, ringCounts).map { $1 > 0 ? $0 / $1 : 0 }
        // Everything inside distance d is the suffix of the rings, accumulated from the centre out.
        var coreMean = [Double](repeating: 0, count: maximumDistance)
        var suffixSum = 0.0
        var suffixCount = 0.0
        for distance in stride(from: maximumDistance - 1, through: 0, by: -1) {
            suffixSum += ringSums[distance]
            suffixCount += ringCounts[distance]
            coreMean[distance] = suffixCount > 0 ? suffixSum / suffixCount : 0
        }
        return (ringMean, ringMaxima, coreMean)
    }

    /// The pad-and-probe sweep, in input-pixel margins.
    public static func probe(
        candidate: CVPixelBuffer, oracle: CVPixelBuffer, scaleFactor: Int, margins: [Int]
    ) throws -> [Probe] {
        let (values, width, height) = try absoluteDifference(candidate, oracle)
        let profile = ringProfile(values: values, width: width, height: height)
        return margins.compactMap { margin in
            let distance = margin * scaleFactor
            guard distance < profile.ringMean.count else { return nil }
            return Probe(
                margin: margin,
                coreMeanAbs255: profile.coreMean[distance],
                ringMeanAbs255: profile.ringMean[distance],
                ringMaxAbs255: profile.ringMax[distance]
            )
        }
    }

    /// The derivation: the first distance at which the border's influence has decayed below one
    /// 8-bit level, doubled for safety.
    ///
    /// The criterion is stated on **mean** absolute error, not maximum, and that is a measurement
    /// result rather than a preference: Level A recorded a max-abs floor of 3–5/255 between a Core
    /// ML artifact and the PyTorch oracle even with full context, so a max-abs criterion of 1/255
    /// can never be met by any margin and would return "no margin suffices" for a model that is in
    /// fact fine. Mean absolute error has a floor of ≈0.27/255, comfortably under the threshold, so
    /// it can actually discriminate.
    public static func derive(
        from probes: [Probe], threshold255: Double = 1.0, safetyFactor: Int = 2
    ) -> (margin: Int, crossing: Int)? {
        guard let crossing = probes.first(where: { $0.ringMeanAbs255 < threshold255 })?.margin else {
            return nil
        }
        return (crossing * safetyFactor, crossing)
    }

    // MARK: - Seam certification

    /// Boundary-versus-interior error against the full-frame oracle.
    ///
    /// A seam is a *local* defect, so a global metric cannot see one: an image can pass Level A
    /// comfortably and still have a visible line every 192 pixels. Splitting the same error field
    /// into the pixels near tile boundaries and everything else is what makes the defect legible.
    public static func seamProfile(
        candidate: CVPixelBuffer, oracle: CVPixelBuffer, grid: TileGrid, boundaryRadius: Int = 4
    ) throws -> SeamProfile {
        let (values, width, height) = try absoluteDifference(candidate, oracle)

        var seamsX: Set<Int> = []
        var seamsY: Set<Int> = []
        for tile in grid.tiles {
            let rect = IntRect(tile.validRect)
            if rect.maxX < width { seamsX.insert(rect.maxX) }
            if rect.maxY < height { seamsY.insert(rect.maxY) }
        }

        var mask = [Float](repeating: 0, count: values.count)
        let rowSamples = width * 3
        var one: Float = 1
        mask.withUnsafeMutableBufferPointer { buffer in
            guard let base = buffer.baseAddress else { return }
            for seam in seamsY {
                let rows = max(0, seam - boundaryRadius)...min(height - 1, seam + boundaryRadius)
                for row in rows {
                    vDSP_vfill(&one, base.advanced(by: row * rowSamples), 1, vDSP_Length(rowSamples))
                }
            }
            for seam in seamsX {
                let columns = max(0, seam - boundaryRadius)...min(width - 1, seam + boundaryRadius)
                for column in columns {
                    // One strided run per channel: three samples per pixel, `height` rows apart.
                    // A single run of `height * 3` would stride past the end of the buffer.
                    for channel in 0..<3 {
                        vDSP_vfill(
                            &one, base.advanced(by: column * 3 + channel), vDSP_Stride(rowSamples),
                            vDSP_Length(height))
                    }
                }
            }
        }

        // The interior is the complement, built by subtraction rather than a second scan.
        var interiorMask = [Float](repeating: 1, count: mask.count)
        vDSP_vsub(mask, 1, interiorMask, 1, &interiorMask, 1, vDSP_Length(mask.count))

        return SeamProfile(
            overall: metrics(values: values, mask: nil),
            boundary: metrics(values: values, mask: mask),
            interior: metrics(values: values, mask: interiorMask)
        )
    }
}
