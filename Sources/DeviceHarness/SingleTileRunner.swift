import CoreGraphics
import CoreVideo
import Foundation
import ImageIO
import UniformTypeIdentifiers
import UpscaleCore

#if canImport(Darwin)
    import Darwin
#endif

/// Outcome of one device-side tile run.
///
/// The timing and memory fields are **best-effort and informational**. They exist so a human can
/// see that a run completed and roughly what it cost, not to support any claim. Benchmark
/// methodology — cold/warm/sustained/throttled, medians of ten, recorded device state — is out of
/// scope here, and a number produced here must never be quoted as a benchmark result.
///
/// **Two memory figures, both high-water marks, deliberately not one:** `peakMemoryBytes` is
/// `resident_size_max` and `peakFootprintBytes` is `phys_footprint`'s peak. The system enforces the
/// second; the first is what earlier single-tile runs were recorded in. Replacing rather than
/// adding would have made every earlier device number incomparable with the next one, so the RSS
/// field keeps its name, its units and its value.
public struct SingleTileResult: Sendable {
    public let success: Bool
    public let outputSize: CGSize
    public let wallClockSeconds: TimeInterval  // informational only — not a benchmark
    /// Model load, split out from inference. On a cold run this is a full `MLModel.compileModel`
    /// and dominates everything; on a warm one it is a cache read. Reporting only the total is how
    /// a compile gets mistaken for a slow model — or a warm run for a fast one.
    public let loadSeconds: TimeInterval
    public let inferenceSeconds: TimeInterval
    /// Whether a compiled model was already on disk when the run started, probed before loading.
    public let warmStart: Bool
    public let peakMemoryBytes: UInt64?  // best-effort; nil where unavailable
    /// Peak `phys_footprint` — a **process high-water mark**, not a point sample. See
    /// `SingleTileRunner.peakFootprintBytes()` for what that costs a reader who forgets it.
    public let peakFootprintBytes: UInt64?  // best-effort; nil where unavailable
    public let computeUnits: String
    public let errorDescription: String?

    public init(
        success: Bool,
        outputSize: CGSize,
        wallClockSeconds: TimeInterval,
        loadSeconds: TimeInterval = 0,
        inferenceSeconds: TimeInterval = 0,
        warmStart: Bool = false,
        peakMemoryBytes: UInt64?,
        peakFootprintBytes: UInt64?,
        computeUnits: String,
        errorDescription: String?
    ) {
        self.success = success
        self.outputSize = outputSize
        self.wallClockSeconds = wallClockSeconds
        self.loadSeconds = loadSeconds
        self.inferenceSeconds = inferenceSeconds
        self.warmStart = warmStart
        self.peakMemoryBytes = peakMemoryBytes
        self.peakFootprintBytes = peakFootprintBytes
        self.computeUnits = computeUnits
        self.errorDescription = errorDescription
    }
}

/// A completed run: the result fields plus the produced pixels, encoded as PNG.
///
/// The image travels as `Data` rather than `CGImage` deliberately: `CGImage` is not `Sendable`, and
/// the app renders on the main actor. Handing back bytes keeps the boundary clean under Swift 6.
public struct SingleTileRun: Sendable {
    public let result: SingleTileResult
    public let outputPNG: Data?

    public init(result: SingleTileResult, outputPNG: Data?) {
        self.result = result
        self.outputPNG = outputPNG
    }
}

/// Runs exactly one fixed-shape tile through the Core ML backend on whatever device this is.
///
/// This answers one question: *does the same model execute and produce sane output on an A14-class
/// device?* Nothing more — no tiling, no memory budgeting, no thermal or energy measurement.
public enum SingleTileRunner {
    /// Missing resources are a first-class outcome, not a crash: the model and tile are dropped in
    /// by hand (they are gitignored), so "you forgot to copy them" must be legible on-device. The
    /// file name travels with the case because with three shapes in the ladder, "model not bundled"
    /// is only actionable if it names the artifact.
    public enum ResourceError: Error, Equatable {
        case modelMissing(fileName: String)
        case tileMissing(fileName: String)

        public var message: String {
            switch self {
            case .modelMissing(let fileName):
                return
                    "Model not bundled. Copy \(fileName) into DeviceHarnessApp/HarnessArtifacts/ and rebuild."
            case .tileMissing(let fileName):
                return
                    "Test tile not bundled. Copy \(fileName) into DeviceHarnessApp/HarnessArtifacts/ and rebuild."
            }
        }
    }

    public static let tileSize = HarnessShape.default256.tileSize
    /// The 256 default's factor, kept for callers that predate per-shape scales.
    public static let scaleFactor = HarnessShape.default256.scaleFactor

    /// Descriptor for a bundled artifact. fp16; the shape supplies the identifier, the fixed input
    /// geometry AND the scale factor — the last because a ×2 model run as though it were ×4
    /// mis-sizes every output surface and memory plan downstream, silently.
    public static func descriptor(
        shape: HarnessShape = .default256, modelURL: URL
    ) -> ModelDescriptor {
        ModelDescriptor(
            identifier: shape.descriptorIdentifier,
            scaleFactor: shape.scaleFactor,
            contextMargin: shape.derivedMargin ?? 76,
            packageURL: modelURL,
            inputTileSize: shape.tileSize,
            precision: .float16
        )
    }

    public static func run(
        shape: HarnessShape = .default256,
        modelURL: URL?,
        tileURL: URL?,
        computeUnits: ComputeUnits = .all
    ) async -> SingleTileRun {
        guard let modelURL else {
            let missing = ResourceError.modelMissing(fileName: shape.modelFileName)
            return failure(missing.message, computeUnits: computeUnits)
        }
        guard let tileURL else {
            let missing = ResourceError.tileMissing(fileName: shape.tileFileName)
            return failure(missing.message, computeUnits: computeUnits)
        }

        func seconds(_ duration: Duration) -> TimeInterval {
            TimeInterval(
                Double(duration.components.attoseconds) / 1e18
                    + Double(duration.components.seconds))
        }

        let clock = ContinuousClock()
        let started = clock.now
        let plan = descriptor(shape: shape, modelURL: modelURL)
        // Probed BEFORE loading: afterwards the cache always exists and the answer is always yes.
        let warm = CoreMLBackend.cachedModelURL(for: plan) != nil
        do {
            let tile = try loadPixelBuffer(from: tileURL)
            let backend = CoreMLBackend()
            let loadStarted = clock.now
            try await backend.load(
                plan, context: InferenceContext(computeUnits: computeUnits))
            let loadEnded = clock.now
            let output = try await backend.predict(tile)
            let finished = clock.now
            let elapsed = seconds(finished - started)
            let loadElapsed = seconds(loadEnded - loadStarted)
            let inferenceElapsed = seconds(finished - loadEnded)

            let size = CGSize(
                width: CVPixelBufferGetWidth(output), height: CVPixelBufferGetHeight(output))
            let png = pngData(from: output)
            return SingleTileRun(
                result: SingleTileResult(
                    success: true,
                    outputSize: size,
                    wallClockSeconds: elapsed,
                    loadSeconds: loadElapsed,
                    inferenceSeconds: inferenceElapsed,
                    warmStart: warm,
                    peakMemoryBytes: peakMemoryBytes(),
                    peakFootprintBytes: peakFootprintBytes(),
                    computeUnits: computeUnits.rawValue,
                    errorDescription: nil
                ),
                outputPNG: png
            )
        } catch {
            let elapsed = TimeInterval(
                Double((clock.now - started).components.attoseconds) / 1e18
                    + Double((clock.now - started).components.seconds))
            return failure(describe(error), computeUnits: computeUnits, elapsed: elapsed)
        }
    }

    /// Error text a human can act on. Typed engine errors keep their own description; anything
    /// else is reported rather than swallowed.
    public static func describe(_ error: Error) -> String {
        if let inferenceError = error as? InferenceError { return "\(inferenceError)" }
        if let resourceError = error as? ResourceError { return resourceError.message }
        return error.localizedDescription
    }

    public static func failure(
        _ message: String,
        computeUnits: ComputeUnits,
        elapsed: TimeInterval = 0
    ) -> SingleTileRun {
        SingleTileRun(
            result: SingleTileResult(
                success: false,
                outputSize: .zero,
                wallClockSeconds: elapsed,
                peakMemoryBytes: peakMemoryBytes(),
                peakFootprintBytes: peakFootprintBytes(),
                computeUnits: computeUnits.rawValue,
                errorDescription: message
            ),
            outputPNG: nil
        )
    }

    // MARK: - Best-effort instrumentation

    /// Peak resident size from `mach_task_basic_info`. Best-effort by contract: `nil` when the
    /// kernel call fails, never a fabricated number.
    public static func peakMemoryBytes() -> UInt64? {
        #if canImport(Darwin)
            var info = mach_task_basic_info()
            var count = mach_msg_type_number_t(
                MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
            let result = withUnsafeMutablePointer(to: &info) { pointer in
                pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
                    task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), rebound, &count)
                }
            }
            return result == KERN_SUCCESS ? info.resident_size_max : nil
        #else
            return nil
        #endif
    }

    /// Peak `phys_footprint` from `task_vm_info` — the metric the system actually enforces (WWDC22
    /// session 10106, WWDC18 session 416), read with the same `task_info(TASK_VM_INFO)` call
    /// `MemoryProbeView.physicalFootprint()` already uses.
    ///
    /// **This is a process high-water mark, not an instantaneous reading.** The probe samples
    /// `phys_footprint` (what the process occupies *now*, four times a second, because a probe
    /// watching a countdown wants exactly that); this function reads `ledger_phys_footprint_peak`
    /// (the largest footprint the process has reached since launch) because the field it sits
    /// beside, `resident_size_max`, is a high-water mark too. Comparing a peak against a point
    /// sample is the mistake this doc comment exists to prevent, and the same property is why
    /// single-shape footprint measurement runs **one launch per run**: a second run in the same
    /// process reports the first run's peak whenever the first was higher.
    ///
    /// Best-effort by contract, exactly like the RSS reading above: `nil` when the kernel call
    /// fails, when the running kernel predates the `task_vm_info` revision that added the field, or
    /// on a non-Darwin host — never a fabricated number, and never a zero dressed as one.
    public static func peakFootprintBytes() -> UInt64? {
        #if canImport(Darwin)
            // `ledger_phys_footprint_peak` arrived in a later `task_vm_info` revision than
            // `phys_footprint`. Ask for the whole struct and let the kernel report how much of it
            // it filled: anything short of this field's own end means the field was never written,
            // and the zero left there by the initializer must not be read as a measurement.
            guard
                let fieldOffset = MemoryLayout<task_vm_info_data_t>.offset(
                    of: \task_vm_info_data_t.ledger_phys_footprint_peak)
            else { return nil }
            let requiredCount = mach_msg_type_number_t(
                (fieldOffset + MemoryLayout<Int64>.size) / MemoryLayout<natural_t>.size)

            var info = task_vm_info_data_t()
            var count = mach_msg_type_number_t(
                MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
            let result = withUnsafeMutablePointer(to: &info) { pointer in
                pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
                    task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), rebound, &count)
                }
            }
            guard result == KERN_SUCCESS, count >= requiredCount,
                info.ledger_phys_footprint_peak > 0
            else { return nil }
            return UInt64(info.ledger_phys_footprint_peak)
        #else
            return nil
        #endif
    }

    /// `phys_footprint` **now** — a point sample, not the high-water mark above it.
    ///
    /// The peak reader beside this one is a *process* high-water mark, which makes it monotone
    /// across a run series: in ten consecutive runs it reports the largest footprint the process
    /// ever reached, not what run 7 cost. One launch per run avoids that; certification spec §6's
    /// protocol is ten runs in one launch and cannot, so the per-run signal has to come from a
    /// point sample. That is what this is, and it is the same reading
    /// `MemoryProbeView.physicalFootprint()` takes four times a second.
    ///
    /// Deliberately written out rather than factored together with `peakFootprintBytes()` above, so
    /// the reviewed peak reader stays exactly as it is. Same `task_info(TASK_VM_INFO)` call, same
    /// offset guard, same best-effort contract — `nil` on failure, never a fabricated number and
    /// never a zero dressed as one.
    public static func currentFootprintBytes() -> UInt64? {
        #if canImport(Darwin)
            guard
                let fieldOffset = MemoryLayout<task_vm_info_data_t>.offset(
                    of: \task_vm_info_data_t.phys_footprint)
            else { return nil }
            let requiredCount = mach_msg_type_number_t(
                (fieldOffset + MemoryLayout<Int64>.size) / MemoryLayout<natural_t>.size)

            var info = task_vm_info_data_t()
            var count = mach_msg_type_number_t(
                MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
            let result = withUnsafeMutablePointer(to: &info) { pointer in
                pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
                    task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), rebound, &count)
                }
            }
            guard result == KERN_SUCCESS, count >= requiredCount, info.phys_footprint > 0
            else { return nil }
            return UInt64(info.phys_footprint)
        #else
            return nil
        #endif
    }

    // MARK: - Utilitarian image I/O (no colour management — ImagePipeline owns that subsystem)

    static func loadPixelBuffer(from url: URL) throws -> CVPixelBuffer {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
            let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            throw InferenceError.predictionFailed("could not decode the bundled tile at \(url.path)")
        }
        var buffer: CVPixelBuffer?
        let attributes: [CFString: Any] = [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, image.width, image.height, kCVPixelFormatType_32BGRA,
            attributes as CFDictionary, &buffer)
        guard status == kCVReturnSuccess, let pixelBuffer = buffer else {
            throw InferenceError.predictionFailed("could not allocate the input buffer")
        }
        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }
        guard
            let context = CGContext(
                data: CVPixelBufferGetBaseAddress(pixelBuffer),
                width: image.width, height: image.height, bitsPerComponent: 8,
                bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer),
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
                    | CGBitmapInfo.byteOrder32Little.rawValue)
        else {
            throw InferenceError.predictionFailed("could not create a drawing context")
        }
        context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        return pixelBuffer
    }

    static func pngData(from buffer: CVPixelBuffer) -> Data? {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard
            let context = CGContext(
                data: CVPixelBufferGetBaseAddress(buffer),
                width: CVPixelBufferGetWidth(buffer), height: CVPixelBufferGetHeight(buffer),
                bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
                    | CGBitmapInfo.byteOrder32Little.rawValue),
            let image = context.makeImage()
        else { return nil }

        let data = NSMutableData()
        guard
            let destination = CGImageDestinationCreateWithData(
                data as CFMutableData, UTType.png.identifier as CFString, 1, nil)
        else { return nil }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { return nil }
        return data as Data
    }
}
