import CoreML
import CoreVideo
import Foundation

extension ComputeUnits {
    /// The mapping the contract's comment promises. Tested directly: the unit test asserts this
    /// function, never Core ML's dispatch behaviour, which is empirical and belongs to
    /// benchmarking.
    var mlComputeUnits: MLComputeUnits {
        switch self {
        case .all: return .all
        case .cpuAndGPU: return .cpuAndGPU
        case .cpuAndNeuralEngine: return .cpuAndNeuralEngine
        case .cpuOnly: return .cpuOnly
        }
    }
}

/// Core ML implementation of `InferenceBackend`, contract version 1.
///
/// Concurrency: a nonisolated final class guarded by a lock, not an actor. `CVPixelBuffer` is
/// explicitly `@_nonSendable` in the SDK, so an actor would force region-transfer semantics on
/// every caller of `predict` — the buffer would become unusable to the caller afterwards. A
/// nonisolated class crosses no isolation boundary, so the contract's signatures work as written.
/// `@unchecked Sendable` is therefore load-bearing and honest: `MLModel` is not `Sendable`, and the
/// lock is what makes sharing this instance safe.
///
/// This backend is deliberately minimal: one fixed-shape tile, no tiling (`TilingEngine`), no
/// memory budgeting (`MemoryBudget`), no timings (`Benchmarking`).
public final class CoreMLBackend: InferenceBackend, @unchecked Sendable {
    private struct LoadedModel {
        let model: MLModel
        let descriptor: ModelDescriptor
        let inputName: String
        let outputName: String
    }

    private let lock = NSLock()
    private var loaded: LoadedModel?

    public init() {}

    // MARK: - Loading

    public func load(_ model: ModelDescriptor, context: InferenceContext) async throws {
        guard FileManager.default.fileExists(atPath: model.packageURL.path) else {
            throw InferenceError.modelNotFound(model.packageURL)
        }

        let configuration = MLModelConfiguration()
        configuration.computeUnits = context.computeUnits.mlComputeUnits
        configuration.modelDisplayName = model.identifier

        let modelURL: URL
        if Self.usesCompiledCache(for: model.packageURL) {
            // Read the cache before compiling.
            if let hit = Self.cachedModelURL(for: model) {
                modelURL = hit
            } else {
                let compiled: URL
                do {
                    compiled = try await MLModel.compileModel(at: model.packageURL)
                } catch {
                    throw InferenceError.modelLoadFailed(
                        "compilation failed: \(error.localizedDescription)")
                }
                modelURL = try Self.moveIntoCaches(compiled, for: model)
            }
        } else {
            // Pre-compiled bundle (.mlmodelc or .mlmodelc.encrypted) supplied by an app.
            // The cache does not apply: the decrypted form is securely held in memory.
            modelURL = model.packageURL
        }

        let mlModel: MLModel
        do {
            mlModel = try await MLModel.load(contentsOf: modelURL, configuration: configuration)
        } catch {
            throw Self.classify(loadError: error)
        }

        let description = mlModel.modelDescription
        let inputs = description.inputDescriptionsByName.keys.sorted()
        let outputs = description.outputDescriptionsByName.keys.sorted()
        guard inputs.count == 1, outputs.count == 1,
            let inputName = inputs.first, let outputName = outputs.first
        else {
            throw InferenceError.modelLoadFailed(
                "expected exactly one input and one output, got inputs \(inputs) outputs \(outputs)"
            )
        }

        lock.withLock {
            loaded = LoadedModel(
                model: mlModel, descriptor: model, inputName: inputName, outputName: outputName
            )
        }
    }

    /// The cache directory name for a descriptor: identifier **and** input geometry.
    ///
    /// Keying on the identifier alone would make the cache single-shape: with 256/384/512
    /// artifacts, any runtime that switches among them under one identifier would evict the
    /// previous shape on every switch (a compile spike exactly when memory is short) and could
    /// instantiate the wrong shape's weights through the window between the move and
    /// `MLModel(contentsOf:)`. Geometry in the key removes both, because two shapes no longer name
    /// the same directory.
    ///
    /// Pure by design — no file I/O, no `FileManager`, no descriptor field that is not part of the
    /// cache's semantic identity — so the derivation is testable on its own. `packageURL` is
    /// deliberately excluded: it is a location, not an identity, and keying on it would
    /// double-cache two copies of the same weights.
    ///
    /// Identifiers that already end in `_<N>x<N>` (the CLI's artifact stems) yield doubled names
    /// such as `RealESRGAN_x4plus_256x256_256x256.mlmodelc`. Accepted deliberately: doubled but
    /// unique, stable, and opaque to users — stripping suffixes would reintroduce the ambiguity
    /// this key exists to remove.
    ///
    /// - Parameter descriptor: the model whose compiled form is being cached.
    /// - Returns: the `.mlmodelc` directory name, unique per identifier + input geometry.
    static func compiledModelName(for descriptor: ModelDescriptor) -> String {
        "\(descriptor.identifier)_\(descriptor.inputTileSize)x\(descriptor.inputTileSize).mlmodelc"
    }

    /// Whether the compiled-model cache applies to this artifact.
    ///
    /// Only a source `.mlpackage` is compiled and cached. A pre-compiled `.mlmodelc` — including an
    /// `.mlmodelc.encrypted` an app ships — is loaded in place: **an encrypted model's decrypted
    /// form is not ours to hold**, and copying one into Caches would move it outside the boundary
    /// Core ML keeps it in. The cache's per-launch saving therefore does not apply to encrypted
    /// assets, which is a cost of encryption and not a defect.
    static func usesCompiledCache(for packageURL: URL) -> Bool {
        packageURL.pathExtension == "mlpackage"
    }

    /// Turn a Core ML load failure into something a person can act on.
    ///
    /// Extracted from `load`'s catch so it can be tested: the three outcomes are indistinguishable
    /// from outside without a real encrypted model, a real network failure and a real corrupt
    /// asset, and code whose branches cannot be asserted is code whose branches are decorative.
    /// This project has now been bitten by that three times.
    ///
    /// The distinction that matters is between **a network failure on first run** — the user waits
    /// or connects, and the model works forever after — and **a broken asset**, where waiting does
    /// nothing. Reporting the first as the second sends the user to the wrong action.
    static func classify(loadError error: any Error) -> InferenceError {
        let nsError = error as NSError
        let description = error.localizedDescription.lowercased()

        // MLModelErrorModelDecryptionKeyFetchFailed == 8. The string checks are a fallback: the
        // code is the reliable signal, and the wording is Apple's to change.
        let isKeyFetch =
            (nsError.domain == MLModelErrorDomain && nsError.code == 8)
            || description.contains("network")
            || description.contains("connection")
            || description.contains("key fetch")
        if isKeyFetch { return .modelRequiresNetworkOnFirstRun }

        let isCorrupt =
            (nsError.domain == MLModelErrorDomain && nsError.code == 3)
            || description.contains("corrupt")
            || description.contains("invalid")
        if isCorrupt { return .modelAssetCorrupted(error.localizedDescription) }

        return .modelLoadFailed("could not instantiate model: \(error.localizedDescription)")
    }

    /// The cached `.mlmodelc` for `descriptor`, or `nil` when there is no usable one.
    ///
    /// The read half of the cache `moveIntoCaches` writes. Deliberately conservative in three ways,
    /// because a wrong hit is worse than a miss — a miss costs one recompile, a wrong hit loads the
    /// wrong weights or a half-written directory:
    ///
    /// 1. **A crashed or killed compile can leave a partial directory.** `.mlmodelc` is a bundle,
    ///    so existence alone proves nothing; `coremldata.bin` is required to be present before the
    ///    directory is trusted. Anything less is treated as a miss and recompiled over.
    /// 2. **The key already carries the geometry**, so a hit cannot be another shape's model. That
    ///    is what makes reading the cache safe at all.
    /// 3. **Any error is a miss, never a throw.** A caches directory that cannot be read is a
    ///    performance problem, not a correctness one; falling back to compilation always works.
    ///
    /// `load` still instantiates and shape-checks whatever this returns, so a stale-but-complete
    /// directory from an older build fails there by name rather than silently. Public so a harness
    /// can report whether a run will be cold or warm BEFORE it loads. Timing a load without knowing
    /// which it was is how a compile gets read as inference.
    public static func cachedModelURL(for descriptor: ModelDescriptor) -> URL? {
        let fileManager = FileManager.default
        guard
            let caches = try? fileManager.url(
                for: .cachesDirectory, in: .userDomainMask, appropriateFor: nil, create: false)
        else { return nil }
        return cachedModelURL(for: descriptor, in: caches)
    }

    /// The same lookup against an explicit caches root, so the guards above are testable.
    ///
    /// Split out because the guards are the whole point of this function and an untested guard
    /// against a *wrong* cache hit is worse than no cache: under mutation testing, dropping the
    /// `coremldata.bin` check survived a suite that only exercised the real caches directory.
    /// `.cachesDirectory` cannot be injected, so the decision is separated from the lookup of where
    /// the cache lives.
    static func cachedModelURL(for descriptor: ModelDescriptor, in caches: URL) -> URL? {
        let fileManager = FileManager.default
        let candidate =
            caches
            .appendingPathComponent("com.tbd.upscaler/CompiledModels", isDirectory: true)
            .appendingPathComponent(compiledModelName(for: descriptor), isDirectory: true)
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: candidate.path, isDirectory: &isDirectory),
            isDirectory.boolValue,
            fileManager.fileExists(atPath: candidate.appendingPathComponent("coremldata.bin").path)
        else { return nil }
        return candidate
    }

    /// Compilation writes to a temporary location the system may reclaim; keep it under Caches so a
    /// second run of the same artifact does not recompile. **That promise is only kept because
    /// `cachedModelURL(for:)` reads this back before `load` compiles.**
    ///
    /// Directories keyed under an older naming scheme go stale and are never read again; each shape
    /// recompiles once on first load. No cleanup is performed — `Caches/` is system-managed.
    private static func moveIntoCaches(_ compiled: URL, for descriptor: ModelDescriptor) throws -> URL {
        let fileManager = FileManager.default
        do {
            let caches = try fileManager.url(
                for: .cachesDirectory, in: .userDomainMask, appropriateFor: nil, create: true
            )
            let directory = caches.appendingPathComponent(
                "com.tbd.upscaler/CompiledModels", isDirectory: true)
            try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
            let destination = directory.appendingPathComponent(
                compiledModelName(for: descriptor), isDirectory: true
            )
            if fileManager.fileExists(atPath: destination.path) {
                try fileManager.removeItem(at: destination)
            }
            try fileManager.moveItem(at: compiled, to: destination)
            return destination
        } catch {
            throw InferenceError.modelLoadFailed(
                "could not cache compiled model: \(error.localizedDescription)")
        }
    }

    // MARK: - Prediction

    public func predict(_ tile: CVPixelBuffer) async throws -> CVPixelBuffer {
        try Task.checkCancellation()

        guard let loaded = lock.withLock({ self.loaded }) else {
            throw InferenceError.predictionFailed("no model loaded — call load(_:context:) first")
        }

        let expected = loaded.descriptor.inputTileSize
        let width = CVPixelBufferGetWidth(tile)
        let height = CVPixelBufferGetHeight(tile)
        guard width == expected, height == expected else {
            // `got` is a single value by contract: report the offending dimension, width first.
            throw InferenceError.invalidTileSize(expected: expected, got: width == expected ? height : width)
        }

        let format = CVPixelBufferGetPixelFormatType(tile)
        guard format == kCVPixelFormatType_32BGRA else {
            throw InferenceError.predictionFailed(
                "unsupported pixel format '\(Self.fourCharacterCode(format))' — 32BGRA required"
            )
        }

        let input = try Self.multiArray(from: tile, tileSize: expected)
        let provider: MLFeatureProvider
        do {
            provider = try MLDictionaryFeatureProvider(
                dictionary: [loaded.inputName: MLFeatureValue(multiArray: input)]
            )
        } catch {
            throw InferenceError.predictionFailed(
                "could not build input features: \(error.localizedDescription)")
        }

        let result: MLFeatureProvider
        do {
            result = try await loaded.model.prediction(from: provider)
        } catch {
            throw InferenceError.predictionFailed(error.localizedDescription)
        }

        guard let output = result.featureValue(for: loaded.outputName)?.multiArrayValue else {
            throw InferenceError.predictionFailed("model produced no '\(loaded.outputName)' multi-array")
        }
        return try Self.pixelBuffer(from: output)
    }

    // MARK: - Marshalling
    //
    // The artifact's Core ML interface is a float32 MLMultiArray (1,3,H,W), not an image feature,
    // so the buffer is converted rather than handed over. Values are [0,1] in, clamped [0,1] out,
    // matching the manifest's declared pre/postprocessing.

    private static func multiArray(from tile: CVPixelBuffer, tileSize: Int) throws -> MLMultiArray {
        let array: MLMultiArray
        do {
            array = try MLMultiArray(
                shape: [1, 3, NSNumber(value: tileSize), NSNumber(value: tileSize)], dataType: .float32
            )
        } catch {
            throw InferenceError.predictionFailed(
                "could not allocate input tensor: \(error.localizedDescription)")
        }

        CVPixelBufferLockBaseAddress(tile, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(tile, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(tile) else {
            throw InferenceError.predictionFailed("could not lock pixel buffer for reading")
        }
        let rowBytes = CVPixelBufferGetBytesPerRow(tile)
        let source = base.assumingMemoryBound(to: UInt8.self)

        array.withUnsafeMutableBytes { raw, strides in
            guard let destination = raw.baseAddress?.assumingMemoryBound(to: Float.self) else { return }
            let (planeStride, rowStride, columnStride) = (strides[1], strides[2], strides[3])
            for y in 0..<tileSize {
                let row = source.advanced(by: y * rowBytes)
                for x in 0..<tileSize {
                    let pixel = row.advanced(by: x * 4)  // BGRA
                    let base = y * rowStride + x * columnStride
                    destination[base] = Float(pixel[2]) / 255.0  // R
                    destination[base + planeStride] = Float(pixel[1]) / 255.0  // G
                    destination[base + 2 * planeStride] = Float(pixel[0]) / 255.0  // B
                }
            }
        }
        return array
    }

    private static func pixelBuffer(from array: MLMultiArray) throws -> CVPixelBuffer {
        let shape = array.shape.map(\.intValue)
        guard shape.count == 4, shape[0] == 1, shape[1] == 3 else {
            throw InferenceError.predictionFailed("unexpected output shape \(shape), expected [1,3,H,W]")
        }
        guard array.dataType == .float32 else {
            throw InferenceError.predictionFailed(
                "unexpected output dtype \(array.dataType.rawValue), expected float32")
        }
        let (height, width) = (shape[2], shape[3])

        var buffer: CVPixelBuffer?
        let attributes: [CFString: Any] = [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA,
            attributes as CFDictionary, &buffer
        )
        guard status == kCVReturnSuccess, let output = buffer else {
            throw InferenceError.predictionFailed(
                "could not allocate \(width)x\(height) output buffer (CVReturn \(status))")
        }

        CVPixelBufferLockBaseAddress(output, [])
        defer { CVPixelBufferUnlockBaseAddress(output, []) }
        guard let base = CVPixelBufferGetBaseAddress(output) else {
            throw InferenceError.predictionFailed("could not lock output pixel buffer for writing")
        }
        let rowBytes = CVPixelBufferGetBytesPerRow(output)
        let destination = base.assumingMemoryBound(to: UInt8.self)

        let strides = array.strides.map(\.intValue)
        guard strides.count == 4 else {
            throw InferenceError.predictionFailed("unexpected output strides \(strides), expected four")
        }
        array.withUnsafeBytes { raw in
            guard let source = raw.baseAddress?.assumingMemoryBound(to: Float.self) else { return }
            let (planeStride, rowStride, columnStride) = (strides[1], strides[2], strides[3])
            for y in 0..<height {
                let row = destination.advanced(by: y * rowBytes)
                for x in 0..<width {
                    let base = y * rowStride + x * columnStride
                    let red = clampToByte(source[base])
                    let green = clampToByte(source[base + planeStride])
                    let blue = clampToByte(source[base + 2 * planeStride])
                    let pixel = row.advanced(by: x * 4)
                    pixel[0] = blue
                    pixel[1] = green
                    pixel[2] = red
                    pixel[3] = 255
                }
            }
        }
        return output
    }

    /// Postprocessing per the model manifest: clamp to [0,1], then quantise to 8 bits.
    private static func clampToByte(_ value: Float) -> UInt8 {
        UInt8((min(max(value, 0.0), 1.0) * 255.0).rounded())
    }

    private static func fourCharacterCode(_ code: OSType) -> String {
        let bytes = [24, 16, 8, 0].map { UInt8((code >> $0) & 0xFF) }
        let scalars = bytes.map { (32...126).contains($0) ? Character(UnicodeScalar($0)) : "?" }
        return String(scalars)
    }
}
