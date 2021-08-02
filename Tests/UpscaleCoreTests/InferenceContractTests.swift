import CoreML
import Foundation
import Testing

@testable import UpscaleCore

/// Contract tests for the inference backend, contract version 1.
///
/// Model-free by design: nothing here loads the `.mlpackage`. These prove the plumbing — descriptor
/// semantics, the error taxonomy, and that each `ComputeUnits` case maps to the right
/// `MLComputeUnits`. What Core ML *does* with those units is empirical and belongs to benchmarking;
/// asserting it here would be benchmark theatre.
struct InferenceContractTests {
    static let packageURL = URL(fileURLWithPath: "/models/RealESRGAN_x4plus/output/x.mlpackage")

    static func descriptor(
        identifier: String = "RealESRGAN_x4plus",
        scaleFactor: Int = 4,
        tile: Int = 256,
        precision: Precision = .float16
    ) -> ModelDescriptor {
        ModelDescriptor(
            identifier: identifier, scaleFactor: scaleFactor, contextMargin: 76,
            packageURL: packageURL, inputTileSize: tile, precision: precision
        )
    }

    // MARK: - Compute-unit mapping

    @Test("every ComputeUnits case maps to its MLComputeUnits counterpart")
    func computeUnitMapping() {
        #expect(ComputeUnits.all.mlComputeUnits == .all)
        #expect(ComputeUnits.cpuAndGPU.mlComputeUnits == .cpuAndGPU)
        #expect(ComputeUnits.cpuAndNeuralEngine.mlComputeUnits == .cpuAndNeuralEngine)
        #expect(ComputeUnits.cpuOnly.mlComputeUnits == .cpuOnly)
    }

    @Test("the mapping is total and injective — no case shares a destination")
    func mappingIsTotalAndInjective() {
        let mapped = ComputeUnits.allCases.map(\.mlComputeUnits)
        #expect(mapped.count == 4, "a new case must be given a mapping, not defaulted")
        #expect(Set(mapped.map(\.rawValue)).count == mapped.count, "two cases map to the same units")
    }

    @Test("compute-unit raw values are the CLI's accepted spellings")
    func computeUnitRawValues() {
        #expect(
            ComputeUnits.allCases.map(\.rawValue) == ["all", "cpuAndGPU", "cpuAndNeuralEngine", "cpuOnly"])
        #expect(ComputeUnits(rawValue: "cpuOnly") == .cpuOnly)
        #expect(ComputeUnits(rawValue: "gpu") == nil)
    }

    // MARK: - Descriptor and context

    @Test("descriptor carries the fixed-shape geometry it was given")
    func descriptorSemantics() {
        let subject = Self.descriptor()
        #expect(subject.identifier == "RealESRGAN_x4plus")
        #expect(subject.scaleFactor == 4)
        #expect(subject.inputTileSize == 256)
        #expect(subject.precision == .float16)
        #expect(subject.packageURL == Self.packageURL)
    }

    @Test("descriptor equality distinguishes every field")
    func descriptorEquality() {
        #expect(Self.descriptor() == Self.descriptor())
        #expect(Self.descriptor() != Self.descriptor(identifier: "other"))
        #expect(Self.descriptor() != Self.descriptor(scaleFactor: 2))
        #expect(Self.descriptor() != Self.descriptor(tile: 384))
        #expect(Self.descriptor() != Self.descriptor(precision: .float32))
    }

    @Test("precision raw values match the manifest vocabulary")
    func precisionRawValues() {
        #expect(Precision.float16.rawValue == "float16")
        #expect(Precision.float32.rawValue == "float32")
    }

    @Test("context carries compute units and is mutable in place")
    func contextSemantics() {
        var context = InferenceContext(computeUnits: .all)
        #expect(context.computeUnits == .all)
        context.computeUnits = .cpuOnly
        #expect(context.computeUnits == .cpuOnly)
    }

    // MARK: - Error taxonomy

    @Test("errors compare by payload, not just by case")
    func errorEquality() {
        let url = Self.packageURL
        #expect(InferenceError.modelNotFound(url) == .modelNotFound(url))
        #expect(InferenceError.modelNotFound(url) != .modelNotFound(URL(fileURLWithPath: "/elsewhere")))
        #expect(InferenceError.modelLoadFailed("a") != .modelLoadFailed("b"))
        #expect(
            InferenceError.invalidTileSize(expected: 256, got: 128)
                == .invalidTileSize(expected: 256, got: 128))
        #expect(
            InferenceError.invalidTileSize(expected: 256, got: 128)
                != .invalidTileSize(expected: 256, got: 512))
        #expect(InferenceError.predictionFailed("x") != .modelLoadFailed("x"))
    }

    @Test("error descriptions name the offending value")
    func errorDescriptions() {
        #expect("\(InferenceError.modelNotFound(Self.packageURL))".contains(Self.packageURL.path))
        #expect("\(InferenceError.invalidTileSize(expected: 256, got: 512))".contains("512"))
        #expect("\(InferenceError.modelLoadFailed("bad header"))".contains("bad header"))
        #expect("\(InferenceError.predictionFailed("shape"))".contains("shape"))
    }

    // MARK: - Backend behaviour reachable without a model

    @Test("loading a package that does not exist reports modelNotFound with the URL")
    func loadMissingPackage() async {
        let missing = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("absent.mlpackage")
        let backend = CoreMLBackend()

        await #expect(throws: InferenceError.modelNotFound(missing)) {
            try await backend.load(
                Self.descriptor(packageURL: missing), context: .init(computeUnits: .cpuOnly))
        }
    }

    @Test("predicting before loading fails rather than crashing")
    func predictWithoutLoad() async throws {
        let backend = CoreMLBackend()
        var buffer: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, 256, 256, kCVPixelFormatType_32BGRA, nil, &buffer)
        let tile = try #require(buffer)

        // Discard the result: CVPixelBuffer is non-Sendable, so returning it from the
        // closure would be a `sending` violation under Swift 6.
        await #expect(throws: InferenceError.self) { _ = try await backend.predict(tile) }
    }
}

extension InferenceContractTests {
    static func descriptor(packageURL: URL) -> ModelDescriptor {
        ModelDescriptor(
            identifier: "RealESRGAN_x4plus", scaleFactor: 4, contextMargin: 76,
            packageURL: packageURL, inputTileSize: 256, precision: .float16
        )
    }
}
