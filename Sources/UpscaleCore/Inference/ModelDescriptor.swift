import Foundation

/// Numeric precision of a converted model artifact.
public enum Precision: String, Sendable { case float16, float32 }

/// Identity and fixed-shape geometry of a converted model artifact.
///
/// Contract version 1. Fixed-shape by design: Core ML tooling optimises static shapes better than
/// flexible ones, so arbitrary images are handled by a tile planner (`TilePlanner`), never by
/// making the model accept arbitrary input.
public struct ModelDescriptor: Sendable, Equatable {
    public let identifier: String
    public let scaleFactor: Int
    public let contextMargin: Int
    public let packageURL: URL  // local .mlpackage
    public let inputTileSize: Int  // fixed-shape model: 256
    public let precision: Precision
    // NOTE (post-MVP intent, do not implement): split into
    // ModelCapabilities / ModelRuntimeConfiguration when tiling lands.

    /// Memberwise initialiser, public so other modules can construct a descriptor.
    /// The contract lists the stored properties only; a public struct's synthesised memberwise
    /// init is internal, which would make the type unusable from UpscaleCLI.
    public init(
        identifier: String,
        scaleFactor: Int,
        contextMargin: Int,
        packageURL: URL,
        inputTileSize: Int,
        precision: Precision
    ) {
        self.identifier = identifier
        self.scaleFactor = scaleFactor
        self.contextMargin = contextMargin
        self.packageURL = packageURL
        self.inputTileSize = inputTileSize
        self.precision = precision
    }
}

/// Per-run inference configuration. One bag, so parameter lists do not grow.
public struct InferenceContext: Sendable {
    public var computeUnits: ComputeUnits
    public init(computeUnits: ComputeUnits) { self.computeUnits = computeUnits }
}

/// Which silicon Core ML may dispatch to. Which one wins is settled empirically, never assumed
/// (certification spec §7.3): if the GPU wins on time and energy, the GPU wins.
public enum ComputeUnits: String, Sendable, CaseIterable {
    case all, cpuAndGPU, cpuAndNeuralEngine, cpuOnly  // maps to MLComputeUnits
}
