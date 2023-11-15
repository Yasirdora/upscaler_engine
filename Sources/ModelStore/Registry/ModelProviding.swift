import Foundation
import UpscaleCore

/// Interface for registering and supplying ML models to the inference pipeline.
///
/// An app implements this protocol to provide secure model URLs (e.g., bundled `.mlmodelc.encrypted` assets)
/// to the engine without hardcoding paths or scanning the bundle directly.
public protocol ModelProviding: Sendable {
    /// Returns the available models as descriptors containing geometry, margin, and physical location.
    func availableModels() async throws -> [ModelDescriptor]
}
