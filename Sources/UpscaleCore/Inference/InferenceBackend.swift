import CoreVideo
import Foundation

/// The inference contract, version 1.
///
/// Core ML is the default backend, not the only possible one: the abstraction exists so a Metal /
/// MPSGraph path stays available if profiling demands it, and so the model itself can be swapped
/// without redesigning the engine.
///
/// Cancellation is structured concurrency only — `Task.checkCancellation()`, never custom tokens.
public protocol InferenceBackend: Sendable {
    func load(_ model: ModelDescriptor, context: InferenceContext) async throws
    func predict(_ tile: CVPixelBuffer) async throws -> CVPixelBuffer
}
