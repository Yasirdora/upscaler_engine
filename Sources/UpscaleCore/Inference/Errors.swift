import Foundation

/// Typed inference failures. Nothing is swallowed: every failure path a caller can encounter
/// is one of these cases, carrying enough detail to diagnose it without a debugger.
public enum InferenceError: Error, Equatable {
    case modelNotFound(URL)
    case modelLoadFailed(String)
    case invalidTileSize(expected: Int, got: Int)
    case predictionFailed(String)
    case modelRequiresNetworkOnFirstRun
    case modelAssetCorrupted(String)
}

extension InferenceError: CustomStringConvertible {
    public var description: String {
        switch self {
        case .modelNotFound(let url):
            return "model not found at \(url.path)"
        case .modelLoadFailed(let reason):
            return "model load failed: \(reason)"
        case .invalidTileSize(let expected, let got):
            return "invalid tile size: expected \(expected)x\(expected), got \(got)x\(got)"
        case .predictionFailed(let reason):
            return "prediction failed: \(reason)"
        case .modelRequiresNetworkOnFirstRun:
            return "model decryption requires an active internet connection on its first run"
        case .modelAssetCorrupted(let reason):
            return "model asset corrupted: \(reason)"
        }
    }
}
