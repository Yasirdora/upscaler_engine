import CoreML
import Foundation
import Testing

@testable import UpscaleCore

/// Encrypted model loading.
///
/// Three mutations against this path — mis-mapping the network error, pushing encrypted assets
/// through the on-disk cache, and deleting the corrupt-asset mapping — each once survived the
/// suite. The code may well have been right; nothing could tell. These tests make each of those
/// mutations fail.
struct EncryptedModelLoadingTests {
    static func error(domain: String, code: Int, message: String = "failed") -> NSError {
        NSError(domain: domain, code: code, userInfo: [NSLocalizedDescriptionKey: message])
    }

    // MARK: - Which assets the compiled-model cache applies to

    /// A source package is compiled and cached; that is the cache's whole point.
    @Test("a .mlpackage uses the compiled-model cache")
    func packageUsesCache() {
        #expect(CoreMLBackend.usesCompiledCache(for: URL(fileURLWithPath: "/m/x.mlpackage")))
    }

    /// The load-bearing one. Copying an encrypted model into Caches would move its decrypted form
    /// outside the boundary Core ML keeps it in — the thing encryption was adopted to preserve.
    @Test("a pre-compiled .mlmodelc bypasses the cache, encrypted or not")
    func compiledBundleBypassesCache() {
        #expect(!CoreMLBackend.usesCompiledCache(for: URL(fileURLWithPath: "/m/x.mlmodelc")))
        #expect(
            !CoreMLBackend.usesCompiledCache(
                for: URL(fileURLWithPath: "/m/x.mlmodelc.encrypted")),
            "an encrypted asset must never be copied into Caches")
    }

    // MARK: - What a load failure is reported as

    /// The distinction that matters to a user: waiting or connecting fixes one and does nothing
    /// for the other. Reporting the first as the second sends them to the wrong action.
    @Test("a decryption-key fetch failure is reported as needing a network, by error CODE")
    func keyFetchFailureIsNetwork() {
        let classified = CoreMLBackend.classify(
            loadError: Self.error(domain: MLModelErrorDomain, code: 8, message: "opaque"))

        #expect(classified == .modelRequiresNetworkOnFirstRun)
    }

    /// The code is the reliable signal; wording is Apple's to change. Both paths are covered so a
    /// future OS that stops saying "network" does not silently reclassify the failure.
    @Test(
        "the wording fallback also catches it",
        arguments: ["The network connection was lost.", "key fetch failed", "no Internet connection"])
    func wordingFallbackCatchesNetwork(message: String) {
        #expect(
            CoreMLBackend.classify(loadError: Self.error(domain: "Other", code: 1, message: message))
                == .modelRequiresNetworkOnFirstRun)
    }

    @Test("a corrupt asset is reported as corrupt, not as a network problem")
    func corruptIsNotNetwork() {
        let byCode = CoreMLBackend.classify(
            loadError: Self.error(domain: MLModelErrorDomain, code: 3, message: "opaque"))
        let byWording = CoreMLBackend.classify(
            loadError: Self.error(domain: "Other", code: 1, message: "The model is corrupt."))

        #expect(byCode == .modelAssetCorrupted("opaque"))
        #expect(byWording == .modelAssetCorrupted("The model is corrupt."))
        #expect(byCode != .modelRequiresNetworkOnFirstRun)
    }

    /// Network is checked first deliberately: a key fetch that fails offline can surface with
    /// wording mentioning an invalid state, and telling the user to reconnect is the useful
    /// instruction of the two.
    @Test("a key-fetch code wins over corrupt-sounding wording")
    func networkTakesPrecedence() {
        #expect(
            CoreMLBackend.classify(
                loadError: Self.error(
                    domain: MLModelErrorDomain, code: 8, message: "invalid model state"))
                == .modelRequiresNetworkOnFirstRun)
    }

    @Test("anything else stays a plain load failure and keeps the underlying text")
    func unrecognisedStaysGeneric() {
        let classified = CoreMLBackend.classify(
            loadError: Self.error(domain: "Other", code: 99, message: "disk on fire"))

        guard case .modelLoadFailed(let reason) = classified else {
            Issue.record("expected .modelLoadFailed, got \(classified)")
            return
        }
        #expect(reason.contains("disk on fire"), "the real cause must survive classification")
    }
}
