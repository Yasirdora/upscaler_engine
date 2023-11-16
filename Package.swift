// swift-tools-version: 6.0

import PackageDescription

// Module boundaries are declared as dependency edges below.
//
// These edges are the intent; they are NOT self-enforcing. With a warm module cache SwiftPM
// compiles and links an undeclared intra-package import with exit 0 (only a clean build rejects
// it). The deterministic enforcement point is the import check in .github/workflows/ci.yml — keep
// the two in sync when adding a target.
//
//   UpscaleCore, ImagePipeline, Benchmarking, TestFixtures -> no internal dependencies.
//     (In particular UpscaleCore may NOT import ImagePipeline or ModelStore.)
//   ModelStore    -> may depend on UpscaleCore.
//   UpscaleCLI    -> may depend on all of the above.
let package = Package(
    name: "Upscaler",
    platforms: [
        .iOS(.v18),
        .macOS(.v15),
    ],
    products: [
        .library(name: "UpscaleCore", targets: ["UpscaleCore"]),
        .library(name: "ImagePipeline", targets: ["ImagePipeline"]),
        .library(name: "ModelStore", targets: ["ModelStore"]),
        .library(name: "Benchmarking", targets: ["Benchmarking"]),
        .library(name: "TestFixtures", targets: ["TestFixtures"]),
        .executable(name: "UpscaleCLI", targets: ["UpscaleCLI"]),
    ],
    dependencies: [
        // No third-party dependencies, by design.
    ],
    targets: [
        .target(name: "UpscaleCore"),
        .target(name: "ImagePipeline"),
        // Depends on UpscaleCore: `ModelProviding` traffics in `ModelDescriptor`.
        .target(name: "ModelStore", dependencies: ["UpscaleCore"]),
        .target(name: "Benchmarking"),
        // corpus/, references/ and workloads/ hold certification data (images, .npy tensors, the
        // committed references_manifest.json) — not Swift sources. Excluded so SwiftPM does not
        // reject them as unhandled files.
        .target(
            name: "TestFixtures",
            exclude: ["corpus", "references", "references_manifest.json", "workloads"]
        ),
        .executableTarget(
            name: "UpscaleCLI",
            dependencies: [
                "UpscaleCore",
                "ImagePipeline",
                "ModelStore",
                "Benchmarking",
                "TestFixtures",
            ]
        ),
        .testTarget(name: "UpscaleCoreTests", dependencies: ["UpscaleCore"]),
        .testTarget(name: "ImagePipelineTests", dependencies: ["ImagePipeline"]),
        .testTarget(name: "ModelStoreTests", dependencies: ["ModelStore"]),
        .testTarget(name: "BenchmarkingTests", dependencies: ["Benchmarking"]),
        .testTarget(name: "TestFixturesTests", dependencies: ["TestFixtures"]),
        .testTarget(name: "UpscaleCLITests", dependencies: ["UpscaleCLI"]),
    ],
    swiftLanguageModes: [.v6]
)
