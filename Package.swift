// swift-tools-version: 6.0

import PackageDescription

// Module boundaries are declared as dependency edges below.
//
// These edges are the intent; they are NOT self-enforcing. With a warm module cache SwiftPM
// compiles and links an undeclared intra-package import with exit 0 (only a clean build rejects
// it). The deterministic enforcement point is the import check in .github/workflows/ci.yml — keep
// the two in sync when adding a target.
//
//   UpscaleCore -> no internal dependencies.
let package = Package(
    name: "Upscaler",
    platforms: [
        .iOS(.v18),
        .macOS(.v15),
    ],
    products: [
        .library(name: "UpscaleCore", targets: ["UpscaleCore"]),
    ],
    dependencies: [
        // No third-party dependencies, by design.
    ],
    targets: [
        .target(name: "UpscaleCore"),
        .testTarget(name: "UpscaleCoreTests", dependencies: ["UpscaleCore"]),
    ],
    swiftLanguageModes: [.v6]
)
