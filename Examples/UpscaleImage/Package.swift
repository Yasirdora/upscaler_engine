// swift-tools-version: 6.0

import PackageDescription

// A standalone package that depends on the engine the way an app would: by product, through a
// package dependency. Run it from the repository root:
//
//   swift run --package-path Examples/UpscaleImage upscale-image \
//       --model models/RealESRGAN_x4plus --input photo.jpg --output photo_x4.png
let package = Package(
    name: "UpscaleImageExample",
    platforms: [
        .macOS(.v15)
    ],
    dependencies: [
        .package(name: "Upscaler", path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "upscale-image",
            dependencies: [
                .product(name: "UpscaleCore", package: "Upscaler"),
                .product(name: "ImagePipeline", package: "Upscaler"),
            ],
            path: "Sources/UpscaleImage"
        )
    ],
    swiftLanguageModes: [.v6]
)
