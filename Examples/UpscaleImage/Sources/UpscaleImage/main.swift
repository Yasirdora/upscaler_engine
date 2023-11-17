// Upscale one photograph end to end with the engine's public API.
//
// The steps are the ones an app performs, in order:
//
//   1. Decode the file into the engine's input contract, keeping its colour profile and metadata.
//   2. Plan the colour: which profile the pixels are in and which one the output must carry.
//   3. Normalise the EXIF orientation, so the model sees the image the way a person does.
//   4. Split off alpha, if the image has any; only colour goes through the model.
//   5. Load the converted Core ML model and upscale tile by tile.
//   6. Upscale alpha separately and recombine.
//   7. Encode, embedding the planned profile and the original metadata.

import CoreGraphics
import CoreVideo
import Foundation
import ImagePipeline
import UpscaleCore

struct Options {
    var modelDirectory: URL
    var input: URL
    var output: URL
    var method: StitchMethod = .centerValidRegion

    static let usage = """
        USAGE: upscale-image --model <models/<id>> --input <image> --output <image>
                             [--method hardCrop|linearFeather|cosineFeather|centerValidRegion]

        --model   a model directory holding manifest.json and output/<id>_<N>x<N>.mlpackage,
                  as produced by tools/convert/convert.py
        --output  the format follows the extension: .png, .jpg, .heic or .tiff
        """

    static func parse(_ arguments: [String]) -> Options? {
        var values: [String: String] = [:]
        var index = 0
        while index + 1 < arguments.count {
            values[arguments[index]] = arguments[index + 1]
            index += 2
        }
        guard index == arguments.count,
            let model = values["--model"], let input = values["--input"],
            let output = values["--output"]
        else { return nil }
        var options = Options(
            modelDirectory: URL(fileURLWithPath: model), input: URL(fileURLWithPath: input),
            output: URL(fileURLWithPath: output))
        if let raw = values["--method"] {
            guard let method = StitchMethod(rawValue: raw) else { return nil }
            options.method = method
        }
        return options
    }
}

enum ExampleError: Error, CustomStringConvertible {
    case unreadableManifest(URL)
    case noColorSpace
    case noContextMargin(String)
    case unsupportedOutputExtension(String)

    var description: String {
        switch self {
        case .unreadableManifest(let url):
            return "cannot read \(url.path) — convert the model first (docs/MODELS.md)"
        case .noColorSpace:
            return "the decoded buffer carries no usable colour space"
        case .noContextMargin(let family):
            return "no derived context margin for \(family); the engine refuses to guess one"
        case .unsupportedOutputExtension(let ext):
            return "unsupported output extension '.\(ext)' — use .png, .jpg, .heic or .tiff"
        }
    }
}

/// The model as its manifest describes it: identity, scale and the fixed input shape.
func loadModel(from directory: URL) throws -> ModelDescriptor {
    struct Manifest: Decodable {
        struct Conversion: Decodable { let inputShape: [Int] }
        let modelId: String
        let scaleFactor: Int
        let conversion: Conversion
    }
    let manifestURL = directory.appendingPathComponent("manifest.json")
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    guard let data = try? Data(contentsOf: manifestURL),
        let manifest = try? decoder.decode(Manifest.self, from: data),
        manifest.conversion.inputShape.count == 4
    else { throw ExampleError.unreadableManifest(manifestURL) }

    let tile = manifest.conversion.inputShape[3]
    let package = directory.appendingPathComponent(
        "output/\(manifest.modelId)_\(tile)x\(tile).mlpackage")

    // Margins are a property of the network, so they are keyed by model family:
    // `RealESRGAN_x4plus_384` is the 384-pixel variant of `RealESRGAN_x4plus`.
    let family = manifest.modelId.replacing(/_\d+$/, with: "")
    guard let margin = ContextMargin.derivedMargin(for: family) else {
        throw ExampleError.noContextMargin(family)
    }
    return ModelDescriptor(
        identifier: manifest.modelId, scaleFactor: manifest.scaleFactor, contextMargin: margin,
        packageURL: package, inputTileSize: tile, precision: .float16)
}

func outputFormat(for url: URL) throws -> Encode.Format {
    switch url.pathExtension.lowercased() {
    case "png": return .png
    case "jpg", "jpeg": return .jpeg
    case "heic", "heif": return .heif
    case "tif", "tiff": return .tiff
    case let other: throw ExampleError.unsupportedOutputExtension(other)
    }
}

func upscale(_ options: Options) async throws {
    let format = try outputFormat(for: options.output)

    // 1–2. Decode, then decide the colour policy. Unsupported or truncated files are refused
    // here, before any pixel is processed.
    let decoded = try Decode.decode(contentsOf: options.input)
    let plan = try Color.plan(for: decoded.color)
    for note in plan.notes { print("colour: \(note)") }

    // 3. Orientation is applied before the model, not left as a tag.
    guard let colorSpace = decoded.color.bufferColorSpace() else { throw ExampleError.noColorSpace }
    let upright = try Metadata.normalise(
        decoded.pixelBuffer, orientation: decoded.metadata.orientation, colorSpace: colorSpace)

    // 4. Only colour goes through the model; alpha is resampled on its own.
    let split = decoded.source.hasAlpha ? try Alpha.split(upright) : nil
    let colour = split?.colour ?? upright

    // 5. Load the model and upscale tile by tile.
    let model = try loadModel(from: options.modelDirectory)
    let backend = CoreMLBackend()
    try await backend.load(model, context: InferenceContext(computeUnits: .all))
    let engine = SequentialTilingEngine(contextMargin: model.contextMargin)

    let width = CVPixelBufferGetWidth(colour), height = CVPixelBufferGetHeight(colour)
    print("upscaling \(width)×\(height) ×\(model.scaleFactor) with \(model.identifier)")
    let started = ContinuousClock.now
    var result = try await engine.upscale(
        colour, model: model, backend: backend, method: options.method,
        progress: { fraction in
            print(String(format: "\r  %3.0f%%", fraction * 100), terminator: "")
            fflush(stdout)
        })
    let elapsed = ContinuousClock.now - started
    print("\r  done in \(elapsed.formatted(.units(allowed: [.seconds], fractionalPart: .show(length: 1))))")

    // 6. Recombine alpha at the output size.
    if let split, split.hasTransparency {
        let target = (width: width * model.scaleFactor, height: height * model.scaleFactor)
        let alpha = try Alpha.scale(split.alpha, from: (split.width, split.height), to: target)
        result = try Alpha.recombine(result, alpha: alpha)
    }

    // 7. Encode with the planned profile and the original metadata, orientation reset to 1.
    let properties = try Metadata.normalisedProperties(from: decoded.metadata)
    try Encode.write(result, to: options.output, format: format, plan: plan, properties: properties)
    print(
        "wrote \(options.output.path) "
            + "(\(CVPixelBufferGetWidth(result))×\(CVPixelBufferGetHeight(result)))")
}

guard let options = Options.parse(Array(CommandLine.arguments.dropFirst())) else {
    print(Options.usage)
    exit(2)
}
do {
    try await upscale(options)
} catch {
    print("error: \(error)")
    exit(1)
}
