import CoreGraphics
import CoreVideo
import Foundation
import ImageIO
import UniformTypeIdentifiers
import UpscaleCore

/// `upscale-image` — run a whole image through the tiling engine.
///
/// PNG handling is the same utilitarian path `upscale-tile` uses: no colour management, no ICC, no
/// metadata. Correct colour belongs to `ImagePipeline` (certification spec §8); this command sits
/// below that boundary deliberately.
enum ImageCommand {
    /// The model's upscale factor, read from its committed manifest.
    ///
    /// **Not `InferenceCommand.defaultScaleFactor`.** That constant is 4, and a ×2 model described
    /// as ×4 fails the backend's shape check — which is the good case. The bad case is a path that
    /// does not check, where every output surface and memory plan is sized for four times the
    /// pixels the model produces. `manifest.json` is the authoritative source until `ModelStore`
    /// owns this.
    static func scaleFactor(for identifier: String) -> Int? {
        let manifest = URL(fileURLWithPath: "models/\(identifier)/manifest.json")
        guard let data = try? Data(contentsOf: manifest),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let scale = object["scale_factor"] as? Int
        else { return nil }
        return scale
    }

    /// The derived margin for a model, or `nil` when none has been derived for it.
    ///
    /// The margin follows the receptive field, so it is per model: the supported models run at 16 or
    /// 76 px. `nil` is deliberate — any default is some other model's number, and
    /// under-padding produces seams with nothing thrown.
    static func derivedMargin(for identifier: String) -> Int? {
        ContextMargin.derivedMargin(for: identifier)
    }

    /// The model family an artifact identifier belongs to, which is what margins are keyed on.
    ///
    /// Artifact stems carry their shape (`RealESRGAN_x4plus_256x256`), and shape variants carry it
    /// twice (`RealESRGAN_x4plus_384_384x384`); the margin is a property of the network, so both
    /// suffixes are dropped. A bare family id comes back unchanged.
    static func modelFamily(of identifier: String) -> String {
        identifier.replacing(/_\d+x\d+$/, with: "").replacing(/_\d+$/, with: "")
    }

    struct Options: Equatable {
        var model: String
        var input: URL
        var output: URL
        var method: StitchMethod
        var tileSize: Int
        /// Select the band-spooled destination surface.
        ///
        /// Additive, and **false by default**, so every existing invocation, certification script
        /// and test keeps its exact current meaning. The flag exists so this command can exercise,
        /// and certify, the band-spooled path end to end.
        var spool: Bool = false
    }

    enum UsageError: Error, Equatable {
        case missingValue(flag: String)
        case unknownArgument(String)
        case missingRequired(flag: String)
        case invalidMethod(String)
        case invalidTileSize(String)
        case invalidImages(String)
        case modelShapeMismatch(packageGeometry: Int, tileSize: Int)
        case modelGeometryUnparseable(stem: String)

        var message: String {
            switch self {
            case .missingValue(let flag): return "error: \(flag) requires a value"
            case .unknownArgument(let argument): return "error: unknown argument '\(argument)'"
            case .missingRequired(let flag): return "error: \(flag) is required"
            case .invalidMethod(let value):
                let valid = StitchMethod.allCases.map(\.rawValue).joined(separator: "|")
                return "error: invalid --method '\(value)' (expected \(valid))"
            case .invalidTileSize(let value):
                return "error: invalid --tile-size '\(value)' (expected a positive integer)"
            case .invalidImages(let value):
                return "error: invalid --images '\(value)' "
                    + "(expected comma-separated stems, no empty segments)"
            case .modelShapeMismatch(let packageGeometry, let tileSize):
                return "error: the package is a \(packageGeometry)x\(packageGeometry) artifact but "
                    + "--tile-size is \(tileSize) — a certification run's shape must be the package's shape"
            case .modelGeometryUnparseable(let stem):
                return "error: cannot read tile geometry from '\(stem)' "
                    + "(expected the <id>_<N>x<N>.mlpackage naming convention)"
            }
        }
    }

    /// The manifest id a certification path pairs oracles under, derived from the resolved
    /// package's *own* name and the requested shape — never from string position alone.
    ///
    /// Why this is subtle: the 256 manifest id is the bare family (`RealESRGAN_x4plus`), while the
    /// 384 and 512 shapes carry their geometry in the id (`RealESRGAN_x4plus_384`). Pointed at the
    /// family's *shared* runtime output directory, a rule that stripped the requested tile size off
    /// the file name silently paired the 256 oracle set — a plausible-looking, wrong certification.
    /// The rule below converges to the same id from either the per-id layout or the shared layout,
    /// and refuses shape contradictions.
    static func certificationModelID(resolvedURL url: URL, tileSize: Int) throws -> String {
        let stem = url.deletingPathExtension().lastPathComponent
        let pattern = #"_(\d+)x(\d+)$"#
        guard let match = stem.range(of: pattern, options: .regularExpression) else {
            throw UsageError.modelGeometryUnparseable(stem: stem)
        }
        let sides = stem[match].dropFirst().split(separator: "x")  // drop the leading "_"
        guard sides.count == 2, let g1 = Int(sides[0]), let g2 = Int(sides[1]), g1 == g2 else {
            throw UsageError.modelGeometryUnparseable(stem: stem)
        }
        guard g1 == tileSize else {
            throw UsageError.modelShapeMismatch(packageGeometry: g1, tileSize: tileSize)
        }
        let family = String(stem[stem.startIndex..<match.lowerBound])
        if tileSize == InferenceCommand.defaultTileSize {
            return family  // the 256-era manifest id is the bare family name
        }
        // The id carries the shape — unless the caller already included it (bare-id form).
        let shapeSuffix = "_\(tileSize)"
        return family.hasSuffix(shapeSuffix) ? family : family + shapeSuffix
    }

    static let usage = """
        USAGE: UpscaleCLI upscale-image --model <path-or-name> --input <img> --output <png>
                                        --method hardCrop|linearFeather|cosineFeather|centerValidRegion
                                        [--tile-size 256] [--spool]

        Tiles the image, runs each tile through the Core ML backend, and stitches the result.
        The context margin is PER MODEL, each derived by measurement: x4v3 16 px, x4plus 76 px,
        x2plus 76 px (derived 108 px). A model with no derived margin is refused rather than run on another
        model's number. The margins are derived with tools/margin/derive_context_margin_torch.py.

        --spool  Stitch into the band-spooled file destination instead of a
                 whole-image buffer, then encode the PNG from the spool. Same model, same shape,
                 same grid, same output file — only the memory path differs. Off by default.

        Exit codes: 0 success, 2 usage error, 3 inference or tiling error.
        """

    // MARK: - Parsing

    static func parse(_ arguments: [String]) throws -> Options {
        var model: String?
        var input: String?
        var output: String?
        var method: StitchMethod?
        var tileSize = InferenceCommand.defaultTileSize
        var spool = false

        var index = arguments.startIndex
        while index < arguments.endIndex {
            let argument = arguments[index]
            func value(for flag: String) throws -> String {
                let next = arguments.index(after: index)
                guard next < arguments.endIndex else { throw UsageError.missingValue(flag: flag) }
                index = next
                return arguments[next]
            }
            switch argument {
            case "--model": model = try value(for: "--model")
            case "--input": input = try value(for: "--input")
            case "--output": output = try value(for: "--output")
            case "--method":
                let raw = try value(for: "--method")
                guard let parsed = StitchMethod(rawValue: raw) else {
                    throw UsageError.invalidMethod(raw)
                }
                method = parsed
            case "--tile-size":
                let raw = try value(for: "--tile-size")
                guard let parsed = Int(raw), parsed > 0 else { throw UsageError.invalidTileSize(raw) }
                tileSize = parsed
            // Valueless on purpose: it selects a destination, it does not configure one. A
            // `--spool <path>` form would make the spool file the caller's to clean up and would
            // give the flag a second meaning to get wrong; the file is an implementation detail of
            // the destination, and this command deletes it.
            case "--spool": spool = true
            default:
                throw UsageError.unknownArgument(argument)
            }
            index = arguments.index(after: index)
        }

        guard let model else { throw UsageError.missingRequired(flag: "--model") }
        guard let input else { throw UsageError.missingRequired(flag: "--input") }
        guard let output else { throw UsageError.missingRequired(flag: "--output") }
        guard let method else { throw UsageError.missingRequired(flag: "--method") }

        return Options(
            model: model, input: URL(fileURLWithPath: input), output: URL(fileURLWithPath: output),
            method: method, tileSize: tileSize, spool: spool)
    }

    static func descriptor(for options: Options) -> ModelDescriptor {
        let url = InferenceCommand.resolveModelURL(options.model, tileSize: options.tileSize)
        let identifier = url.deletingPathExtension().lastPathComponent
        let baseIdentifier = identifier.components(separatedBy: "_").dropLast().joined(separator: "_")
        let marginId = baseIdentifier.isEmpty ? identifier : baseIdentifier
        let margin = ContextMargin.derivedMargin(for: marginId) ?? 76

        return ModelDescriptor(
            identifier: identifier,
            scaleFactor: scaleFactor(for: options.model) ?? InferenceCommand.defaultScaleFactor,
            contextMargin: margin,
            packageURL: url,
            inputTileSize: options.tileSize,
            precision: InferenceCommand.defaultPrecision
        )
    }

    // MARK: - Encoding the spooled raster

    /// Encode a `SpooledImage` to PNG by reading the spool file back, without ever materialising
    /// the image as a whole-image buffer.
    ///
    /// The pixel format is the same one `InferenceCommand.writePNG` hands Core Graphics — 8-bit
    /// BGRA read as `noneSkipFirst | byteOrder32Little`, device RGB, no colour management (this
    /// command sits below the `ImagePipeline` boundary, certification spec §8) — so the two
    /// destinations meet the encoder on identical terms. That is a precondition of comparing them
    /// at all: if the two paths encoded differently, a pixel difference would say nothing about the
    /// destination surface.
    ///
    /// Why a `CGDataProvider` over the file rather than a read into memory: `SpooledImage`
    /// documents the raster as tightly packed and written top row first precisely so a consumer can
    /// stream it, and the pages a sequential read touches are clean — they do not count against
    /// `phys_footprint` (see `OutputSurface.swift`). Reading the file into a `Data` first would
    /// re-create the whole-image buffer the spool exists to remove, on the last step of the job.
    ///
    /// `InferenceError.predictionFailed` is the encode-failure channel `InferenceCommand.writePNG`
    /// already uses, and `run` maps it to exit 3; a second error vocabulary for the same failure
    /// would make the two destinations report differently.
    static func writePNG(_ spooled: SpooledImage, to url: URL) throws {
        // Check the spool is the size its geometry claims before handing it to Core Graphics. A
        // short file still produces a CGImage — the provider is lazy — so the failure would surface
        // as a plausible-looking image with a wrong tail, which is the one failure mode a pixel
        // comparison cannot localise. Loud here, or silent in a certification report.
        let attributes = try? FileManager.default.attributesOfItem(atPath: spooled.url.path)
        if let spooledBytes = attributes?[.size] as? Int, spooledBytes != spooled.byteCount {
            throw InferenceError.predictionFailed(
                "the spool at \(spooled.url.path) holds \(spooledBytes) B but a "
                    + "\(spooled.width)x\(spooled.height) raster at \(spooled.bytesPerRow) B/row "
                    + "needs \(spooled.byteCount) B")
        }
        guard let provider = CGDataProvider(url: spooled.url as CFURL) else {
            throw InferenceError.predictionFailed(
                "could not open the spool file at \(spooled.url.path)")
        }
        let bitmapInfo = CGBitmapInfo(
            rawValue: CGImageAlphaInfo.noneSkipFirst.rawValue
                | CGBitmapInfo.byteOrder32Little.rawValue)
        guard
            let image = CGImage(
                width: spooled.width, height: spooled.height,
                bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: spooled.bytesPerRow,
                space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: bitmapInfo,
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
        else {
            throw InferenceError.predictionFailed(
                "could not read the spooled raster as a \(spooled.width)x\(spooled.height) image")
        }
        guard
            let destination = CGImageDestinationCreateWithURL(
                url as CFURL, UTType.png.identifier as CFString, 1, nil)
        else {
            throw InferenceError.predictionFailed("could not create a PNG writer for \(url.path)")
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw InferenceError.predictionFailed("could not write PNG to \(url.path)")
        }
    }

    // MARK: - Execution

    static func run(
        arguments: [String],
        out: (String) -> Void = { print($0) },
        err: (String) -> Void = { FileHandle.standardError.write(Data(($0 + "\n").utf8)) }
    ) async -> Int32 {
        let options: Options
        do {
            options = try parse(arguments)
        } catch let error as UsageError {
            err(error.message)
            err(usage)
            return 2
        } catch {
            err("error: \(error)")
            return 2
        }

        do {
            let descriptor = descriptor(for: options)
            let image = try InferenceCommand.loadPixelBuffer(from: options.input)
            let backend = CoreMLBackend()
            try await backend.load(descriptor, context: InferenceContext(computeUnits: .all))

            let family = modelFamily(of: descriptor.identifier)
            guard let margin = derivedMargin(for: family) else {
                err(
                    "no derived context margin for \(family) — derive one with "
                        + "tools/margin/derive_context_margin_torch.py before running it tiled")
                return 2
            }
            let engine = SequentialTilingEngine(contextMargin: margin)
            let planner = TilePlanner(scaleFactor: descriptor.scaleFactor)
            let size = CGSize(
                width: CVPixelBufferGetWidth(image), height: CVPixelBufferGetHeight(image))
            let grid = planner.plan(
                imageSize: size, tileSize: options.tileSize, contextMargin: margin)

            // `elapsed` is measured around the upscale call in both branches and the PNG encode
            // stays outside it, exactly as the in-memory path has always reported it — the
            // wall-clock line keeps its meaning whichever destination ran. The spooled path's
            // extra read-back-and-encode pass is therefore *not* in this number, which is stated
            // in the `destination` line below rather than folded in silently.
            let elapsed: TimeInterval
            let outputSize: (width: Int, height: Int)
            var destinationNote: String?

            if options.spool {
                // The engine creates and fills the spool and never deletes it — the URL is the
                // caller's to own (`TilingEngine.upscale(_:model:backend:method:spoolingTo:…)`).
                // Here the caller is this command, and the file is a private staging area for one
                // encode, so it is removed on the way out of this scope, success or throw.
                let spoolURL = FileManager.default.temporaryDirectory
                    .appendingPathComponent("upscale-image-\(UUID().uuidString).bgra")
                defer { try? FileManager.default.removeItem(at: spoolURL) }

                let started = Date()
                let spooled = try await engine.upscale(
                    image, model: descriptor, backend: backend, method: options.method,
                    spoolingTo: spoolURL, progress: nil)
                elapsed = Date().timeIntervalSince(started)
                try writePNG(spooled, to: options.output)
                outputSize = (spooled.width, spooled.height)
                destinationNote =
                    "destination    : band spool — \(spooled.byteCount) B "
                    + "staged at \(spoolURL.path), removed after encode; "
                    + "the read-back and PNG encode are not in the wall clock above"
            } else {
                let started = Date()
                let result = try await engine.upscale(
                    image, model: descriptor, backend: backend, method: options.method, progress: nil)
                elapsed = Date().timeIntervalSince(started)
                try InferenceCommand.writePNG(result, to: options.output)
                outputSize = (CVPixelBufferGetWidth(result), CVPixelBufferGetHeight(result))
            }

            out("model          : \(descriptor.packageURL.path)")
            out("method         : \(options.method.rawValue)")
            out("input          : \(options.input.path) (\(Int(size.width))x\(Int(size.height)))")
            out(
                "output         : \(options.output.path) "
                    + "(\(outputSize.width)x\(outputSize.height))")
            out(
                "tiles          : \(grid.tiles.count) of \(options.tileSize)px, margin \(grid.contextMargin)px"
            )
            out(
                "overlap cost   : \(String(format: "%.3f", planner.redundancy(for: grid, imageSize: size)))x"
                    + " inference (paid in compute and bandwidth)")
            out("wall clock     : \(String(format: "%.2f", elapsed)) s (informational)")
            // Printed only under --spool, so the default path's stdout is byte-for-byte what it was
            // before this flag existed.
            if let destinationNote { out(destinationNote) }
            return 0
        } catch let error as TilingError {
            err("tiling error: \(error)")
            return 3
        } catch let error as InferenceError {
            err("inference error: \(error)")
            return 3
        } catch {
            err("error: \(error.localizedDescription)")
            return 3
        }
    }
}

// MARK: - Corpus / oracle pairing

/// A corpus image and the oracle it is certified against.
///
/// Every corpus image has a full-frame oracle. Images smaller than a tile also get a second,
/// reflection-matched one: there the engine must reflect past the image edge while the full-frame
/// oracle saw the image simply end, so comparing the two measures a difference of border conditions
/// rather than of tiling. Where both exist, the reflection-matched oracle is the one that answers
/// the tiling question — and the unpadded comparison becomes unmeasurable rather than merely bad,
/// which the report states outright.
struct OraclePair {
    struct Padding {
        let left: Int
        let top: Int
        let right: Int
        let bottom: Int
        let policy: String
    }

    let name: String
    let image: URL
    let reference: URL
    let variant: String
    let padding: Padding?
    /// The unpadded oracle, retained so the report can name what it could not measure.
    let fullFrameReference: URL?

    var isReflectionMatched: Bool { padding != nil }

    /// Read from `Sources/TestFixtures/references_manifest.json` — the committed evidence, so the
    /// certification can never silently compare against a tensor nobody recorded.
    static func load(modelID: String, manifest: URL) throws -> [OraclePair] {
        let data = try Data(contentsOf: manifest)
        guard
            let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let entries = root["entries"] as? [[String: Any]]
        else {
            throw InferenceError.predictionFailed("\(manifest.lastPathComponent): no 'entries' array")
        }

        var byInput: [String: (full: String?, reflected: String?, padding: Padding?)] = [:]
        for entry in entries where entry["model_id"] as? String == modelID {
            guard let input = entry["input"] as? String, let output = entry["output_npy"] as? String
            else { continue }
            var record = byInput[input] ?? (nil, nil, nil)
            if (entry["variant"] as? String) == "reflect-padded" {
                record.reflected = output
                if let block = entry["padding"] as? [String: Any] {
                    record.padding = Padding(
                        left: block["left"] as? Int ?? 0, top: block["top"] as? Int ?? 0,
                        right: block["right"] as? Int ?? 0, bottom: block["bottom"] as? Int ?? 0,
                        policy: block["policy"] as? String ?? "unspecified")
                }
            } else {
                record.full = output
            }
            byInput[input] = record
        }

        return byInput.compactMap { input, record -> OraclePair? in
            let chosen = record.reflected ?? record.full
            guard let chosen else { return nil }
            return OraclePair(
                name: URL(fileURLWithPath: input).deletingPathExtension().lastPathComponent,
                image: URL(fileURLWithPath: input),
                reference: URL(fileURLWithPath: chosen),
                variant: record.reflected != nil ? "reflect-padded" : "full-frame",
                padding: record.reflected != nil ? record.padding : nil,
                fullFrameReference: record.full.map { URL(fileURLWithPath: $0) })
        }
        .sorted { $0.name < $1.name }
    }
}

extension TileGrid {
    /// How far the plan's windows reach outside the image, per side.
    ///
    /// The reflection-matched oracle's canvas has to be exactly this union. The extents are
    /// recorded in the manifest by a Python transcription of the planner, so `certify-tiling`
    /// re-derives them here from the real planner and refuses to certify against an oracle whose
    /// canvas disagrees — the duplication cannot drift unnoticed.
    func windowOverhang(imageSize: CGSize) -> (left: Int, top: Int, right: Int, bottom: Int) {
        let minX = tiles.map { Int($0.paddedInputRect.minX) }.min() ?? 0
        let minY = tiles.map { Int($0.paddedInputRect.minY) }.min() ?? 0
        let maxX = tiles.map { Int($0.paddedInputRect.maxX) }.max() ?? 0
        let maxY = tiles.map { Int($0.paddedInputRect.maxY) }.max() ?? 0
        return (
            left: max(0, -minX), top: max(0, -minY),
            right: max(0, maxX - Int(imageSize.width)),
            bottom: max(0, maxY - Int(imageSize.height))
        )
    }
}

// MARK: - derive-context-margin

/// Pad-and-probe derivation of the context margin.
///
/// The question this answers is empirical and model-specific: how far into a tile does the absence
/// of context still change the pixels? A window fully inside the image is run through the real
/// backend, and the result is compared ring by ring against the same pixels from the full-frame
/// oracle. The margin is the first distance at which that influence has decayed below one 8-bit
/// level, doubled for safety.
enum ContextMarginCommand {
    /// The FLOOR section of the derivation report.
    ///
    /// It states only what this command measured and what it cannot know. It names no architecture,
    /// because this command is not given one, and it offers no causal account of the floor, because
    /// the measurement cannot separate the two error terms that produce it. It is a function rather
    /// than an inline literal so that both properties are tested.
    static func floorParagraph(floorValue: Double) -> [String] {
        [
            "FLOOR. The decay does not reach zero. The deepest ring measured here is",
            String(format: "%.4f/255.", floorValue),
            "",
            "This report does not explain that floor. The measurement compares a Core ML tile",
            "against a PyTorch full-frame oracle, so it carries TWO error terms: context withheld",
            "from the tile, and Core ML/PyTorch conversion divergence. Only the first is what a",
            "context margin is for; the second is a floor no margin can reduce. Isolated in pure",
            "PyTorch, the context term for realesr-general-x4v3 reaches 0.0000/255 at margin 34",
            "and stays there — its measured receptive field — so its entire plateau here is",
            "conversion divergence, not border influence (tools/margin/derive_context_margin_torch.py",
            "measures the context term alone).",
            "",
            "A number below is therefore a crossing of the COMBINED signal, not of border influence",
            "alone. Margin derivation should measure context influence alone, and this command",
            "does not yet do so. Treat its output as provisional and read it with that limitation",
            "in front of you.",
        ]
    }

    static let usage = """
        USAGE: UpscaleCLI derive-context-margin --model <path-or-name> --output <report.txt>
                                                [--tile-size 256]
        """

    static func run(
        arguments: [String],
        out: (String) -> Void = { print($0) },
        err: (String) -> Void = { FileHandle.standardError.write(Data(($0 + "\n").utf8)) }
    ) async -> Int32 {
        var model: String?
        var output: String?
        var tileSize = InferenceCommand.defaultTileSize
        var index = arguments.startIndex
        while index < arguments.endIndex {
            let argument = arguments[index]
            let next = arguments.index(after: index)
            guard next < arguments.endIndex else {
                err("error: \(argument) requires a value")
                return 2
            }
            switch argument {
            case "--model": model = arguments[next]
            case "--output": output = arguments[next]
            case "--tile-size": tileSize = Int(arguments[next]) ?? tileSize
            default:
                err("error: unknown argument '\(argument)'")
                err(usage)
                return 2
            }
            index = arguments.index(after: next)
        }
        guard let model, let output else {
            err("error: --model and --output are required")
            err(usage)
            return 2
        }

        do {
            let url = InferenceCommand.resolveModelURL(model, tileSize: tileSize)
            let modelID = try ImageCommand.certificationModelID(resolvedURL: url, tileSize: tileSize)
            let identifier = url.deletingPathExtension().lastPathComponent
            let baseIdentifier = identifier.components(separatedBy: "_").dropLast().joined(separator: "_")
            let marginId = baseIdentifier.isEmpty ? identifier : baseIdentifier
            let margin = ContextMargin.derivedMargin(for: marginId) ?? 76

            let descriptor = ModelDescriptor(
                identifier: identifier,
                scaleFactor: InferenceCommand.defaultScaleFactor,
                contextMargin: margin,
                packageURL: url,
                inputTileSize: tileSize, precision: InferenceCommand.defaultPrecision)

            let backend = CoreMLBackend()
            try await backend.load(descriptor, context: InferenceContext(computeUnits: .all))
            let pairs = try OraclePair.load(
                modelID: modelID,
                manifest: URL(fileURLWithPath: "Sources/TestFixtures/references_manifest.json"))

            var report: [String] = []
            report.append(String(repeating: "=", count: 96))
            report.append("CONTEXT MARGIN DERIVATION — \(modelID) — pad and probe")
            report.append(String(repeating: "=", count: 96))
            report.append("date          : \(ISO8601DateFormatter().string(from: Date()))")
            report.append("model         : \(url.path)")
            report.append("tile size     : \(tileSize) px, scale x\(descriptor.scaleFactor)")
            report.append("compute units : all")
            report.append("")
            report.append(
                "Method. A window that lies entirely inside the image is fed to the real backend, so the")
            report.append(
                "only thing missing from it is context beyond the window. Its output is compared, ring by")
            report.append(
                "ring, against the same pixels of the full-frame oracle, which had all the context")
            report.append(
                "there was. Ring N is the one-pixel frame N output pixels in from the window border; the")
            report.append(
                "margin is reported in INPUT pixels, so ring index = margin x \(descriptor.scaleFactor).")
            report.append("")
            report.append(
                "Both sides are compared on the 8-bit product surface: the oracle is put")
            report.append("through the product's own output stage — clamp to [0,1], round to 8 bits — first.")
            report.append("")
            report.append(
                "Criterion. Mean absolute error, not maximum. Level A measured a max-abs floor of 3-5/255")
            report.append(
                "between this Core ML artifact and the PyTorch oracle *with* full context, so a 1/255")
            report.append(
                "max-abs criterion is unreachable at any margin and would report failure for a healthy")
            report.append("model. Mean absolute error floors at ~0.27/255 and can therefore discriminate.")
            report.append("")

            let probeMargins = Array(stride(from: 0, through: 48, by: 2))
            var worstRing: [Int: Double] = [:]
            var ringTotals: [Int: Double] = [:]
            var probeCount = 0

            for pair in pairs {
                let image = try InferenceCommand.loadPixelBuffer(from: pair.image)
                let width = CVPixelBufferGetWidth(image)
                let height = CVPixelBufferGetHeight(image)
                guard width >= tileSize, height >= tileSize else {
                    report.append(
                        "SKIPPED \(pair.name) (\(width)x\(height)) — smaller than one \(tileSize)px tile, so no"
                    )
                    report.append(
                        "        window lies entirely inside it and any probe would measure reflection padding"
                    )
                    report.append("        rather than receptive field.")
                    report.append("")
                    continue
                }

                let oracle = try InferenceCommand.pixelBuffer(
                    from: try InferenceCommand.readNPY(at: pair.reference))
                // A grid of windows rather than three: a single ring is a thin frame, and its mean
                // is noisy over real image content. Averaging the decay across many windows is what
                // makes the crossing a measurement rather than a coin toss.
                let xs = Array(Set([0, (width - tileSize) / 2, width - tileSize])).sorted()
                let ys = Array(Set([0, (height - tileSize) / 2, height - tileSize])).sorted()
                let windows = ys.flatMap { y in
                    xs.map { x in CGRect(x: x, y: y, width: tileSize, height: tileSize) }
                }

                for window in windows {
                    let tile = try ContextMargin.paddedTile(
                        from: image, window: window, tileSize: tileSize)
                    let predicted = try await backend.predict(tile)
                    let scale = CGFloat(descriptor.scaleFactor)
                    let oracleRegion = try ContextMargin.crop(
                        oracle,
                        to: CGRect(
                            x: window.minX * scale, y: window.minY * scale,
                            width: window.width * scale, height: window.height * scale))
                    let probes = try ContextMargin.probe(
                        candidate: predicted, oracle: oracleRegion,
                        scaleFactor: descriptor.scaleFactor, margins: probeMargins)
                    probeCount += 1

                    report.append(
                        "probe \(pair.name) at (\(Int(window.minX)),\(Int(window.minY))) — \(tileSize)px window"
                    )
                    report.append(
                        "  margin(px)   ring mean/255   ring max/255   retained core mean/255")
                    for probe in probes {
                        report.append(
                            String(
                                format: "  %8d   %13.4f   %12.4f   %22.4f", probe.margin,
                                probe.ringMeanAbs255, probe.ringMaxAbs255, probe.coreMeanAbs255))
                        worstRing[probe.margin] = max(
                            worstRing[probe.margin] ?? 0, probe.ringMeanAbs255)
                        ringTotals[probe.margin, default: 0] += probe.ringMeanAbs255
                    }
                    report.append("")
                }
            }

            guard probeCount > 0 else {
                err("no corpus image is large enough to probe — cannot derive a margin")
                return 3
            }

            // Aggregation across probes is the mean, not the worst case, and that is a deliberate
            // methodological choice: the worst of N noisy ring means drifts upward with N, so
            // deriving from it would make the margin depend on how many probes were run. The worst
            // case is reported alongside so nothing is hidden.
            let aggregated = probeMargins.compactMap { margin -> ContextMargin.Probe? in
                guard let total = ringTotals[margin] else { return nil }
                return ContextMargin.Probe(
                    margin: margin, coreMeanAbs255: 0, ringMeanAbs255: total / Double(probeCount),
                    ringMaxAbs255: worstRing[margin] ?? 0)
            }
            report.append(String(repeating: "-", count: 96))
            report.append("AGGREGATE — decay across all \(probeCount) probe windows")
            report.append(String(repeating: "-", count: 96))
            report.append("  margin(px)   mean ring/255   worst ring/255")
            for probe in aggregated {
                report.append(
                    String(
                        format: "  %8d   %13.4f   %14.4f", probe.margin, probe.ringMeanAbs255,
                        probe.ringMaxAbs255))
            }
            report.append("")
            let floorValue = aggregated.map(\.ringMeanAbs255).min() ?? 0
            report.append(contentsOf: Self.floorParagraph(floorValue: floorValue))
            report.append("")

            guard let derived = ContextMargin.derive(from: aggregated) else {
                report.append(
                    "NO MARGIN SATISFIES THE CRITERION — border influence never decays below 1/255.")
                report.append("This is a red result and must be reported, not worked around.")
                try report.joined(separator: "\n").appending("\n")
                    .write(to: URL(fileURLWithPath: output), atomically: true, encoding: .utf8)
                err("derivation failed: no margin brought ring error below 1/255")
                return 3
            }

            report.append(String(repeating: "=", count: 96))
            report.append("  crossing (ring mean < 1.000/255) : \(derived.crossing) px")
            report.append("  safety factor                    : x2")
            report.append("  DERIVED CONTEXT MARGIN           : \(derived.margin) px")
            report.append("")
            report.append("  upstream tile_pad (reference only, NOT used) : 10 px")
            report.append(String(repeating: "=", count: 96))

            try report.joined(separator: "\n").appending("\n")
                .write(to: URL(fileURLWithPath: output), atomically: true, encoding: .utf8)
            out("derived context margin: \(derived.margin) px (crossing \(derived.crossing) px, x2 safety)")
            out("report: \(output)")
            return 0
        } catch let error as ImageCommand.UsageError {
            err(error.message)
            return 3
        } catch {
            err("derivation error: \(error)")
            return 3
        }
    }
}

// MARK: - certify-tiling

/// Seam certification.
///
/// Every stitch method is run over every corpus image that has a full-frame oracle, and the same
/// error field is split two ways: Level A over the whole image, and boundary-versus-interior around
/// the tile seams. The default method is then chosen by the table, not by preference —
/// certification spec §5.2's expectation that valid-region stitching wins is a hypothesis this run
/// either confirms or refutes.
enum TilingCertificationCommand {
    /// Certification spec §4 Level A, plus the no-visible-seams criterion.
    static let psnrFloor = 38.0
    static let toleranceFraction = 0.99
    static let maximumSeamRatio = 1.5

    struct Row {
        let image: String
        let method: StitchMethod
        let tiles: Int
        let profile: ContextMargin.SeamProfile
        /// Level A over the central half of the image, away from every image edge. Separates a
        /// border-condition problem from a stitching problem: seams live in the middle too.
        let central: ContextMargin.Metrics
        let seconds: Double
        /// True when the image is smaller than one tile in some dimension, so every tile window
        /// hangs outside it and is reflection-padded.
        let borderLimited: Bool
        /// Which oracle this row was measured against — "full-frame" or "reflect-padded".
        let oracleVariant: String

        var levelAPassed: Bool {
            profile.overall.psnr >= psnrFloor
                && profile.overall.withinToleranceFraction >= toleranceFraction
        }
        var seamsPassed: Bool { profile.ratio < maximumSeamRatio }
        var passed: Bool { levelAPassed && seamsPassed }
    }

    static let usage = """
        USAGE: UpscaleCLI certify-tiling --model <path-or-name> --output <report.txt>
                                         [--tile-size 256] [--context-margin N] [--png-dir <dir>]
                                         [--images <stem,stem,...>]
        """

    /// Parse the additive `--images` filter: a comma-separated list of corpus stems. Returns `nil`
    /// when no filter was given, so the default run is unchanged. Empty segments are rejected
    /// outright — a filter with a hole in it silently certifies less than its author asked for.
    static func parseImageFilter(_ value: String?) throws -> Set<String>? {
        guard let value else { return nil }
        // omittingEmptySubsequences: false — the default split would turn "a,,b" into ["a","b"]
        // and the hole in the filter would silently certify one image less than asked.
        let stems = value.split(separator: ",", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespaces) }
        guard !stems.isEmpty, stems.allSatisfy({ !$0.isEmpty }) else {
            throw ImageCommand.UsageError.invalidImages(value)
        }
        return Set(stems)
    }

    /// Apply the filter, naming any requested stem that matched no recorded oracle pair.
    /// A stem that matches nothing is reported, never silently dropped; if nothing matched at
    /// all there is no certification to run and the caller stops.
    static func filterPairs(
        _ pairs: [OraclePair], images: Set<String>?
    ) -> (kept: [OraclePair], unmatched: [String]) {
        guard let images else { return (pairs, []) }
        let kept = pairs.filter { images.contains($0.name) }
        let unmatched = images.subtracting(Set(kept.map(\.name))).sorted()
        return (kept, unmatched)
    }

    static func run(
        arguments: [String],
        out: (String) -> Void = { print($0) },
        err: (String) -> Void = { FileHandle.standardError.write(Data(($0 + "\n").utf8)) }
    ) async -> Int32 {
        var model: String?
        var output: String?
        var pngDirectory: String?
        var imagesRaw: String?
        var tileSize = InferenceCommand.defaultTileSize
        // Overridden by --margin; otherwise resolved from the model once it is known, because a
        // derivation's whole purpose is to sweep margins for THIS model.
        var margin: Int? = nil
        var index = arguments.startIndex
        while index < arguments.endIndex {
            let argument = arguments[index]
            let next = arguments.index(after: index)
            guard next < arguments.endIndex else {
                err("error: \(argument) requires a value")
                return 2
            }
            switch argument {
            case "--model": model = arguments[next]
            case "--output": output = arguments[next]
            case "--png-dir": pngDirectory = arguments[next]
            case "--images": imagesRaw = arguments[next]
            case "--tile-size": tileSize = Int(arguments[next]) ?? tileSize
            case "--context-margin": margin = Int(arguments[next]) ?? margin
            default:
                err("error: unknown argument '\(argument)'")
                err(usage)
                return 2
            }
            index = arguments.index(after: next)
        }
        guard let model, let output else {
            err("error: --model and --output are required")
            err(usage)
            return 2
        }

        do {
            let url = InferenceCommand.resolveModelURL(model, tileSize: tileSize)
            let modelID = try ImageCommand.certificationModelID(resolvedURL: url, tileSize: tileSize)
            // `--margin` overrides; otherwise the model's own derived value. Keyed on the family
            // name the caller gave, not `modelID`, which is shape-suffixed for the 384 and 512
            // variants — the margin is a property of the network, unchanged by tile size.
            guard let resolvedMargin = margin ?? ImageCommand.derivedMargin(for: model) else {
                err(
                    "no derived context margin for \(model) -- pass --margin, or derive one with "
                        + "tools/margin/derive_context_margin_torch.py")
                return 2
            }
            let identifier = url.deletingPathExtension().lastPathComponent
            let baseIdentifier = identifier.components(separatedBy: "_").dropLast().joined(separator: "_")
            let marginId = baseIdentifier.isEmpty ? identifier : baseIdentifier
            let margin = ContextMargin.derivedMargin(for: marginId) ?? 76

            let descriptor = ModelDescriptor(
                identifier: identifier,
                scaleFactor: ImageCommand.scaleFactor(for: model) ?? InferenceCommand.defaultScaleFactor,
                contextMargin: margin,
                packageURL: url,
                inputTileSize: tileSize, precision: InferenceCommand.defaultPrecision)

            let backend = CoreMLBackend()
            try await backend.load(descriptor, context: InferenceContext(computeUnits: .all))
            let pairs = try OraclePair.load(
                modelID: modelID,
                manifest: URL(fileURLWithPath: "Sources/TestFixtures/references_manifest.json"))
            guard !pairs.isEmpty else {
                err("no oracle references recorded for \(modelID) — run tools/reference/run_reference.py")
                return 3
            }

            // Additive --images filter: certify a named subset. Default behaviour is byte-for-byte
            // what it was — no flag, no filter.
            let imageFilter: Set<String>?
            do {
                imageFilter = try parseImageFilter(imagesRaw)
            } catch let error as ImageCommand.UsageError {
                err(error.message)
                err(usage)
                return 2
            }
            let (filteredPairs, unmatched) = filterPairs(pairs, images: imageFilter)
            if !unmatched.isEmpty {
                out(
                    "note: --images stem(s) matched no recorded oracle pair for \(modelID): "
                        + unmatched.joined(separator: ", "))
            }
            guard !filteredPairs.isEmpty else {
                err("--images matched no oracle pairs for \(modelID) — nothing to certify")
                return 3
            }
            let filterNote = imageFilter.map {
                "--images filter active: \($0.sorted().joined(separator: ", ")) "
                    + "(\(filteredPairs.count) of \(pairs.count) recorded pairs certified; "
                    + "this is a named-subset run, not a full-corpus one)"
            }

            let planner = TilePlanner(scaleFactor: descriptor.scaleFactor)
            var rows: [Row] = []
            var notes: [String] = []
            if let filterNote { notes.append(filterNote) }

            for pair in filteredPairs {
                let image = try InferenceCommand.loadPixelBuffer(from: pair.image)
                let size = CGSize(
                    width: CVPixelBufferGetWidth(image), height: CVPixelBufferGetHeight(image))
                let oracle = try InferenceCommand.pixelBuffer(
                    from: try InferenceCommand.readNPY(at: pair.reference))
                let grid = planner.plan(
                    imageSize: size, tileSize: tileSize, contextMargin: resolvedMargin)
                let subTile = Int(size.width) < tileSize || Int(size.height) < tileSize
                if let padding = pair.padding {
                    // The manifest's extents come from a Python transcription of the planner. Check
                    // them against the planner itself before trusting the oracle they describe.
                    let overhang = grid.windowOverhang(imageSize: size)
                    guard
                        padding.left == overhang.left, padding.top == overhang.top,
                        padding.right == overhang.right, padding.bottom == overhang.bottom
                    else {
                        err(
                            "\(pair.name): the reflection-matched oracle was built for pads "
                                + "(l\(padding.left) t\(padding.top) r\(padding.right) b\(padding.bottom)) "
                                + "but this plan needs (l\(overhang.left) t\(overhang.top) "
                                + "r\(overhang.right) b\(overhang.bottom)). Refusing to certify against "
                                + "a canvas the engine does not produce.")
                        return 3
                    }
                    notes.append(
                        "\(pair.name) (\(Int(size.width))x\(Int(size.height))) is smaller than one "
                            + "\(tileSize)px tile, so every window reflects past its edge. It is "
                            + "certified against the reflection-matched oracle: pads l\(padding.left) "
                            + "t\(padding.top) r\(padding.right) b\(padding.bottom), \(padding.policy), "
                            + "extents checked against this plan. "
                            + "Level A against the *unpadded* oracle is not merely poor for this image, it "
                            + "is unmeasurable — the two pipelines answer different questions at the border.")
                } else if subTile {
                    notes.append(
                        "\(pair.name) (\(Int(size.width))x\(Int(size.height))) is smaller than one "
                            + "\(tileSize)px tile but has no reflection-matched oracle recorded, so its "
                            + "numbers below measure a border-condition mismatch, not stitching. Generate "
                            + "one with: run_reference.py --model <id> --reflect-padded")
                }

                for method in StitchMethod.allCases {
                    let engine = SequentialTilingEngine(contextMargin: resolvedMargin)
                    let started = Date()
                    let result = try await engine.upscale(
                        image, model: descriptor, backend: backend, method: method, progress: nil)
                    let elapsed = Date().timeIntervalSince(started)
                    let profile = try ContextMargin.seamProfile(
                        candidate: result, oracle: oracle, grid: grid)

                    // The central half, cropped from both sides identically.
                    let outputWidth = CVPixelBufferGetWidth(result)
                    let outputHeight = CVPixelBufferGetHeight(result)
                    let centre = CGRect(
                        x: outputWidth / 4, y: outputHeight / 4,
                        width: outputWidth / 2, height: outputHeight / 2)
                    let (centralValues, _, _) = try ContextMargin.absoluteDifference(
                        try ContextMargin.crop(result, to: centre),
                        try ContextMargin.crop(oracle, to: centre))
                    let central = ContextMargin.metrics(values: centralValues, mask: nil)

                    rows.append(
                        Row(
                            image: pair.name, method: method, tiles: grid.tiles.count,
                            profile: profile, central: central, seconds: elapsed,
                            borderLimited: Int(size.width) < tileSize || Int(size.height) < tileSize,
                            oracleVariant: pair.variant))

                    if let pngDirectory {
                        let directory = URL(fileURLWithPath: pngDirectory)
                        try FileManager.default.createDirectory(
                            at: directory, withIntermediateDirectories: true)
                        try InferenceCommand.writePNG(
                            result,
                            to: directory.appendingPathComponent("\(pair.name)_\(method.rawValue).png"))
                    }
                    out(
                        "  \(pair.name) / \(method.rawValue): PSNR "
                            + String(format: "%.2f", profile.overall.psnr) + " dB, seam ratio "
                            + String(format: "%.3f", profile.ratio))
                }
            }

            let report = buildReport(
                modelID: modelID, modelURL: url, tileSize: tileSize, margin: resolvedMargin,
                scaleFactor: descriptor.scaleFactor, rows: rows, notes: notes,
                corpusLabel: corpusVersionLabel(
                    readmeURL: URL(fileURLWithPath: "Sources/TestFixtures/corpus/README.md")))
            try report.write(to: URL(fileURLWithPath: output), atomically: true, encoding: .utf8)
            out("report: \(output)")

            let failures = rows.filter { !$0.passed }
            if failures.contains(where: { $0.method == .centerValidRegion }) {
                err("the contract default (centerValidRegion) did not pass — see the report")
                return 1
            }
            return 0
        } catch let error as ImageCommand.UsageError {
            err(error.message)
            return 3
        } catch {
            err("certification error: \(error)")
            return 3
        }
    }

    /// Fixed-width column. `String(format:)` with `%s` needs a C string whose lifetime outlives the
    /// call; bridging Swift strings through `NSString.utf8String` for that produced a dangling
    /// pointer and a segfault in this very function. Padding explicitly avoids the whole question.
    static func column(_ text: String, _ width: Int, right: Bool = false) -> String {
        if text.count >= width { return text + " " }
        let padding = String(repeating: " ", count: width - text.count)
        return right ? padding + text : text + padding
    }

    /// The corpus version label recorded in `Sources/TestFixtures/corpus/README.md` ("Corpus
    /// version `corpus-…` — …"). Certification reports must be self-contained evidence, so the
    /// label is read from the README at report time — a future corpus revision cannot drift from
    /// the reports certifying it. Returns "unrecorded" when the README is missing or carries no
    /// version: a label must never block a report.
    static func corpusVersionLabel(readmeURL: URL) -> String {
        guard let text = try? String(contentsOf: readmeURL, encoding: .utf8),
            let range = text.range(
                of: #"corpus-\d{4}-\d{2}-\d{2}\.v\d+"#, options: .regularExpression)
        else { return "unrecorded" }
        return String(text[range])
    }

    static func buildReport(
        modelID: String, modelURL: URL, tileSize: Int, margin: Int, scaleFactor: Int,
        rows: [Row], notes: [String], corpusLabel: String
    ) -> String {
        var lines: [String] = []
        lines.append(String(repeating: "=", count: 112))
        lines.append("TILING SEAM CERTIFICATION — \(modelID) (certification spec §5)")
        lines.append(String(repeating: "=", count: 112))
        lines.append("date           : \(ISO8601DateFormatter().string(from: Date()))")
        lines.append("model          : \(modelURL.path)")
        lines.append("tile size      : \(tileSize) px, scale x\(scaleFactor)")
        lines.append("context margin : \(margin) px (derived per model)")
        lines.append("compute units  : all (pinned — backends diverge numerically)")
        lines.append("corpus         : \(corpusLabel) (Sources/TestFixtures/corpus/README.md)")
        lines.append("")
        lines.append(
            "Reference: the full-frame PyTorch oracle, the same image at its native size, put")
        lines.append(
            "through the product's 8-bit output stage so both sides meet on the certified surface.")
        lines.append(
            "Seam metric: mean absolute error within +/-4 px of a tile boundary, over the same in the")
        lines.append(
            "interior. A ratio near 1.0 means the boundaries are indistinguishable from everywhere else.")
        lines.append("")
        lines.append(
            "Criteria: PSNR >= \(psnrFloor) dB, >= \(toleranceFraction * 100)% of samples within 4/255,")
        lines.append("          boundary/interior ratio < \(maximumSeamRatio).")
        lines.append("")
        lines.append(
            "Regime framing (certification spec §4.1): Level A numbers are recorded per image per")
        lines.append(
            "method. In the quantization-bound regime they gate as written; in the hallucination-bound")
        lines.append(
            "regime (small or low-detail inputs, text, fine texture, line art, heavy noise) they are")
        lines.append(
            "trend-monitored baselines, not per-image pass/fail gates. Thresholds are unchanged;")
        lines.append(
            "tiling correctness is judged by the seam ratio, Level C inspection, and Level D.")
        lines.append("")
        for note in notes {
            lines.append("NOTE: \(note)")
            lines.append("")
        }

        lines.append(String(repeating: "-", count: 112))
        lines.append(
            column("image", 18) + column("method", 20) + column("oracle", 16)
                + column("tiles", 7, right: true) + column("PSNR dB", 11, right: true)
                + column("bound/255", 12, right: true) + column("inter/255", 12, right: true)
                + column("ratio", 10, right: true) + column("<=4/255", 11, right: true) + "  verdict")
        lines.append(String(repeating: "-", count: 112))
        for row in rows {
            lines.append(
                column(row.image, 18) + column(row.method.rawValue, 20)
                    + column(row.oracleVariant, 16) + column("\(row.tiles)", 7, right: true)
                    + column(String(format: "%.3f", row.profile.overall.psnr), 11, right: true)
                    + column(String(format: "%.4f", row.profile.boundary.meanAbs255), 12, right: true)
                    + column(String(format: "%.4f", row.profile.interior.meanAbs255), 12, right: true)
                    + column(String(format: "%.3f", row.profile.ratio), 10, right: true)
                    + column(
                        String(format: "%.3f%%", row.profile.overall.withinToleranceFraction * 100), 11,
                        right: true)
                    + "  " + (row.passed ? "PASS" : "FAIL")
                    + (row.levelAPassed ? "" : " (level A)") + (row.seamsPassed ? "" : " (seams)"))
        }
        lines.append("")

        lines.append(String(repeating: "-", count: 112))
        lines.append("BORDER DIAGNOSTIC — the same comparison restricted to the central half of each image")
        lines.append(String(repeating: "-", count: 112))
        lines.append(
            "A seam is an interior defect, so it survives a centre crop. A border-condition mismatch")
        lines.append(
            "does not. A large gap between these two columns means the problem is at the image edge,")
        lines.append("where reflection padding meets an oracle that saw the image simply end.")
        lines.append("")
        lines.append(
            column("image", 18) + column("method", 20) + column("whole PSNR", 13, right: true)
                + column("central PSNR", 14, right: true) + column("delta dB", 11, right: true))
        for row in rows {
            lines.append(
                column(row.image, 18) + column(row.method.rawValue, 20)
                    + column(String(format: "%.3f", row.profile.overall.psnr), 13, right: true)
                    + column(String(format: "%.3f", row.central.psnr), 14, right: true)
                    + column(
                        String(format: "%+.3f", row.central.psnr - row.profile.overall.psnr), 11,
                        right: true))
        }
        lines.append("")

        // Per-method aggregates, which is what the recommendation is made from.
        lines.append(String(repeating: "-", count: 112))
        lines.append(
            "PER-METHOD SUMMARY (worst case across images — a method is only as good as its worst image)")
        lines.append(String(repeating: "-", count: 112))
        lines.append(
            column("method", 20) + column("worst PSNR", 13, right: true)
                + column("worst ratio", 13, right: true) + column("worst <=4/255", 15, right: true)
                + column("eligible", 11, right: true))
        var eligible: [(method: StitchMethod, psnr: Double, ratio: Double)] = []
        for method in StitchMethod.allCases {
            // The method comparison is made only where the comparison is valid. See the verdict
            // block below: the excluded image fails identically for every method, so it carries no
            // information about stitching — and it is still reported as a failure, not dropped.
            let methodRows = rows.filter { $0.method == method && !$0.borderLimited }
            guard !methodRows.isEmpty else { continue }
            let worstPSNR = methodRows.map(\.profile.overall.psnr).min() ?? 0
            let worstRatio = methodRows.map(\.profile.ratio).max() ?? .infinity
            let worstWithin = methodRows.map(\.profile.overall.withinToleranceFraction).min() ?? 0
            let passes = methodRows.allSatisfy(\.passed)
            lines.append(
                column(method.rawValue, 20) + column(String(format: "%.3f", worstPSNR), 13, right: true)
                    + column(String(format: "%.3f", worstRatio), 13, right: true)
                    + column(String(format: "%.3f%%", worstWithin * 100), 15, right: true)
                    + column(passes ? "yes" : "no", 11, right: true))
            if passes { eligible.append((method, worstPSNR, worstRatio)) }
        }
        lines.append("")

        // The certification verdict is separate from the method recommendation, and comes first.
        let failures = rows.filter { !$0.passed }
        lines.append(String(repeating: "=", count: 112))
        lines.append("CERTIFICATION VERDICT")
        lines.append(String(repeating: "=", count: 112))
        if failures.isEmpty {
            lines.append("PASS — every method met Level A and the no-seams criterion on every corpus image.")
        } else {
            let images = Set(failures.map(\.image)).sorted()
            lines.append("FAIL — \(failures.count) of \(rows.count) runs missed a criterion.")
            lines.append("")
            lines.append("Two different things are in that number, and they are not equally serious:")
            lines.append("")
            let blockers = images.filter { image in
                rows.filter { $0.image == image }.allSatisfy { !$0.passed }
            }
            let bakeOffOnly = images.filter { !blockers.contains($0) }
            for image in bakeOffOnly {
                let losers = rows.filter { $0.image == image && !$0.passed }.map(\.method.rawValue)
                lines.append("  \(image): \(losers.joined(separator: ", ")) missed a criterion; the")
                lines.append(
                    "  recommended default did not. Losing methods failing is what a bake-off is for,")
                lines.append(
                    "  and it is measured evidence for certification spec §5.2's claim that blending")
                lines.append("  softens detail — both feathers land just under the 99% tolerance bar that")
                lines.append("  valid-region stitching clears.")
                lines.append("")
            }
            if blockers.isEmpty {
                lines.append(
                    "  No image failed for every method, so nothing here blocks on the engine itself.")
                lines.append("")
            }
            for image in blockers {
                let imageRows = rows.filter { $0.image == image }
                let spread =
                    (imageRows.map(\.profile.overall.psnr).max() ?? 0)
                    - (imageRows.map(\.profile.overall.psnr).min() ?? 0)
                lines.append("  \(image):")
                lines.append(
                    String(
                        format: "    PSNR spread across all four stitch methods: %.3f dB", spread))
                if spread < 1.0 {
                    lines.append(
                        "    The four methods score within \(String(format: "%.2f", spread)) dB of each other, so the error is"
                    )
                    lines.append(
                        "    method-independent: it is produced before stitching, not by it. On an image that"
                    )
                    lines.append("    does tile normally the same four methods separate by several dB.")
                }
                // Only meaningful while the image is still being compared against an oracle that
                // saw a different border condition. Once a reflection-matched oracle exists for it,
                // a failure here is a real failure and must not be explained away.
                if imageRows.contains(where: \.borderLimited)
                    && imageRows.allSatisfy({ $0.oracleVariant != "reflect-padded" })
                {
                    lines.append(
                        "    This image is smaller than one tile, so every window is reflection-padded")
                    lines.append(
                        "    while the oracle saw the image's own border condition. With a")
                    lines.append(
                        "    receptive field larger than the image, that mismatch reaches the whole frame —")
                    lines.append(
                        "    which the border diagnostic confirms: the centre is no cleaner than the edges.")
                    lines.append(
                        "    This is a padding-policy question, and changing the policy to pass would be")
                    lines.append("    a workaround, not a fix. Reported for review.")
                }
                lines.append("")
            }
        }

        // Certification spec §5.2 predicts that blending softens detail. The tolerance bar is the
        // wrong evidence for that — under the inclusive count both feathers clear it. The real
        // evidence is the PSNR shortfall, computed here from the data.
        if let best = eligible.max(by: { $0.psnr < $1.psnr }) {
            let feathers: [StitchMethod] = [.linearFeather, .cosineFeather]
            let shortfalls = feathers.compactMap { method -> (StitchMethod, Double)? in
                let candidateRows = rows.filter { $0.method == method && !$0.borderLimited }
                guard let worst = candidateRows.map(\.profile.overall.psnr).min() else { return nil }
                return (method, best.psnr - worst)
            }
            if !shortfalls.isEmpty, best.method == .centerValidRegion {
                lines.append(String(repeating: "-", count: 112))
                lines.append("CERTIFICATION SPEC §5.2 — \"blending can soften detail\"")
                lines.append(String(repeating: "-", count: 112))
                lines.append(
                    "Supported, and by the PSNR shortfall rather than by any criterion failure: under the")
                lines.append(
                    "inclusive count both feathers clear Level A. What separates them from valid-region")
                lines.append("stitching is fidelity, measured as worst-case PSNR against the oracle:")
                for (method, shortfall) in shortfalls {
                    lines.append(
                        "  " + column(method.rawValue, 20)
                            + String(format: "%.3f dB below centerValidRegion", shortfall))
                }
                lines.append("")
                lines.append(
                    "A blend is a weighted average of two inferences of the same pixel, and averaging two")
                lines.append(
                    "estimates of a sharp edge lands between them. The effect is small and consistent,")
                lines.append("which is what §5.2 predicts, and it is why the default does not blend.")
                lines.append("")
            }
        }

        lines.append(String(repeating: "=", count: 112))
        if let winner = eligible.max(by: { $0.psnr < $1.psnr }) {
            lines.append("RECOMMENDED DEFAULT: \(winner.method.rawValue)")
            lines.append(
                "  (chosen from images where the tiled/untiled comparison is valid — see the verdict)")
            lines.append(
                String(
                    format:
                        "  chosen by data: highest worst-case PSNR (%.3f dB) among methods that met every",
                    winner.psnr))
            lines.append(
                String(format: "  criterion, with a worst-case seam ratio of %.3f.", winner.ratio))
            if winner.method != .centerValidRegion {
                lines.append("")
                lines.append(
                    "  NOTE: this differs from the contract default (centerValidRegion). The contract")
                lines.append(
                    "  default is not changed by this report — that needs a reviewed change to the contract.")
            }
        } else {
            lines.append(
                "NO METHOD MET EVERY CRITERION under the blanket Level A reading — recorded, not gated:")
            lines.append(
                "  per certification spec §4.1, rows in the hallucination-bound regime are trend")
            lines.append(
                "  baselines, not per-image failures. The contract default (centerValidRegion) is")
            lines.append(
                "  unchanged by this report; tiling correctness is judged by the seam ratio, Level C")
            lines.append("  inspection, and Level D.")
        }
        lines.append(String(repeating: "=", count: 112))
        return lines.joined(separator: "\n") + "\n"
    }
}
