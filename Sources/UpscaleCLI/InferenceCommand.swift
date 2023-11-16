import Accelerate
import CoreGraphics
import CoreVideo
import Foundation
import ImageIO
import UniformTypeIdentifiers
import UpscaleCore

/// `upscale-tile` — run one fixed-shape tile through the Core ML backend.
///
/// The PNG handling here is deliberately utilitarian: no colour management, no ICC handling, no
/// metadata preservation. Correct colour is a whole subsystem (certification spec §8) and belongs
/// to `ImagePipeline` — this command sits below that boundary on purpose.
enum InferenceCommand {
    struct Options: Equatable {
        var model: String
        var input: URL
        var output: URL
        var computeUnits: ComputeUnits
        /// Optional dump of the 8-bit product surface widened to float32, for the certification
        /// harness. Defaults to nil, so this command's behaviour is unchanged when the flag is
        /// absent.
        var emitNPY: URL?
        /// The fixed input shape of the model being run. Defaults to `defaultTileSize`, so every
        /// invocation written before the 384 and 512 variants existed is unchanged. `ImageCommand`
        /// wires its `--tile-size` the same way.
        var tileSize: Int
    }

    enum UsageError: Error, Equatable {
        case missingValue(flag: String)
        case unknownArgument(String)
        case missingRequired(flag: String)
        case invalidComputeUnits(String)
        case invalidTileSize(String)

        var message: String {
            switch self {
            case .missingValue(let flag): return "error: \(flag) requires a value"
            case .unknownArgument(let argument): return "error: unknown argument '\(argument)'"
            case .missingRequired(let flag): return "error: \(flag) is required"
            case .invalidComputeUnits(let value):
                let valid = ComputeUnits.allCases.map(\.rawValue).joined(separator: "|")
                return "error: invalid --compute-units '\(value)' (expected \(valid))"
            case .invalidTileSize(let value):
                return "error: invalid --tile-size '\(value)' (expected a positive integer)"
            }
        }
    }

    // Utilitarian defaults. The certification path sources these from the model manifest instead of
    // assuming them; the reference artifact is x4, 256px, fp16.
    static let defaultScaleFactor = 4
    static let defaultTileSize = 256
    static let defaultPrecision = Precision.float16

    static let usage = """
        USAGE: UpscaleCLI upscale-tile --model <path-or-name> --input <png> --output <png>
                                       [--compute-units all|cpuAndGPU|cpuAndNeuralEngine|cpuOnly]
                                       [--emit-npy <path.npy>] [--tile-size 256]

        Runs a single fixed-shape tile through the Core ML backend (default shape
        \(defaultTileSize)x\(defaultTileSize); --tile-size selects another converted variant).

        --emit-npy additionally writes the 8-bit product surface widened to float32, as a NumPy
        `.npy` file: NCHW (1,3,H,W), C order, little-endian — the convention that
        tools/reference/run_reference.py uses. This is what `InferenceBackend.predict` returns and
        what the user receives; it is not the model's pre-quantisation float output.
        Exit codes: 0 success, 2 usage error, 3 inference error.
        """

    // MARK: - Argument parsing (pure, so it is testable without a model)

    static func parse(_ arguments: [String]) throws -> Options {
        var model: String?
        var input: String?
        var output: String?
        var computeUnits = ComputeUnits.all
        var emitNPY: String?
        var tileSize = defaultTileSize

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
            case "--compute-units":
                let raw = try value(for: "--compute-units")
                guard let parsed = ComputeUnits(rawValue: raw) else {
                    throw UsageError.invalidComputeUnits(raw)
                }
                computeUnits = parsed
            case "--emit-npy": emitNPY = try value(for: "--emit-npy")
            case "--tile-size":
                let raw = try value(for: "--tile-size")
                guard let parsed = Int(raw), parsed > 0 else { throw UsageError.invalidTileSize(raw) }
                tileSize = parsed
            default:
                throw UsageError.unknownArgument(argument)
            }
            index = arguments.index(after: index)
        }

        guard let model else { throw UsageError.missingRequired(flag: "--model") }
        guard let input else { throw UsageError.missingRequired(flag: "--input") }
        guard let output else { throw UsageError.missingRequired(flag: "--output") }

        return Options(
            model: model,
            input: URL(fileURLWithPath: input),
            output: URL(fileURLWithPath: output),
            computeUnits: computeUnits,
            emitNPY: emitNPY.map { URL(fileURLWithPath: $0) },
            tileSize: tileSize
        )
    }

    /// `--model` accepts a path to a `.mlpackage` or a bare model id, resolved to the conversion
    /// pipeline's layout: `models/<id>/output/<id>_<tile>x<tile>.mlpackage`.
    static func resolveModelURL(_ value: String, tileSize: Int = defaultTileSize) -> URL {
        if value.hasSuffix(".mlpackage") || value.contains("/") {
            return URL(fileURLWithPath: value)
        }
        return URL(fileURLWithPath: "models/\(value)/output/\(value)_\(tileSize)x\(tileSize).mlpackage")
    }

    static func descriptor(for options: Options) -> ModelDescriptor {
        let url = resolveModelURL(options.model, tileSize: options.tileSize)
        let identifier = url.deletingPathExtension().lastPathComponent
        let baseIdentifier = identifier.components(separatedBy: "_").dropLast().joined(separator: "_")
        let marginId = baseIdentifier.isEmpty ? identifier : baseIdentifier
        let margin = ContextMargin.derivedMargin(for: marginId) ?? 76

        return ModelDescriptor(
            identifier: identifier,
            scaleFactor: defaultScaleFactor,
            contextMargin: margin,
            packageURL: url,
            inputTileSize: options.tileSize,
            precision: defaultPrecision
        )
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

        let descriptor = descriptor(for: options)
        do {
            let tile = try loadPixelBuffer(from: options.input)
            let backend = CoreMLBackend()
            try await backend.load(descriptor, context: InferenceContext(computeUnits: options.computeUnits))
            let result = try await backend.predict(tile)
            try writePNG(result, to: options.output)
            if let npyURL = options.emitNPY { try writeNPY(result, to: npyURL) }

            out("model         : \(descriptor.packageURL.path)")
            out("compute units : \(options.computeUnits.rawValue)")
            out(
                "input         : \(options.input.path) (\(CVPixelBufferGetWidth(tile))x\(CVPixelBufferGetHeight(tile)))"
            )
            out(
                "output        : \(options.output.path) (\(CVPixelBufferGetWidth(result))x\(CVPixelBufferGetHeight(result)))"
            )
            if let npyURL = options.emitNPY {
                let shape = "1x3x\(CVPixelBufferGetHeight(result))x\(CVPixelBufferGetWidth(result))"
                out("tensor        : \(npyURL.path) (float32 NCHW \(shape))")
            }
            return 0
        } catch let error as InferenceError {
            err("inference error: \(error)")
            return 3
        } catch {
            err("inference error: \(error.localizedDescription)")
            return 3
        }
    }

    // MARK: - PNG <-> CVPixelBuffer

    static func loadPixelBuffer(from url: URL) throws -> CVPixelBuffer {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
            let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            throw InferenceError.predictionFailed("could not decode image at \(url.path)")
        }

        let width = image.width
        let height = image.height
        var buffer: CVPixelBuffer?
        let attributes: [CFString: Any] = [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA,
            attributes as CFDictionary, &buffer
        )
        guard status == kCVReturnSuccess, let pixelBuffer = buffer else {
            throw InferenceError.predictionFailed("could not allocate \(width)x\(height) input buffer")
        }

        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }
        guard
            let context = CGContext(
                data: CVPixelBufferGetBaseAddress(pixelBuffer),
                width: width, height: height, bitsPerComponent: 8,
                bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer),
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
            )
        else {
            throw InferenceError.predictionFailed("could not create a drawing context for \(url.path)")
        }
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return pixelBuffer
    }

    static func writePNG(_ buffer: CVPixelBuffer, to url: URL) throws {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }

        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        guard
            let context = CGContext(
                data: CVPixelBufferGetBaseAddress(buffer),
                width: width, height: height, bitsPerComponent: 8,
                bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
            ), let image = context.makeImage()
        else {
            throw InferenceError.predictionFailed("could not render the output buffer")
        }

        guard
            let destination = CGImageDestinationCreateWithURL(
                url as CFURL, UTType.png.identifier as CFString, 1, nil
            )
        else {
            throw InferenceError.predictionFailed("could not create a PNG writer for \(url.path)")
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw InferenceError.predictionFailed("could not write PNG to \(url.path)")
        }
    }

    // MARK: - NumPy tensor dump
    //
    // The certification harness compares this command's output against the PyTorch oracle, whose
    // tensors are written by `numpy.save(..., allow_pickle=False)`. Emitting that same container —
    // rather than inventing a format and teaching Python to read it — is what makes the comparison
    // honest: the harness loads both sides with one `np.load`.
    //
    // NPY v1.0 (numpy/lib/format.py): magic "\u{93}NUMPY", version 1.0, a little-endian uint16
    // header length, then a header dict padded with spaces so the array data begins on a 64-byte
    // boundary, terminated by a newline. Byte-compatibility is asserted in NPYWriterTests against
    // constants captured from numpy itself, never against this code's own output.
    //
    // What this writes is the **8-bit product surface widened to float32** — never call it a raw
    // float32 tensor. `InferenceBackend.predict` returns a `CVPixelBuffer`, 8 bits per channel, so
    // k/255 is the precision the user actually receives, and certifying it is certifying the
    // product. The model's pre-quantisation float output is reachable only inside `UpscaleCore`; a
    // `predictRaw` float path is a contract-v2 candidate, not built here.

    static let npyMagic: [UInt8] = [0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59, 0x01, 0x00]
    static let npyAlignment = 64

    /// The exact bytes numpy writes ahead of a C-ordered little-endian float32 array.
    static func npyHeader(shape: [Int]) -> Data {
        let dimensions = shape.map(String.init).joined(separator: ", ")
        let descriptor = "{'descr': '<f4', 'fortran_order': False, 'shape': (\(dimensions)), }"
        // numpy counts the terminating newline in the declared length, then pads to the
        // alignment; a header that already lands on the boundary is padded by a whole 64.
        let declared = descriptor.utf8.count + 1
        let padding = npyAlignment - ((npyMagic.count + 2 + declared) % npyAlignment)
        let length = UInt16(declared + padding)

        var data = Data(npyMagic)
        data.append(UInt8(length & 0xFF))
        data.append(UInt8(length >> 8))
        data.append(contentsOf: descriptor.utf8)
        data.append(contentsOf: Array(repeating: UInt8(0x20), count: padding))
        data.append(0x0A)
        return data
    }

    /// float32 NCHW (1,3,H,W), C order: one contiguous plane per channel, R then G then B.
    /// Apple platforms are little-endian, which is what the '<f4' descriptor above declares.
    static func npyData(from buffer: CVPixelBuffer) -> Data {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }

        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        var data = npyHeader(shape: [1, 3, height, width])
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return data }

        let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
        let source = base.assumingMemoryBound(to: UInt8.self)
        var plane = [Float](repeating: 0, count: width * height)
        for channel in [2, 1, 0] {  // the buffer is BGRA in memory; NCHW wants R, G, B
            for y in 0..<height {
                let row = source.advanced(by: y * rowBytes)
                for x in 0..<width {
                    plane[y * width + x] = Float(row[x * 4 + channel]) / 255.0
                }
            }
            plane.withUnsafeBufferPointer { data.append(Data(buffer: $0)) }
        }
        return data
    }

    static func writeNPY(_ buffer: CVPixelBuffer, to url: URL) throws {
        do {
            try npyData(from: buffer).write(to: url, options: .atomic)
        } catch {
            throw InferenceError.predictionFailed(
                "could not write tensor to \(url.path): \(error.localizedDescription)")
        }
    }

    // MARK: - NumPy tensor reading (shared helper)
    //
    // The counterpart of the writer above. Tiling certification compares tiled output against the
    // full-frame PyTorch oracle, whose tensors are `.npy` files, so the CLI has to be able to read
    // what `tools/reference/run_reference.py` wrote. Reading is deliberately strict: a tensor that
    // is not C-ordered little-endian float32 NCHW is refused rather than reinterpreted.

    struct NPYTensor {
        var shape: [Int]
        var values: [Float]

        var height: Int { shape.count == 4 ? shape[2] : 0 }
        var width: Int { shape.count == 4 ? shape[3] : 0 }
    }

    static func readNPY(at url: URL) throws -> NPYTensor {
        let data = try Data(contentsOf: url)
        guard data.count > 10, Array(data.prefix(6)) == Array(npyMagic.prefix(6)) else {
            throw InferenceError.predictionFailed("\(url.lastPathComponent): not a NumPy .npy file")
        }
        let headerLength = Int(data[8]) | (Int(data[9]) << 8)
        let headerStart = 10
        guard data.count >= headerStart + headerLength else {
            throw InferenceError.predictionFailed("\(url.lastPathComponent): truncated .npy header")
        }
        let header = String(decoding: data[headerStart..<(headerStart + headerLength)], as: UTF8.self)

        guard header.contains("'descr': '<f4'") else {
            throw InferenceError.predictionFailed(
                "\(url.lastPathComponent): expected little-endian float32, header says \(header.trimmingCharacters(in: .whitespacesAndNewlines))"
            )
        }
        guard header.contains("'fortran_order': False") else {
            throw InferenceError.predictionFailed(
                "\(url.lastPathComponent): Fortran-ordered tensors are not read")
        }
        guard let open = header.firstIndex(of: "("), let close = header[open...].firstIndex(of: ")") else {
            throw InferenceError.predictionFailed("\(url.lastPathComponent): no shape in the .npy header")
        }
        let shape = header[header.index(after: open)..<close]
            .split(separator: ",")
            .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        let expected = shape.reduce(1, *)

        let body = data[(headerStart + headerLength)...]
        guard body.count == expected * MemoryLayout<Float>.size else {
            throw InferenceError.predictionFailed(
                "\(url.lastPathComponent): shape \(shape) needs \(expected * 4) bytes, file has \(body.count)"
            )
        }
        var values = [Float](repeating: 0, count: expected)
        _ = values.withUnsafeMutableBytes { destination in
            body.copyBytes(to: destination)
        }
        return NPYTensor(shape: shape, values: values)
    }

    /// The oracle tensor, put through the product's own output stage: clamp to [0,1], round to 8
    /// bits. The 8-bit surface is what gets certified, and the tiled candidate only exists at 8
    /// bits, so both sides meet here.
    static func pixelBuffer(from tensor: NPYTensor) throws -> CVPixelBuffer {
        guard tensor.shape.count == 4, tensor.shape[0] == 1, tensor.shape[1] == 3 else {
            throw InferenceError.predictionFailed("expected a [1,3,H,W] tensor, got \(tensor.shape)")
        }
        let height = tensor.height
        let width = tensor.width
        var buffer: CVPixelBuffer?
        let attributes: [CFString: Any] = [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA,
            attributes as CFDictionary, &buffer)
        guard status == kCVReturnSuccess, let output = buffer else {
            throw InferenceError.predictionFailed("could not allocate a \(width)x\(height) oracle buffer")
        }

        CVPixelBufferLockBaseAddress(output, [])
        defer { CVPixelBufferUnlockBaseAddress(output, []) }
        guard let base = CVPixelBufferGetBaseAddress(output) else {
            throw InferenceError.predictionFailed("could not lock the oracle buffer")
        }
        let rowBytes = CVPixelBufferGetBytesPerRow(output)
        let destination = base.assumingMemoryBound(to: UInt8.self)
        memset(base, 0xFF, rowBytes * height)  // opaque alpha; the colour channels overwrite below

        var low: Float = 0
        var high: Float = 1
        var scale: Float = 255
        var row = [Float](repeating: 0, count: width)
        let plane = width * height
        // NCHW planes are R, G, B; BGRA bytes are B, G, R, A.
        for (planeIndex, byteOffset) in [(0, 2), (1, 1), (2, 0)] {
            for y in 0..<height {
                tensor.values.withUnsafeBufferPointer { source in
                    guard let start = source.baseAddress?.advanced(by: planeIndex * plane + y * width)
                    else { return }
                    vDSP_vclip(start, 1, &low, &high, &row, 1, vDSP_Length(width))
                    vDSP_vsmul(row, 1, &scale, &row, 1, vDSP_Length(width))
                    vDSP_vfixru8(
                        row, 1, destination.advanced(by: y * rowBytes + byteOffset), 4, vDSP_Length(width))
                }
            }
        }
        return output
    }
}
