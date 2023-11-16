import Foundation
import UpscaleCore

/// DeviceHarness — on-device benchmark scene for A14 / A17 Pro / M1 / M3–M4.
///
/// The real entry point the iOS harness app calls. The app is deliberately thin: it owns a button
/// and a result panel, and nothing else. All logic lives here so it can be unit-tested on the host
/// without a device, a simulator, or the model.
///
/// Scope: one fixed-shape tile, to answer *does the same model run on an A14?* Device claims
/// require DeviceHarness runs on hardware, never Mac runs.
public enum DeviceHarness {
    /// The app bundles the artifacts inside a copied folder rather than as compiled resources: The
    /// folder must NOT be called "Resources": a top-level `Resources` directory inside an iOS
    /// `.app` makes the installer parse the bundle as macOS-style and fail with "Missing bundle
    /// ID". Xcode would otherwise compile a `.mlpackage` into `.mlmodelc` at build time, and the
    /// backend compiles the package itself. A folder reference keeps the artifact byte-identical to
    /// the one the conversion pipeline produced and verified.
    public static let resourceFolderName = "HarnessArtifacts"
    public static let modelFileName = "RealESRGAN_x4plus_256x256.mlpackage"
    public static let tileFileName = "input_tile.png"

    public static func bundledModelURL(in bundle: Bundle = .main) -> URL? {
        existingURL(bundle: bundle, fileName: modelFileName)
    }

    public static func bundledTileURL(in bundle: Bundle = .main) -> URL? {
        existingURL(bundle: bundle, fileName: tileFileName)
    }

    static func existingURL(bundle: Bundle, fileName: String) -> URL? {
        guard let directory = artifactsURL(in: bundle) else { return nil }
        let candidate = directory.appendingPathComponent(fileName)
        return FileManager.default.fileExists(atPath: candidate.path) ? candidate : nil
    }

    // MARK: - Fixed-shape variants

    /// The drop-in folder itself, or `nil` when the bundle has no resources at all.
    public static func artifactsURL(in bundle: Bundle = .main) -> URL? {
        bundle.resourceURL?.appendingPathComponent(resourceFolderName, isDirectory: true)
    }

    /// Every fixed-shape variant actually present in the bundle, smallest tile first.
    ///
    /// Discovery, not configuration: whichever `<identifier>_<N>x<N>.mlpackage` files were copied
    /// in are the shapes the harness offers. An empty result means nothing was copied in — the
    /// caller falls back to `HarnessShape.default256` so the "you forgot the model" message stays
    /// the standard one.
    public static func availableShapes(in bundle: Bundle = .main) -> [HarnessShape] {
        shapes(inArtifactsDirectory: artifactsURL(in: bundle))
    }

    static func shapes(inArtifactsDirectory directory: URL?) -> [HarnessShape] {
        guard let directory,
            let entries = try? FileManager.default.contentsOfDirectory(atPath: directory.path)
        else { return [] }
        return entries.compactMap(HarnessShape.parse(modelFileName:)).sorted()
    }

    public static func bundledModelURL(for shape: HarnessShape, in bundle: Bundle = .main) -> URL? {
        existingURL(bundle: bundle, fileName: shape.modelFileName)
    }

    /// One selectable input tile: what the picker shows, and what the runner reads.
    public struct TileChoice: Hashable, Sendable {
        public let label: String
        public let url: URL
    }

    /// Every tile bundled for `shape`, discovery-not-configuration like `availableShapes`.
    ///
    /// Two naming forms are accepted. The canonical `input_tile_<N>x<N>.png` is the certified tile
    /// and is always listed first as `"standard"`, so a comparison run keeps a fixed reference.
    /// Anything named `input_tile_<label>_<N>x<N>.png` is offered under `<label>` — drop a file in,
    /// get an option, exactly as models work.
    ///
    /// **The same tile must reach every model being compared.** That is why selection is a URL the
    /// caller passes explicitly rather than something the runner re-resolves per model.
    public static func bundledTileChoices(
        for shape: HarnessShape, in bundle: Bundle = .main
    ) -> [TileChoice] {
        tileChoices(
            inArtifactsDirectory: artifactsURL(in: bundle), for: shape,
            standard: bundledTileURL(for: shape, in: bundle))
    }

    /// The directory-scoped form, so discovery is testable without a bundle.
    static func tileChoices(
        inArtifactsDirectory directory: URL?, for shape: HarnessShape, standard: URL? = nil
    ) -> [TileChoice] {
        guard let directory,
            let entries = try? FileManager.default.contentsOfDirectory(atPath: directory.path)
        else { return [] }

        let suffix = "_\(shape.tileSize)x\(shape.tileSize).png"
        var choices: [TileChoice] = []
        let canonical =
            standard
            ?? {
                let url = directory.appendingPathComponent(
                    "input_tile_\(shape.tileSize)x\(shape.tileSize).png")
                return FileManager.default.fileExists(atPath: url.path) ? url : nil
            }()
        if let canonical {
            choices.append(TileChoice(label: "standard", url: canonical))
        }
        for entry in entries.sorted() where entry.hasSuffix(suffix) && entry.hasPrefix("input_tile_") {
            let middle = entry.dropFirst("input_tile_".count).dropLast(suffix.count)
            guard !middle.isEmpty else { continue }  // the canonical name, already added
            choices.append(
                TileChoice(
                    label: String(middle), url: directory.appendingPathComponent(entry)))
        }
        return choices
    }

    /// Run `shape` against an explicitly chosen tile.
    ///
    /// The shape-resolved overload below is unchanged and remains what the certification path
    /// uses; this exists so a human comparing two models can hold the input fixed and vary only
    /// the model.
    public static func runSingleTile(
        shape: HarnessShape,
        tileURL: URL?,
        in bundle: Bundle = .main,
        computeUnits: ComputeUnits = .all
    ) async -> SingleTileRun {
        await SingleTileRunner.run(
            shape: shape,
            modelURL: bundledModelURL(for: shape, in: bundle),
            tileURL: tileURL,
            computeUnits: computeUnits
        )
    }

    /// First accepted tile name that is actually there — see `HarnessShape.tileFileNames` for why
    /// only the 256 shape accepts the unsuffixed name.
    public static func bundledTileURL(for shape: HarnessShape, in bundle: Bundle = .main) -> URL? {
        for fileName in shape.tileFileNames {
            if let url = existingURL(bundle: bundle, fileName: fileName) { return url }
        }
        return nil
    }

    public static func runSingleTile(
        modelURL: URL? = bundledModelURL(),
        tileURL: URL? = bundledTileURL(),
        computeUnits: ComputeUnits = .all
    ) async -> SingleTileRun {
        await SingleTileRunner.run(modelURL: modelURL, tileURL: tileURL, computeUnits: computeUnits)
    }

    /// Runs one tile of `shape`, resolving both artifacts from the bundle by that shape's names.
    ///
    /// The measurements carry the same contract as the 256 path: best-effort and informational,
    /// never a benchmark result.
    public static func runSingleTile(
        shape: HarnessShape,
        in bundle: Bundle = .main,
        computeUnits: ComputeUnits = .all
    ) async -> SingleTileRun {
        await SingleTileRunner.run(
            shape: shape,
            modelURL: bundledModelURL(for: shape, in: bundle),
            tileURL: bundledTileURL(for: shape, in: bundle),
            computeUnits: computeUnits
        )
    }

    // MARK: - Display

    /// Rows the app renders verbatim. Formatting lives here, not in the view, so it is testable
    /// without a UI — and so the informational status of the numbers is stated once, in one place.
    public static func displayRows(for result: SingleTileResult) -> [(label: String, value: String)] {
        var rows: [(label: String, value: String)] = [
            ("Status", result.success ? "SUCCESS" : "FAILED"),
            ("Compute units", result.computeUnits),
        ]
        if result.success {
            rows.append(
                ("Output size", "\(Int(result.outputSize.width)) × \(Int(result.outputSize.height))"))
        }
        rows.append(("Wall clock", format(seconds: result.wallClockSeconds) + "  (informational)"))
        if result.success {
            // Split out because the total is the sum of two very different things, and reading it
            // as one number is how a cold compile gets attributed to the model. The start state is
            // stated rather than inferred: COLD pays a full MLModel.compileModel, WARM reads the
            // compiled model back from the cache.
            rows.append(
                (
                    "  ├ model load",
                    format(seconds: result.loadSeconds)
                        + (result.warmStart ? "  (WARM — cache hit)" : "  (COLD — compiled now)")
                ))
            rows.append(("  └ inference", format(seconds: result.inferenceSeconds)))
            if !result.warmStart {
                rows.append(
                    (
                        "  ⚠︎ comparability",
                        "cold run — run again for a warm number before comparing models"
                    ))
            }
        }
        rows.append(("Peak memory", format(bytes: result.peakMemoryBytes) + "  (best-effort)"))
        // Alongside, never instead of: the row above is `resident_size_max`, the metric earlier
        // single-tile runs were recorded in; this one is what the system enforces.
        rows.append(
            (
                "Peak footprint",
                format(footprintBytes: result.peakFootprintBytes)
                    + "  (phys_footprint high-water, best-effort)"
            ))
        if let errorDescription = result.errorDescription {
            rows.append(("Error", errorDescription))
        }
        return rows
    }

    public static func format(seconds: TimeInterval) -> String {
        seconds < 1 ? String(format: "%.0f ms", seconds * 1000) : String(format: "%.2f s", seconds)
    }

    public static func format(bytes: UInt64?) -> String {
        guard let bytes else { return "unavailable" }
        let megabytes = Double(bytes) / (1024 * 1024)
        return String(format: "%.1f MB", megabytes)
    }

    /// MiB, and labelled MiB.
    ///
    /// `format(bytes:)` above divides by 1024² and calls the answer "MB". That naming is kept as-is
    /// because existing single-tile records are written in those exact strings, and correcting the
    /// unit name would silently restate them. The footprint row is new, and the ceilings it will be
    /// read against are quoted in MiB (3.07–3.12 GiB on the floor device), so it is written
    /// correctly from the start rather than inheriting a defect for symmetry's sake.
    public static func format(footprintBytes bytes: UInt64?) -> String {
        guard let bytes else { return "unavailable" }
        return String(format: "%.1f MiB", Double(bytes) / (1024 * 1024))
    }

    /// The one sentence the harness must always show, so no screenshot of it can be mistaken for a
    /// benchmark result.
    public static let measurementDisclaimer =
        "Timing and memory are best-effort and informational. Benchmark methodology "
        + "(cold/warm/sustained, medians) is out of scope here — never quote these as results."
}
