import Foundation
import UpscaleCore

/// One fixed-shape model variant the harness can run, named by the artifact itself.
///
/// The adaptive fallback ladder (certification spec §5.5: 512 → 384 → 256) cannot exist without
/// 384×384 and 512×512 artifacts alongside the 256×256 one, so the harness has to run *a* shape
/// rather than *the* shape.
///
/// The shape is **parsed from the artifact's file name, never configured**: the conversion pipeline
/// names every variant `<identifier>_<N>x<N>.mlpackage` and the CLI resolves models by that same
/// convention (`InferenceCommand.resolveModelURL`). Reading the geometry off the file keeps the
/// drop-in pattern intact — the operator copies an artifact into `HarnessArtifacts/` and the
/// harness knows what it is — and makes it impossible for a setting somewhere to disagree with the
/// weights actually bundled.
public struct HarnessShape: Sendable, Hashable, Comparable {
    /// Model family, e.g. `RealESRGAN_x4plus` — the part of the file name before the geometry.
    /// Deliberately shape-free: the shape lives in `tileSize`, exactly as `ModelDescriptor` carries
    /// it in `inputTileSize`.
    public let identifier: String

    /// Side length in pixels of the square input tile: 256, 384 or 512 today.
    public let tileSize: Int

    public init(identifier: String, tileSize: Int) {
        self.identifier = identifier
        self.tileSize = tileSize
    }

    /// The original 256 variant. It stays the default everywhere, so a bundle holding only the 256
    /// artifacts behaves exactly as a single-shape harness would.
    public static let default256 = HarnessShape(identifier: "RealESRGAN_x4plus", tileSize: 256)

    /// The conversion pipeline's name for this variant — also the name to drop into
    /// `HarnessArtifacts/`, which is why the missing-resource message can quote it.
    public var modelFileName: String {
        "\(identifier)_\(tileSize)x\(tileSize).mlpackage"
    }

    /// The identifier handed to `ModelDescriptor` — and, inside the backend, **one of the two
    /// components** of the compiled-model cache directory's key
    /// (`CoreMLBackend.compiledModelName`); the other is the descriptor's input geometry.
    ///
    /// With the identifier alone as the key, two shapes sharing it would race on one directory —
    /// the loser instantiating the other shape's weights, reproduced on the host with parallel
    /// loads. The geometry component fixes that at the backend, for every caller.
    ///
    /// 256 keeps the family-level name. Every other shape takes the artifact's own stem instead —
    /// which is what the CLI already does (`InferenceCommand.descriptor`). With geometry in the
    /// cache key that suffixing is redundant for cache uniqueness rather than wrong, and it is kept
    /// as belt and braces at zero cost.
    public var descriptorIdentifier: String {
        self == .default256 ? identifier : "\(identifier)_\(tileSize)x\(tileSize)"
    }

    /// The shape-suffixed test-tile name — the convention for the 384 and 512 variants.
    var suffixedTileFileName: String {
        "input_tile_\(tileSize)x\(tileSize).png"
    }

    /// The tile name the harness asks for when the file is absent, and the first one it looks for.
    /// The 256 shape keeps the unsuffixed `input_tile.png`, which is what
    /// `DeviceHarnessApp/README.md` documents.
    public var tileFileName: String {
        self == .default256 ? DeviceHarness.tileFileName : suffixedTileFileName
    }

    /// Test-tile names accepted for this shape, in the order they are tried.
    ///
    /// 256 also accepts the suffixed spelling, so a bundle holding all three shapes can name its
    /// tiles uniformly. The other shapes never accept the unsuffixed name: that would feed a 256
    /// tile to a 512 model and report a wrong-shape run as a missing file.
    public var tileFileNames: [String] {
        self == .default256 ? [tileFileName, suffixedTileFileName] : [tileFileName]
    }

    /// How the shape is written for humans — button titles, pickers, the footprint table.
    public var displayName: String {
        "\(tileSize)×\(tileSize)"
    }

    /// The margin this model needs, or `nil` when none has been derived for it.
    ///
    /// Keyed on `identifier`, the MODEL name — not `descriptorIdentifier`, which is
    /// shape-suffixed. The margin is a property of the network's receptive field and does not
    /// change with tile size; suffixing would miss every variant but 256.
    public var derivedMargin: Int? {
        ContextMargin.derivedMargin(for: identifier)
    }

    /// The model's upscale factor, inferred from its identifier.
    ///
    /// **A harness heuristic, and only defensible because this is an evaluation tool.** The real
    /// answer lives in the model's `manifest.json` (`scale_factor`), which is not bundled — reading
    /// it is `ModelStore`'s job, and this should be deleted when that exists. Until then a
    /// hardcoded 4 would silently mis-size every output surface and memory plan for a ×2 model,
    /// which is worse than a named guess.
    public var scaleFactor: Int {
        identifier.lowercased().contains("x2") ? 2 : 4
    }

    /// A short model tag — `RealESRGAN_x4plus` → `x4plus`, `realesr-general-x4v3` → `x4v3`.
    ///
    /// The trailing token after the last `_` or `-`, which is how both naming conventions in this
    /// repository end. Falls back to the whole identifier rather than to an empty string.
    public var modelLabel: String {
        let token = identifier.split(whereSeparator: { $0 == "_" || $0 == "-" }).last
        return token.map(String.init) ?? identifier
    }

    /// What a picker shows. `displayName` alone is ambiguous the moment two models are bundled at
    /// the same tile size — the segments read identically and the tester cannot tell which is
    /// which, which is exactly the comparison the harness is being used for.
    public var pickerLabel: String {
        "\(tileSize) \(modelLabel)"
    }

    /// Recovers the shape from an artifact file name, or `nil` when the name is not one this
    /// convention would have produced.
    ///
    /// Strict on purpose: non-square geometry, a padded or signed number, a missing identifier or a
    /// different extension all mean "this is not a variant of ours", and guessing would put an
    /// unverified artifact in the picker.
    ///
    /// - Parameter modelFileName: A file name such as `RealESRGAN_x4plus_384x384.mlpackage`, not a path.
    /// - Returns: The parsed shape, or `nil` if the name does not follow the convention.
    public static func parse(modelFileName: String) -> HarnessShape? {
        let packageExtension = ".mlpackage"
        guard modelFileName.hasSuffix(packageExtension) else { return nil }
        let stem = modelFileName.dropLast(packageExtension.count)
        guard let separator = stem.lastIndex(of: "_") else { return nil }

        let identifier = String(stem[stem.startIndex..<separator])
        let geometry = stem[stem.index(after: separator)...]
        guard !identifier.isEmpty else { return nil }

        let sides = geometry.split(separator: "x", omittingEmptySubsequences: false)
        guard sides.count == 2, sides[0] == sides[1],
            let tileSize = Int(sides[0]), tileSize > 0, String(tileSize) == sides[0]
        else { return nil }

        return HarnessShape(identifier: identifier, tileSize: tileSize)
    }

    /// Ascending by tile size — the order the ladder is written in, smallest (safest) first.
    public static func < (lhs: HarnessShape, rhs: HarnessShape) -> Bool {
        (lhs.tileSize, lhs.identifier) < (rhs.tileSize, rhs.identifier)
    }
}
