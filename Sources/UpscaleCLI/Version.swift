/// Version identity for the macOS certification CLI.
///
/// The CLI proves determinism and conversion correctness on macOS; device claims always require
/// DeviceHarness runs.
enum Version {
    /// Executable name, as printed by `--version`.
    static let productName = "UpscaleCLI"

    /// Semantic version of the CLI.
    static let semanticVersion = "0.1.0"

    /// Single line printed by `--version`, e.g. `UpscaleCLI 0.1.0`.
    static var line: String { "\(productName) \(semanticVersion)" }
}
