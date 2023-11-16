// The engine imports exist to prove the dependency edges declared in Package.swift actually
// link. UpscaleCLI is the one target permitted to depend on every engine package.
import Benchmarking
import Foundation
import ImagePipeline
import ModelStore
import TestFixtures
import UpscaleCore

/// Entry point for the certification / benchmark CLI.
///
/// Subcommands: `upscale-tile`, `upscale-image`, `derive-context-margin` and `certify-tiling`, plus
/// `--version` and `--help`.
@main
struct Entry {
    static func main() async {
        let arguments = Array(CommandLine.arguments.dropFirst())
        // Subcommands are async; `run` stays synchronous so its behaviour remains testable
        // without a task context.
        switch arguments.first {
        case "upscale-tile":
            exit(await InferenceCommand.run(arguments: Array(arguments.dropFirst())))
        case "upscale-image":
            exit(await ImageCommand.run(arguments: Array(arguments.dropFirst())))
        case "derive-context-margin":
            exit(await ContextMarginCommand.run(arguments: Array(arguments.dropFirst())))
        case "certify-tiling":
            exit(await TilingCertificationCommand.run(arguments: Array(arguments.dropFirst())))
        default:
            break
        }
        exit(run(arguments: arguments))
    }

    /// Exit codes: 0 success, 2 usage error.
    /// Returns the process exit status so the behaviour is testable without spawning a process.
    static func run(
        arguments: [String],
        out: (String) -> Void = { print($0) },
        err: (String) -> Void = { FileHandle.standardError.write(Data(($0 + "\n").utf8)) }
    ) -> Int32 {
        switch arguments.first {
        case "--version", "-v":
            out(Version.line)
            return 0
        case "--help", "-h", nil:
            out(usage)
            return 0
        case let unknown?:
            err("error: unknown argument '\(unknown)'")
            err(usage)
            return 2
        }
    }

    static let usage = """
        \(Version.line)

        USAGE: UpscaleCLI <command>

        COMMANDS:
          upscale-tile           Run one fixed-shape tile through the Core ML backend.
          upscale-image          Tile a whole image and stitch the result.
          derive-context-margin  Pad-and-probe derivation of the tiling context margin.
          certify-tiling         Seam certification: every stitch method against the oracle.

        OPTIONS:
          -v, --version   Print the version and exit.
          -h, --help      Print this message and exit.
        """
}
