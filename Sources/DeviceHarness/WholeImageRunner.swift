import CoreVideo
import Foundation
import UpscaleCore

#if canImport(Darwin)
    import Darwin
#endif

/// One of certification spec §7.1's three workloads, named by the fixture file that carries it.
///
/// The files are generated locally (`tools/fixtures/make_fixtures.py`) and arrive on device
/// through `HarnessArtifacts/` — the harness app's folder-reference drop-in — because that is the
/// only folder the app copies verbatim. Names are load-bearing exactly as the model names are.
public enum HarnessWorkload: String, Sendable, CaseIterable {
    case a
    case b
    case c

    public var fileName: String {
        switch self {
        case .a: return "A_1080p.jpg"
        case .b: return "B_12MP.jpg"
        case .c: return "C_48MP.jpg"
        }
    }

    /// The letter the spec and the run sheets use. Upper case; the raw value stays lower case so
    /// `-workload b` and `-workload B` both parse.
    public var letter: String { rawValue.uppercased() }

    /// What the workload is *for*, as certification spec §7.1 states it.
    public var role: String {
        switch self {
        case .a: return "A — 1080p → 4×, headline benchmark"
        case .b: return "B — 12MP → 4×, primary workload; the §6 protocol's workload"
        case .c: return "C — 48MP → 4×, survival"
        }
    }

    /// Case-insensitive, and strict: an unrecognised letter is `nil`, never a default. A run that
    /// silently substituted workload B for a mistyped C would be a mislabelled certification record.
    public static func parse(_ text: String) -> HarnessWorkload? {
        HarnessWorkload(rawValue: text.lowercased())
    }
}

/// Whether the app was cold or warm when the series started — **declared by the operator, never
/// inferred.**
///
/// Certification spec §6 requires the protocol "app warm and cold", and the process cannot honestly
/// tell which it is: "cold" in §7.3's sense means first launch after install or reboot with the
/// system's caches cold, which is a property of the device the operator establishes, not a flag the
/// runner can read. So the runner records the operator's declaration and labels it as such.
/// `.unrecorded` is the honest default — the same answer `MemoryBudget.Figure.unavailable` gives.
///
/// One thing *is* derivable and is reported separately: run 1 of a series is the first run of its
/// process by construction, so within one launch runs 2…N are process-warm whatever the device was.
public enum LaunchState: String, Sendable, CaseIterable {
    case cold
    case warm
    case unrecorded

    public static func parse(_ text: String) -> LaunchState? {
        LaunchState(rawValue: text.lowercased())
    }
}

/// What one invocation is asked to run: a workload, a repeat count, and the state the operator says
/// the device was in.
///
/// A series is one *request* rather than N taps because certification spec §6's "10 consecutive
/// runs" is a property of the protocol, not of a user interface.
public struct WholeImageSeriesRequest: Sendable, Equatable {
    public let workload: HarnessWorkload
    public let runs: Int
    public let launchState: LaunchState
    public let computeUnits: ComputeUnits
    public let method: StitchMethod
    /// The rung the ladder starts the **first** image at. `nil` means "whatever `TileShapeLadder`
    /// ships as its default", which is the floor rung — see `WholeImageRunner.run` for why the
    /// runner declines to seed it on its own initiative.
    public let startingShape: TileShape?
    /// Stop the series at the first failed run by default; the opposite is available so a survival
    /// run that degrades does not hide the runs after it.
    public let stopsOnFailure: Bool

    public init(
        workload: HarnessWorkload,
        runs: Int,
        launchState: LaunchState = .unrecorded,
        computeUnits: ComputeUnits = .all,
        method: StitchMethod = WholeImageRunner.contractDefaultMethod,
        startingShape: TileShape? = nil,
        stopsOnFailure: Bool = true
    ) {
        self.workload = workload
        self.runs = runs
        self.launchState = launchState
        self.computeUnits = computeUnits
        self.method = method
        self.startingShape = startingShape
        self.stopsOnFailure = stopsOnFailure
    }

    // MARK: Launch-argument invocation

    /// The argument that asks for a whole-image series at all. Follows `-memoryprobe` and
    /// `-autorun`: the app branches on it and is otherwise unchanged.
    public static let launchArgument = "-wholeimage"

    /// The outcome of reading a launch-argument vector.
    ///
    /// Three cases rather than an `Optional`, because "not asked for" and "asked for, badly" are
    /// different events and only one of them is silence. A malformed value must never fall back to
    /// a default: `ContentView`'s `-shape` handling already rules that an automated check "must
    /// never quietly run a different variant than the one it asked for", and a certification run
    /// that recorded workload B under a mistyped `-workload BB` would be worse than no run.
    public enum ParseResult: Sendable, Equatable {
        case notRequested
        case invalid(String)
        case requested(WholeImageSeriesRequest)
    }

    /// Read a series request out of a process argument vector.
    ///
    /// Grammar, all optional except the trigger:
    ///
    ///     -wholeimage                       run a series (required trigger)
    ///     -workload A|B|C                   default B — §6's workload
    ///     -runs N                           default 1
    ///     -state cold|warm                  default unrecorded
    ///     -computeunits all|cpuAndGPU|cpuAndNeuralEngine|cpuOnly    default all
    ///     -method hardCrop|linearFeather|cosineFeather|centerValidRegion   default centerValidRegion
    ///     -startshape 256|384|512           default: the ladder's own starting rung
    ///     -continueonfailure                default: stop at the first failure
    ///
    /// `-runs` defaults to **1, not 10**, deliberately. §6's protocol is ten consecutive runs, so
    /// ten is what the operator types — and typing it means the count is recorded in the command
    /// line that produced the record. The safe default also keeps `-wholeimage -workload C` from
    /// writing ten 3.07 GB spool files to a phone by omission.
    ///
    /// - Parameter arguments: the process argument vector, `argv[0]` included and ignored.
    /// - Returns: `.notRequested`, `.invalid` naming the offending argument, or `.requested`.
    public static func parse(arguments: [String]) -> ParseResult {
        guard arguments.contains(launchArgument) else { return .notRequested }

        func value(for flag: String) -> String? {
            guard let position = arguments.firstIndex(of: flag), position + 1 < arguments.count
            else { return nil }
            return arguments[position + 1]
        }

        var workload = HarnessWorkload.b
        if let text = value(for: "-workload") {
            guard let parsed = HarnessWorkload.parse(text) else {
                return .invalid("-workload \(text) — expected A, B or C")
            }
            workload = parsed
        }

        var runs = 1
        if let text = value(for: "-runs") {
            guard let parsed = Int(text), parsed >= 1 else {
                return .invalid("-runs \(text) — expected a positive integer")
            }
            runs = parsed
        }

        var launchState = LaunchState.unrecorded
        if let text = value(for: "-state") {
            guard let parsed = LaunchState.parse(text), parsed != .unrecorded else {
                return .invalid("-state \(text) — expected cold or warm")
            }
            launchState = parsed
        }

        var computeUnits = ComputeUnits.all
        if let text = value(for: "-computeunits") {
            guard let parsed = ComputeUnits(rawValue: text) else {
                return .invalid(
                    "-computeunits \(text) — expected one of "
                        + ComputeUnits.allCases.map(\.rawValue).joined(separator: ", "))
            }
            computeUnits = parsed
        }

        var method = WholeImageRunner.contractDefaultMethod
        if let text = value(for: "-method") {
            guard let parsed = StitchMethod(rawValue: text) else {
                return .invalid(
                    "-method \(text) — expected one of "
                        + StitchMethod.allCases.map(\.rawValue).joined(separator: ", "))
            }
            method = parsed
        }

        var startingShape: TileShape?
        if let text = value(for: "-startshape") {
            guard let size = Int(text), let parsed = TileShape(rawValue: size) else {
                return .invalid(
                    "-startshape \(text) — expected one of "
                        + TileShape.allCases.sorted().map { String($0.inputTileSize) }
                        .joined(separator: ", "))
            }
            startingShape = parsed
        }

        return .requested(
            WholeImageSeriesRequest(
                workload: workload, runs: runs, launchState: launchState,
                computeUnits: computeUnits, method: method, startingShape: startingShape,
                stopsOnFailure: !arguments.contains("-continueonfailure")))
    }
}

/// A number of bytes that knows whether it is a measurement — the JSON form of
/// `MemoryBudget.Figure`, which the harness cannot encode directly because that type is not
/// `Codable`, and making it so would widen `UpscaleCore`'s API for a harness concern.
///
/// A figure without provenance is a defect. `nil` bytes and `unavailable` provenance travel
/// together, so no reader can mistake a failed kernel call for a zero.
///
/// **`provenance` is the field a consumer branches on, never the presence of `bytes`.** Swift's
/// synthesized `Encodable` omits a `nil` optional rather than writing an explicit null, so an
/// unavailable reading is `{"provenance": "unavailable", "source": …}` with no `bytes` key at all —
/// which is uniform with every other optional in this record, and unambiguous because the label is
/// always there. What must never appear is a `bytes` key holding a zero that was really a failure.
public struct MemoryReading: Sendable, Equatable, Encodable {
    public let bytes: UInt64?
    /// Mebibytes, and labelled MiB — the unit the memory ceilings are quoted in.
    public let mebibytes: Double?
    public let provenance: String
    public let source: String

    init(bytes: UInt64?, provenance: MemoryBudget.Provenance, source: String) {
        self.bytes = bytes
        self.mebibytes = bytes.map { Double($0) / 1_048_576 }
        self.provenance = provenance.rawValue
        self.source = source
    }

    /// A best-effort kernel reading: measured when the call returned a figure, unavailable when it
    /// did not. Never a fabricated number and never a zero dressed as one — the contract
    /// `SingleTileRunner.peakFootprintBytes()` already carries.
    static func kernel(_ bytes: UInt64?, source: String) -> MemoryReading {
        MemoryReading(bytes: bytes, provenance: bytes == nil ? .unavailable : .measured, source: source)
    }

    /// A `MemoryBudget.Figure` carried across with its own provenance and citation intact. An
    /// `.unavailable` figure stores zero bytes so its arithmetic stays total; that zero is dropped
    /// here rather than published as a reading.
    static func figure(_ figure: MemoryBudget.Figure) -> MemoryReading {
        guard figure.provenance != .unavailable, figure.bytes >= 0 else {
            return MemoryReading(bytes: nil, provenance: .unavailable, source: figure.source)
        }
        return MemoryReading(
            bytes: UInt64(figure.bytes), provenance: figure.provenance, source: figure.source)
    }
}

/// Every memory reading one run took, at the boundaries that make them interpretable.
///
/// **Point samples and high-water marks are not interchangeable, and this type keeps them apart on
/// purpose.** `SingleTileRunner.peakFootprintBytes()` and `peakMemoryBytes()` are *process*
/// high-water marks: in a ten-run series they are monotone, so run 7's peak is not "the peak run 7
/// caused" — it is the largest the process ever reached, which may have been run 1. Single-shape
/// footprint measurement therefore uses one launch per run. A series cannot do that, so the per-run
/// signal has to come from the point samples, and both are recorded rather than one being quietly
/// reinterpreted.
///
/// Read the series like this: `processPeakFootprint` on the last run is the series' peak;
/// `footprintAtStart` across runs 1…N is what says whether the process is growing run over run,
/// which is the whole reason §6 asks for ten consecutive runs rather than one.
public struct MemorySample: Sendable, Equatable, Encodable {
    /// `os_proc_available_memory()` at run start — the device-validated countdown (it reached 6 MB
    /// one chunk before jetsam fired on the floor device).
    public let availableAtStart: MemoryReading
    public let availableAtEnd: MemoryReading
    /// `phys_footprint` **now**, before the workload file is decoded.
    public let footprintAtStart: MemoryReading
    /// `phys_footprint` **now**, with the decoded source image live and nothing else started. This
    /// is the budget's input-surface term, measured rather than estimated.
    public let footprintAfterDecode: MemoryReading
    /// `phys_footprint` **now**, with the run finished and the spool file not yet deleted.
    public let footprintAtEnd: MemoryReading
    /// `ledger_phys_footprint_peak` — process high-water, monotone across the series.
    public let processPeakFootprint: MemoryReading
    /// `resident_size_max` — process high-water, monotone across the series. Kept alongside the
    /// footprint so single-tile and whole-image series stay comparable.
    public let processPeakResidentSize: MemoryReading
}

/// What the job actually was, in pixels and bytes.
public struct RunGeometry: Sendable, Equatable, Encodable {
    public let inputWidth: Int
    public let inputHeight: Int
    public let outputWidth: Int
    public let outputHeight: Int
    /// Bytes the spool file held when the run finished, read off the file system.
    public let spoolBytes: Int?
    /// Bytes `SpooledImage.byteCount` said it should hold. The only completion check available
    /// without re-reading 3 GB of pixels: certification spec §6 forbids corrupt output absolutely,
    /// and a truncated spool is the one corruption a size comparison can catch for free.
    public let spoolBytesExpected: Int
}

/// One whole-image run's record — the row of the §6 results table, in machine-readable form.
///
/// **Every timing here is informational:** benchmark methodology — cold, warm, sustained,
/// throttled, medians of ten, recorded device state (certification spec §7.3) — is out of this
/// instrument's scope, and no figure it produces may be quoted as a performance result. The memory
/// figures are measurements and are labelled as such; they are evidence for the §6 memory protocol,
/// not a benchmark.
public struct WholeImageRunResult: Sendable, Equatable, Encodable {
    /// 1-based position in the series. Run 1 is the first run of its process by construction.
    public let index: Int
    public let workload: String
    public let workloadFile: String
    public let launchState: String
    public let computeUnits: String
    public let method: String
    public let contextMargin: Int
    public let success: Bool
    /// The whole run: decode, plan, upscale, spool teardown.
    public let wallClockSeconds: Double
    /// The engine call alone, so the decode and the file work are separable from the compute.
    /// `nil` when the run failed before the engine was reached.
    public let upscaleSeconds: Double?
    public let geometry: RunGeometry?
    /// The rung the budget selected before the ladder was consulted — the proactive half of §6's
    /// "proactive budgeting … demonstrated", visible in the record rather than asserted in prose.
    public let plannedShape: Int?
    public let plannedSelection: String?
    public let plannedFit: String?
    /// ρ × available — the budget the engine granted itself.
    public let allowanceBytes: Int?
    /// What the selected rung was estimated to cost: model + input surface + destination.
    ///
    /// Recorded because "proactive budgeting demonstrated" (certification spec §6) is only a claim
    /// until the estimate can be subtracted from the measured peak. With
    /// `memory.processPeakFootprint` beside it, the prediction error is a number a reader can
    /// compute; without it, it is not.
    public let plannedTotalBytes: Int?
    public let plannedHeadroomBytes: Int?
    public let plannedProvenance: String?
    /// ρ, the safety fraction. Explicit in code and in the report.
    public let safetyFraction: Double?
    /// The rung the ladder carried into this image, before `beginImage`.
    public let startingShape: Int?
    /// The rung the last tile ran at.
    public let finalShape: Int?
    public let stepDowns: Int?
    public let tilesPredicted: Int?
    /// Tiles a step down discarded and re-ran — bounded at one row per step.
    public let tilesRecomputed: Int?
    /// The floor rung was held under a trigger: the job completed **degraded**, which is what
    /// certification spec §6's 48 MP rule accepts.
    public let isDegraded: Bool?
    /// Every ladder decision, in order, rendered by `TileShapeLadder.Outcome`'s own description.
    public let ladderOutcomes: [String]
    public let thermalStateAtStart: String
    public let thermalStateAtEnd: String
    /// The worst pressure level the monitor was carrying when the run ended.
    public let pressureLevelAtEnd: String
    public let memory: MemorySample
    public let errorDescription: String?
}

/// The device a series ran on. Certification spec §6 records peak footprint **per device**, so the
/// record has to name one.
public struct DeviceIdentity: Sendable, Equatable, Encodable {
    /// `uname` machine identifier — `iPhone13,3` is the floor device (iPhone 12 Pro). Deliberately
    /// the raw identifier and not a product name: there is no headless table mapping the two, and
    /// inventing one would put an unverified string in a certification record.
    public let machine: String
    public let systemName: String
    public let systemVersion: String
    public let processorCount: Int
    public let physicalMemoryBytes: UInt64

    public static func current() -> DeviceIdentity {
        let info = ProcessInfo.processInfo
        #if os(iOS)
            let system = "iOS"
        #elseif os(macOS)
            let system = "macOS"
        #else
            let system = "unknown"
        #endif
        return DeviceIdentity(
            machine: machineIdentifier(), systemName: system,
            systemVersion: info.operatingSystemVersionString,
            processorCount: info.processorCount, physicalMemoryBytes: info.physicalMemory)
    }

    /// Best-effort by the same contract the memory readings carry: `"unavailable"` when `uname`
    /// fails, never a guess.
    static func machineIdentifier() -> String {
        #if canImport(Darwin)
            var info = utsname()
            guard uname(&info) == 0 else { return "unavailable" }
            let machine = withUnsafeBytes(of: &info.machine) { raw -> String in
                guard let base = raw.baseAddress else { return "" }
                return String(cString: base.assumingMemoryBound(to: CChar.self))
            }
            return machine.isEmpty ? "unavailable" : machine
        #else
            return "unavailable"
        #endif
    }
}

/// A whole series, and the JSON document a device run produces.
///
/// `Encodable` only. The derived summary fields (`runsCompleted`, `runsSucceeded`, `stoppedEarly`)
/// are computed rather than stored — an inconsistent summary in a certification record would be
/// worse than no summary — and `encode(to:)` writes them alongside the stored ones so a human
/// reading the file does not have to count rows.
public struct WholeImageSeries: Sendable, Equatable, Encodable {
    /// Bumped when the JSON shape changes, so a later reader can tell which schema it has.
    public static let schemaVersion = 1

    public let request: WholeImageSeriesRequest
    public let device: DeviceIdentity
    public let startedAt: Date
    public let runs: [WholeImageRunResult]

    public var runsCompleted: Int { runs.count }
    public var runsSucceeded: Int { runs.filter(\.success).count }

    /// Fewer records than runs asked for.
    ///
    /// **This is the jetsam signal**, and it is the only one available: a jetsam kill terminates the
    /// process without notice, so nothing inside it can report its own termination. The series file
    /// is rewritten after every run for exactly this reason — a kill during run 7 leaves a file
    /// holding runs 1…6 and `runsCompleted == 6` against `runs == 10`, which is the evidence.
    /// A series that stopped early on a *reported* failure has the same shape plus a non-nil
    /// `errorDescription` on its last row; the two are told apart by that field, never by this one.
    public var stoppedEarly: Bool { runsCompleted < request.runs }

    public var allSucceeded: Bool {
        runsCompleted == request.runs && runsSucceeded == request.runs
    }

    enum CodingKeys: String, CodingKey {
        case schemaVersion, workload, workloadFile, workloadRole, runsRequested, launchState
        case computeUnits, method, requestedStartingShape, stopsOnFailure, contextMargin
        case device, startedAt, runsCompleted, runsSucceeded, stoppedEarly, allSucceeded
        case measurementDisclaimer, terminationNote, runs
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(Self.schemaVersion, forKey: .schemaVersion)
        try container.encode(request.workload.letter, forKey: .workload)
        try container.encode(request.workload.fileName, forKey: .workloadFile)
        try container.encode(request.workload.role, forKey: .workloadRole)
        try container.encode(request.runs, forKey: .runsRequested)
        try container.encode(request.launchState.rawValue, forKey: .launchState)
        try container.encode(request.computeUnits.rawValue, forKey: .computeUnits)
        try container.encode(request.method.rawValue, forKey: .method)
        // `requestedStartingShape`, not `startingShape`: the run rows carry a `startingShape` too
        // and it means something else — the rung the ladder had actually *reached* by that run.
        // One key name for two quantities in one file is how a record gets misread.
        //
        // `encodeIfPresent`, so the whole document follows one rule: an optional with no value is
        // **omitted**, exactly as the synthesized encoders do for every run-row field. A header
        // that wrote an explicit null while the rows omitted theirs would be a second convention
        // for a reader to learn (see `MemoryReading` for why the label, not the key's presence, is
        // what a consumer branches on).
        try container.encodeIfPresent(
            request.startingShape?.inputTileSize, forKey: .requestedStartingShape)
        try container.encode(request.stopsOnFailure, forKey: .stopsOnFailure)
        // The margin the runs ACTUALLY used, not a constant. Margins are per model, so a
        // static value here would label a series with another model's number.
        try container.encode(runs.first?.contextMargin ?? -1, forKey: .contextMargin)
        try container.encode(device, forKey: .device)
        try container.encode(
            ISO8601DateFormatter().string(from: startedAt), forKey: .startedAt)
        try container.encode(runsCompleted, forKey: .runsCompleted)
        try container.encode(runsSucceeded, forKey: .runsSucceeded)
        try container.encode(stoppedEarly, forKey: .stoppedEarly)
        try container.encode(allSucceeded, forKey: .allSucceeded)
        try container.encode(DeviceHarness.measurementDisclaimer, forKey: .measurementDisclaimer)
        try container.encode(WholeImageRunner.terminationNote, forKey: .terminationNote)
        try container.encode(runs, forKey: .runs)
    }
}

/// One completed run: the record, plus the ladder to carry into the next image.
///
/// The ladder travels beside the record rather than inside it because `TileShapeLadder` is engine
/// state, not a measurement — `SequentialTilingEngine.upscale(_:shapes:ladder:…)` takes it by value
/// and hands it back so a job that throws leaves the caller's ladder exactly as it was. Same shape
/// as `SingleTileRun`, which carries its pixels beside its result for the same kind of reason.
public struct WholeImageRun: Sendable {
    public let result: WholeImageRunResult
    /// `nil` when the run never reached the engine, in which case the caller keeps what it had.
    public let ladder: TileShapeLadder?
}

/// Supplies one rung's artifact and a loaded backend, out of whatever was dropped into
/// `HarnessArtifacts/` — the harness half of `TileShapeBackendProviding`.
///
/// Which artifact belongs to which rung is the caller's knowledge (the engine says so), and here
/// that knowledge is `HarnessShape`'s file-name convention: discovery, not configuration. Holds
/// `URL`s rather than a `Bundle` because `Bundle` is not `Sendable` and this must cross into the
/// engine's concurrency domain.
struct BundledShapeBackendProvider: TileShapeBackendProviding {
    let models: [HarnessShape: URL]
    let computeUnits: ComputeUnits
    /// Model family used to name a rung that was never dropped in, so the error says which file.
    let identifier: String

    func backend(for shape: TileShape) async throws -> ShapedBackend {
        guard let entry = models.first(where: { $0.key.tileSize == shape.inputTileSize }) else {
            // The rung the ladder wants has no artifact. Reported as the same missing-resource
            // message the single-tile path uses, naming the file to copy in — a step down that
            // dies on "not found" must be actionable on the phone, not in a debugger.
            throw SingleTileRunner.ResourceError.modelMissing(
                fileName: HarnessShape(identifier: identifier, tileSize: shape.inputTileSize)
                    .modelFileName)
        }
        let descriptor = SingleTileRunner.descriptor(shape: entry.key, modelURL: entry.value)
        let backend = CoreMLBackend()
        try await backend.load(descriptor, context: InferenceContext(computeUnits: computeUnits))
        return ShapedBackend(model: descriptor, backend: backend)
    }
}

/// Runs a **whole image** through the tiling engine on device, and a series of them in one
/// invocation — the instrument certification spec §6's protocol is executed with.
///
/// `SingleTileRunner` answers whether the model executes on device at all, with one fixed tile. The
/// memory protocol asks a different question — can the floor device take a 12 MP image without
/// jetsam? — and every part of it is a whole-image job: ten consecutive 12 MP → 4× runs warm and
/// cold, memory warnings handled, 48 MP survived.
///
/// **Headless, and that is the scope line.** Nothing here imports UIKit, AppKit or SwiftUI; the
/// app's whole integration is one launch-argument branch, following `-memoryprobe`. Anything that
/// needs a view hierarchy or user-facing state belongs to the app, not here.
///
/// **What it does not do.** No progress display, no results browser, no benchmark methodology, no
/// encode of the spooled output — output equivalence and timing methodology are measured elsewhere.
/// The spool file is measured, then deleted.
public enum WholeImageRunner {
    /// The context margin for a shape, or `nil` when that model has no derived one.
    ///
    /// Never a single constant: the margin follows the model's receptive field, and the supported
    /// models run at 16 or 76 px. Running one on another's number under-pads it, and under-padding
    /// is visible seams with nothing thrown.
    public static func contextMargin(for shape: HarnessShape) -> Int? {
        shape.derivedMargin
    }

    /// `centerValidRegion` — the tiling contract's default, which seam certification across all
    /// four stitch methods left unchanged. Chosen by the table, never by preference.
    public static let contractDefaultMethod = StitchMethod.centerValidRegion

    /// The sentence a series file carries so nobody reads a short run as a clean one.
    public static let terminationNote =
        "A jetsam termination cannot be self-reported: it kills the process without notice. This "
        + "file is rewritten after every run, so runsCompleted < runsRequested with no "
        + "errorDescription on the last row is what a termination looks like from inside."

    /// The workload file for `workload`, if it was dropped into `HarnessArtifacts/`.
    public static func workloadURL(
        _ workload: HarnessWorkload, in bundle: Bundle = .main
    ) -> URL? {
        DeviceHarness.existingURL(bundle: bundle, fileName: workload.fileName)
    }

    /// Missing workload fixtures get the same treatment missing models get: an instruction naming
    /// the file, on screen, rather than a crash. The fixture is committed to the repository but is
    /// still a manual copy into `HarnessArtifacts/`, so "you forgot it" is a live outcome.
    public enum ResourceError: Error, Equatable {
        case workloadMissing(fileName: String)

        public var message: String {
            switch self {
            case .workloadMissing(let fileName):
                return
                    "Workload image not bundled. Copy \(fileName) from "
                    + "Sources/TestFixtures/workloads/ into DeviceHarnessApp/HarnessArtifacts/ "
                    + "and rebuild."
            }
        }
    }

    // MARK: - One run

    /// Run one whole image, carrying `ladder` in and handing it back out.
    ///
    /// ## Why the ladder is threaded rather than rebuilt
    ///
    /// `TileShapeLadder` adopts the budget's plan on its first image and afterwards climbs only on
    /// **sustained** headroom — `dwellImages` completed images in a row, each with the next rung up
    /// clearing the ρ-derived margin. A ladder rebuilt per run could never climb, and the climb is
    /// exactly what a ten-run series is in a position to observe. So run 1 gets a fresh ladder and
    /// runs 2…N get the one the previous run returned, which is what the engine's own documentation
    /// asks for ("handed back in the result … for the caller to carry into the next image").
    ///
    /// **The runner does not seed the ladder on its own initiative.** It could:
    /// `TileShapeLadder.init(startingAt:)` is public and `-startshape` exposes it. It does not,
    /// because seeding would be the instrument inventing policy — and because what the shipped
    /// policy actually does across ten runs is the measurement the placeholder dwell number is
    /// waiting for. `plannedShape` and `startingShape` are both in the record so the policy's
    /// behaviour is visible in the data.
    ///
    /// - Parameters:
    ///   - request: the series this run belongs to; supplies workload, units, method and shape.
    ///   - index: 1-based position in the series.
    ///   - ladder: the ladder the previous run returned, or `nil` for the first run.
    ///   - bundle: where `HarnessArtifacts/` lives.
    /// - Returns: the run's record, and the ladder to carry forward.
    public static func run(
        _ request: WholeImageSeriesRequest,
        index: Int,
        carrying ladder: TileShapeLadder?,
        in bundle: Bundle = .main
    ) async -> WholeImageRun {
        let clock = ContinuousClock()
        let started = clock.now
        let thermalAtStart = describe(ProcessInfo.processInfo.thermalState)
        let availableAtStart = AvailableMemory.now()
        let footprintAtStart = MemoryReading.kernel(
            SingleTileRunner.currentFootprintBytes(), source: "phys_footprint, before decode")

        func sample(afterDecode: UInt64?, atEnd: UInt64?) -> MemorySample {
            MemorySample(
                availableAtStart: .figure(availableAtStart),
                availableAtEnd: .figure(AvailableMemory.now()),
                footprintAtStart: footprintAtStart,
                footprintAfterDecode: .kernel(
                    afterDecode, source: "phys_footprint, source image decoded"),
                footprintAtEnd: .kernel(atEnd, source: "phys_footprint, run finished"),
                processPeakFootprint: .kernel(
                    SingleTileRunner.peakFootprintBytes(),
                    source: "ledger_phys_footprint_peak — process high-water since launch"),
                processPeakResidentSize: .kernel(
                    SingleTileRunner.peakMemoryBytes(),
                    source: "resident_size_max — process high-water since launch"))
        }

        func failed(_ message: String, afterDecode: UInt64? = nil) -> WholeImageRun {
            WholeImageRun(
                result: result(
                    // -1: a failure can precede model selection, so there may be no margin in force.
                    // A plausible-looking 76 would label the row with a number nothing used.
                    request: request, index: index, contextMargin: -1, success: false,
                    wallClockSeconds: seconds(clock.now - started), upscaleSeconds: nil,
                    geometry: nil, plan: nil, startingShape: ladder?.shape, run: nil,
                    thermalAtStart: thermalAtStart,
                    thermalAtEnd: describe(ProcessInfo.processInfo.thermalState),
                    pressureLevelAtEnd: describe(MemoryPressureLevel.normal),
                    memory: sample(afterDecode: afterDecode, atEnd: nil),
                    errorDescription: message),
                ladder: ladder)
        }

        guard let imageURL = workloadURL(request.workload, in: bundle) else {
            return failed(ResourceError.workloadMissing(fileName: request.workload.fileName).message)
        }
        let discovered = DeviceHarness.availableShapes(in: bundle)
        var models: [HarnessShape: URL] = [:]
        for shape in discovered {
            if let url = DeviceHarness.bundledModelURL(for: shape, in: bundle) {
                models[shape] = url
            }
        }
        guard let floorEntry = models.min(by: { $0.key < $1.key }) else {
            return failed(
                SingleTileRunner.ResourceError
                    .modelMissing(fileName: HarnessShape.default256.modelFileName).message)
        }

        let monitor = MemoryPressureMonitor()
        monitor.start()
        defer { monitor.stop() }

        do {
            // The decoded source stays live from the first tile to the last — the budget's
            // input-surface term, 46.5 MiB at 12 MP and 186.0 MiB at 48 MP. `footprintAfterDecode`
            // is where that term stops being an estimate.
            let image = try SingleTileRunner.loadPixelBuffer(from: imageURL)
            let footprintAfterDecode = SingleTileRunner.currentFootprintBytes()
            let width = CVPixelBufferGetWidth(image)
            let height = CVPixelBufferGetHeight(image)

            guard let margin = WholeImageRunner.contextMargin(for: floorEntry.key) else {
                return failed(
                    "no derived context margin for \(floorEntry.key.identifier). Deriving one is "
                        + "tools/margin/derive_context_margin_torch.py; running without it would "
                        + "use another model's number and produce seams.")
            }
            let engine = SequentialTilingEngine(contextMargin: margin)
            // Only `scaleFactor` is read out of this descriptor — `workload(…)` ignores
            // `inputTileSize` by contract, because the shape is what the budget solves for. Going
            // through the engine's own seam rather than building an `ImageWorkload` by hand is what
            // makes it impossible to cost the budget against a margin the engine is not running.
            let workload = engine.workload(
                imageWidth: width, imageHeight: height,
                model: SingleTileRunner.descriptor(
                    shape: floorEntry.key, modelURL: floorEntry.value),
                method: request.method, destination: .bandSpool)
            let budget = MemoryBudget(available: availableAtStart)
            let plan = budget.plan(for: workload)

            let carried =
                ladder
                ?? TileShapeLadder(
                    parameters: .defaults(),
                    startingAt: request.startingShape ?? TileShape.floorRung)
            let startingShape = carried.shape

            let provider = BundledShapeBackendProvider(
                models: models, computeUnits: request.computeUnits,
                identifier: floorEntry.key.identifier)

            // The engine creates and fills the spool and never deletes it — the URL is the
            // caller's to own. Here the caller is a measurement, so the file is measured and then
            // removed: at 12MP→4× it is 768 MB and at 48MP→4× 3.07 GB, and ten of them would fill
            // the phone. Removed on the way out of this scope, success or throw.
            let spoolURL = FileManager.default.temporaryDirectory
                .appendingPathComponent("wholeimage-\(UUID().uuidString).bgra")
            defer { try? FileManager.default.removeItem(at: spoolURL) }

            let upscaleStarted = clock.now
            let laddered = try await engine.upscale(
                image, shapes: provider, ladder: carried, budget: plan, pressure: monitor,
                method: request.method, spoolingTo: spoolURL, progress: nil)
            let upscaleSeconds = seconds(clock.now - upscaleStarted)
            let footprintAtEnd = SingleTileRunner.currentFootprintBytes()

            let geometry = RunGeometry(
                inputWidth: width, inputHeight: height,
                outputWidth: laddered.image.width, outputHeight: laddered.image.height,
                spoolBytes: fileSize(at: spoolURL), spoolBytesExpected: laddered.image.byteCount)

            return WholeImageRun(
                result: result(
                    request: request, index: index, contextMargin: margin, success: true,
                    wallClockSeconds: seconds(clock.now - started),
                    upscaleSeconds: upscaleSeconds, geometry: geometry, plan: plan,
                    startingShape: startingShape, run: laddered, thermalAtStart: thermalAtStart,
                    thermalAtEnd: describe(ProcessInfo.processInfo.thermalState),
                    pressureLevelAtEnd: describe(monitor.level),
                    memory: sample(afterDecode: footprintAfterDecode, atEnd: footprintAtEnd),
                    errorDescription: nil),
                ladder: laddered.ladder)
        } catch {
            return failed(describe(error), afterDecode: SingleTileRunner.currentFootprintBytes())
        }
    }

    // MARK: - The series

    /// Run `request.runs` whole images back to back in one invocation.
    ///
    /// Certification spec §6's "10 consecutive runs" is a property of the protocol, not of a user
    /// interface, so it is one call rather than ten taps. `onProgress` fires after **every** run
    /// with the series as it stands, which is what lets the caller persist the file incrementally —
    /// see `WholeImageSeries.stoppedEarly` for why that is the jetsam evidence and not a
    /// convenience.
    ///
    /// The ladder is threaded from each run into the next; a run that never reached the engine
    /// leaves it untouched.
    ///
    /// - Parameters:
    ///   - request: what to run, how many times.
    ///   - device: the device identity stamped into the record.
    ///   - startedAt: series start timestamp.
    ///   - runOne: performs one run. Injected so the series arithmetic is testable without a model.
    ///   - onProgress: called after each run with the series so far.
    /// - Returns: the finished series.
    static func runSeries(
        _ request: WholeImageSeriesRequest,
        device: DeviceIdentity = .current(),
        startedAt: Date = Date(),
        running runOne: (Int, TileShapeLadder?) async -> WholeImageRun,
        onProgress: ((WholeImageSeries) -> Void)? = nil
    ) async -> WholeImageSeries {
        var records: [WholeImageRunResult] = []
        var ladder: TileShapeLadder?
        var series = WholeImageSeries(
            request: request, device: device, startedAt: startedAt, runs: records)

        for index in 1...max(1, request.runs) where index <= request.runs {
            let outcome = await runOne(index, ladder)
            records.append(outcome.result)
            ladder = outcome.ladder ?? ladder
            series = WholeImageSeries(
                request: request, device: device, startedAt: startedAt, runs: records)
            onProgress?(series)
            if !outcome.result.success, request.stopsOnFailure { break }
        }
        return series
    }

    /// The device-side entry point: run the series described by `request`, against the real engine.
    public static func runSeries(
        _ request: WholeImageSeriesRequest,
        in bundle: Bundle = .main,
        onProgress: ((WholeImageSeries) -> Void)? = nil
    ) async -> WholeImageSeries {
        await runSeries(
            request,
            running: { index, ladder in
                await run(request, index: index, carrying: ladder, in: bundle)
            },
            onProgress: onProgress)
    }

    // MARK: - JSON export

    /// The series as JSON — sorted keys and pretty printing, so two runs of the same series differ
    /// only where the measurements differ and a diff of two records is readable.
    public static func json(for series: WholeImageSeries) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(series)
    }

    /// File name for a series' record. Carries the workload, the run count and the operator's
    /// declared launch state, so cold and warm series cannot be confused after the fact.
    public static func fileName(for request: WholeImageSeriesRequest, startedAt: Date) -> String {
        let stamp = DateFormatter.seriesStamp.string(from: startedAt)
        return "wholeimage-\(request.workload.letter)-\(request.runs)x-"
            + "\(request.launchState.rawValue)-\(stamp).json"
    }

    /// Where a device run leaves its record: the app's Documents directory, which Xcode's
    /// Devices and Simulators window can download as a container. `nil` when there is no such
    /// directory, in which case the caller still has the printed copy.
    public static func documentsURL() -> URL? {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
    }

    // MARK: - Launch-argument invocation

    /// Run a series if the launch arguments ask for one, and do nothing at all if they do not.
    ///
    /// The whole of the app's integration. Absent `-wholeimage` this returns immediately and the
    /// harness is unchanged — the same shape as `-memoryprobe`'s branch.
    ///
    /// The record goes two places on purpose: printed, so the Xcode console has it verbatim with no
    /// container download; and written to Documents after every run, so a jetsam kill leaves the
    /// completed rows on the device rather than in a console that died with the process.
    ///
    /// - Parameters:
    ///   - arguments: the process argument vector.
    ///   - bundle: where `HarnessArtifacts/` lives.
    /// - Returns: the finished series, or `nil` when no series was requested or the request was
    ///   malformed.
    @discardableResult
    public static func runSeriesIfRequested(
        arguments: [String] = ProcessInfo.processInfo.arguments,
        in bundle: Bundle = .main
    ) async -> WholeImageSeries? {
        switch WholeImageSeriesRequest.parse(arguments: arguments) {
        case .notRequested:
            return nil
        case .invalid(let message):
            print("[whole-image] bad launch argument: \(message)")
            return nil
        case .requested(let request):
            let startedAt = Date()
            let destination = documentsURL()?
                .appendingPathComponent(fileName(for: request, startedAt: startedAt))
            print("[whole-image] \(request.runs)× \(request.workload.role)")
            print("[whole-image] record: \(destination?.path ?? "not writable — console only")")

            let series = await runSeries(
                request, device: .current(), startedAt: startedAt,
                running: { index, ladder in
                    await run(request, index: index, carrying: ladder, in: bundle)
                },
                onProgress: { series in
                    guard let destination, let data = try? json(for: series) else { return }
                    try? data.write(to: destination, options: .atomic)
                })

            if let data = try? json(for: series), let text = String(data: data, encoding: .utf8) {
                print(text)
            }
            return series
        }
    }

    // MARK: - Rendering, and the arithmetic behind it

    /// Flattens one run into its record. One place knows the mapping, so the success path and the
    /// failure path cannot drift into recording different things.
    static func result(
        request: WholeImageSeriesRequest,
        index: Int,
        contextMargin: Int,
        success: Bool,
        wallClockSeconds: Double,
        upscaleSeconds: Double?,
        geometry: RunGeometry?,
        plan: MemoryBudget.Plan?,
        startingShape: TileShape?,
        run: LadderedRun?,
        thermalAtStart: String,
        thermalAtEnd: String,
        pressureLevelAtEnd: String,
        memory: MemorySample,
        errorDescription: String?
    ) -> WholeImageRunResult {
        WholeImageRunResult(
            index: index, workload: request.workload.letter,
            workloadFile: request.workload.fileName, launchState: request.launchState.rawValue,
            computeUnits: request.computeUnits.rawValue, method: request.method.rawValue,
            contextMargin: contextMargin, success: success, wallClockSeconds: wallClockSeconds,
            upscaleSeconds: upscaleSeconds, geometry: geometry,
            plannedShape: plan?.shape.inputTileSize,
            plannedSelection: plan.map { describe($0.selection) }, plannedFit: plan?.fit.rawValue,
            allowanceBytes: plan?.allowance?.bytes, plannedTotalBytes: plan?.selected.totalBytes,
            plannedHeadroomBytes: plan?.headroomBytes,
            plannedProvenance: plan?.provenance.rawValue,
            safetyFraction: plan == nil ? nil : MemoryBudget.SafetyFraction.standard.value,
            startingShape: startingShape?.inputTileSize, finalShape: run?.shape.inputTileSize,
            stepDowns: run?.stepDowns, tilesPredicted: run?.tilesPredicted,
            tilesRecomputed: run?.tilesRecomputed, isDegraded: run?.isDegraded,
            ladderOutcomes: run?.outcomes.map(\.description) ?? [],
            thermalStateAtStart: thermalAtStart, thermalStateAtEnd: thermalAtEnd,
            pressureLevelAtEnd: pressureLevelAtEnd, memory: memory,
            errorDescription: errorDescription)
    }

    /// Error text a human can act on, extended with this file's own resource case. Typed engine
    /// errors keep their own description, exactly as the single-tile path reports them.
    public static func describe(_ error: Error) -> String {
        if let resourceError = error as? ResourceError { return resourceError.message }
        if let tilingError = error as? TilingError { return "\(tilingError)" }
        return SingleTileRunner.describe(error)
    }

    /// Thermal state as a word. `ProcessInfo.thermalState` is Foundation, so reading it costs no UI
    /// framework — certification spec §7.3 records device state with every measurement, and this is
    /// the part of it the process can read for itself.
    public static func describe(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    public static func describe(_ level: MemoryPressureLevel) -> String {
        switch level {
        case .normal: return "normal"
        case .warning: return "warning"
        case .critical: return "critical"
        }
    }

    /// The budget's selection, spelled out. `MemoryBudget.Selection` carries its payload in the
    /// case, and a record that dropped it would lose the shortfall — the one number that says how
    /// far off "does not fit" was.
    public static func describe(_ selection: MemoryBudget.Selection) -> String {
        switch selection {
        case .largestThatFits(let shape):
            return "largestThatFits(\(shape.inputTileSize))"
        case .floorRungAvailabilityUnknown(let shape):
            return "floorRungAvailabilityUnknown(\(shape.inputTileSize))"
        case .noRungFits(let floor, let shortfallBytes):
            return "noRungFits(floor: \(floor.inputTileSize), shortfallBytes: \(shortfallBytes))"
        }
    }

    /// `Duration` to seconds, by the same arithmetic `SingleTileRunner.run` uses.
    static func seconds(_ duration: Duration) -> TimeInterval {
        TimeInterval(duration.components.seconds)
            + Double(duration.components.attoseconds) / 1e18
    }

    /// Bytes the file holds, or `nil` when it cannot be read. Never zero standing in for unknown.
    static func fileSize(at url: URL) -> Int? {
        let attributes = try? FileManager.default.attributesOfItem(atPath: url.path)
        return (attributes?[.size] as? NSNumber)?.intValue
    }
}

extension DateFormatter {
    /// `20260829-142530` — sortable, no separators a file system dislikes, UTC so two devices in
    /// two time zones produce comparable file names.
    static let seriesStamp: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return formatter
    }()
}
