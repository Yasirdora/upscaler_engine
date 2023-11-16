import Foundation
import Testing
import UpscaleCore

@testable import DeviceHarness

/// Model-free tests for the whole-image runner.
///
/// **Nothing here loads a `.mlpackage`, decodes a workload image or touches a device.** What is
/// covered is exactly the part that can be tested without one: the launch-argument grammar,
/// artifact and workload resolution, the series arithmetic, and the JSON a device run leaves
/// behind. Every expectation is a written-out literal — a record format checked against itself
/// would pass while producing an unreadable file.
struct WholeImageRunnerTests {

    // MARK: - Fixtures

    static let stampedDate = Date(timeIntervalSince1970: 1_788_013_530)  // 2026-08-29T14:25:30Z

    static func memory(
        available: UInt64? = 2_147_483_648,
        footprintStart: UInt64? = 104_857_600,
        footprintEnd: UInt64? = 209_715_200,
        peakFootprint: UInt64? = 314_572_800,
        peakResident: UInt64? = 419_430_400
    ) -> MemorySample {
        MemorySample(
            availableAtStart: .kernel(available, source: "os_proc_available_memory()"),
            availableAtEnd: .kernel(available, source: "os_proc_available_memory()"),
            footprintAtStart: .kernel(footprintStart, source: "phys_footprint, before decode"),
            footprintAfterDecode: .kernel(nil, source: "phys_footprint, source image decoded"),
            footprintAtEnd: .kernel(footprintEnd, source: "phys_footprint, run finished"),
            processPeakFootprint: .kernel(peakFootprint, source: "ledger_phys_footprint_peak"),
            processPeakResidentSize: .kernel(peakResident, source: "resident_size_max"))
    }

    static func record(
        index: Int,
        request: WholeImageSeriesRequest,
        success: Bool = true,
        startingShape: Int? = nil,
        error: String? = nil
    ) -> WholeImageRunResult {
        WholeImageRunner.result(
            request: request, index: index, contextMargin: 76, success: success, wallClockSeconds: 42.5,
            upscaleSeconds: 41.0,
            geometry: RunGeometry(
                inputWidth: 4032, inputHeight: 3024, outputWidth: 16128, outputHeight: 12096,
                spoolBytes: 780_337_152, spoolBytesExpected: 780_337_152),
            plan: nil, startingShape: startingShape.flatMap(TileShape.init(rawValue:)), run: nil,
            thermalAtStart: "nominal", thermalAtEnd: "fair", pressureLevelAtEnd: "normal",
            memory: memory(), errorDescription: error)
    }

    /// A stub run: a pure function of the index, so the series can be driven without a model and
    /// without mutable capture. `startingShape` echoes back whatever ladder the series handed in,
    /// which is how the threading assertions observe it.
    static func stub(
        request: WholeImageSeriesRequest,
        failingAt failureIndex: Int? = nil,
        handingBack ladder: TileShapeLadder? = nil
    ) -> (Int, TileShapeLadder?) async -> WholeImageRun {
        { index, carried in
            WholeImageRun(
                result: record(
                    index: index, request: request, success: index != failureIndex,
                    startingShape: carried?.shape.inputTileSize,
                    error: index == failureIndex ? "stubbed failure" : nil),
                ladder: ladder)
        }
    }

    // MARK: - Invocation: the launch-argument grammar

    @Test("no trigger argument means no series — the harness is untouched")
    func absentTriggerIsSilence() {
        #expect(
            WholeImageSeriesRequest.parse(arguments: ["app", "-autorun", "-shape", "512"])
                == .notRequested)
        #expect(WholeImageSeriesRequest.parse(arguments: []) == .notRequested)
    }

    @Test("the bare trigger runs one workload-B series at the contract defaults")
    func bareTriggerDefaults() throws {
        guard
            case .requested(let request) = WholeImageSeriesRequest.parse(
                arguments: ["app", "-wholeimage"])
        else {
            Issue.record("expected a request")
            return
        }
        #expect(request.workload == .b)
        #expect(request.runs == 1, "the safe default — §6's ten is typed, so it is recorded")
        #expect(request.launchState == .unrecorded)
        #expect(request.computeUnits == .all)
        #expect(request.method == .centerValidRegion)
        #expect(request.startingShape == nil)
        #expect(request.stopsOnFailure)
    }

    @Test("the §6 protocol's own invocation parses to the §6 protocol")
    func protocolInvocation() throws {
        guard
            case .requested(let request) = WholeImageSeriesRequest.parse(
                arguments: [
                    "app", "-wholeimage", "-workload", "B", "-runs", "10", "-state", "cold",
                    "-computeunits", "cpuAndNeuralEngine", "-method", "hardCrop", "-startshape", "512",
                    "-continueonfailure",
                ])
        else {
            Issue.record("expected a request")
            return
        }
        #expect(request.workload == .b)
        #expect(request.runs == 10)
        #expect(request.launchState == .cold)
        #expect(request.computeUnits == .cpuAndNeuralEngine)
        #expect(request.method == .hardCrop)
        #expect(request.startingShape == .n512)
        #expect(request.stopsOnFailure == false)
    }

    /// The rule `ContentView`'s `-shape` handling already states: an automated check must never
    /// quietly run something other than what it asked for. A mistyped value is an error, never a
    /// silent fallback to the default.
    @Test(
        "a malformed value is reported by name, never silently defaulted",
        arguments: [
            (["app", "-wholeimage", "-workload", "D"], "-workload D"),
            (["app", "-wholeimage", "-runs", "0"], "-runs 0"),
            (["app", "-wholeimage", "-runs", "ten"], "-runs ten"),
            (["app", "-wholeimage", "-state", "tepid"], "-state tepid"),
            (["app", "-wholeimage", "-computeunits", "gpuOnly"], "-computeunits gpuOnly"),
            (["app", "-wholeimage", "-method", "bilinear"], "-method bilinear"),
            (["app", "-wholeimage", "-startshape", "1024"], "-startshape 1024"),
        ])
    func malformedValuesAreRejected(arguments: [String], expectedPrefix: String) {
        guard case .invalid(let message) = WholeImageSeriesRequest.parse(arguments: arguments)
        else {
            Issue.record("expected .invalid for \(arguments)")
            return
        }
        #expect(message.hasPrefix(expectedPrefix))
    }

    @Test("`-state unrecorded` is rejected: it is a default, not a declaration")
    func unrecordedCannotBeDeclared() {
        guard
            case .invalid = WholeImageSeriesRequest.parse(
                arguments: ["app", "-wholeimage", "-state", "unrecorded"])
        else {
            Issue.record("expected .invalid")
            return
        }
    }

    @Test("a flag with no value after it keeps the default rather than reading past the end")
    func danglingFlagIsSafe() {
        guard
            case .requested(let request) = WholeImageSeriesRequest.parse(
                arguments: ["app", "-wholeimage", "-runs"])
        else {
            Issue.record("expected a request")
            return
        }
        #expect(request.runs == 1)
    }

    @Test("workload letters parse in either case and nothing else parses at all")
    func workloadParsing() {
        #expect(HarnessWorkload.parse("A") == .a)
        #expect(HarnessWorkload.parse("c") == .c)
        #expect(HarnessWorkload.parse("B") == .b)
        #expect(HarnessWorkload.parse("BB") == nil)
        #expect(HarnessWorkload.parse("") == nil)
    }

    // MARK: - Shape and artifact selection

    @Test("workload file names are the committed fixture names, verbatim")
    func workloadFileNames() {
        #expect(HarnessWorkload.a.fileName == "A_1080p.jpg")
        #expect(HarnessWorkload.b.fileName == "B_12MP.jpg")
        #expect(HarnessWorkload.c.fileName == "C_48MP.jpg")
        #expect(HarnessWorkload.b.letter == "B")
    }

    @Test("a bundle without the drop-in folder resolves every workload to nil, not a bogus URL")
    func workloadResolutionIsHonest() {
        for workload in HarnessWorkload.allCases {
            #expect(WholeImageRunner.workloadURL(workload, in: .main) == nil)
        }
    }

    @Test("a missing workload names the file and both folders it travels between")
    func missingWorkloadIsActionable() {
        let message = WholeImageRunner.ResourceError
            .workloadMissing(fileName: "C_48MP.jpg").message
        #expect(message.contains("C_48MP.jpg"))
        #expect(message.contains("Sources/TestFixtures/workloads/"))
        #expect(message.contains("DeviceHarnessApp/HarnessArtifacts/"))
        #expect(
            WholeImageRunner.describe(
                WholeImageRunner.ResourceError.workloadMissing(fileName: "C_48MP.jpg")) == message)
    }

    /// A step down that dies because the rung below was never copied in has to say which file to
    /// copy. The provider is the only place that knows, and the message it throws is the one the
    /// single-tile path already uses.
    @Test("a rung with no bundled artifact throws a message naming the .mlpackage to copy in")
    func missingRungNamesItsArtifact() async {
        let provider = BundledShapeBackendProvider(
            models: [:], computeUnits: .all, identifier: "RealESRGAN_x4plus")
        await #expect(
            throws: SingleTileRunner.ResourceError.modelMissing(
                fileName: "RealESRGAN_x4plus_384x384.mlpackage")
        ) {
            _ = try await provider.backend(for: .n384)
        }
    }

    @Test("a run in a bundle with no artifacts fails legibly and leaves the ladder untouched")
    func runWithoutArtifactsIsLegible() async {
        let request = WholeImageSeriesRequest(workload: .b, runs: 1)
        let carried = TileShapeLadder(startingAt: .n384)
        let run = await WholeImageRunner.run(request, index: 1, carrying: carried, in: .main)

        #expect(run.result.success == false)
        #expect(run.result.errorDescription?.contains("Workload image not bundled") == true)
        #expect(run.result.index == 1)
        #expect(run.result.workload == "B")
        #expect(run.result.geometry == nil)
        #expect(run.result.ladderOutcomes.isEmpty)
        #expect(run.ladder?.shape == .n384, "a run that never started must not move the ladder")
    }

    // MARK: - Series logic

    @Test("a ten-run request runs ten times and reports a clean series")
    func fullSeries() async {
        let request = WholeImageSeriesRequest(workload: .b, runs: 10, launchState: .warm)
        let series = await WholeImageRunner.runSeries(
            request, device: .current(), startedAt: Self.stampedDate,
            running: Self.stub(request: request))

        #expect(series.runsCompleted == 10)
        #expect(series.runsSucceeded == 10)
        #expect(series.stoppedEarly == false)
        #expect(series.allSucceeded)
        #expect(series.runs.map(\.index) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    }

    @Test("the series stops at the first failure and records the runs it did complete")
    func stopsOnFailure() async {
        let request = WholeImageSeriesRequest(workload: .b, runs: 10)
        let series = await WholeImageRunner.runSeries(
            request, device: .current(), startedAt: Self.stampedDate,
            running: Self.stub(request: request, failingAt: 4))

        #expect(series.runsCompleted == 4, "the failed run is recorded, not discarded")
        #expect(series.runsSucceeded == 3)
        #expect(series.stoppedEarly)
        #expect(series.allSucceeded == false)
        #expect(series.runs.last?.errorDescription == "stubbed failure")
    }

    @Test("-continueonfailure runs the whole series and records the one that failed")
    func continuesOnFailure() async {
        let request = WholeImageSeriesRequest(workload: .c, runs: 5, stopsOnFailure: false)
        let series = await WholeImageRunner.runSeries(
            request, device: .current(), startedAt: Self.stampedDate,
            running: Self.stub(request: request, failingAt: 2))

        #expect(series.runsCompleted == 5)
        #expect(series.runsSucceeded == 4)
        #expect(series.stoppedEarly == false)
        #expect(series.allSucceeded == false)
    }

    /// The ladder is threaded so the climb `TileShapeLadder` gates on dwell can happen at all — a
    /// ladder rebuilt per run could never climb, and observing the climb is what the placeholder
    /// dwell number needs.
    @Test("the ladder each run hands back is the one the next run is given")
    func ladderIsThreaded() async {
        let request = WholeImageSeriesRequest(workload: .b, runs: 3)
        let series = await WholeImageRunner.runSeries(
            request, device: .current(), startedAt: Self.stampedDate,
            running: Self.stub(request: request, handingBack: TileShapeLadder(startingAt: .n512)))

        #expect(series.runs.map(\.startingShape) == [nil, 512, 512])
    }

    @Test("progress fires once per run, each time with the series as it stands")
    func progressFiresPerRun() async {
        let request = WholeImageSeriesRequest(workload: .b, runs: 4)
        let observed = Counts()
        _ = await WholeImageRunner.runSeries(
            request, device: .current(), startedAt: Self.stampedDate,
            running: Self.stub(request: request),
            onProgress: { observed.append($0.runsCompleted) })

        #expect(observed.values == [1, 2, 3, 4], "the file is rewritten after every run")
    }

    /// Not `@Sendable`-shared state: the series is sequential and `onProgress` is called on the
    /// same task, so a plain box is the honest model of what happens.
    final class Counts {
        private(set) var values: [Int] = []
        func append(_ value: Int) { values.append(value) }
    }

    // MARK: - JSON export

    static func encodedSeries(
        request: WholeImageSeriesRequest, runs: [WholeImageRunResult]
    ) throws -> [String: Any] {
        let series = WholeImageSeries(
            request: request,
            device: DeviceIdentity(
                machine: "iPhone13,3", systemName: "iOS", systemVersion: "Version 18.0",
                processorCount: 6, physicalMemoryBytes: 6_442_450_944),
            startedAt: stampedDate, runs: runs)
        let data = try WholeImageRunner.json(for: series)
        let object = try JSONSerialization.jsonObject(with: data)
        guard let dictionary = object as? [String: Any] else {
            throw CocoaError(.coderInvalidValue)
        }
        return dictionary
    }

    @Test("the series header names the workload, the protocol and the device")
    func seriesHeaderJSON() throws {
        let request = WholeImageSeriesRequest(
            workload: .b, runs: 10, launchState: .cold, computeUnits: .cpuAndNeuralEngine)
        let json = try Self.encodedSeries(request: request, runs: [Self.record(index: 1, request: request)])

        #expect(json["schemaVersion"] as? Int == 1)
        #expect(json["workload"] as? String == "B")
        #expect(json["workloadFile"] as? String == "B_12MP.jpg")
        #expect(json["runsRequested"] as? Int == 10)
        #expect(json["runsCompleted"] as? Int == 1)
        #expect(json["runsSucceeded"] as? Int == 1)
        #expect(json["stoppedEarly"] as? Bool == true)
        #expect(json["allSucceeded"] as? Bool == false)
        #expect(json["launchState"] as? String == "cold")
        #expect(json["computeUnits"] as? String == "cpuAndNeuralEngine")
        #expect(json["method"] as? String == "centerValidRegion")
        #expect(json["contextMargin"] as? Int == 76)
        // Distinct from a run row's `startingShape`, which is the rung the ladder had reached.
        #expect(json["requestedStartingShape"] == nil)
        #expect(json["startingShape"] == nil, "the series header must not shadow the run-row key")

        let seeded = try Self.encodedSeries(
            request: WholeImageSeriesRequest(workload: .b, runs: 1, startingShape: .n512),
            runs: [])
        #expect(seeded["requestedStartingShape"] as? Int == 512)
        #expect(json["startedAt"] as? String == "2026-08-29T14:25:30Z")
        #expect((json["device"] as? [String: Any])?["machine"] as? String == "iPhone13,3")
    }

    /// A short series file is the only jetsam evidence there is. The file has to say so on its
    /// face, because the person reading it will not have the doc comment.
    @Test("the record carries the termination note and the measurement disclaimer")
    func seriesCarriesItsCaveats() throws {
        let request = WholeImageSeriesRequest(workload: .b, runs: 10)
        let json = try Self.encodedSeries(request: request, runs: [])

        let note = json["terminationNote"] as? String
        #expect(note?.contains("cannot be self-reported") == true)
        #expect(note?.contains("runsCompleted < runsRequested") == true)
        #expect((json["measurementDisclaimer"] as? String)?.contains("never quote") == true)
        #expect(json["runsCompleted"] as? Int == 0)
        #expect(json["stoppedEarly"] as? Bool == true)
    }

    @Test("a run row carries both memory metrics, each with its own provenance")
    func runRowJSON() throws {
        let request = WholeImageSeriesRequest(workload: .b, runs: 1)
        let json = try Self.encodedSeries(request: request, runs: [Self.record(index: 1, request: request)])
        guard let rows = json["runs"] as? [[String: Any]], let row = rows.first,
            let memory = row["memory"] as? [String: Any]
        else {
            Issue.record("expected one run row with a memory block")
            return
        }

        #expect(row["index"] as? Int == 1)
        #expect(row["success"] as? Bool == true)
        #expect(row["thermalStateAtStart"] as? String == "nominal")
        #expect(row["thermalStateAtEnd"] as? String == "fair")
        #expect(row["pressureLevelAtEnd"] as? String == "normal")

        // 300 × 1024² = 314,572,800 B; 400 × 1024² = 419,430,400 B. Hand-computed.
        let footprint = memory["processPeakFootprint"] as? [String: Any]
        let resident = memory["processPeakResidentSize"] as? [String: Any]
        #expect(footprint?["bytes"] as? UInt64 == 314_572_800)
        #expect(footprint?["mebibytes"] as? Double == 300.0)
        #expect(footprint?["provenance"] as? String == "measured")
        #expect(resident?["bytes"] as? UInt64 == 419_430_400)
        #expect(resident?["mebibytes"] as? Double == 400.0)
        #expect(
            resident != nil,
            "RSS is reported alongside phys_footprint, never instead of it")
    }

    @Test("an unavailable reading omits its bytes and says so, never encoding a zero")
    func unavailableReadingJSON() throws {
        let request = WholeImageSeriesRequest(workload: .b, runs: 1)
        let json = try Self.encodedSeries(request: request, runs: [Self.record(index: 1, request: request)])
        guard let rows = json["runs"] as? [[String: Any]],
            let memory = rows.first?["memory"] as? [String: Any],
            let decode = memory["footprintAfterDecode"] as? [String: Any]
        else {
            Issue.record("expected a footprintAfterDecode block")
            return
        }

        // The key is absent rather than zero: `provenance` is what a consumer branches on, and
        // a `bytes` key holding 0 would be a failed kernel call wearing a measurement's clothes.
        #expect(decode["bytes"] == nil)
        #expect(decode["mebibytes"] == nil)
        #expect(decode["provenance"] as? String == "unavailable")
        #expect(decode["source"] as? String == "phys_footprint, source image decoded")
    }

    @Test("the spool size the engine promised travels beside the size the file actually had")
    func geometryJSON() throws {
        let request = WholeImageSeriesRequest(workload: .b, runs: 1)
        let json = try Self.encodedSeries(request: request, runs: [Self.record(index: 1, request: request)])
        guard let rows = json["runs"] as? [[String: Any]],
            let geometry = rows.first?["geometry"] as? [String: Any]
        else {
            Issue.record("expected a geometry block")
            return
        }

        #expect(geometry["inputWidth"] as? Int == 4032)
        #expect(geometry["outputWidth"] as? Int == 16128)
        #expect(geometry["spoolBytes"] as? Int == 780_337_152)
        #expect(geometry["spoolBytesExpected"] as? Int == 780_337_152)
    }

    @Test("keys are sorted, so two records differ only where the measurements differ")
    func jsonIsDeterministic() throws {
        let request = WholeImageSeriesRequest(workload: .a, runs: 1)
        let series = WholeImageSeries(
            request: request, device: .current(), startedAt: Self.stampedDate, runs: [])
        let first = try WholeImageRunner.json(for: series)
        let second = try WholeImageRunner.json(for: series)

        #expect(first == second)
        guard let text = String(data: first, encoding: .utf8) else {
            Issue.record("expected UTF-8 JSON")
            return
        }
        #expect(text.contains("\"allSucceeded\""))
        #expect(text.contains("\\/") == false, "slashes are not escaped, so paths stay readable")
    }

    @Test("the record's file name says which workload, how many runs and which launch state")
    func seriesFileName() {
        let request = WholeImageSeriesRequest(workload: .c, runs: 3, launchState: .cold)
        #expect(
            WholeImageRunner.fileName(for: request, startedAt: Self.stampedDate)
                == "wholeimage-C-3x-cold-20260829-142530.json")
    }

    // MARK: - Proactive budgeting, recorded rather than asserted

    /// Certification spec §6 requires proactive budgeting **demonstrated**, and the demonstration
    /// is a number: the prediction error (peak − estimate) is what makes "proactive budgeting
    /// demonstrated" a number rather than an assertion. That subtraction needs the estimate in the
    /// record, so this pins that it is there and that ρ travels with it.
    ///
    /// Model-free: `MemoryBudget.plan(for:)` is pure arithmetic over six integers.
    @Test("the record carries the budget the engine granted itself, ρ included")
    func budgetTelemetryIsRecorded() throws {
        let request = WholeImageSeriesRequest(workload: .b, runs: 1)
        // 2 GiB available; ρ = 0.5 ⇒ allowance 1 GiB. Both hand-computed powers of two.
        let budget = MemoryBudget(
            available: MemoryBudget.Figure(
                bytes: 2_147_483_648, provenance: .measured, source: "test"))
        let plan = budget.plan(
            for: ImageWorkload(
                inputWidth: 4032, inputHeight: 3024, scaleFactor: 4, contextMargin: 52,
                method: .centerValidRegion, destination: .bandSpool))
        let row = WholeImageRunner.result(
            request: request, index: 1, contextMargin: 76, success: true,
            wallClockSeconds: 1, upscaleSeconds: 1,
            geometry: nil, plan: plan, startingShape: .n256, run: nil, thermalAtStart: "nominal",
            thermalAtEnd: "nominal", pressureLevelAtEnd: "normal", memory: Self.memory(),
            errorDescription: nil)

        #expect(row.allowanceBytes == 1_073_741_824)
        #expect(row.safetyFraction == 0.5, "ρ is explicit in the record, not only in the code")
        #expect(row.startingShape == 256)
        #expect(row.plannedShape == plan.shape.inputTileSize)
        #expect(row.plannedFit == plan.fit.rawValue)
        // Not recomputed here — only pinned as present, so `peak − estimate` is computable.
        #expect((row.plannedTotalBytes ?? 0) > 0)
        #expect(row.plannedProvenance != nil)
        #expect(row.plannedSelection?.isEmpty == false)
    }

    @Test("a run that never reached the budget records no budget at all, rather than zeros")
    func absentBudgetIsAbsent() {
        let request = WholeImageSeriesRequest(workload: .b, runs: 1)
        let row = Self.record(index: 1, request: request, success: false, error: "no artifacts")

        #expect(row.allowanceBytes == nil)
        #expect(row.plannedTotalBytes == nil)
        #expect(row.plannedShape == nil)
        #expect(row.safetyFraction == nil)
    }

    // MARK: - Readings, and the words the record uses

    @Test("a reading converts to MiB and says it was measured")
    func readingArithmetic() {
        // 1.5 × 1024² = 1,572,864 B, hand-computed.
        let reading = MemoryReading.kernel(1_572_864, source: "test")
        #expect(reading.bytes == 1_572_864)
        #expect(reading.mebibytes == 1.5)
        #expect(reading.provenance == "measured")
    }

    @Test("a failed kernel call is unavailable, not zero")
    func readingHonesty() {
        let reading = MemoryReading.kernel(nil, source: "test")
        #expect(reading.bytes == nil)
        #expect(reading.mebibytes == nil)
        #expect(reading.provenance == "unavailable")
    }

    /// `MemoryBudget.Figure.unavailable` stores zero bytes so its arithmetic stays total. That zero
    /// must not be published as a reading — it is the exact confusion the provenance label exists
    /// to prevent.
    @Test("an unavailable budget figure carries across as nil bytes, never as its stored zero")
    func unavailableFigureDropsItsZero() {
        let reading = MemoryReading.figure(.unavailable(source: "macOS has no such API"))
        #expect(reading.bytes == nil)
        #expect(reading.provenance == "unavailable")
        #expect(reading.source == "macOS has no such API")

        let measured = MemoryReading.figure(
            MemoryBudget.Figure(bytes: 2_097_152, provenance: .measured, source: "device run"))
        #expect(measured.bytes == 2_097_152)
        #expect(measured.mebibytes == 2.0)
        #expect(measured.provenance == "measured")
    }

    @Test("thermal states are recorded as the words certification spec §7.3 uses")
    func thermalWords() {
        #expect(WholeImageRunner.describe(ProcessInfo.ThermalState.nominal) == "nominal")
        #expect(WholeImageRunner.describe(ProcessInfo.ThermalState.fair) == "fair")
        #expect(WholeImageRunner.describe(ProcessInfo.ThermalState.serious) == "serious")
        #expect(WholeImageRunner.describe(ProcessInfo.ThermalState.critical) == "critical")
    }

    @Test("pressure levels are recorded by name")
    func pressureWords() {
        #expect(WholeImageRunner.describe(MemoryPressureLevel.normal) == "normal")
        #expect(WholeImageRunner.describe(MemoryPressureLevel.warning) == "warning")
        #expect(WholeImageRunner.describe(MemoryPressureLevel.critical) == "critical")
    }

    /// The shortfall is the one number that says how far off "does not fit" was. A record that
    /// dropped the payload would lose it.
    @Test("the budget's selection is spelled out, payload included")
    func selectionWords() {
        #expect(
            WholeImageRunner.describe(MemoryBudget.Selection.largestThatFits(.n512))
                == "largestThatFits(512)")
        #expect(
            WholeImageRunner.describe(
                MemoryBudget.Selection.floorRungAvailabilityUnknown(.n256))
                == "floorRungAvailabilityUnknown(256)")
        #expect(
            WholeImageRunner.describe(
                MemoryBudget.Selection.noRungFits(floor: .n256, shortfallBytes: 12_345))
                == "noRungFits(floor: 256, shortfallBytes: 12345)")
    }

    @Test("the current footprint is either a real figure or nil — never fabricated")
    func currentFootprintIsBestEffort() {
        if let bytes = SingleTileRunner.currentFootprintBytes() {
            #expect(bytes > 0, "a reported figure must be a real measurement")
        }
    }

    /// The point sample and the high-water mark are different readings and the series depends on
    /// them being different. If they were the same call the per-run signal would vanish.
    @Test("the point sample never exceeds the process high-water mark")
    func pointSampleSitsUnderThePeak() {
        guard let current = SingleTileRunner.currentFootprintBytes(),
            let peak = SingleTileRunner.peakFootprintBytes()
        else { return }
        #expect(current <= peak)
    }

    @Test("the launch argument is the one the app branches on")
    func launchArgumentIsStable() {
        #expect(WholeImageSeriesRequest.launchArgument == "-wholeimage")
        // Margins are per model, read from the shared `ContextMargin` table; these pin the values
        // the runner records for the default shape and for x2plus.
        #expect(WholeImageRunner.contextMargin(for: .default256) == 76)
        #expect(
            WholeImageRunner.contextMargin(
                for: HarnessShape(identifier: "RealESRGAN_x2plus", tileSize: 256)) == 76)
        #expect(
            WholeImageRunner.contextMargin(
                for: HarnessShape(identifier: "unmeasured", tileSize: 256)) == nil)
        #expect(WholeImageRunner.contractDefaultMethod == .centerValidRegion)
    }

    @Test("no series is run, and nothing is printed, when the arguments do not ask for one")
    func ifRequestedIsSilentWithoutTheArgument() async {
        #expect(await WholeImageRunner.runSeriesIfRequested(arguments: ["app"]) == nil)
        #expect(
            await WholeImageRunner.runSeriesIfRequested(
                arguments: ["app", "-wholeimage", "-workload", "Q"]) == nil,
            "a malformed request runs nothing rather than running the default")
    }
}

/// Shape labels when more than one model is bundled for an on-device comparison.
struct HarnessShapeLabelTests {
    @Test("the picker label names the model, so two models at one tile size are distinguishable")
    func pickerLabelDisambiguates() {
        let plus = HarnessShape(identifier: "RealESRGAN_x4plus", tileSize: 256)
        let v3 = HarnessShape(identifier: "realesr-general-x4v3", tileSize: 256)

        #expect(plus.displayName == v3.displayName, "the premise: displayName alone is ambiguous")
        #expect(plus.pickerLabel == "256 x4plus")
        #expect(v3.pickerLabel == "256 x4v3")
        #expect(plus.pickerLabel != v3.pickerLabel)
    }
}

/// Discovery of the tile inputs the picker offers.
struct TileChoiceDiscoveryTests {
    static func makeArtifacts(_ names: [String]) throws -> URL {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("tiles-\(UUID().uuidString)/HarnessArtifacts", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        for name in names {
            try Data("x".utf8).write(to: root.appendingPathComponent(name))
        }
        return root
    }

    /// Discovery, not configuration — the same rule the shape picker follows.
    @Test("a labelled tile is offered under its label, and the canonical one is not duplicated")
    func labelledTilesAreDiscovered() throws {
        let root = try Self.makeArtifacts([
            "input_tile_256x256.png",
            "input_tile_portrait_256x256.png",
            "input_tile_sky_256x256.png",
            "input_tile_room_512x512.png",
        ])
        defer { try? FileManager.default.removeItem(at: root.deletingLastPathComponent()) }

        let labels = DeviceHarness.tileChoices(
            inArtifactsDirectory: root, for: HarnessShape(identifier: "M", tileSize: 256)
        ).map(\.label)

        #expect(labels == ["standard", "portrait", "sky"])
        #expect(!labels.contains(""), "the canonical name must not reappear as an empty label")
    }

    @Test("tiles for another shape are not offered")
    func otherShapesAreExcluded() throws {
        let root = try Self.makeArtifacts([
            "input_tile_256x256.png", "input_tile_room_512x512.png",
        ])
        defer { try? FileManager.default.removeItem(at: root.deletingLastPathComponent()) }

        let labels = DeviceHarness.tileChoices(
            inArtifactsDirectory: root, for: HarnessShape(identifier: "M", tileSize: 256)
        ).map(\.label)

        #expect(labels == ["standard"])
    }
}

/// 6.38 s (cold, 512) against 102 ms (warm, 256) reads like a model difference; it was mostly
/// compile-versus-cache. The total alone could not tell them apart.
struct LoadInferenceSplitTests {
    static func result(load: TimeInterval, inference: TimeInterval, warm: Bool) -> SingleTileResult {
        SingleTileResult(
            success: true, outputSize: CGSize(width: 1024, height: 1024),
            wallClockSeconds: load + inference, loadSeconds: load, inferenceSeconds: inference,
            warmStart: warm, peakMemoryBytes: nil, peakFootprintBytes: nil,
            computeUnits: "all", errorDescription: nil)
    }

    @Test("a cold run says so, and warns that it is not comparable")
    func coldRunIsLabelled() {
        let rows = DeviceHarness.displayRows(for: Self.result(load: 6.2, inference: 0.18, warm: false))
        let text = rows.map { "\($0.label) \($0.value)" }.joined(separator: "\n")

        #expect(text.contains("COLD — compiled now"))
        #expect(text.contains("run again for a warm number"))
    }

    @Test("a warm run says so, and does not warn")
    func warmRunIsLabelled() {
        let rows = DeviceHarness.displayRows(for: Self.result(load: 0.02, inference: 0.08, warm: true))
        let text = rows.map { "\($0.label) \($0.value)" }.joined(separator: "\n")

        #expect(text.contains("WARM — cache hit"))
        #expect(!text.contains("run again for a warm number"))
    }

    /// The split must be reported, not just the total — the total is what misled.
    @Test("load and inference are both shown separately")
    func bothPhasesAreShown() {
        let rows = DeviceHarness.displayRows(for: Self.result(load: 6.2, inference: 0.18, warm: false))
        let labels = rows.map(\.label)

        #expect(labels.contains { $0.contains("model load") })
        #expect(labels.contains { $0.contains("inference") })
    }
}

/// Scale inference when RealESRGAN_x2plus is bundled beside the ×4 models.
struct HarnessShapeScaleTests {
    @Test("x2plus is ×2 and everything else is ×4")
    func scaleIsDerivedFromTheIdentifier() {
        #expect(HarnessShape(identifier: "RealESRGAN_x2plus", tileSize: 256).scaleFactor == 2)
        #expect(HarnessShape(identifier: "RealESRGAN_x4plus", tileSize: 256).scaleFactor == 4)
        #expect(HarnessShape(identifier: "realesr-general-x4v3", tileSize: 256).scaleFactor == 4)
    }

    /// The descriptor must carry the shape's scale, not a constant — a ×2 model described as ×4
    /// mis-sizes every output surface and memory plan, and nothing throws.
    @Test("the descriptor carries the shape's own scale factor")
    func descriptorCarriesTheScale() {
        let url = URL(fileURLWithPath: "/tmp/m.mlpackage")
        let two = SingleTileRunner.descriptor(
            shape: HarnessShape(identifier: "RealESRGAN_x2plus", tileSize: 256), modelURL: url)
        let four = SingleTileRunner.descriptor(
            shape: HarnessShape(identifier: "RealESRGAN_x4plus", tileSize: 256), modelURL: url)

        #expect(two.scaleFactor == 2)
        #expect(four.scaleFactor == 4)
    }
}
