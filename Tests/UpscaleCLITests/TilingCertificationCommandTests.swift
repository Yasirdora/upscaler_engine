import Foundation
import Testing
import UpscaleCore

@testable import UpscaleCLI

/// Tests for the additive `--images` filter on `certify-tiling`.
///
/// Model-free by design: the filter is a pure function over parsed arguments and recorded oracle
/// pairs. The certification run itself needs the corpus and a converted model.
struct TilingCertificationCommandTests {
    private func pair(_ name: String) -> OraclePair {
        OraclePair(
            name: name,
            image: URL(fileURLWithPath: "corpus/\(name).jpg"),
            reference: URL(fileURLWithPath: "references/\(name).npy"),
            variant: "full-frame",
            padding: nil,
            fullFrameReference: nil)
    }

    @Test("no --images argument means no filter — the default run is unchanged")
    func noFilterWhenFlagAbsent() throws {
        #expect(try TilingCertificationCommand.parseImageFilter(nil) == nil)
    }

    @Test("a comma-separated list parses to a stem set, whitespace tolerated")
    func parsesStemList() throws {
        let filter = try TilingCertificationCommand.parseImageFilter(" desert_figure , 30_anime_wikipetan ")
        #expect(filter == ["desert_figure", "30_anime_wikipetan"])
    }

    @Test("an empty segment is rejected — a hole in the filter would silently certify less")
    func rejectsEmptySegments() {
        #expect(throws: ImageCommand.UsageError.invalidImages("a,,b")) {
            try TilingCertificationCommand.parseImageFilter("a,,b")
        }
        #expect(throws: ImageCommand.UsageError.invalidImages("")) {
            try TilingCertificationCommand.parseImageFilter("")
        }
        #expect(throws: ImageCommand.UsageError.invalidImages("a,")) {
            try TilingCertificationCommand.parseImageFilter("a,")
        }
    }

    @Test("without a filter every recorded pair is kept and nothing is unmatched")
    func filterPairsWithoutFilter() {
        let pairs = [pair("a"), pair("b"), pair("c")]
        let (kept, unmatched) = TilingCertificationCommand.filterPairs(pairs, images: nil)
        #expect(kept.map(\.name) == ["a", "b", "c"])
        #expect(unmatched.isEmpty)
    }

    @Test("the filter keeps exactly the named pairs and names what matched nothing")
    func filterPairsSelectsAndReports() {
        let pairs = [pair("a"), pair("b"), pair("c")]
        let (kept, unmatched) = TilingCertificationCommand.filterPairs(
            pairs, images: ["c", "a", "not_in_corpus"])
        #expect(kept.map(\.name) == ["a", "c"], "pair order is the recorded order, not the filter's")
        #expect(
            unmatched == ["not_in_corpus"],
            "a stem with no oracle pair is reported, never dropped silently")
    }

    // The tiling report must be self-contained evidence — the corpus label and the Level A regime
    // framing are emitted by the generator, not left to adjacent documents.

    private func report(corpusLabel: String) -> String {
        TilingCertificationCommand.buildReport(
            modelID: "m", modelURL: URL(fileURLWithPath: "/tmp/m.mlpackage"),
            tileSize: 256, margin: 52, scaleFactor: 4, rows: [], notes: [],
            corpusLabel: corpusLabel)
    }

    @Test("the corpus version label is parsed from the corpus README")
    func corpusVersionLabelParsed() throws {
        let readme = FileManager.default.temporaryDirectory
            .appendingPathComponent("corpus-readme-\(UUID().uuidString).md")
        try "**Corpus version `corpus-2026-08-22.v1` — 45 images**, staged during curation"
            .write(to: readme, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: readme) }
        #expect(TilingCertificationCommand.corpusVersionLabel(readmeURL: readme) == "corpus-2026-08-22.v1")
    }

    @Test("the corpus version label degrades to 'unrecorded', never fails")
    func corpusVersionLabelFallback() throws {
        let missing = URL(fileURLWithPath: "/tmp/no-such-readme-\(UUID().uuidString).md")
        #expect(TilingCertificationCommand.corpusVersionLabel(readmeURL: missing) == "unrecorded")
        let noVersion = FileManager.default.temporaryDirectory
            .appendingPathComponent("corpus-readme-\(UUID().uuidString).md")
        try "# Certification corpus\nNo version line here.".write(
            to: noVersion, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: noVersion) }
        #expect(TilingCertificationCommand.corpusVersionLabel(readmeURL: noVersion) == "unrecorded")
    }

    @Test("the report header carries the corpus label and the Level A regime framing")
    func reportHeaderCarriesFraming() {
        let text = report(corpusLabel: "corpus-2026-08-22.v1")
        #expect(
            text.contains("corpus         : corpus-2026-08-22.v1 (Sources/TestFixtures/corpus/README.md)"))
        #expect(text.contains("Regime framing (certification spec §4.1)"))
        #expect(text.contains("trend-monitored baselines, not per-image pass/fail gates"))
    }

    @Test("the no-eligible-method conclusion names the regime reading, not a blanket red")
    func verdictNamesRegimes() {
        let text = report(corpusLabel: "corpus-2026-08-22.v1")
        #expect(
            text.contains(
                "NO METHOD MET EVERY CRITERION under the blanket Level A reading — recorded, not gated:"))
        #expect(!text.contains("this is a red result and is reported as one"))
    }
}
