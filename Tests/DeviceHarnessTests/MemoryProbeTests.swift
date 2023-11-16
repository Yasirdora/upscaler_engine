import Foundation
import Testing

/// Model-free tests for the memory probe's pure logic.
///
/// The probe's subject under test lives in the app target
/// (`DeviceHarnessApp/DeviceHarnessApp/MemoryProbeView.swift`) and is symlinked into this
/// directory, because SwiftPM cannot reach an Xcode app target and the probe may not live in
/// `Sources/`. The allocation loop itself is device-side and untested by design; everything a
/// screenshot or a stored reading depends on — units, arithmetic, encoding — is pinned here.
struct MemoryProbeTests {
    static func reading(
        chunks: Int = 10,
        available: UInt64? = 1_000_000_000,
        footprint: UInt64? = 600_000_000,
        elapsed: TimeInterval = 2.5
    ) -> MemoryProbe.Reading {
        MemoryProbe.Reading(
            chunks: chunks, availableBytes: available, footprintBytes: footprint,
            elapsedSeconds: elapsed)
    }

    // MARK: - The launch gate

    @Test("the probe is reached only by its own argument, never by default")
    func launchGate() {
        #expect(MemoryProbe.isRequested(arguments: ["DeviceHarnessApp"]) == false)
        #expect(MemoryProbe.isRequested(arguments: ["DeviceHarnessApp", "-memoryprobe"]))
        #expect(MemoryProbe.isRequested(arguments: ["-autorun", "-shape", "512"]) == false)
        #expect(MemoryProbe.isRequested(arguments: ["-memoryprobe-please"]) == false)
        #expect(MemoryProbe.isRequested(arguments: []) == false)
    }

    @Test("the loop constants are the ones the protocol quotes")
    func loopConstants() {
        #expect(MemoryProbe.chunkBytes == 52_428_800)  // 50 MiB
        #expect(MemoryProbe.chunkIntervalMilliseconds == 250)  // ≈4 chunks/second
    }

    // MARK: - Arithmetic

    @Test("allocated bytes are chunks × 50 MiB", arguments: [0, 1, 40, 200])
    func allocatedBytes(chunks: Int) {
        let reading = Self.reading(chunks: chunks)

        #expect(reading.allocatedBytes == UInt64(chunks) * 52_428_800)
    }

    @Test("allocated plus available adds up, and is absent when the query failed")
    func allocatedPlusAvailable() {
        #expect(Self.reading(chunks: 2, available: 100).allocatedPlusAvailableBytes == 104_857_700)
        #expect(Self.reading(available: nil).allocatedPlusAvailableBytes == nil)
    }

    // MARK: - Units

    @Test("byte figures read in MiB and say MiB")
    func mebibyteFormatting() {
        #expect(MemoryProbe.format(bytes: 1024 * 1024) == "1.0 MiB")
        #expect(MemoryProbe.format(bytes: 52_428_800) == "50.0 MiB")
        #expect(MemoryProbe.format(bytes: 2_200_000_000) == "2098.1 MiB")
        #expect(MemoryProbe.format(bytes: 0) == "0.0 MiB")
    }

    @Test("a failed kernel query reads as unavailable, never as zero")
    func unavailableIsNotZero() {
        #expect(MemoryProbe.format(bytes: nil) == "unavailable")
    }

    @Test("elapsed time stays readable past a minute")
    func secondsFormatting() {
        #expect(MemoryProbe.format(seconds: 0) == "0.0 s")
        #expect(MemoryProbe.format(seconds: 12.34) == "12.3 s")
        #expect(MemoryProbe.format(seconds: 59.94) == "59.9 s")
        #expect(MemoryProbe.format(seconds: 61) == "1m 01.0 s")
        #expect(MemoryProbe.format(seconds: 125.4) == "2m 05.4 s")
    }

    // MARK: - What the screen says

    @Test("the rows are the five the protocol asks the user to report, in order")
    func rowOrder() {
        let labels = MemoryProbe.rows(for: Self.reading()).map(\.label)

        #expect(
            labels == [
                "Allocated (dirtied)", "os_proc_available_memory", "phys_footprint",
                "Allocated + available", "Elapsed",
            ])
    }

    @Test("the allocated row shows the chunk count behind the total")
    func allocatedRowShowsChunks() {
        let rows = MemoryProbe.rows(for: Self.reading(chunks: 7))

        #expect(rows.first?.value == "350.0 MiB  (7 × 50 MiB)")
    }

    @Test("no row inherits the harness's MB mislabel")
    func noRowSaysMB() {
        for row in MemoryProbe.rows(for: Self.reading()) {
            #expect(row.value.contains(" MB") == false, "\(row.label) must be labelled MiB")
        }
    }

    @Test("a zero from the availability query keeps its number and gains a caveat")
    func zeroAvailabilityIsAReading() {
        let rows = MemoryProbe.rows(for: Self.reading(available: 0))
        let values = Dictionary(uniqueKeysWithValues: rows.map { ($0.label, $0.value) })

        // Apple documents 0 as "already exceeded its memory limit" or "not an app" — hiding it
        // behind "unavailable" would discard the reading the whole run exists to capture.
        #expect(values["os_proc_available_memory"] == "0.0 MiB  (no headroom reported)")
        #expect(MemoryProbe.availabilityNote(for: 0)?.contains("returned 0") == true)
        #expect(MemoryProbe.availabilityNote(for: 1_000) == nil)
        #expect(MemoryProbe.availabilityNote(for: nil) == nil)
    }

    @Test("failed queries degrade to unavailable rather than dropping the row")
    func rowsSurviveFailedQueries() {
        let rows = MemoryProbe.rows(for: Self.reading(available: nil, footprint: nil))
        let values = Dictionary(uniqueKeysWithValues: rows.map { ($0.label, $0.value) })

        #expect(rows.count == 5)
        #expect(values["os_proc_available_memory"] == "unavailable")
        #expect(MemoryProbe.availability(of: 52_428_800) == "50.0 MiB")
        #expect(values["phys_footprint"] == "unavailable")
        #expect(values["Allocated + available"] == "unavailable")
    }

    @Test("the safety line says what will happen and that it is intended")
    func safetyNoticeIsExplicit() {
        let notice = MemoryProbe.safetyNotice

        #expect(notice.contains("kill"))
        #expect(notice.contains("app, not the device"))
        #expect(notice.contains("intended"))
    }

    // MARK: - Surviving the kill

    @Test("a reading round-trips through storage, so a missed screenshot costs nothing")
    func encodingRoundTrips() throws {
        let original = Self.reading(chunks: 33, available: 12_345_678, footprint: nil, elapsed: 8.25)
        let encoded = try #require(MemoryProbe.encode(original))

        #expect(MemoryProbe.decode(encoded) == original)
    }

    @Test(
        "malformed stored text decodes to nil rather than crashing a relaunch",
        arguments: [
            "", "not json", "{}", "{\"chunks\":3}", "[1,2,3]", "null",
        ])
    func malformedStorageIsIgnored(text: String) {
        #expect(MemoryProbe.decode(text) == nil)
    }

    @Test("the storage key is namespaced so it cannot collide with harness state")
    func storageKeyIsNamespaced() {
        #expect(MemoryProbe.storageKey == "upscaler.memoryProbe.lastReading")
    }
}
