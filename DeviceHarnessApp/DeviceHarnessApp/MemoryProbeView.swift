import Foundation

#if canImport(UIKit)
    import SwiftUI
    import os
#endif

/// Allocate-until-killed memory probe — a measured jetsam ceiling for the floor device (iPhone 12
/// Pro), which nothing else provides.
///
/// Apple publishes no per-device app memory limit, and `os_proc_available_memory` is the only
/// sanctioned way to ask. This probe asks the question from the other side: it dirties memory in
/// fixed chunks until the system terminates the app, so the ceiling becomes a measured number
/// instead of a forum post about a different phone.
///
/// Reached only via the `-memoryprobe` launch argument. Without it the app is the single-tile
/// harness, untouched — this file adds no code to that path.
///
/// **Layout note:** everything outside the `canImport(UIKit)` guards is deliberately platform-free.
/// This file is symlinked into `Tests/DeviceHarnessTests/` so the formatting, arithmetic and
/// encoding below are unit-tested on the host, where UIKit does not exist. Keep iOS-only API inside
/// the guards.
enum MemoryProbe {
    /// Swaps the root view for the probe. Absent → the harness, unchanged.
    static let launchArgument = "-memoryprobe"

    /// 50 MiB per chunk. Coarse enough to reach a 6 GB-class ceiling in a couple of minutes, fine
    /// enough that the last reading before the kill brackets that ceiling to within one chunk.
    static let chunkBytes = 50 * 1024 * 1024

    /// ≈4 chunks per second: legible on screen, and far from a spin loop — CPU churn is the enemy,
    /// in a diagnostic as much as in the engine.
    static let chunkIntervalMilliseconds = 250

    /// Where the last reading is parked so a missed screenshot does not cost a run. The kill is
    /// instantaneous and unannounced; relaunching shows what the screen last said.
    static let storageKey = "upscaler.memoryProbe.lastReading"

    /// Shown on screen, not only in the protocol: a person watching their phone kill an app is
    /// owed an explanation before it happens.
    static let safetyNotice =
        "Expect the system to kill this app. Jetsam terminates the app, not the device — that "
        + "termination is the measurement working as intended. Relaunch to read the last figures."

    static func isRequested(arguments: [String] = ProcessInfo.processInfo.arguments) -> Bool {
        arguments.contains(launchArgument)
    }

    /// One on-screen line of the run, and the unit the user reports back.
    ///
    /// `availableBytes` and `footprintBytes` are optional because both kernel queries can fail and
    /// neither is available on every platform. A failed query reads "unavailable" — never zero,
    /// which would look like a measurement.
    struct Reading: Codable, Equatable, Sendable {
        var chunks: Int
        var availableBytes: UInt64?
        var footprintBytes: UInt64?
        var elapsedSeconds: TimeInterval

        var allocatedBytes: UInt64 { UInt64(chunks) * UInt64(chunkBytes) }

        /// Allocated plus what the system still offers. Roughly constant across a run — it is the
        /// limit minus whatever the app already occupied at baseline — so a drifting value is
        /// itself information (see the protocol).
        var allocatedPlusAvailableBytes: UInt64? {
            guard let availableBytes else { return nil }
            return allocatedBytes + availableBytes
        }
    }

    /// Rows the probe renders verbatim, in the order a reader needs them.
    static func rows(for reading: Reading) -> [(label: String, value: String)] {
        [
            (
                "Allocated (dirtied)",
                "\(format(bytes: reading.allocatedBytes))  (\(reading.chunks) × 50 MiB)"
            ),
            ("os_proc_available_memory", availability(of: reading.availableBytes)),
            ("phys_footprint", format(bytes: reading.footprintBytes)),
            ("Allocated + available", format(bytes: reading.allocatedPlusAvailableBytes)),
            ("Elapsed", format(seconds: reading.elapsedSeconds)),
        ]
    }

    /// How the availability query reads on screen.
    ///
    /// Zero is a documented answer, not a failed one: Apple defines it as "the calling process
    /// isn't an app, or ... has already exceeded its memory limit", and the iOS Simulator reports
    /// it too (observed while developing this probe). Rendering it as "unavailable" would hide the
    /// one reading the run exists to capture, so zero keeps its number and gets a caveat.
    static func availability(of bytes: UInt64?) -> String {
        guard let bytes else { return "unavailable" }
        return bytes == 0 ? "0.0 MiB  (no headroom reported)" : format(bytes: bytes)
    }

    /// The sentence that belongs under a zero reading, or `nil` when the number speaks for itself.
    static func availabilityNote(for bytes: UInt64?) -> String? {
        guard bytes == 0 else { return nil }
        return
            "os_proc_available_memory returned 0. Apple documents that as \"already exceeded its "
            + "memory limit\" or \"not an app\" — and the Simulator returns it unconditionally. On "
            + "the phone, treat a 0 here as a real reading; in the Simulator, ignore this row."
    }

    /// MiB, and labelled MiB. The harness's own "MB" is bytes ÷ 1024² under a wrong name
    /// (`Sources/DeviceHarness/DeviceHarness.swift`); this probe does not inherit it.
    static func format(bytes: UInt64?) -> String {
        guard let bytes else { return "unavailable" }
        return String(format: "%.1f MiB", Double(bytes) / (1024 * 1024))
    }

    static func format(seconds: TimeInterval) -> String {
        if seconds < 60 { return String(format: "%.1f s", seconds) }
        let minutes = Int(seconds) / 60
        return String(format: "%dm %04.1f s", minutes, seconds - Double(minutes * 60))
    }

    /// JSON, so a reading written by one build is readable by the next and a malformed one is
    /// simply absent rather than a crash on launch.
    static func encode(_ reading: Reading) -> String? {
        guard let data = try? JSONEncoder().encode(reading) else { return nil }
        return String(data: data, encoding: .utf8)
    }

    static func decode(_ text: String) -> Reading? {
        guard let data = text.data(using: .utf8) else { return nil }
        return try? JSONDecoder().decode(Reading.self, from: data)
    }
}

#if canImport(UIKit)

    extension MemoryProbe {
        /// Bytes the app may still allocate before it hits its limit, or `nil` when the query is
        /// unavailable. Apple documents this as the cheap query and `task_info` as the expensive
        /// one, which is why it leads.
        static func availableMemory() -> UInt64 {
            UInt64(os_proc_available_memory())
        }

        /// `phys_footprint` from `task_vm_info` — the metric the system actually enforces (WWDC18
        /// session 416). Read once per chunk, four times a second: `task_info` is expensive
        /// relative to the call above, not expensive in absolute terms.
        static func physicalFootprint() -> UInt64? {
            var info = task_vm_info_data_t()
            var count = mach_msg_type_number_t(
                MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
            let result = withUnsafeMutablePointer(to: &info) { pointer in
                pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
                    task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), rebound, &count)
                }
            }
            return result == KERN_SUCCESS ? UInt64(info.phys_footprint) : nil
        }
    }

    /// Holds the allocations outside SwiftUI's state graph.
    ///
    /// The pointers are never freed: the run ends when the system ends it, and a probe that tidied
    /// up would be measuring its own bookkeeping. Keeping them off `@State` also means the view
    /// redraws from the reading alone, four times a second, rather than from a growing array.
    @MainActor
    final class ChunkBag {
        private(set) var pointers: [UnsafeMutableRawPointer] = []

        var count: Int { pointers.count }

        /// Allocates one chunk and writes every byte of it.
        ///
        /// Writing the whole chunk rather than one byte per page is deliberate: it costs a few
        /// milliseconds per chunk and it assumes nothing about the page size. Untouched pages are
        /// not dirty, and dirty is what the enforced footprint metric counts.
        func allocateAndDirty(bytes: Int) -> Bool {
            guard let pointer = malloc(bytes) else { return false }
            memset(pointer, 0xA5, bytes)
            pointers.append(pointer)
            return true
        }
    }

    /// The probe screen. One button, four numbers, one warning.
    struct MemoryProbeView: View {
        @State private var bag = ChunkBag()
        @State private var reading: MemoryProbe.Reading?
        @State private var previous: MemoryProbe.Reading?
        @State private var isRunning = false
        @State private var stoppedNote: String?

        /// The live run when there is one, otherwise whatever the previous run left behind.
        private var displayed: MemoryProbe.Reading? { reading ?? previous }

        /// Shared with the harness: `-autorun` means "start without waiting for a tap".
        private var autorunRequested: Bool {
            ProcessInfo.processInfo.arguments.contains("-autorun")
        }

        var body: some View {
            NavigationStack {
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        Text(MemoryProbe.safetyNotice)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)

                        Button(action: start) {
                            HStack {
                                if isRunning { ProgressView().padding(.trailing, 4) }
                                Text(isRunning ? "Allocating…" : "Start probe")
                            }
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 10)
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(isRunning)

                        if let displayed {
                            panel(
                                reading == nil
                                    ? "Last reading before the previous run ended" : "Current run",
                                reading: displayed)
                        } else {
                            Text("Tap Start. The app allocates 50 MiB every 250 ms and dirties it.")
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                        }

                        // The caveat follows whichever reading is on screen — a restored one is
                        // read just as literally as a live one.
                        if let note = MemoryProbe.availabilityNote(for: displayed?.availableBytes) {
                            Text(note)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                        }

                        if let stoppedNote {
                            Text(stoppedNote)
                                .font(.footnote)
                                .foregroundStyle(.primary)
                                .fixedSize(horizontal: false, vertical: true)
                        }

                        Text(
                            "Figures are informational. The number that settles the ceiling is in the "
                                + "jetsam event report afterwards — see the protocol."
                        )
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    }
                    .padding()
                }
                .navigationTitle("Memory ceiling probe")
                .navigationBarTitleDisplayMode(.inline)
            }
            .task {
                previous = UserDefaults.standard.string(forKey: MemoryProbe.storageKey)
                    .flatMap(MemoryProbe.decode)
                if autorunRequested, !isRunning, reading == nil { start() }
            }
        }

        @ViewBuilder
        private func panel(_ title: String, reading: MemoryProbe.Reading) -> some View {
            VStack(alignment: .leading, spacing: 8) {
                Text(title)
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                ForEach(MemoryProbe.rows(for: reading), id: \.label) { row in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(row.label)
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                        Text(row.value)
                            .font(.title3.monospaced())
                            .textSelection(.enabled)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .accessibilityIdentifier("probeRows")
            }
            .padding(12)
            .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 10))
        }

        private func start() {
            guard !isRunning else { return }
            isRunning = true
            stoppedNote = nil
            previous = nil
            reading = nil
            UserDefaults.standard.removeObject(forKey: MemoryProbe.storageKey)

            let clock = ContinuousClock()
            let started = clock.now
            Task { @MainActor in
                while isRunning {
                    guard bag.allocateAndDirty(bytes: MemoryProbe.chunkBytes) else {
                        stoppedNote =
                            "malloc returned nil after \(bag.count) chunks — the run stopped without a "
                            + "system kill, so this is not a jetsam ceiling. Report it as such."
                        isRunning = false
                        break
                    }
                    let elapsed = clock.now - started
                    let current = MemoryProbe.Reading(
                        chunks: bag.count,
                        availableBytes: MemoryProbe.availableMemory(),
                        footprintBytes: MemoryProbe.physicalFootprint(),
                        elapsedSeconds: TimeInterval(elapsed.components.seconds)
                            + Double(elapsed.components.attoseconds) / 1e18
                    )
                    reading = current
                    if let encoded = MemoryProbe.encode(current) {
                        UserDefaults.standard.set(encoded, forKey: MemoryProbe.storageKey)
                    }
                    try? await Task.sleep(for: .milliseconds(MemoryProbe.chunkIntervalMilliseconds))
                }
            }
        }
    }

    #Preview {
        MemoryProbeView()
    }

#endif
