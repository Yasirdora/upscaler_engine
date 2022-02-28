import Foundation

#if os(iOS) || os(tvOS) || os(watchOS) || os(visionOS)
    import os
#endif

// MARK: - Shapes

/// The three fixed-shape rungs the engine supports: 256, 384 and 512.
///
/// This enumerates artifacts, not policy. *Choosing* a rung when the budget is short, stepping
/// between rungs, dwell and hysteresis belong to `TileShapeLadder` and are deliberately absent
/// here.
public enum TileShape: Int, Sendable, CaseIterable, Comparable {
    case n256 = 256
    case n384 = 384
    case n512 = 512

    /// Input tile side in pixels — the model's fixed shape (`ModelDescriptor.inputTileSize`).
    public var inputTileSize: Int { rawValue }

    /// The floor rung: "256 when unknown".
    public static let floorRung = TileShape.n256

    public static func < (lhs: TileShape, rhs: TileShape) -> Bool { lhs.rawValue < rhs.rawValue }
}

// MARK: - Destination

/// Which of the two shipped destinations a job will write into (`OutputSurface.swift`).
///
/// The budget models both because the engine offers both: `SequentialTilingEngine.upscale` still
/// allocates a whole-image `CVPixelBuffer`, and `upscale(_:model:backend:method:spoolingTo:)` band
/// spools. A budget that only knew the spool would understate a caller on the in-memory path by the
/// whole 744.2 MiB it was built to remove.
public enum DestinationKind: String, Sendable, CaseIterable {
    /// The default path (`BandSpoolSurface`): residency is a sliding row window.
    case bandSpool
    /// The control (`PixelBufferSurface`): one whole-image buffer, every row open.
    case wholeImage
}

// MARK: - Workload

/// One image's worth of work, described in the terms a budget needs — and nothing more.
///
/// Deliberately carries no tile size: the shape is the variable the budget solves for, so
/// `destinationBytes(at:)` takes it as an argument.
public struct ImageWorkload: Sendable, Equatable {
    public let inputWidth: Int
    public let inputHeight: Int
    public let scaleFactor: Int
    /// Input-space context margin the engine is running with
    /// (`SequentialTilingEngine.contextMargin`). It sets the band height, so it is a budget input.
    public let contextMargin: Int
    public let method: StitchMethod
    public let destination: DestinationKind

    public init(
        inputWidth: Int,
        inputHeight: Int,
        scaleFactor: Int,
        contextMargin: Int,
        method: StitchMethod,
        destination: DestinationKind = .bandSpool
    ) {
        self.inputWidth = inputWidth
        self.inputHeight = inputHeight
        self.scaleFactor = scaleFactor
        self.contextMargin = contextMargin
        self.method = method
        self.destination = destination
    }

    public var outputWidth: Int { inputWidth * scaleFactor }
    public var outputHeight: Int { inputHeight * scaleFactor }

    /// Bytes one output row occupies. Tight BGRA, matching `BandSpoolSurface`'s packed raster.
    public var outputRowBytes: Int { outputWidth * PixelView.bytesPerPixel }

    /// The **source** image, resident for the whole job.
    ///
    /// `SequentialTilingEngine.upscale` reads its input `CVPixelBuffer` tile by tile from first
    /// tile to last, so the whole decoded image is live from start to finish. At BGRA8 that is 46.5
    /// MiB for Workload B and 186.0 MiB for Workload C (certification spec §7.1). It is carried as
    /// its own labelled term rather than folded into another so a reader can see it and disagree
    /// with it: a budget that omitted 186 MiB of guaranteed residency at 48 MP would not be an
    /// honest one.
    ///
    /// Row padding is assumed zero — true for these widths, and an assumption worth re-checking for
    /// others.
    public var inputSurfaceBytes: Int { inputWidth * inputHeight * PixelView.bytesPerPixel }

    /// What the destination surface holds resident while this workload runs at `shape`.
    ///
    /// For `.bandSpool` this is the band spool's own cost model — `DestinationSurface.openRowCount`
    /// rows, clamped to the image the way `BandSpoolSurface.init` clamps it — **not** the 744.2 MiB
    /// whole-image figure it replaced. Residency therefore scales with output *width*, not area.
    public func destinationBytes(at shape: TileShape) -> Int {
        switch destination {
        case .wholeImage:
            return outputRowBytes * outputHeight
        case .bandSpool:
            let tileSize = shape.inputTileSize
            let grid = TileGrid(
                tileSize: tileSize,
                contextMargin: Self.plannedMargin(tileSize: tileSize, requested: contextMargin),
                tiles: [])
            let rows = BandSpoolSurface.openRowCount(
                for: grid, scaleFactor: scaleFactor, method: method)
            return outputRowBytes * min(rows, outputHeight)
        }
    }

    /// The margin the planner will actually use, given a requested one.
    ///
    /// Mirrors the clamp in `TilePlanner.plan` — a margin that leaves no core would produce an
    /// infinite grid, so the planner takes the largest that still tiles. Budgeting the *requested*
    /// margin instead would understate the band whenever the clamp bites, so the two must agree;
    /// `MemoryTests.plannedMarginAgreesWithTheShippedPlanner` pins that they do.
    static func plannedMargin(tileSize: Int, requested: Int) -> Int {
        min(max(0, requested), max(0, (tileSize - 1) / 2))
    }
}

// MARK: - The budget

/// Does this workload fit, at which shape, with how much headroom — answered **before** the first
/// tile of an image is started.
///
/// Certification spec §6 makes this mandatory: proactive budgeting ("available memory × model
/// footprint × tile footprint × concurrency") demonstrated, because "reactive-only warning handling
/// is insufficient".
///
/// **A value type with injected inputs.** `os_proc_available_memory()` is never called from in here
/// — it arrives as a `Figure` from `AvailableMemory.now()`, which is the only place in the engine
/// that calls it. That is what makes every number below assertable against hand-computed literals
/// on a host with no device, no model and no Core ML.
///
/// **The arithmetic, per rung:**
///
///     total = measuredPeak(shape) × concurrency   // measured   — floor device
///           + inputSurfaceBytes                   // estimated  — exact BGRA arithmetic
///           + destinationBytes(shape)             // estimated  — the destination's cost model
///     fits  = total ≤ ρ × os_proc_available_memory()
///
/// The input-surface term is carried as its own labelled term because the engine holds the whole
/// source image live for the whole job, and at 48 MP that is 186 MiB the other terms would have
/// silently swallowed.
///
/// **Cost.** Nothing here polls, sleeps or subscribes. `plan(for:)` is pure arithmetic over three
/// rungs and is meant to be called **once per image**, never per tile.
///
/// **What this deliberately does not do:** step up, step down, dwell, hysteresis, or react to
/// pressure. This type states a fact; something else acts on it — `TileShapeLadder` is that
/// something, and it consumes `Plan.assessments` rather than assuming the rungs descend in cost.
public struct MemoryBudget: Sendable, Equatable {

    // MARK: Provenance

    /// The label every budget figure carries: a number without provenance is a defect. Ordered
    /// weakest-last, so combining figures is `max`.
    public enum Provenance: String, Sendable, CaseIterable, Comparable {
        /// Read off a device run, with the run cited in the figure's `source`.
        case measured
        /// Derived arithmetic over measured or exactly-known quantities.
        case estimated
        /// No figure exists. The budget says so rather than inventing one.
        case unavailable

        private var rank: Int {
            switch self {
            case .measured: return 0
            case .estimated: return 1
            case .unavailable: return 2
            }
        }

        public static func < (lhs: Provenance, rhs: Provenance) -> Bool { lhs.rank < rhs.rank }

        /// The weakest link in a chain of figures — an estimate plus a measurement is an estimate.
        public static func weakest(_ values: [Provenance]) -> Provenance {
            values.max() ?? .unavailable
        }
    }

    /// A number of bytes that knows where it came from.
    public struct Figure: Sendable, Equatable, CustomStringConvertible {
        public let bytes: Int
        public let provenance: Provenance
        /// Citation: the device run, the document section, or the derivation.
        public let source: String

        public init(bytes: Int, provenance: Provenance, source: String) {
            self.bytes = bytes
            self.provenance = provenance
            self.source = source
        }

        /// A figure measured in mebibytes, the unit the harness prints.
        public static func mebibytes(_ value: Double, provenance: Provenance, source: String) -> Figure {
            Figure(
                bytes: Int((value * 1_048_576).rounded()), provenance: provenance, source: source)
        }

        /// A figure that does not exist. Zero bytes, so arithmetic stays total; the provenance is
        /// what callers must branch on, never the zero.
        public static func unavailable(source: String) -> Figure {
            Figure(bytes: 0, provenance: .unavailable, source: source)
        }

        public var mebibytes: Double { Double(bytes) / 1_048_576 }

        public var description: String {
            String(format: "%.1f MiB [%@] (%@)", mebibytes, provenance.rawValue, source)
        }
    }

    // MARK: The safety fraction

    /// ρ — the fraction of `os_proc_available_memory()` the engine is willing to occupy at peak.
    ///
    /// **ρ = 0.5**, permitted range [0.4, 0.6]. The lower bound absorbs memory warnings caused by
    /// other activity on the device — a phone call, not this job (WWDC18 session 416, "iOS Memory
    /// Deep Dive"). ρ also absorbs the difference between the RSS the per-shape table was measured
    /// in and the `phys_footprint` the system actually enforces.
    ///
    /// A type rather than a bare `Double` so the permitted range is enforced at construction and ρ
    /// can never appear inline as a magic number.
    public struct SafetyFraction: Sendable, Equatable {
        /// The permitted range. Anything outside it is refused, not clamped.
        public static let permittedRange: ClosedRange<Double> = 0.4...0.6

        /// ρ = 0.5, the standard value.
        public static let standard = SafetyFraction(unchecked: 0.5)

        public let value: Double

        /// The step-up margin the ladder derives from ρ — **1 − ρ**, 0.5 at the standard ρ.
        ///
        /// The derivation: ρ is the share of the device's available memory the engine will occupy,
        /// so (1 − ρ) is the share it deliberately leaves alone. Requiring the same proportional
        /// reserve *inside* the allowance before climbing means a step up must clear the safety
        /// margin a second time, at the candidate rung, on its own arithmetic.
        ///
        /// It lives on the budget type because ρ does; `TileShapeLadder.Parameters` consumes it and
        /// carries the provenance label. **The derivation is a design choice, not a measurement** —
        /// pending device runs, and labelled `.estimated` where it is used.
        public var stepUpMarginFraction: Double { 1 - value }

        /// Fails rather than clamps: a caller asking for 0.9 has misunderstood the range, and
        /// silently giving them 0.6 would hide that.
        public init?(_ value: Double) {
            guard Self.permittedRange.contains(value) else { return nil }
            self.value = value
        }

        private init(unchecked value: Double) { self.value = value }
    }

    // MARK: Stored inputs

    /// `os_proc_available_memory()` at job start, injected. The primary budget input: on the floor
    /// device it counted down to 6.05 MB one chunk before jetsam fired, so it is a real countdown
    /// on real hardware.
    public let available: Figure

    /// ρ. Defaults to the standard 0.5.
    public let safetyFraction: SafetyFraction

    public init(available: Figure, safetyFraction: SafetyFraction = .standard) {
        self.available = available
        self.safetyFraction = safetyFraction
    }

    /// The concurrency term of certification spec §6's formula, **pinned at 1**: the engine is
    /// sequential (`SequentialTilingEngine`).
    ///
    /// It is not free to unpin. The per-shape figures below are *whole-process* peaks, so a second
    /// worker would double the app, the Core ML runtime and the mapped weights along with the tile
    /// — and the measured data cannot currently separate the fixed and marginal parts (the implied
    /// shape-dependent term changes sign between rungs). Whoever raises this owes that separation
    /// first.
    public static let concurrency = 1

    // MARK: The measured per-shape table

    /// Median whole-process peak per shape, measured on the floor device.
    ///
    /// Measured on an iPhone 12 Pro (A14, 6 GB) with `cpuAndNeuralEngine`, with three caveats: the
    /// reading is RSS, not `phys_footprint`; it includes a PNG encode of the tile output; and it
    /// includes an in-window `MLModel.compileModel`. All three make it an over-estimate of the
    /// engine's steady-state per-tile cost, which is the safe direction for a budget and is why it
    /// is used unadjusted.
    ///
    /// The table is not monotonic in shape — 384 measures *below* 256 — because the N² scaling
    /// model does not hold on device: all three rungs sit in a 204–271 MiB band. That is recorded
    /// here rather than smoothed away.
    public static func measuredPeak(at shape: TileShape) -> Figure {
        switch shape {
        case .n256:
            return .mebibytes(
                227.7, provenance: .measured,
                source: "measured on iPhone 12 Pro (A14), cpuAndNeuralEngine, "
                    + "single run, RSS")
        case .n384:
            return .mebibytes(
                203.9, provenance: .measured,
                source: "measured on iPhone 12 Pro (A14), cpuAndNeuralEngine, "
                    + "median of 3 runs, RSS")
        case .n512:
            return .mebibytes(
                267.1, provenance: .measured,
                source: "measured on iPhone 12 Pro (A14), cpuAndNeuralEngine, "
                    + "median of 3 runs, RSS")
        }
    }

    // MARK: Allowance

    /// ρ × available, floored to whole bytes. `nil` when availability is unavailable — the budget
    /// refuses to invent an allowance, and `plan(for:)` falls back to the floor rung instead.
    public var allowance: Figure? {
        guard available.provenance != .unavailable else { return nil }
        let bytes = Int((Double(available.bytes) * safetyFraction.value).rounded(.down))
        return Figure(
            bytes: bytes, provenance: available.provenance,
            source: "ρ = \(safetyFraction.value) × \(available.source)")
    }

    // MARK: Assessment

    /// Whether one shape fits, and by how much.
    public enum Fit: String, Sendable, CaseIterable {
        case fits
        case doesNotFit
        /// Availability is unavailable, so the question has no answer — not a "no".
        case unknown
    }

    /// One rung, costed.
    public struct Assessment: Sendable, Equatable {
        public let shape: TileShape
        /// Measured per-shape whole-process peak × the pinned concurrency term.
        public let model: Figure
        /// The source image, resident for the whole job (`ImageWorkload.inputSurfaceBytes`).
        public let input: Figure
        /// Requirement 2's destination cost model at this shape.
        public let destination: Figure
        /// ρ × available, or `nil` when availability is unavailable.
        public let allowance: Figure?

        public var totalBytes: Int { model.bytes + input.bytes + destination.bytes }

        public var fit: Fit {
            guard let allowance else { return .unknown }
            return totalBytes <= allowance.bytes ? .fits : .doesNotFit
        }

        /// Allowance − total. Negative is a shortfall; `nil` means unknown, never zero.
        public var headroomBytes: Int? {
            guard let allowance else { return nil }
            return allowance.bytes - totalBytes
        }

        /// Weakest link across every term.
        public var provenance: Provenance {
            Provenance.weakest([
                model.provenance, input.provenance, destination.provenance,
                allowance?.provenance ?? .unavailable,
            ])
        }
    }

    /// Cost one shape against this budget. Pure arithmetic; no allocation, no syscall.
    public func assess(_ workload: ImageWorkload, at shape: TileShape) -> Assessment {
        let peak = Self.measuredPeak(at: shape)
        let model = Figure(
            bytes: peak.bytes * Self.concurrency, provenance: peak.provenance,
            source: "\(peak.source) × concurrency \(Self.concurrency)")
        let input = Figure(
            bytes: workload.inputSurfaceBytes, provenance: .estimated,
            source: "source image resident for the whole job — "
                + "\(workload.inputWidth)×\(workload.inputHeight) BGRA8")
        let destination = Figure(
            bytes: workload.destinationBytes(at: shape), provenance: .estimated,
            source: "\(workload.destination.rawValue) destination — "
                + "DestinationSurface.openRowCount × \(workload.outputRowBytes) B/row")
        return Assessment(
            shape: shape, model: model, input: input, destination: destination,
            allowance: allowance)
    }

    // MARK: Selection

    /// Why the plan named the shape it named.
    public enum Selection: Sendable, Equatable {
        /// The largest shape whose measured median fits ρ × available.
        case largestThatFits(TileShape)
        /// "256 when unknown": availability could not be read.
        case floorRungAvailabilityUnknown(TileShape)
        /// No rung fits. The floor rung is reported with what it is short by; certification spec
        /// §6's 48 MP rule (complete or degrade gracefully, never crash) is `TileShapeLadder`'s to
        /// act on.
        case noRungFits(floor: TileShape, shortfallBytes: Int)

        public var shape: TileShape {
            switch self {
            case .largestThatFits(let shape): return shape
            case .floorRungAvailabilityUnknown(let shape): return shape
            case .noRungFits(let floor, _): return floor
            }
        }
    }

    /// The answer: does this workload fit, at which shape, with how much headroom.
    public struct Plan: Sendable, Equatable {
        public let workload: ImageWorkload
        public let allowance: Figure?
        /// Every rung, ascending by shape, so a caller can see the whole ladder without re-costing
        /// it — and can see, for instance, that the floor rung is not always the cheapest one.
        public let assessments: [Assessment]
        public let selection: Selection
        /// The selected rung, costed. Stored rather than looked up so there is no unreachable
        /// "shape not found" branch to leave untested.
        public let selected: Assessment

        public var shape: TileShape { selection.shape }
        public var fit: Fit { selected.fit }
        public var headroomBytes: Int? { selected.headroomBytes }
        public var provenance: Provenance { selected.provenance }
    }

    /// Plan one image. **Once per image** — never per tile, and there is nothing here that would
    /// reward per-tile re-evaluation: the inputs do not change within an image.
    public func plan(for workload: ImageWorkload) -> Plan {
        let assessments = TileShape.allCases.sorted().map { assess(workload, at: $0) }
        let selection: Selection
        let selected: Assessment

        // The floor rung is re-costed rather than looked up in `assessments`, which would hand back
        // an Optional and an unreachable branch with it. `assess` is arithmetic on six integers.
        if let allowance {
            if let largest = assessments.last(where: { $0.fit == .fits }) {
                selection = .largestThatFits(largest.shape)
                selected = largest
            } else {
                let floor = assess(workload, at: .floorRung)
                selection = .noRungFits(
                    floor: .floorRung, shortfallBytes: floor.totalBytes - allowance.bytes)
                selected = floor
            }
        } else {
            selection = .floorRungAvailabilityUnknown(.floorRung)
            selected = assess(workload, at: .floorRung)
        }

        return Plan(
            workload: workload, allowance: allowance, assessments: assessments,
            selection: selection, selected: selected)
    }
}

// MARK: - The injection seam

/// The **only** place in the engine that calls `os_proc_available_memory()`.
///
/// It is a free function rather than a method on `MemoryBudget` on purpose: a budget that read the
/// host's memory from inside itself could not be tested against hand-computed values, and every
/// arithmetic assertion in `MemoryTests` would become a device claim.
///
/// Apple's rules, followed here: the value is advisory and must not be cached — so this is read at
/// job start, per image, and handed in. It is also documented as *much* cheaper than `task_info`,
/// which is what makes a per-image read affordable.
public enum AvailableMemory {

    /// Bytes the process may still allocate, read now.
    ///
    /// Returns an `.unavailable` figure in the two cases Apple's documentation defines away: on
    /// platforms without the API (macOS, the host that runs this package's tests), and when the API
    /// returns 0, which Apple defines as "the calling process isn't an app, or ... has already
    /// exceeded its memory limit". Both readings are handled the same way, because both mean the
    /// same thing to a budget: do not guess. The planner then selects the floor rung.
    public static func now() -> MemoryBudget.Figure {
        #if os(iOS) || os(tvOS) || os(watchOS) || os(visionOS)
            let bytes = Int(os_proc_available_memory())
            guard bytes > 0 else {
                return .unavailable(
                    source: "os_proc_available_memory() returned 0 — not an app, or the limit is "
                        + "already exceeded (Apple docs)")
            }
            return MemoryBudget.Figure(
                bytes: bytes, provenance: .measured,
                source: "os_proc_available_memory(), read at job start")
        #else
            return .unavailable(
                source: "os_proc_available_memory() is unavailable on this platform "
                    + "(macOS is absent from Apple's availability list)")
        #endif
    }
}

// MARK: - Engine seam

extension SequentialTilingEngine {
    /// Describe a job to the budget — **once per image**, never per tile.
    ///
    /// Purely descriptive. It selects nothing and actuates nothing; `TileShapeLadder` owns what
    /// happens when the answer is "does not fit". Its only job is to make the engine, rather than
    /// the caller, supply `contextMargin`, so the budget can never be costed against a margin the
    /// engine is not running with.
    ///
    /// `model.inputTileSize` is deliberately ignored: the shape is what the budget solves for.
    public func workload(
        imageWidth: Int,
        imageHeight: Int,
        model: ModelDescriptor,
        method: StitchMethod,
        destination: DestinationKind
    ) -> ImageWorkload {
        ImageWorkload(
            inputWidth: imageWidth, inputHeight: imageHeight, scaleFactor: model.scaleFactor,
            contextMargin: contextMargin, method: method, destination: destination)
    }
}
