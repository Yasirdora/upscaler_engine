import DeviceHarness
import SwiftUI
import UpscaleCore

struct ContentView: View {
    @State private var run: SingleTileRun?
    @State private var isRunning = false
    @State private var computeUnits: ComputeUnits = .all

    /// The variants actually dropped into `HarnessArtifacts/`, discovered once at launch. Empty
    /// when nothing was copied in; the selection then stays on 256 so the missing-model message is
    /// the standard one.
    @State private var shapes: [HarnessShape]
    @State private var shape: HarnessShape
    /// Selected input tile. Reset whenever the shape changes, since tiles are per-shape.
    @State private var tile: DeviceHarness.TileChoice?

    init() {
        let discovered = DeviceHarness.availableShapes()
        _shapes = State(initialValue: discovered)
        let initialShape =
            Self.requestedShape(among: discovered) ?? discovered.first ?? .default256
        _shape = State(initialValue: initialShape)
        _tile = State(initialValue: DeviceHarness.bundledTileChoices(for: initialShape).first)
    }

    /// `-autorun` lets an automated simulator check exercise the same path a human's tap does,
    /// without a UI-test target. Humans never see it.
    private var autorunRequested: Bool {
        ProcessInfo.processInfo.arguments.contains("-autorun")
    }

    /// `-shape 512` preselects a bundled variant, because the one thing a script cannot do is tap
    /// the picker. Ignored when the shape is not bundled — an automated check must never quietly
    /// run a different variant than the one it asked for.
    private static func requestedShape(among shapes: [HarnessShape]) -> HarnessShape? {
        let arguments = ProcessInfo.processInfo.arguments
        guard let flag = arguments.firstIndex(of: "-shape"), flag + 1 < arguments.count,
            let size = Int(arguments[flag + 1])
        else { return nil }
        return shapes.first { $0.tileSize == size }
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    // Only shown once more than one variant is bundled: with a single 256 drop-in
                    // there is nothing to choose, and the screen stays as simple as it can be.
                    if shapes.count > 1 {
                        Picker("Tile shape", selection: $shape) {
                            ForEach(shapes, id: \.self) { candidate in
                                Text(candidate.pickerLabel).tag(candidate)
                            }
                        }
                        .pickerStyle(.segmented)
                        .disabled(isRunning)
                        // A result panel does not name its shape, so a stale one next to a changed
                        // picker is a mislabelled screenshot waiting to happen.
                        .onChange(of: shape) {
                            run = nil
                            tile = DeviceHarness.bundledTileChoices(for: shape).first
                        }
                    }

                    // Input picker. Shown only when more than one tile is bundled for this shape,
                    // on the same discovery rule as the shapes above.
                    //
                    // Comparing two models is only meaningful with the input held fixed, so the
                    // selection is carried explicitly into the run rather than re-resolved per
                    // model — and changing it clears the result, for the same reason changing the
                    // shape does.
                    let tiles = DeviceHarness.bundledTileChoices(for: shape)
                    if tiles.count > 1 {
                        Picker("Input", selection: $tile) {
                            ForEach(tiles, id: \.self) { choice in
                                Text(choice.label).tag(Optional(choice))
                            }
                        }
                        .pickerStyle(.segmented)
                        .disabled(isRunning)
                        .onChange(of: tile) { run = nil }
                    }

                    Picker("Compute units", selection: $computeUnits) {
                        ForEach(ComputeUnits.allCases, id: \.self) { units in
                            Text(units.rawValue).tag(units)
                        }
                    }
                    .pickerStyle(.segmented)
                    .disabled(isRunning)
                    // Same hazard as the shape picker: the panel does not name its compute
                    // units either, so a stale result beside a changed picker mislabels a
                    // screenshot. (Observed in the field: a cpuOnly panel under "all".)
                    .onChange(of: computeUnits) { run = nil }

                    Button(action: start) {
                        HStack {
                            if isRunning { ProgressView().padding(.trailing, 4) }
                            Text(isRunning ? "Running…" : "Run \(shape.displayName) tile")
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 10)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(isRunning)

                    if let run {
                        resultPanel(run)
                    } else {
                        Text("Tap the button to run one fixed-shape tile through Core ML.")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }

                    Text(DeviceHarness.measurementDisclaimer)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                .padding()
            }
            .navigationTitle("Device Harness")
            .navigationBarTitleDisplayMode(.inline)
            // A toolbar link rather than a redesign: the single-tile screen keeps its layout, so
            // screenshots of it stay comparable across runs.
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    NavigationLink { UpscaleViewerView() } label: {
                        Label("Inspect", systemImage: "photo.on.rectangle.angled")
                    }
                }
            }
        }
        .task {
            if autorunRequested, run == nil { start() }
        }
    }

    @ViewBuilder
    private func resultPanel(_ run: SingleTileRun) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(DeviceHarness.displayRows(for: run.result), id: \.label) { row in
                // Label above value, with no fixed column width: at large Dynamic Type sizes a
                // fixed column truncates the labels to "Sta-tus" / "Com-pute units", and this panel
                // has to stay readable on whatever device the harness is run on.
                VStack(alignment: .leading, spacing: 2) {
                    Text(row.label)
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                    Text(row.value)
                        .font(.subheadline.monospaced())
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .accessibilityIdentifier("resultRows")

            if let data = run.outputPNG, let image = UIImage(data: data) {
                Text("Output — 4× upscale of the bundled \(shape.displayName) tile")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .accessibilityIdentifier("outputImage")
                    .border(.secondary)
            }
        }
        .padding(12)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 10))
    }

    private func start() {
        isRunning = true
        let units = computeUnits
        let selected = shape
        let chosenTile = tile?.url ?? DeviceHarness.bundledTileURL(for: selected)
        Task {
            let outcome = await DeviceHarness.runSingleTile(
                shape: selected, tileURL: chosenTile, computeUnits: units)
            await MainActor.run {
                run = outcome
                isRunning = false
            }
        }
    }
}

#Preview {
    ContentView()
}
