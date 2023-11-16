import CoreImage
import CoreVideo
import DeviceHarness
import ImageIO
import PhotosUI
import SwiftUI
import UIKit
import UpscaleCore

/// On-device visual inspection.
///
/// Certification spec §9 makes human review — not any metric — the final authority on quality. That
/// means *looking* at a 4× result on the device it will run on, at 100 %, which needs zoom. A PSNR
/// figure cannot answer "does this look right", and host-side inspection does not answer "does it
/// look right on a phone".
///
/// **It is an inspection instrument, not an app.** No document model, no persistence, no export, no
/// navigation architecture.
///
/// It runs the **plain in-memory path** (`upscale(_:model:backend:method:progress:)`), which is the
/// certified control path — not the spooled or laddered paths. Whatever appears here is the
/// certified pipeline's own output, so a defect on screen is a defect in the engine rather than in
/// this viewer. Derived by measurement (`ContextMargin`), not copied from upstream.
///
/// File scope rather than a static on the view: `View` is `@MainActor` in Swift 6, which would
/// isolate the constant to the main actor and put it out of reach of the `nonisolated` upscale path
/// below. (Duplicated from `WholeImageRunner.contextMargin` and `ImageCommand`; the single home
/// would be `UpscaleCore`.)
private let viewerContextMargin = 52

struct UpscaleViewerView: View {
    @State private var source: UIImage?
    @State private var result: UIImage?
    @State private var showingResult = true
    @State private var status = "Ready."
    @State private var isRunning = false
    @State private var elapsed: Duration?

    @State private var shape: HarnessShape = .default256
    @State private var computeUnits: ComputeUnits = .all

    /// Every image sitting in the drop-in folder, so the choice of subject is the inspector's. No
    /// image is hardcoded: one smaller than a single 512 tile never exercises tiling, the ladder or
    /// the spool. A small image is a fine subject for judging quality and a misleading one for
    /// judging memory behaviour — the 12 MP and 48 MP workloads are the ones certification spec §6
    /// is written around.
    private let images: [URL] = Self.bundledImages()
    @State private var selected: URL?

    /// Photos chosen from the library, appended to the bundled ones so there is a single
    /// selection mechanism. Each is copied to a temp file first: the whole run path downstream
    /// takes a URL, and a `PhotosPickerItem` is not one — copying keeps that path untouched
    /// rather than teaching it a second way to find its input.
    @State private var imported: [URL] = []
    @State private var photoItem: PhotosPickerItem?
    @State private var importFailure: String?

    private var allImages: [URL] { images + imported }

    var body: some View {
        VStack(spacing: 12) {
            if allImages.isEmpty {
                ContentUnavailableView(
                    "No images bundled",
                    systemImage: "photo",
                    description: Text(
                        "Drop a .jpg or .png into DeviceHarnessApp/HarnessArtifacts/ and rebuild."))
            } else {
                controls
                Divider()
                canvas
                footer
            }
        }
        .padding()
        .navigationTitle("Inspect")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear {
            if selected == nil { selected = allImages.first }
            loadSource()
        }
    }

    // MARK: - Controls

    @ViewBuilder private var controls: some View {
        // Always offered, even with one bundled image: the point is to bring your own.
        PhotosPicker(selection: $photoItem, matching: .images, photoLibrary: .shared()) {
            Label("Choose photo from library", systemImage: "photo.on.rectangle")
        }
        .disabled(isRunning)
        .onChange(of: photoItem) {
            Task { await importPickedPhoto() }
        }
        if let importFailure {
            Text(importFailure).font(.caption).foregroundStyle(.red)
        }

        if allImages.count > 1 {
            Picker("Image", selection: $selected) {
                ForEach(allImages, id: \.self) { url in
                    Text(url.lastPathComponent).tag(Optional(url))
                }
            }
            .pickerStyle(.menu)
            .disabled(isRunning)
            // A stale result beside a changed picker is a mislabelled screenshot waiting to
            // happen — the same hazard ContentView guards for its own pickers.
            .onChange(of: selected) {
                result = nil
                elapsed = nil
                loadSource()
            }
        }

        let shapes = DeviceHarness.availableShapes()
        if shapes.count > 1 {
            Picker("Tile shape", selection: $shape) {
                ForEach(shapes, id: \.self) { Text($0.pickerLabel).tag($0) }
            }
            .pickerStyle(.segmented)
            .disabled(isRunning)
            .onChange(of: shape) { result = nil }
        }

        Picker("Compute units", selection: $computeUnits) {
            ForEach(ComputeUnits.allCases, id: \.self) { Text($0.rawValue).tag($0) }
        }
        .pickerStyle(.segmented)
        .disabled(isRunning)
        // cpuOnly is never a shipping configuration (certification spec §11) — its error is
        // gradient-correlated structure rather than noise. It stays selectable here precisely so
        // that claim can be seen rather than taken on trust.
        .onChange(of: computeUnits) { result = nil }

        HStack {
            Button(action: start) {
                Text(isRunning ? "Upscaling…" : "Upscale 4×")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .disabled(isRunning || selected == nil)

            if result != nil {
                Picker("", selection: $showingResult) {
                    Text("After").tag(true)
                    Text("Before").tag(false)
                }
                .pickerStyle(.segmented)
                .frame(width: 160)
            }
        }
    }

    // MARK: - The canvas

    /// Before and after are shown in the **same** zoomable frame, so A/B is a toggle rather than a
    /// second gesture to re-align. Judging a 4× upscale means comparing the same region at the same
    /// magnification; two independently panned views cannot do that.
    @ViewBuilder private var canvas: some View {
        let shown = showingResult ? (result ?? source) : source
        if let shown {
            ZoomableImage(image: shown)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color(.secondarySystemBackground))
                .clipShape(RoundedRectangle(cornerRadius: 8))
        } else {
            ProgressView().frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    @ViewBuilder private var footer: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(status).font(.caption).foregroundStyle(.secondary)
            if let source {
                let outputText =
                    result.map { " → \(Int($0.size.width))×\(Int($0.size.height))" } ?? ""
                Text("\(Int(source.size.width))×\(Int(source.size.height))\(outputText)")
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
            }
            if let elapsed {
                // Informational only. Nothing here is a performance claim and this number must
                // never be quoted as one.
                Text(verbatim: "\(elapsed) — informational, not a benchmark")
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: - Work

    /// Copy the picked photo into a temp file and select it.
    ///
    /// Failures are surfaced, not swallowed: a picker that silently does nothing is worse than one
    /// that says why, and "I chose a photo and nothing happened" is unfalsifiable from outside.
    @MainActor private func importPickedPhoto() async {
        guard let photoItem else { return }
        importFailure = nil
        do {
            guard let data = try await photoItem.loadTransferable(type: Data.self) else {
                importFailure = "The picked item returned no image data."
                return
            }
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent("picked-\(UUID().uuidString).img")
            try data.write(to: url)
            guard UIImage(contentsOfFile: url.path) != nil else {
                importFailure = "The picked item is not a decodable image."
                try? FileManager.default.removeItem(at: url)
                return
            }
            imported.append(url)
            selected = url
            result = nil
            elapsed = nil
            loadSource()
        } catch {
            importFailure = "Import failed: \(error)"
        }
    }

    private func loadSource() {
        guard let url = selected else { return }
        source = UIImage(contentsOfFile: url.path)
        status = source == nil ? "Could not decode \(url.lastPathComponent)." : "Ready."
    }

    private func start() {
        guard let url = selected, let modelURL = DeviceHarness.bundledModelURL(for: shape) else {
            status = "Model \(shape.modelFileName) is not bundled."
            return
        }
        isRunning = true
        result = nil
        elapsed = nil
        status = "Upscaling at \(shape.displayName), \(computeUnits.rawValue)…"

        Task {
            let clock = ContinuousClock()
            let started = clock.now
            do {
                // PNG `Data` crosses back rather than a `CVPixelBuffer` or `CGImage`: neither is
                // `Sendable`, and Swift 6 strict concurrency is not negotiable on this project.
                let png = try await Self.upscaleToPNG(
                    imageURL: url, modelURL: modelURL, shape: shape, computeUnits: computeUnits)
                let took = clock.now - started
                await MainActor.run {
                    result = UIImage(data: png)
                    elapsed = took
                    showingResult = true
                    status = result == nil ? "Upscale produced an undecodable image." : "Done."
                    isRunning = false
                }
            } catch {
                await MainActor.run {
                    // Reported verbatim. On any unexpected result the house rule is STOP and
                    // report with the numbers — never a friendly message that hides a defect.
                    status = "FAILED: \(error)"
                    isRunning = false
                }
            }
        }
    }

    /// Decode → upscale → encode, entirely off the main actor.
    ///
    /// The engine is handed the image exactly as `UpscaleCLI` hands it over, so this screen is a
    /// viewer onto the certified path and not a second implementation of it.
    private nonisolated static func upscaleToPNG(
        imageURL: URL, modelURL: URL, shape: HarnessShape, computeUnits: ComputeUnits
    ) async throws -> Data {
        let input = try loadPixelBuffer(from: imageURL)
        let descriptor = SingleTileRunner.descriptor(shape: shape, modelURL: modelURL)
        let backend = CoreMLBackend()
        try await backend.load(descriptor, context: InferenceContext(computeUnits: computeUnits))

        let engine = SequentialTilingEngine(contextMargin: viewerContextMargin)
        let output = try await engine.upscale(
            input, model: descriptor, backend: backend, method: .centerValidRegion, progress: nil)

        guard let png = pngData(from: output) else { throw ViewerError.encodeFailed }
        return png
    }

    /// Every bundled image, in name order.
    ///
    /// `HarnessArtifacts` is a *folder reference* in the app target, so whatever is dropped into it
    /// is copied into the bundle verbatim — which is why adding a subject needs no project edit.
    private static func bundledImages() -> [URL] {
        guard let directory = DeviceHarness.artifactsURL() else { return [] }
        let contents =
            (try? FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: nil)) ?? []
        return
            contents
            .filter { ["jpg", "jpeg", "png"].contains($0.pathExtension.lowercased()) }
            // The single-tile inputs are model fixtures, not subjects for visual inspection.
            .filter { !$0.lastPathComponent.hasPrefix("input_tile") }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    /// Decode to the 32BGRA buffer the engine's contract requires.
    ///
    /// Written out here rather than reused: `SingleTileRunner.loadPixelBuffer` is internal to
    /// `DeviceHarness`, and widening its access for a viewer is not worth the API surface. No
    /// colour management — `ImagePipeline` owns that subsystem — so this is the same utilitarian
    /// decode the harness already does.
    private nonisolated static func loadPixelBuffer(from url: URL) throws -> CVPixelBuffer {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
            let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else { throw ViewerError.decodeFailed }

        var buffer: CVPixelBuffer?
        let attributes: [CFString: Any] = [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, image.width, image.height, kCVPixelFormatType_32BGRA,
            attributes as CFDictionary, &buffer)
        guard status == kCVReturnSuccess, let pixelBuffer = buffer else {
            throw ViewerError.decodeFailed
        }

        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }
        guard
            let context = CGContext(
                data: CVPixelBufferGetBaseAddress(pixelBuffer),
                width: image.width, height: image.height, bitsPerComponent: 8,
                bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer),
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
                    | CGBitmapInfo.byteOrder32Little.rawValue)
        else { throw ViewerError.decodeFailed }

        context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        return pixelBuffer
    }

    private nonisolated static func pngData(from buffer: CVPixelBuffer) -> Data? {
        let ciImage = CIImage(cvPixelBuffer: buffer)
        let context = CIContext()
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return nil }
        return UIImage(cgImage: cgImage).pngData()
    }

    enum ViewerError: Error, CustomStringConvertible {
        case decodeFailed
        case encodeFailed
        var description: String {
            switch self {
            case .decodeFailed: return "the source image could not be decoded to 32BGRA"
            case .encodeFailed: return "the upscaled buffer could not be encoded for display"
            }
        }
    }
}

/// Pinch-zoom and pan, with a double-tap between fit and 1:1.
///
/// 1:1 is the magnification that matters: a 4× result judged at fit-to-screen is judged at roughly
/// one eighth of its real resolution, which is where upscaling artefacts hide.
private struct ZoomableImage: View {
    let image: UIImage

    @State private var scale: CGFloat = 1
    @State private var committed: CGFloat = 1
    @State private var offset: CGSize = .zero
    @State private var committedOffset: CGSize = .zero

    var body: some View {
        GeometryReader { geometry in
            Image(uiImage: image)
                .resizable()
                .interpolation(.none)  // Never let the display resampler flatter the result.
                .aspectRatio(contentMode: .fit)
                .frame(width: geometry.size.width, height: geometry.size.height)
                .scaleEffect(scale)
                .offset(offset)
                .gesture(
                    SimultaneousGesture(
                        MagnificationGesture()
                            .onChanged { scale = max(1, min(committed * $0, 40)) }
                            .onEnded { _ in
                                committed = scale
                                if scale == 1 {
                                    offset = .zero
                                    committedOffset = .zero
                                }
                            },
                        DragGesture()
                            .onChanged {
                                offset = CGSize(
                                    width: committedOffset.width + $0.translation.width,
                                    height: committedOffset.height + $0.translation.height)
                            }
                            .onEnded { _ in committedOffset = offset }
                    )
                )
                .onTapGesture(count: 2) {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        if scale > 1 {
                            scale = 1
                            committed = 1
                            offset = .zero
                            committedOffset = .zero
                        } else {
                            // Fit → 1:1 for the image's longest edge, capped so a 48MP result does
                            // not ask for a magnification nobody can navigate.
                            let fitted = max(image.size.width / max(geometry.size.width, 1), 1)
                            scale = min(fitted, 12)
                            committed = scale
                        }
                    }
                }
                .clipped()
        }
    }
}
