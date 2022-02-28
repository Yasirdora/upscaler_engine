import CoreGraphics
import Testing

@testable import UpscaleCore

/// Region-scoped planning.
///
/// A preview is only worth showing if it matches the final render. Two things make that true, and
/// both are properties of the PLAN rather than of the stitcher, which is why they are asserted
/// here: tile origins must land where a whole-image run puts them, and each tile's context must
/// come from the real image rather than reflected at the crop's edge.
struct CroppedTilePlanTests {
    static let planner = TilePlanner(scaleFactor: 4)
    static let size = CGSize(width: 1024, height: 1024)
    static let tile = 256
    static let margin = 16

    static func full() -> TileGrid {
        planner.plan(imageSize: size, tileSize: tile, contextMargin: margin)
    }

    static func cropped(_ rect: CGRect) -> CroppedTileGrid {
        planner.plan(imageSize: size, tileSize: tile, contextMargin: margin, croppingTo: rect)
    }

    // MARK: - The two properties that make a preview honest

    /// Condition (a) of finding F7. Planned on its own grid, a crop would place boundaries
    /// elsewhere and `centerValidRegion` would stitch different seams.
    @Test("kept tiles are a subset of the full grid, at the full grid's origins")
    func tilesComeFromTheFullGrid() {
        let fullPadded = Set(Self.full().tiles.map(\.paddedInputRect.debugDescription))
        let crop = Self.cropped(CGRect(x: 300, y: 300, width: 200, height: 200))

        #expect(!crop.grid.tiles.isEmpty)
        #expect(crop.grid.tiles.count < Self.full().tiles.count, "a crop must not need every tile")
        for region in crop.grid.tiles {
            #expect(
                fullPadded.contains(region.paddedInputRect.debugDescription),
                "every kept tile must sit exactly where the whole-image plan put it")
        }
    }

    /// Condition (b). `paddedInputRect` stays in the FULL image's coordinates, so the engine reads
    /// real neighbouring pixels. Translating it would make the engine read from the crop and
    /// reflect at its edge — which is exactly why today's preview disagrees with the final render.
    @Test("paddedInputRect is left in full-image coordinates, untranslated")
    func contextComesFromTheFullImage() {
        let crop = Self.cropped(CGRect(x: 512, y: 512, width: 128, height: 128))

        for region in crop.grid.tiles {
            #expect(
                region.paddedInputRect.origin.x >= 400 || region.paddedInputRect.origin.y >= 400,
                "a tile serving a crop at (512,512) cannot have been rebased to the crop's origin")
        }
    }

    /// **Nothing is rebased, and that is the correction.** An earlier draft translated `validRect`
    /// into the cropped frame while leaving `paddedInputRect` absolute. `Stitcher.geometry` places
    /// a tile at `paddedInputRect.origin * scaleFactor` and writes the part `validRect` selects, so
    /// the two must share one coordinate space — splitting them put every tile 192 px out and
    /// produced a preview matching nothing. Every structural test still passed; only the
    /// end-to-end pixel comparison caught it.
    @Test("both rects stay absolute, so the stitcher's coordinate spaces still agree")
    func rectsStayInOneSpace() {
        let rect = CGRect(x: 300, y: 300, width: 200, height: 200)
        let crop = Self.cropped(rect)
        let scale = CGFloat(4)

        for region in crop.grid.tiles {
            // The invariant the stitcher depends on: the valid region sits inside the padded tile,
            // offset by exactly the margin, in one shared space.
            let paddedInOutput = CGRect(
                x: region.paddedInputRect.origin.x * scale,
                y: region.paddedInputRect.origin.y * scale,
                width: region.paddedInputRect.width * scale,
                height: region.paddedInputRect.height * scale)
            #expect(
                paddedInOutput.contains(region.validRect),
                "validRect must lie inside its own padded tile; splitting the two spaces produced a preview that matched nothing"
            )
        }
    }

    /// The kept tiles must be byte-identical to the full grid's, not merely similar: the preview's
    /// correctness rests on running the same tiles the final render will.
    @Test("kept tiles are exactly the full grid's tiles, unmodified")
    func keptTilesAreUnmodified() {
        let crop = Self.cropped(CGRect(x: 300, y: 300, width: 200, height: 200))
        let full = Set(Self.full().tiles.map { "\($0.validRect)|\($0.paddedInputRect)" })

        for region in crop.grid.tiles {
            #expect(full.contains("\(region.validRect)|\(region.paddedInputRect)"))
        }
    }

    // MARK: - The overhang the plan review flagged

    /// Tiles are kept when they *intersect*, so what they own generally exceeds the crop. Sizing a
    /// buffer to the crop and writing these would run out of bounds or truncate the edge tiles.
    @Test("covered is the union of kept tiles and contains the requested crop")
    func coveredContainsTheRequest() {
        let crop = Self.cropped(CGRect(x: 300, y: 300, width: 200, height: 200))
        let requestedAbsolute = crop.requested.offsetBy(
            dx: crop.originInFullOutput.x, dy: crop.originInFullOutput.y)

        #expect(crop.covered.contains(requestedAbsolute), "the write region must contain the trim")
        #expect(crop.covered.origin == crop.originInFullOutput, "covered is in full-output space")
    }

    /// A crop aligned to the grid needs no overhang at all — the degenerate case that proves the
    /// union is doing real work in the others.
    @Test("a crop already on a tile boundary is covered exactly")
    func alignedCropNeedsNoOverhang() {
        let core = CGFloat(Self.tile - 2 * Self.margin)
        let crop = Self.cropped(CGRect(x: 0, y: 0, width: core, height: core))

        #expect(crop.covered.size == crop.requested.size)
    }

    // MARK: - Degenerate inputs

    @Test("a crop outside the image keeps no tiles and reports an empty region")
    func cropOutsideTheImage() {
        let crop = Self.cropped(CGRect(x: 5000, y: 5000, width: 100, height: 100))

        #expect(crop.grid.tiles.isEmpty)
        #expect(crop.covered == .zero)
    }

    @Test("a crop covering the whole image keeps every tile")
    func fullCropKeepsEverything() {
        let crop = Self.cropped(CGRect(origin: .zero, size: Self.size))

        #expect(crop.grid.tiles.count == Self.full().tiles.count)
    }

    /// The margin must survive into the cropped grid: the engine reads it back to size context.
    @Test("tile size and margin are carried through unchanged")
    func geometryIsCarried() {
        let crop = Self.cropped(CGRect(x: 100, y: 100, width: 100, height: 100))

        #expect(crop.grid.tileSize == Self.tile)
        #expect(crop.grid.contextMargin == Self.margin)
    }
}
