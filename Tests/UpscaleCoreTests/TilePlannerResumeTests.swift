import CoreGraphics
import Foundation
import Testing

@testable import UpscaleCore

/// The planner's resume entry point, contract version 2.
///
/// Pure geometry, model-free. Every expectation below is either a hand-computed literal written out
/// in the test, or a structural property (exact cover, the context invariant) checked independently
/// of the planner's arithmetic. Nothing is recomputed with the code under test — the one place two
/// planner outputs are compared to each other is the tail-identity property, which *is* the claim.
///
/// The arithmetic the literals come from, stated once so a reader can re-derive any of them: `core
/// = tileSize - 2 * margin`, valid rows start at `resumeRow + k * core` and are `min(core, height -
/// validY)` tall, and the window origin along an axis is `clamp(validOrigin + validExtent / 2 -
/// tileSize / 2, 0, imageExtent - tileSize)` when the image is at least one tile long, or
/// `floor((imageExtent - tileSize) / 2)` when it is not. Output-space rects are input-space values
/// times the scale factor, 4 throughout.
struct TilePlannerResumeTests {
    static let scale = 4
    static let planner = TilePlanner(scaleFactor: scale)

    /// The remainder's valid regions must partition output rows `[bandMinY, bandMaxY)` exactly: no
    /// gap (a hole the resumed pass would leave unwritten) and no overlap (a pixel two tiles claim).
    /// Counted into a plain array here rather than derived from the grid, so a planner that
    /// double-covers or skips a row cannot agree with itself.
    static func assertCoversBandExactly(
        _ grid: TileGrid, imageSize: CGSize, bandMinY: Int, bandMaxY: Int
    ) {
        let outputWidth = Int(imageSize.width) * scale
        let height = bandMaxY - bandMinY
        var covered = [UInt8](repeating: 0, count: outputWidth * max(0, height))
        for tile in grid.tiles {
            for y in Int(tile.validRect.minY)..<Int(tile.validRect.maxY) {
                #expect(y >= bandMinY && y < bandMaxY, "tile row \(y) falls outside the remainder band")
                for x in Int(tile.validRect.minX)..<Int(tile.validRect.maxX) {
                    covered[(y - bandMinY) * outputWidth + x] &+= 1
                }
            }
        }
        #expect(covered.allSatisfy { $0 == 1 }, "the remainder must partition the unassembled band")
    }

    /// One (image, tile size, margin) triple for the parameterised cases below.
    struct Shape: Sendable {
        let size: CGSize
        let tileSize: Int
        let margin: Int
    }

    // MARK: - Row 0 is the full plan

    /// The resume path is a separate loop from the version-1 `plan` (which is kept byte-for-byte
    /// unchanged), so this executes both and compares them. It is the guard against the two
    /// drifting apart.
    @Test(
        "resuming at row 0 reproduces the full plan",
        arguments: [
            Shape(size: CGSize(width: 1000, height: 1000), tileSize: 256, margin: 32),
            Shape(size: CGSize(width: 575, height: 576), tileSize: 256, margin: 32),
            Shape(size: CGSize(width: 577, height: 576), tileSize: 256, margin: 32),
            Shape(size: CGSize(width: 100, height: 80), tileSize: 256, margin: 32),
            Shape(size: CGSize(width: 2000, height: 2000), tileSize: 512, margin: 52),
            Shape(size: CGSize(width: 20, height: 400), tileSize: 256, margin: 32),
            Shape(size: CGSize(width: 400, height: 400), tileSize: 256, margin: 200),
            Shape(size: CGSize(width: 512, height: 512), tileSize: 256, margin: 0),
        ])
    func resumingAtRowZeroReproducesTheFullPlan(shape: Shape) {
        let full = Self.planner.plan(
            imageSize: shape.size, tileSize: shape.tileSize, contextMargin: shape.margin)
        let resumed = Self.planner.plan(
            imageSize: shape.size, tileSize: shape.tileSize, contextMargin: shape.margin,
            resumingAtImageRow: 0)

        #expect(resumed == full)
    }

    @Test("a negative resume row is treated as row 0")
    func negativeResumeRowClampsToZero() {
        let size = CGSize(width: 1000, height: 1000)
        let full = Self.planner.plan(imageSize: size, tileSize: 256, contextMargin: 32)
        let resumed = Self.planner.plan(
            imageSize: size, tileSize: 256, contextMargin: 32, resumingAtImageRow: -400)

        // 1000 / (256 - 64) = 6 rows x 6 columns, hand-counted.
        #expect(full.tiles.count == 36)
        #expect(resumed == full)
    }

    // MARK: - The key property: same shape, identical tail

    /// **The property the resume path lives or dies on.** If a plan resumed at a row boundary is
    /// exactly the tail of the full plan, then re-gridding the remainder at the same shape changes
    /// no pixel, which is what makes the ladder's step down safe to take mid-image.
    @Test("a plan resumed at a row boundary is exactly the tail of the full plan")
    func resumedPlanIsTheTailOfTheFullPlan() {
        // 1000x1000, tile 256, margin 32 => core 192, 6 columns x 6 rows. Hand-counted.
        let size = CGSize(width: 1000, height: 1000)
        let full = Self.planner.plan(imageSize: size, tileSize: 256, contextMargin: 32)
        #expect(full.tiles.count == 36)

        for tileRow in 0...5 {
            let resumed = Self.planner.plan(
                imageSize: size, tileSize: 256, contextMargin: 32,
                resumingAtImageRow: tileRow * 192)

            #expect(resumed.tileSize == 256)
            #expect(resumed.contextMargin == 32)
            #expect(resumed.tiles.count == (6 - tileRow) * 6)
            #expect(resumed.tiles == Array(full.tiles[(tileRow * 6)...]))
            // The head that was already assembled plus the re-gridded remainder is the whole plan,
            // tile for tile: a step down at a fixed shape rewrites nothing it should not.
            #expect(Array(full.tiles[..<(tileRow * 6)]) + resumed.tiles == full.tiles)
        }
    }

    /// The same claim again, against literals rather than against the full plan, so a mutation that
    /// broke *both* code paths in the same way could not slip past the comparison above.
    @Test("the tiles of a resumed plan match hand-computed geometry")
    func resumedTileGeometryIsHandComputed() {
        // Resume at tile row 2 of the 1000x1000 / 256 / 32 grid: image row 2 * 192 = 384.
        let resumed = Self.planner.plan(
            imageSize: CGSize(width: 1000, height: 1000), tileSize: 256, contextMargin: 32,
            resumingAtImageRow: 384)

        #expect(resumed.tiles.count == 24)

        // First tile: valid (0, 384) 192x192 input -> (0, 1536) 768x768 output.
        // Window y: 384 + 96 - 128 = 352, inside [0, 744]. Window x: 0 + 96 - 128 = -32, clamped to 0.
        #expect(resumed.tiles[0].validRect == CGRect(x: 0, y: 1536, width: 768, height: 768))
        #expect(resumed.tiles[0].paddedInputRect == CGRect(x: 0, y: 352, width: 256, height: 256))

        // Last tile: valid (960, 960) 40x40 input -> (3840, 3840) 160x160 output.
        // Window on both axes: 960 + 20 - 128 = 852, clamped down to 1000 - 256 = 744.
        #expect(resumed.tiles[23].validRect == CGRect(x: 3840, y: 3840, width: 160, height: 160))
        #expect(resumed.tiles[23].paddedInputRect == CGRect(x: 744, y: 744, width: 256, height: 256))
    }

    /// The defect a translated sub-image plan would have, pinned at concrete numbers: a 52 px
    /// margin, and a resumed row that must still see 52 px above it.
    @Test("the resumed row keeps the context a translated sub-image plan would have lost")
    func resumedRowKeepsItsContextAbove() {
        // 2000x2000, tile 512, margin 52 => core 408, 5 columns x 5 rows. Resume at row 1: y = 408.
        let size = CGSize(width: 2000, height: 2000)
        let full = Self.planner.plan(imageSize: size, tileSize: 512, contextMargin: 52)
        let resumed = Self.planner.plan(
            imageSize: size, tileSize: 512, contextMargin: 52, resumingAtImageRow: 408)

        #expect(full.tiles.count == 25)
        #expect(resumed.tiles.count == 20)

        // Window y: 408 + 204 - 256 = 356, inside [0, 1488]. The window top sits 52 rows *above* the
        // resume row. Option (b) — planning rows 408..2000 as a 2000x1592 sub-image — would have
        // computed 0 + 204 - 256 = -52, clamped to 0, i.e. image row 408: no context above at all.
        let first = resumed.tiles[0]
        #expect(first.paddedInputRect == CGRect(x: 0, y: 356, width: 512, height: 512))
        #expect(first.validRect == CGRect(x: 0, y: 1632, width: 1632, height: 1632))
        #expect(first.validRect.minY / 4 - first.paddedInputRect.minY == 52)
        #expect(first.paddedInputRect.minY != 408, "the clamp defect of a translated sub-image plan")

        // And it is the very context the full plan's row 1 gets — index 1 * 5 columns + 0.
        #expect(first == full.tiles[5])
        TilePlannerTests.assertContextInvariant(resumed, imageSize: size)
    }

    // MARK: - Boundaries of the resume row itself

    @Test("resuming at the final tile row plans exactly that row")
    func resumingAtTheFinalRow() {
        let size = CGSize(width: 1000, height: 1000)
        let full = Self.planner.plan(imageSize: size, tileSize: 256, contextMargin: 32)
        let resumed = Self.planner.plan(
            imageSize: size, tileSize: 256, contextMargin: 32, resumingAtImageRow: 960)

        #expect(resumed.tiles.count == 6)
        #expect(resumed.tiles == Array(full.tiles[30...]))
        // Valid (0, 960) 192x40 input -> (0, 3840) 768x160 output; window y clamped to 1000 - 256.
        #expect(resumed.tiles[0].validRect == CGRect(x: 0, y: 3840, width: 768, height: 160))
        #expect(resumed.tiles[0].paddedInputRect == CGRect(x: 0, y: 744, width: 256, height: 256))
        Self.assertCoversBandExactly(resumed, imageSize: size, bandMinY: 3840, bandMaxY: 4000)
        TilePlannerTests.assertContextInvariant(resumed, imageSize: size)
    }

    @Test(
        "resuming at or past the last image row leaves nothing to plan",
        arguments: [1000, 1001, 5000, Int.max / 2])
    func resumingPastTheLastRow(resumeRow: Int) {
        let grid = Self.planner.plan(
            imageSize: CGSize(width: 1000, height: 1000), tileSize: 256, contextMargin: 32,
            resumingAtImageRow: resumeRow)

        #expect(grid.tiles.isEmpty)
        // Still a well-formed grid: an empty remainder is a state, not an error.
        #expect(grid.tileSize == 256)
        #expect(grid.contextMargin == 32)
    }

    @Test("an exhausted remainder still reports the clamped margin")
    func exhaustedRemainderKeepsTheClampedMargin() {
        let grid = Self.planner.plan(
            imageSize: CGSize(width: 400, height: 400), tileSize: 256, contextMargin: 200,
            resumingAtImageRow: 400)

        #expect(grid.tiles.isEmpty)
        #expect(grid.contextMargin == 127, "(256 - 1) / 2 — the largest margin leaving a core")
    }

    @Test("a resume row off a tile boundary starts the grid exactly on that row")
    func resumeRowOffATileBoundary() {
        // 1000x1000 / 256 / 32, resume at 500. Rows land at 500, 692, 884 — never at 576 or 768,
        // which is where the full plan's rows are. Heights 192, 192, 116; 192+192+116 = 500.
        let size = CGSize(width: 1000, height: 1000)
        let full = Self.planner.plan(imageSize: size, tileSize: 256, contextMargin: 32)
        let resumed = Self.planner.plan(
            imageSize: size, tileSize: 256, contextMargin: 32, resumingAtImageRow: 500)

        #expect(resumed.tiles.count == 18)
        let columnZero = [resumed.tiles[0], resumed.tiles[6], resumed.tiles[12]]
        #expect(columnZero.map(\.validRect.minY) == [2000, 2768, 3536])
        #expect(columnZero.map(\.validRect.height) == [768, 768, 464])
        // Window tops: 500 + 96 - 128 = 468; 692 + 96 - 128 = 660; 884 + 58 - 128 = 814 -> 744.
        #expect(columnZero.map(\.paddedInputRect.minY) == [468, 660, 744])
        #expect(!full.tiles.contains { $0.validRect.minY == 2000 }, "off-boundary rows are new rows")

        Self.assertCoversBandExactly(resumed, imageSize: size, bandMinY: 2000, bandMaxY: 4000)
        TilePlannerTests.assertContextInvariant(resumed, imageSize: size)
    }

    // MARK: - The cases certification spec §5.3 names

    @Test(
        "one pixel short of, equal to, and one pixel over a core multiple resume correctly",
        arguments: [575, 576, 577])
    func offByOneDimensionsResume(width: Int) {
        // Height 576 = 3 * 192 exactly, so image row 192 is tile row 1 in the full plan.
        let size = CGSize(width: width, height: 576)
        let full = Self.planner.plan(imageSize: size, tileSize: 256, contextMargin: 32)
        let resumed = Self.planner.plan(
            imageSize: size, tileSize: 256, contextMargin: 32, resumingAtImageRow: 192)

        let columns = width == 577 ? 4 : 3
        #expect(full.tiles.count == columns * 3)
        #expect(resumed.tiles.count == columns * 2)
        #expect(resumed.tiles == Array(full.tiles[columns...]))
        #expect(resumed.tiles.allSatisfy { $0.paddedInputRect.width == 256 })
        Self.assertCoversBandExactly(resumed, imageSize: size, bandMinY: 768, bandMaxY: 2304)
        TilePlannerTests.assertContextInvariant(resumed, imageSize: size)
    }

    @Test("an extreme aspect ratio resumes along its long axis")
    func extremeAspectRatioResume() {
        // 20x400: 1 column, ceil(400 / 192) = 3 rows. Resume at 192 leaves 2 of them.
        let size = CGSize(width: 20, height: 400)
        let full = Self.planner.plan(imageSize: size, tileSize: 256, contextMargin: 32)
        let resumed = Self.planner.plan(
            imageSize: size, tileSize: 256, contextMargin: 32, resumingAtImageRow: 192)

        #expect(full.tiles.count == 3)
        #expect(resumed.tiles.count == 2)
        #expect(resumed.tiles == Array(full.tiles[1...]))
        // Width 20 < 256, so the window is centred on the whole image: floor((20 - 256) / 2) = -118.
        #expect(resumed.tiles.allSatisfy { $0.paddedInputRect.minX == -118 })
        Self.assertCoversBandExactly(resumed, imageSize: size, bandMinY: 768, bandMaxY: 1600)
        TilePlannerTests.assertContextInvariant(resumed, imageSize: size)
    }

    @Test("an image shorter than one tile can still be resumed inside")
    func subTileImageResume() {
        // 100x80 with a 256 tile: both axes are shorter than the window, so it is centred on the
        // whole image and reflection fills the rest — including the rows above the resume row.
        let size = CGSize(width: 100, height: 80)
        let resumed = Self.planner.plan(
            imageSize: size, tileSize: 256, contextMargin: 32, resumingAtImageRow: 40)

        #expect(resumed.tiles.count == 1)
        // Valid (0, 40) 100x40 input -> (0, 160) 400x160 output.
        #expect(resumed.tiles[0].validRect == CGRect(x: 0, y: 160, width: 400, height: 160))
        // floor((100 - 256) / 2) = -78, floor((80 - 256) / 2) = -88 — unchanged by the resume row.
        #expect(resumed.tiles[0].paddedInputRect == CGRect(x: -78, y: -88, width: 256, height: 256))
        Self.assertCoversBandExactly(resumed, imageSize: size, bandMinY: 160, bandMaxY: 320)
    }

    @Test(
        "a degenerate image produces no tiles rather than a trap, resumed or not",
        arguments: [
            CGSize(width: 0, height: 100), CGSize(width: 100, height: 0), CGSize(width: 0, height: 0),
        ])
    func degenerateImageResume(size: CGSize) {
        #expect(
            Self.planner.plan(
                imageSize: size, tileSize: 256, contextMargin: 32, resumingAtImageRow: 50
            ).tiles.isEmpty)
    }

    @Test("a non-positive tile size refuses the same way the version-1 plan does")
    func nonPositiveTileSizeResume() {
        let grid = Self.planner.plan(
            imageSize: CGSize(width: 100, height: 100), tileSize: 0, contextMargin: 32,
            resumingAtImageRow: 10)

        #expect(grid.tiles.isEmpty)
        #expect(grid.tileSize == 0)
        #expect(grid.contextMargin == 32, "the unreachable-margin clamp needs a tile size to clamp to")
    }

    // MARK: - The real use: resuming at a *smaller* shape

    /// What the ladder actually does. The grid necessarily differs — a smaller core moves every row
    /// and column boundary — so the claim here is **coverage and context, never identical tiles**.
    @Test("a remainder re-gridded at a smaller tile size covers the band exactly, with full context")
    func remainderAtASmallerTileSizeCoversTheBand() {
        // 300x300, abandoned at image row 100. Remainder at tile 128 / margin 16 => core 96:
        // columns ceil(300/96) = 4, rows ceil(200/96) = 3 => 12 tiles. Rows at 100, 196, 292.
        let size = CGSize(width: 300, height: 300)
        let stepped = Self.planner.plan(
            imageSize: size, tileSize: 128, contextMargin: 16, resumingAtImageRow: 100)

        #expect(stepped.tiles.count == 12)
        #expect(stepped.tileSize == 128)
        #expect(stepped.contextMargin == 16)
        // Abuts the assembled part exactly: the first output row planned is 100 * 4.
        #expect(stepped.tiles[0].validRect.minY == 400)
        // Window y: 100 + 48 - 64 = 84, inside [0, 172] — 16 rows of real context above the resume
        // row, the same margin an interior row of a full plan at this shape would get.
        #expect(stepped.tiles[0].paddedInputRect.minY == 84)
        #expect(stepped.tiles[0].validRect.minY / 4 - stepped.tiles[0].paddedInputRect.minY == 16)

        Self.assertCoversBandExactly(stepped, imageSize: size, bandMinY: 400, bandMaxY: 1200)
        TilePlannerTests.assertContextInvariant(stepped, imageSize: size)
    }

    /// The negative half of the same statement, asserted rather than assumed: a step down is *not*
    /// tile-identical, so no caller may treat it as one — ladder-path equivalence is a Level A/B
    /// quality claim, not a pixel-identity claim.
    @Test("a remainder at a different tile size is a genuinely different grid")
    func remainderAtASmallerTileSizeIsNotTheSameGrid() {
        let size = CGSize(width: 300, height: 300)
        let atLargeShape = Self.planner.plan(
            imageSize: size, tileSize: 256, contextMargin: 32, resumingAtImageRow: 100)
        let atSmallShape = Self.planner.plan(
            imageSize: size, tileSize: 128, contextMargin: 16, resumingAtImageRow: 100)

        // ceil(300/192) = 2 columns, ceil(200/192) = 2 rows against 4 columns x 3 rows.
        #expect(atLargeShape.tiles.count == 4)
        #expect(atSmallShape.tiles.count == 12)
        #expect(atLargeShape.tiles != atSmallShape.tiles)
        // Both still start on the abandoned row, which is the part that must hold across shapes.
        #expect(atLargeShape.tiles[0].validRect.minY == 400)
        #expect(atSmallShape.tiles[0].validRect.minY == 400)
    }

    // MARK: - Contract surface

    @Test("the resume entry point is on the protocol, not only on the concrete planner")
    func resumeEntryPointIsPartOfTheContract() {
        let planning: any TilePlanning = TilePlanner(scaleFactor: 4)
        let grid = planning.plan(
            imageSize: CGSize(width: 1000, height: 1000), tileSize: 256, contextMargin: 32,
            resumingAtImageRow: 768)

        #expect(grid.tiles.count == 12, "rows at 768 and 960, 6 columns each")
    }
}
