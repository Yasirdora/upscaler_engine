import CoreGraphics
import Foundation

/// How a tile's output is written into the assembled image (certification spec §5.2).
///
/// All four are under test, on the same grid, so the bake-off compares stitching and nothing else:
/// `hardCrop` keeps every tile's full output and lets later tiles overwrite earlier ones, which is
/// the naive baseline that should show seams; `centerValidRegion` keeps only the uncontaminated
/// core; the two feathers blend across the overlap. The corpus decides which one is the default —
/// §5.2 records a preference for valid-region stitching because blending can soften detail, but a
/// preference is a hypothesis, not a result.
public enum StitchMethod: String, Sendable, CaseIterable {
    case hardCrop, linearFeather, cosineFeather, centerValidRegion
}

/// One tile of work: what the model is fed, and what part of the result this tile owns.
///
/// The two rects live in different spaces on purpose. `paddedInputRect` is input pixels and is
/// always exactly `tileSize` square, because the converted artifact is fixed-shape; `validRect` is
/// output pixels, so the stitcher never has to re-derive the scale factor while it is writing.
public struct TileRegion: Sendable, Equatable {
    /// Output-space region this tile owns.
    public let validRect: CGRect
    /// Input-space rect including context; always `tileSize` square.
    public let paddedInputRect: CGRect

    public init(validRect: CGRect, paddedInputRect: CGRect) {
        self.validRect = validRect
        self.paddedInputRect = paddedInputRect
    }
}

/// A grid restricted to a crop, plus the output region it actually covers.
///
/// `covered` is **not** the requested crop. Kept tiles are those whose `validRect` *intersects* the
/// crop, so the union of what they own generally overhangs it — writing them into a buffer sized to
/// the crop would either run out of bounds or silently truncate the edge tiles. Carrying the union
/// lets the caller allocate for what will actually be written and trim afterwards, leaving the
/// certified stitcher untouched.
public struct CroppedTileGrid: Sendable, Equatable {
    public let grid: TileGrid
    /// Output-space union of the kept tiles' `validRect`s, in FULL-image coordinates.
    public let covered: CGRect
    /// Where `covered` sits in the full image's output space. `covered` + this = absolute.
    public let originInFullOutput: CGPoint
    /// The requested crop in the cropped frame's coordinates — what to trim to.
    public let requested: CGRect

    public init(
        grid: TileGrid, covered: CGRect, originInFullOutput: CGPoint, requested: CGRect
    ) {
        self.grid = grid
        self.covered = covered
        self.originInFullOutput = originInFullOutput
        self.requested = requested
    }
}

/// A complete plan for one image: deterministic, row-major, and fully described by its inputs.
public struct TileGrid: Sendable, Equatable {
    public let tileSize: Int
    public let contextMargin: Int
    /// Deterministic ordering, row-major.
    public let tiles: [TileRegion]

    public init(tileSize: Int, contextMargin: Int, tiles: [TileRegion]) {
        self.tileSize = tileSize
        self.contextMargin = contextMargin
        self.tiles = tiles
    }

    /// Input-space side of the region each interior tile owns. Context is discarded on both sides.
    public var validCoreSize: Int { max(1, tileSize - 2 * contextMargin) }
}

/// The tiling contract's planner surface — **contract version 2**.
///
/// Version 1 was `plan(imageSize:tileSize:contextMargin:)` alone. Version 2 adds the resume entry
/// point below and changes nothing else: the version-1 member is byte-for-byte unchanged, so every
/// existing caller, test and certification keeps its exact current meaning.
public protocol TilePlanning: Sendable {
    func plan(imageSize: CGSize, tileSize: Int, contextMargin: Int) -> TileGrid

    /// Plan the part of an image that has **not been assembled yet**, starting at
    /// `resumingAtImageRow`.
    ///
    /// This is where the ladder's step down enters: *abandon the current tile at the current shape,
    /// retry it at the smaller shape, and re-grid the remainder of the image from that tile
    /// forward.* Tiles already stitched are never re-gridded, so the planner is asked only for the
    /// remainder — but it is still asked about the **whole image**, which is the entire point (see
    /// the implementation's note on why a translated sub-image plan was rejected).
    ///
    /// `resumingAtImageRow` is an **input-space image row**, not a tile index and not an
    /// output-space coordinate: it is the first row of the source image that no assembled tile
    /// owns. Values below 0 are treated as 0. A value at or past the image height leaves nothing to
    /// plan and yields an empty grid rather than a trap, matching the version-1 member's
    /// non-throwing contract.
    ///
    /// Resumption is at **row granularity**, never mid-row. Row-major assembly means the rows above
    /// the resume row are complete and the rows below are untouched, which is the only granularity
    /// a re-grid at a *different* tile size can even be expressed at — a new tile size moves the
    /// column boundaries too, so "the rest of this row, at a new shape" has no meaning. The caller
    /// therefore passes the top of the abandoned tile's row, and the tiles to its left in that row
    /// are re-planned and re-run.
    func plan(
        imageSize: CGSize, tileSize: Int, contextMargin: Int, resumingAtImageRow row: Int
    ) -> TileGrid
}

/// Row-major tile planner.
///
/// The scale factor is a property of the planner rather than a parameter of `plan`, because
/// `TileRegion.validRect` is specified in output space and the contract's `plan` signature carries
/// no scale: a planner is built for a model, then asked about images.
///
/// **Overlap is paid twice.** Every pixel of context is a pixel the model infers and then discards,
/// and it is also bandwidth: read into the tile, written out of the model, read again by the
/// stitcher. `redundancy(for:)` reports the multiplier so the cost is stated rather than assumed —
/// at 256px tiles with a 32px margin, a large image costs ≈1.78× the inference of a
/// seamless-but-impossible single-shot run.
public struct TilePlanner: TilePlanning {
    public let scaleFactor: Int

    public init(scaleFactor: Int) {
        self.scaleFactor = scaleFactor
    }

    /// The full plan, restricted to the tiles a crop needs — the preview path.
    ///
    /// **Two properties make a preview honest, and both are structural rather than checked.**
    ///
    /// 1. *The full grid is planned first and then filtered.* Tile origins therefore land exactly
    ///    where a whole-image run would put them. A crop planned on its own grid would place
    ///    boundaries elsewhere, so `centerValidRegion` would stitch different seams and the pixels
    ///    would legitimately differ from the final render — the surprise the preview exists to
    ///    prevent.
    /// 2. *`paddedInputRect` is carried through untouched.* It stays in the **full image's** input
    ///    coordinates, so the engine reads each tile's context from real neighbouring pixels instead
    ///    of reflecting at the crop's edge. Reflection at a crop boundary is precisely what would make
    ///    a preview disagree with the final render.
    ///
    /// **Both rects are returned absolute.** Nothing is rebased here: `Stitcher.geometry` derives a
    /// tile's destination position from `paddedInputRect`, so `validRect` and `paddedInputRect`
    /// must remain in one space. The caller renders into a canvas the size of `covered` and offsets
    /// its writes by `originInFullOutput`.
    ///
    /// - Parameters:
    ///   - imageSize: the FULL image, never the crop — the grid must match the whole-image plan.
    ///   - tileSize: fixed input geometry of the converted model.
    ///   - contextMargin: the model's own derived margin (`ContextMargin.derivedMargin(for:)`).
    ///   - inputCropRect: the region of interest, in **input** pixels.
    /// - Returns: the kept tiles with `validRect` rebased to the cropped frame, together with the
    ///   region they cover (in full-output coordinates), where it starts, and what to trim to.
    public func plan(
        imageSize: CGSize, tileSize: Int, contextMargin: Int, croppingTo inputCropRect: CGRect
    ) -> CroppedTileGrid {
        let full = plan(imageSize: imageSize, tileSize: tileSize, contextMargin: contextMargin)
        let scale = CGFloat(scaleFactor)
        let cropOutput = CGRect(
            x: inputCropRect.origin.x * scale, y: inputCropRect.origin.y * scale,
            width: inputCropRect.width * scale, height: inputCropRect.height * scale)

        let kept = full.tiles.filter { $0.validRect.intersects(cropOutput) }
        guard !kept.isEmpty else {
            return CroppedTileGrid(
                grid: TileGrid(tileSize: full.tileSize, contextMargin: full.contextMargin, tiles: []),
                covered: .zero, originInFullOutput: cropOutput.origin,
                requested: CGRect(origin: .zero, size: cropOutput.size))
        }

        // The union of what the kept tiles own — generally larger than the crop, because a tile is
        // kept when it merely intersects. This is what the stitcher will actually write.
        let union = kept.dropFirst().reduce(kept[0].validRect) { $0.union($1.validRect) }
        // BOTH rects stay absolute. `Stitcher.geometry` places a tile at
        // `paddedInputRect.origin * scaleFactor` and then writes the part `validRect` selects, so
        // the two must share one coordinate space. Translating `validRect` alone — which an earlier
        // draft of this method did — silently placed every tile 192 px out and produced a preview
        // that matched nothing. The end-to-end pixel test caught it; every structural test passed.
        // The canvas offset belongs to the engine, which applies `originInFullOutput` at write time.
        return CroppedTileGrid(
            grid: TileGrid(
                tileSize: full.tileSize, contextMargin: full.contextMargin, tiles: kept),
            covered: union,
            originInFullOutput: union.origin,
            requested: cropOutput.offsetBy(dx: -union.origin.x, dy: -union.origin.y))
    }

    public func plan(imageSize: CGSize, tileSize: Int, contextMargin: Int) -> TileGrid {
        let width = Int(imageSize.width.rounded(.down))
        let height = Int(imageSize.height.rounded(.down))
        guard width > 0, height > 0, tileSize > 0 else {
            return TileGrid(tileSize: max(0, tileSize), contextMargin: max(0, contextMargin), tiles: [])
        }

        // A margin that leaves no core would produce an infinite grid. Clamped rather than trapped:
        // `plan` is non-throwing by contract, and a caller asking for an impossible margin gets the
        // largest one that still tiles, which every test below pins.
        let margin = min(max(0, contextMargin), (tileSize - 1) / 2)
        let core = tileSize - 2 * margin

        let columns = (width + core - 1) / core
        let rows = (height + core - 1) / core

        var tiles: [TileRegion] = []
        tiles.reserveCapacity(rows * columns)
        for row in 0..<rows {
            for column in 0..<columns {
                let validX = column * core
                let validY = row * core
                let validWidth = min(core, width - validX)
                let validHeight = min(core, height - validY)
                tiles.append(
                    TileRegion(
                        validRect: CGRect(
                            x: validX * scaleFactor, y: validY * scaleFactor,
                            width: validWidth * scaleFactor, height: validHeight * scaleFactor
                        ),
                        paddedInputRect: CGRect(
                            x: Self.windowOrigin(
                                validOrigin: validX, validExtent: validWidth,
                                tileSize: tileSize, imageExtent: width),
                            y: Self.windowOrigin(
                                validOrigin: validY, validExtent: validHeight,
                                tileSize: tileSize, imageExtent: height),
                            width: tileSize, height: tileSize
                        )
                    )
                )
            }
        }
        return TileGrid(tileSize: tileSize, contextMargin: margin, tiles: tiles)
    }

    /// The resume entry point — contract version 2, additive.
    ///
    /// ## The property this exists to preserve
    ///
    /// The alternative — planning the remainder as its own translated sub-image — needs no contract
    /// change but changes pixels, and the mechanism is `windowOrigin`: it clamps the window to the
    /// extent it is handed. Given only the remainder, the resumed row's `validOrigin` is 0,
    /// `centred` is negative, and the clamp puts the window top **exactly on the resume row** —
    /// that row takes no context above it at all, where an interior row of a full-image plan sees a
    /// full `margin` of real context. A seam at every step down.
    ///
    /// This member avoids that by construction: **only the starting row changes.** `width`/`height`
    /// stay the whole image's, and `windowOrigin` is handed `imageExtent` for the whole image, so a
    /// resumed row's window slides above the resume row into pixels the caller still has — they are
    /// input pixels, and the input buffer is live from the first tile to the last. Nothing about
    /// the window depends on where assembly happens to have stopped.
    ///
    /// The consequence is exact and is what makes the step down safe: **at the same `tileSize` and
    /// `contextMargin`, resuming at row `r * core` produces precisely the tail of the full plan
    /// from tile row `r` onward** — same `validRect`s, same windows, same context, same order.
    /// Re-gridding the remainder is therefore pixel-neutral at a fixed shape.
    ///
    /// **At a different tile size the claim is weaker, and deliberately so.** A smaller shape has a
    /// smaller core, so the grid genuinely differs: different row and column boundaries, a
    /// different tile count, different windows, and — measured — output differing by ≤ 0.007 dB.
    /// What still holds, and is what the retry actually needs, is that the remainder's valid
    /// regions **exactly cover** image rows `[resumeRow, height)` with no gap and no overlap,
    /// abutting the assembled part at the resume row, and that every tile keeps the same context
    /// invariant the version-1 plan guarantees: each edge of a valid region either has `margin`
    /// pixels of window inside it or sits on the image boundary. Pixel identity across shapes is
    /// not claimed and is arithmetically unavailable.
    ///
    /// ## Why this repeats the version-1 loop instead of sharing one
    ///
    /// `plan(imageSize:tileSize:contextMargin:)` is kept **byte-for-byte unchanged** so that adding
    /// this member could not move any existing certification, so it is not refactored into a shared
    /// helper here. The duplication is the price of that guarantee and is deliberate.
    /// `TilePlannerResumeTests.resumingAtRowZeroReproducesTheFullPlan` executes both paths and
    /// compares them, so the two cannot drift silently.
    public func plan(
        imageSize: CGSize, tileSize: Int, contextMargin: Int, resumingAtImageRow row: Int
    ) -> TileGrid {
        let width = Int(imageSize.width.rounded(.down))
        let height = Int(imageSize.height.rounded(.down))
        guard width > 0, height > 0, tileSize > 0 else {
            return TileGrid(tileSize: max(0, tileSize), contextMargin: max(0, contextMargin), tiles: [])
        }

        let margin = min(max(0, contextMargin), (tileSize - 1) / 2)
        let core = tileSize - 2 * margin

        // A resume point at or past the last row means the image is already assembled. That is a
        // legitimate state for the engine to reach — the abandoned tile can be in the final row —
        // so it yields an empty remainder rather than a trap, the same way an impossible margin is
        // clamped rather than trapped.
        let resumeY = min(max(0, row), height)

        let columns = (width + core - 1) / core
        let rows = (height - resumeY + core - 1) / core

        var tiles: [TileRegion] = []
        tiles.reserveCapacity(rows * columns)
        for rowIndex in 0..<rows {
            for column in 0..<columns {
                let validX = column * core
                let validY = resumeY + rowIndex * core
                let validWidth = min(core, width - validX)
                let validHeight = min(core, height - validY)
                tiles.append(
                    TileRegion(
                        validRect: CGRect(
                            x: validX * scaleFactor, y: validY * scaleFactor,
                            width: validWidth * scaleFactor, height: validHeight * scaleFactor
                        ),
                        paddedInputRect: CGRect(
                            // `imageExtent` is the whole image on both axes, never the remainder.
                            // This single argument is the difference between this member and a
                            // translated sub-image plan.
                            x: Self.windowOrigin(
                                validOrigin: validX, validExtent: validWidth,
                                tileSize: tileSize, imageExtent: width),
                            y: Self.windowOrigin(
                                validOrigin: validY, validExtent: validHeight,
                                tileSize: tileSize, imageExtent: height),
                            width: tileSize, height: tileSize
                        )
                    )
                )
            }
        }
        return TileGrid(tileSize: tileSize, contextMargin: margin, tiles: tiles)
    }

    /// Where to put the fixed-size window along one axis.
    ///
    /// Centred on the region the tile owns, then slid back inside the image when the image is large
    /// enough to hold the window. Sliding matters: at an image edge it buys real context instead of
    /// reflected context, and it preserves the invariant the engine depends on — every edge of
    /// `validRect` is either at least `margin` inside the window or coincident with the image
    /// boundary, where no context exists for anyone, the full-frame oracle included.
    static func windowOrigin(validOrigin: Int, validExtent: Int, tileSize: Int, imageExtent: Int) -> Int {
        guard imageExtent >= tileSize else {
            // Smaller than one tile: centre the window and let reflection padding fill both sides.
            return Int((Double(imageExtent - tileSize) / 2.0).rounded(.down))
        }
        let centred = validOrigin + validExtent / 2 - tileSize / 2
        return min(max(centred, 0), imageExtent - tileSize)
    }

    /// Inference cost multiplier: pixels the model must produce ÷ pixels the image actually needs.
    /// Stated explicitly because overlap is paid in compute *and* bandwidth.
    public func redundancy(for grid: TileGrid, imageSize: CGSize) -> Double {
        let outputPixels = imageSize.width * imageSize.height * Double(scaleFactor * scaleFactor)
        guard outputPixels > 0 else { return 0 }
        let tilePixels =
            Double(grid.tiles.count)
            * Double(grid.tileSize * scaleFactor)
            * Double(grid.tileSize * scaleFactor)
        return tilePixels / outputPixels
    }
}
