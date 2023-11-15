import CoreGraphics
import Foundation

/// The colour policy: which space the pipeline works in, and which profile the encoder must embed.
///
/// **The risk is preservation, not conversion.** A null test establishes that Core Graphics already
/// honours a source profile on read — max delta 0 — so nothing here rebuilds conversion. What has
/// to be guaranteed is that the profile is carried through to the *written* file: otherwise a
/// Display P3 photograph comes back sRGB and the out-of-gamut colour is gone. Measured on a
/// synthetic P3 sweep, that loss is a **max per-channel delta of 117/255**, 46% of full scale — a
/// visible loss rather than a technicality.
///
/// This module decides; it moves no pixels. `Encode` embeds `Plan.outputProfile` when it writes.
public enum Color {
    /// Whether a space is a fact about the file or a decision made on its behalf.
    ///
    /// The distinction is the whole point: assuming sRGB for untagged input is correct and
    /// unavoidable, but an assumption recorded as a fact is how a wrong guess becomes permanent.
    public enum Provenance: Sendable, Equatable {
        /// The file said so.
        case tagged
        /// The file was untagged and policy supplied sRGB.
        case assumedSRGB
        /// The source space could not back an 8-bit bitmap, so decode normalised to sRGB. The
        /// buffer really is sRGB; the *source's* space is what was lost.
        case normalisedFromUnsupportedSpace
    }

    public enum ColorError: Error, Equatable {
        /// The buffer's own profile could not be rebuilt. Unrecoverable: nothing downstream can
        /// know what the samples mean.
        case unusableBufferProfile
    }

    /// What the encoder needs to know, decided once and carried.
    public struct Plan: Sendable, Equatable {
        /// The space the samples are in while the model runs.
        public let workingProfile: Data
        /// The profile the encoder must embed. Preservation lives here.
        public let outputProfile: Data
        public let provenance: Provenance
        /// Core Graphics' name for the output space (e.g. `kCGColorSpaceDisplayP3`),
        /// when it has one. Identification only — this is the CG identifier, not
        /// ImageIO's human-facing `sourceProfileName`.
        public let outputProfileName: String?
        /// Plain-language record of anything a user or an audit would want to know.
        public let notes: [String]

        public func workingColorSpace() -> CGColorSpace? {
            CGColorSpace(iccData: workingProfile as CFData)
        }

        public func outputColorSpace() -> CGColorSpace? {
            CGColorSpace(iccData: outputProfile as CFData)
        }

        /// The gamma-correctness guarantee, stated as a property rather than a promise.
        ///
        /// §3.1 requires no processing in the wrong space. The strongest available guarantee is
        /// that **no conversion happens at all** between decode and the model: the working space
        /// *is* the buffer's space, so there is no step in which a transfer function could be
        /// applied, skipped or applied twice. A plan where these differ would mean a conversion
        /// was introduced, which is exactly the condition to fail on.
        public var isConversionFree: Bool { workingProfile == outputProfile }
    }

    /// Decide the plan for a decoded image.
    ///
    /// The output profile is the **buffer's** profile, never the source's. They are the same for a
    /// preserved decode, and deliberately different when decode had to normalise: writing the
    /// source's profile over normalised samples would mislabel them, which is worse than the
    /// normalisation itself.
    public static func plan(for carriage: Decode.ColorCarriage) throws -> Plan {
        guard let bufferSpace = carriage.bufferColorSpace() else {
            throw ColorError.unusableBufferProfile
        }
        var notes: [String] = []
        let provenance: Provenance
        switch (carriage.disposition, carriage.sourceProfile) {
        case (.normalisedToSRGB, _):
            provenance = .normalisedFromUnsupportedSpace
            notes.append(
                "source space could not back an 8-bit bitmap; samples were normalised to sRGB "
                    + "and the output is tagged sRGB accordingly")
        case (.preservedSourceSpace, .none):
            provenance = .assumedSRGB
            notes.append(
                "input carried no ICC profile; sRGB assumed and written explicitly so the "
                    + "assumption is not inherited as untagged")
        case (.preservedSourceSpace, .some):
            provenance = .tagged
            if let name = carriage.sourceProfileName {
                notes.append("source profile preserved end to end: \(name)")
            } else {
                notes.append("source profile preserved end to end")
            }
        }
        return Plan(
            workingProfile: carriage.bufferProfile,
            outputProfile: carriage.bufferProfile,
            provenance: provenance,
            outputProfileName: bufferSpace.name as String?,
            notes: notes)
    }

    /// The profile sRGB is written with when it is assumed rather than read.
    ///
    /// Built from `CGColorSpace.sRGB` rather than from a literal blob so it cannot drift from what
    /// Core Graphics actually means by sRGB.
    public static func assumedSRGBProfile() -> Data? {
        CGColorSpace(name: CGColorSpace.sRGB)?.copyICCData() as Data?
    }
}
