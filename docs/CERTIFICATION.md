# Engine certification specification

This document defines what the engine may call "correct", "fast enough" and "memory safe", and how
each claim is measured. Source comments cite it as *certification spec §N*. Section numbers are
stable; a change to a threshold or a definition is a reviewed change to this file, never an edit
made to get a failing run to pass.

---

## 1. Scope

The engine runs fixed-shape super-resolution models through Core ML on Apple devices, tile by tile,
and assembles the result. Certification answers four questions:

1. Does the model convert to Core ML and execute correctly? (§4)
2. Does tiling preserve quality, with no visible seams? (§5)
3. Can the device floor process a 12 MP photograph without being terminated for memory? (§6)
4. Is the result fast and efficient enough on the target devices? (§7)

Human visual review (§9) sits above all four: a model that passes every metric and still looks
wrong does not pass.

---

## 2. Definitions of done

| Term | Means |
|---|---|
| **Conversion successful** | `.mlpackage` loads, executes a fixed-shape tile, and passes Level A thresholds (§4) against the reference oracle |
| **Correct** | Level A + Level B green on the full certification corpus; Level C artifact checklist clean at 100% / 200% / 400% inspection |
| **Fast enough** | Thresholds in §7.2 met on DeviceHarness, methodology per §7.3 |
| **Memory safe** | §6 protocol: 10 consecutive 12 MP → 4× runs on a 4 GB A14 device, zero jetsam, adaptive fallback exercised and verified |
| **Quality** | Level D human review (§9) — the final authority, above all metrics |
| **Seam-free** | Zero *human-visible* seams at 100% / 200% / 400% inspection across the corpus — not pixel identity |

---

## 3. Input contract

### 3.1 Supported (certified)

- Formats: JPEG, PNG, HEIF, TIFF.
- Colour: SDR RGB, 8-bit; sRGB and Display P3 (ICC profiles honoured). No processing in the wrong
  colour space.
- EXIF orientation: all values (1 / 3 / 6 / 8 explicitly tested — §8.2).
- Alpha (PNG): RGB is upscaled by the model; the alpha channel is upscaled separately with
  high-quality resampling and recombined. Certified by test.

### 3.2 Tolerated, not promised

16-bit TIFF, ProRAW and HDR must be accepted without crashing (input normalisation and fp16
overflow guards are tested), but no RAW/HDR quality claim is made.

### 3.3 Unsupported (clear pre-processing errors, never silent failure)

Corrupt or truncated files, animated formats, video, CMYK, and files exceeding device-memory
feasibility (§10).

### 3.4 Metadata policy

EXIF and ICC are preserved by default through the pipeline; orientation is normalised to 1 on
export; GPS and camera metadata are preserved (processing is on-device). Round-trip certified per
format (§8.4).

---

## 4. Model certification — four levels

| Level | Name | Metrics / method | Initial thresholds* |
|---|---|---|---|
| A | Numerical correctness | max-abs pixel error, MAE, PSNR vs. the reference oracle | PSNR ≥ 38 dB corpus median; max-abs ≤ 4/255 on ≥ 99% of pixels (fp16 tolerance) |
| B | Perceptual similarity | SSIM; LPIPS (offline harness only, pinned `lpips`; on-device Level B is SSIM only) | SSIM ≥ 0.98 corpus median; LPIPS recorded and trend-monitored |
| C | Artifact detection | Inspection at 100% / 200% / 400%: seams, ringing, halos, checkerboarding, colour shift, oversharpening, plastic skin, text deformation, false texture | Zero occurrences rated "visible at 100%" on the corpus |
| D | Human visual quality | Blind review (§9). **Final authority — overrides A–C.** | §9.3 |

\* **Ratchet rule:** initial thresholds are starting positions. They move only with evidence
attached, and the expectation is that they tighten — never that they loosen so a failing run
passes.

**LPIPS caveat:** perceptual metrics can reward sharp-but-hallucinated detail. LPIPS is a signal,
never a verdict.

### 4.1 Level A regimes

Measured across the full corpus (45 images × 4 stitch methods), tiled-versus-full-frame divergence
is content-dependent and tracks how much the model must invent. Category medians run from 23.3 dB
(anime) to 47.6 dB (rich real detail); per-block mean absolute error runs 0.5–0.9/255 where the
model refines real detail and 8–19/255 where it must invent the subject. The tiling machinery is
exact wherever the mapping is deterministic (seam ratios: corpus median 1.044, p95 1.385, no
genuine seam defect in 180 runs).

1. **Quantisation-bound regime** (rich input detail, smooth content): Level A thresholds apply as
   written.
2. **Hallucination-bound regime** (small or low-detail inputs, text, fine texture, line art, heavy
   noise): the full-frame oracle is one sample from a distribution of plausible outputs. Level A
   numbers are recorded and trend-monitored per image per method, but are **not per-image pass/fail
   gates**. Tiling correctness is judged by the seam boundary/interior ratio, Level C inspection and
   Level D.
3. **Sub-tile images** are certified against reflection-matched oracles, so the comparison measures
   tiling rather than a difference in border conditions.
4. This changes Level A's *interpretation* per regime, never its values (ratchet rule).

---

## 5. Tiling certification

### 5.1 Context margin derivation

Measured per model by receptive-field testing (pad-and-probe), not copied from upstream
(`tile_pad ≈ 10` is a reference point only). The criterion is the first margin at which mean ring
error falls below 1/255, doubled for safety.

### 5.2 Stitching methods

Hard crop · linear feather · cosine feather · centre-valid-region, all under test on the same grid.
Default hypothesis: context-pad → infer → crop valid region → stitch, because blending can soften
detail. The corpus decides.

### 5.3 Boundary tests

Tiles at image edges and corners, images smaller than one tile, dimensions 1 px short of / equal
to / 1 px over tile multiples, extreme aspect ratios (1:20).

### 5.4 Seam protocol

Every corpus image is inspected at 100% / 200% / 400% along tile boundaries; the result is recorded
in the certification manifest.

### 5.5 Adaptive fallback

Forced-failure tests must demonstrate 512 → 384 → 256 retry and successful completion.

---

## 6. Memory certification

- **Device floor:** A14 / 4 GB RAM.
- **Protocol:** 10 consecutive 12 MP → 4× runs, app warm and cold, background-app pressure applied;
  zero jetsam terminations; memory warnings handled gracefully (caches purged, model retained).
- **Budgets:** peak footprint measured and recorded per device; proactive budgeting (available
  memory × model footprint × tile footprint × concurrency) demonstrated — reactive-only warning
  handling is insufficient.
- **48 MP survival:** must complete or gracefully degrade (smaller tiles, a notice to the user) —
  never crash, never corrupt output.

---

## 7. Performance and energy certification

### 7.1 Workloads

| Workload | Size | Role |
|---|---|---|
| A | 1080p → 4× (2 MP → ~33 MP) | Headline benchmark |
| **B** | **12 MP → 4× (~192 MP out)** | **Primary workload** |
| C | 48 MP → 4× | Survival — not a performance target |

`tools/fixtures/make_fixtures.py` generates synthetic images at these sizes.

### 7.2 Thresholds (workload B)

| Device | Target | Threshold |
|---|---|---|
| A14 (iPhone 12) | < 60 s | < 120 s, zero jetsam |
| A17 Pro | < 25 s | < 45 s |
| M1 | < 20 s | < 35 s |
| M3 Pro/Max, M4 | < 8 s | < 10 s |

**Quality is never sacrificed to hit a timing target** — a miss means switch models, not degrade
output.

### 7.3 Methodology

- Four states measured and reported separately: **cold, warm, sustained (10-image run), thermally
  throttled**. First-run numbers alone are benchmark theatre.
- Median of ≥ 10 warm runs; device state (battery, thermal, background load) recorded.
- Compute-unit matrix per model × device × OS: `.all` / `.cpuAndGPU` / `.cpuAndNeuralEngine` /
  `.cpuOnly`. If the GPU wins on time and energy, the GPU wins.
- **Energy:** joules per image and battery % per 12 MP upscale where reliably obtainable. Every
  energy figure is labelled **measured / estimated / unavailable**.

---

## 8. Image pipeline certification

Even at SDR-only scope, incorrect colour is a broken image.

1. **Colour (§8.1):** sRGB and Display P3 round-trips; ICC profile preservation; gamma-correct
   processing; 8-bit / 16-bit decode correctness.
2. **Orientation (§8.2):** JPEG orientation 1 / 3 / 6 / 8 → decode → normalise → upscale → export →
   reopen → verify pixels and orientation.
3. **Alpha (§8.3):** per the §3.1 policy — the recombined result is inspected for fringing at alpha
   edges.
4. **Formats (§8.4):** JPEG / PNG / HEIF / TIFF round-trip with metadata intact (EXIF, ICC,
   timestamps, GPS per §3.4).
5. **Deferred (§8.5):** a full RAW / ProRAW / HDR colour pipeline (linear vs. gamma, wide gamut,
   transfer functions, tone mapping) is documented, not certified.

---

## 9. Human visual review (Level D)

### 9.1 Comparison set

The same source images through the reference PyTorch implementation, this engine's Core ML
pipeline, and other upscalers. A same-weights implementation should tie with the reference; a loss
there means the pipeline, not the model, is at fault.

### 9.2 Images

Drawn from the adversarial corpus: faces, hair, skin, foliage, fine text, fabric, architecture,
fine edges, skies, noise, heavy JPEG artifacts, anime / illustration.

### 9.3 Review

Three to five competent reviewers, blind to pipeline identity, in randomised order, inspecting at
100% / 200% / 400% and full-image. They score per category, then answer the question that matters:
*"which result would you actually export?"* — not "which has the highest PSNR". A model that is
consistently ranked last is the problem, not the engine: replace the model.

---

## 10. Failure behaviour

| Condition | Required behaviour |
|---|---|
| Memory pressure mid-job | Adaptive tile fallback (512 → 384 → 256) → retry → continue; whole-job failure is a defect |
| Unsupported input | Clear pre-processing error naming the limitation — never silent, never mid-job |
| Corrupt file | Decode error with a comprehensible message; no partial output files left behind |
| Cancellation | Honoured within one tile boundary; temporary artifacts cleaned |
| Thermal throttling | Reduce tile concurrency; ETA adapts; job continues |
| Interruption | Job state designed so resume is implementable |

---

## 11. Certification manifest

Every certification run emits `CERTIFICATION_MANIFEST.json`, complete enough to answer, months
later — including after a Core ML or OS update — *"why did model X produce this output?"*:

```json
{
  "certification_id": "", "date": "",
  "model_id": "", "weights_sha256": "", "mlpackage_sha256": "",
  "conversion": {"script_version": "", "coremltools": "", "python": "", "precision": "", "input_shape": "", "parameters": ""},
  "reference": {"implementation": "", "commit": "", "output_hashes": []},
  "coreml_output_hashes": [],
  "corpus_version": "",
  "device": "", "os_version": "", "coreml_version": "", "compute_units": "",
  "results": {"level_a": {}, "level_b": {}, "level_c": {}, "performance": {}, "energy": {"value": null, "kind": "measured|estimated|unavailable"}},
  "tiles": {"tile_set": "<provenance label, e.g. 'starter (12 tiles, 1 source image)'>", "entries": [{"source": "", "coords": [], "metrics": {}}]},
  "status": "PASS|FAIL"
}
```

- `tiles` carries per-tile metrics and a mandatory `tile_set` provenance label, so a thin corpus
  can never be cited as full certification by someone reading only the summary.
- Manifests live at `certification/<compute-units>/CERTIFICATION_MANIFEST.json` — one per pinned
  backend, because backends diverge numerically and unpinned runs are not certifiable.
- **8-bit surface ceiling: 58.9349 dB** (the oracle against its own 8-bit rounding). Level A on the
  8-bit product surface is quantisation-bound, not model-bound; a float output path would lift this
  ceiling.
- **`cpuOnly` is never a shipping configuration:** its error is gradient-correlated structure, not
  noise.

Conversion identity is recorded separately in `models/<id>/manifest.json`, and reference outputs in
`Sources/TestFixtures/references_manifest.json`.

---

## 12. Device matrix

**Must pass:** A14 · A17 Pro · M1.
**Certified before any device-specific claim is made:** A15, A16, A18+, M2, M3, M4.
**Informational:** everything else.
