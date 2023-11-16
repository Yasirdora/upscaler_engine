# tools/certify — Level A/B certification harness

    python tools/certify/certify.py --model RealESRGAN_x4plus --compute-units all
    python tools/certify/certify.py --model RealESRGAN_x4plus --compute-units all --check

Exit `0` PASS/valid · `1` a threshold or a check failed · `2` stopped deliberately. `--compute-units`
is **required and has no default**: Core ML numerics diverge by backend, so an
unpinned run compares two things that were never the same thing. Each backend gets its own manifest.

The candidate is the product path — `UpscaleCLI upscale-tile … --emit-npy`, one subprocess per tile
— never a Python coremltools call. The oracle is `tools/reference/run_reference.py`. Geometry,
precision and pipeline come from `models/<id>/manifest.json`; thresholds from `config.json`
(certification spec §4). Neither is hardcoded in the harness.

Tiles: centre + four corners + stride-128 interior crops per corpus image, deduplicated, then shuffled
with the recorded seed. An image smaller than one tile yields none and is recorded as excluded, because
padding it would certify a padding policy this harness does not own.

    certification/tiles/       gitignored — the crops both sides read
    certification/reference/   gitignored — oracle .npy + 8-bit .png view
    certification/<units>/     CERTIFICATION_MANIFEST.json + report.txt — the committed evidence

`--check` re-hashes every recorded artifact **and** recomputes Level A + SSIM from the tensors, then
re-derives the verdict; LPIPS is not recomputed (needs torch). Any mismatch names the file. The suite,
`pytest tools/certify/tests/`, is offline, numpy-only and model-free.
