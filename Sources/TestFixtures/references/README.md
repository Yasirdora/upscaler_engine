# Reference outputs

The PyTorch oracle's outputs for the certification corpus live here locally and are **never
committed**: they are derived from corpus images, which are not distributed, and they are
regenerated on demand.

    python tools/reference/run_reference.py --model RealESRGAN_x4plus

Each output is recorded in `../references_manifest.json` with the SHA256 of its input, its weights
and itself, so `run_reference.py --verify` detects both bit-rot and environment drift. The committed
manifest is empty until you generate references for your own corpus.

Layout: `references/<model>/<stem>.npy` (float32 NCHW, the comparison tensor) and
`references/<model>/<stem>.png` (an 8-bit view for human inspection).
