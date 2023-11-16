# tools/reference — the permanent PyTorch oracle

    python tools/reference/run_reference.py --model RealESRGAN_x4plus            # generate
    python tools/reference/run_reference.py --model RealESRGAN_x4plus --verify   # check

Exit `0` ok · `1` a reference did not verify · `2` stopped deliberately (missing config, unsupported
pipeline). Every Level A/B judgement is made against these outputs (certification spec §4), so they are only
worth their attribution: the manifest records model, weights SHA256, input + its SHA256, both output
hashes, shape, dtype, torch/python versions and timestamp.

`--verify` answers two separate questions and needs both. It re-hashes the files on disk (catching
bit-rot or edits) **and** recomputes the forward pass, comparing the fresh tensor's hash to the
recorded one (catching environment drift or nondeterminism). Checking only the first would miss a
torch upgrade quietly changing the oracle. Any mismatch names the offending file.

    Sources/TestFixtures/corpus/                  local, gitignored — inputs
    Sources/TestFixtures/references/<model>/      gitignored — .npy (exact float32) + .png (8-bit view)
    Sources/TestFixtures/references_manifest.json committed — the evidence

Weights come from `tools/artifacts/fetch.py` (imported, never re-implemented). The architecture comes
from `tools/convert/model_loader.py` — the same definition the conversion pipeline uses, so oracle and
candidate cannot diverge. Preprocessing is read from the model's `manifest.json`; a manifest declaring
a pipeline this tool does not implement stops the run rather than silently applying a different one.

An empty corpus is not a failure: the tool reports "corpus empty" and exits 0 with zero references.
