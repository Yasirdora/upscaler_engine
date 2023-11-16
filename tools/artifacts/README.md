# tools/artifacts — fetch and verify model artifacts

    python tools/artifacts/fetch.py --model RealESRGAN_x4plus    # fetch what is missing
    python tools/artifacts/fetch.py --all                        # every manifest under models/
    python tools/artifacts/fetch.py --verify-only [--model X]    # verify cache, never downloads

Exit `0` ok · `1` a hash did not verify · `2` missing or malformed manifest. Stdlib only.

**Identity policy.** Verification targets the **weights SHA256** in `models/<id>/manifest.json`, because the weights are
a reproducible input. A `.mlpackage` hash is never a rebuild-verification target: Core ML compilation
is not deterministic, so two identical runs produced different package hashes with bit-identical
metrics — it pins only the binary that shipped, and a rebuild is re-derived by conversion.

Artifacts live under `models/`:

    models/<id>/manifest.json    committed — the identity document
    models/<id>/weights/         gitignored — fetched and verified here
    models/<id>/output/          gitignored — .mlpackage from tools/convert/convert.py

Downloads land in a temp file inside the destination directory, are hashed there, and are renamed into
place only on a match — a failed fetch leaves no partial or unverified file. An already verified
artifact is not re-downloaded. A local file disagreeing with the manifest is never overwritten: it is
preserved for inspection and the run fails. `--verify-only` treats an absent artifact as "nothing to
verify", so CI with a cold cache stays green.
