# Workload images (certification spec §7.1)

**These are workload images for performance, memory, thermal and survival measurement — not quality
evidence and not certification corpus.** They are generated, never committed:

    python tools/fixtures/make_fixtures.py --workloads

| File | Size | Workload | Purpose |
|---|---|---|---|
| `A_1080p.jpg` | 1920×1080 | A — headline benchmark | timing runs |
| `B_12MP.jpg` | 4032×3024 | B — primary workload | the §6 protocol (10 consecutive runs) |
| `C_48MP.jpg` | 8064×6048 | C — survival | 48 MP survival; the 4× output surface is 2,976.8 MiB |

Each image is a montage of deterministic 512×512 synthetic cells (gradients, gratings, texture,
edges and strokes), so it is not trivially compressible and carries no one's photograph. Content is
irrelevant to memory, time and survival measurement; it is not suitable for judging quality.

To run a workload on device, copy it into `DeviceHarnessApp/HarnessArtifacts/` (see
`DeviceHarnessApp/README.md`).
