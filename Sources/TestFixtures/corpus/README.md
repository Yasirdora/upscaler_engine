# Certification corpus

Corpus images live here locally and are **never committed**: the certification corpus is personal
to whoever runs it, and many usable sources do not permit redistribution. Only this README and
`../references_manifest.json` are tracked. A reference entry is therefore verifiable only on a
machine that holds the corresponding image; the manifest records each input's SHA256 so a
substituted file is detected rather than silently re-referenced.

## Layout

    Sources/TestFixtures/corpus/<category>/<name>.<ext>    # subdirectories are optional and recursive

Supported extensions: `.png .jpg .jpeg .tif .tiff .bmp .webp`. **Filename stems must be unique
across the whole corpus**, including across subdirectories: a reference is written as
`references/<model>/<stem>.npy`, so two images sharing a stem would collide. The oracle refuses to
run on a collision and names both files.

## Categories

A useful corpus is 100–200 deliberately adversarial images (certification spec §9.2). Curation is a
human task — the engine runs on whatever is present.

| Category | What it probes |
|---|---|
| `faces`, `skin`, `hair` | plastic skin, false detail, identity drift — the hardest content |
| `foliage`, `fabric`, `fine-edges` | fine organic and woven texture; oversharpening |
| `text`, `screenshots` | character deformation, ringing on hard edges |
| `architecture` | repetitive structure, moiré, checkerboarding |
| `skies`, `gradients` | banding, blotching, tile seams on smooth areas |
| `noise`, `jpeg-artifacts` | does the model restore or amplify degradation |
| `anime-illustration` | flat regions and line art, a different failure mode from photos |
| `extremes` | 1:20 aspect ratios, sub-tile images, 16-bit/ProRAW robustness |

## Provenance and versioning

Record the source and licence of every image you add, here, per image. Use only images you have the
right to use (your own photographs, or sources whose licence permits this use). Any addition,
removal or replacement bumps the corpus version, recorded on a line of this form, which
`certify-tiling` reads into its reports:

    Corpus version `corpus-YYYY-MM-DD.vN` — <count> images
