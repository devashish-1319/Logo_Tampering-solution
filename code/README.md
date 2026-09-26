# Logo Tampering Detection — Approach 1 (alignment + structural/colour/geometric scoring)

Detects subtle tampering (stretch, recolour, re-spacing, letterform swap) in
photos of NORVIA-branded packaging, by registering each photo against the 3
clean reference images and scoring the aligned result. See the top-level
write-up for full problem framing, the approaches considered but not
pursued, and an honest discussion of what this experiment did and didn't
show.

## Run it (single command)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run.py
```

Takes about 30 seconds. Writes to `outputs/`:
- `real_scores.csv` — per-image scores for all 50 candidate photos
- `synthetic_calibration.csv` — every synthetic calibration example generated, with known ground truth
- `debug/top_flagged/` — reference-vs-candidate crops for the 8 most unusual real images
- `debug/calib_*.jpg` — one example synthetic scene per tamper type, for a sanity look

## Pipeline

For each candidate photo, against each of the 3 references independently:

1. **Localize** — ORB keypoints + Lowe ratio test + RANSAC homography, matching the candidate directly against each reference image.
2. **Rectify** — warp the candidate into the reference's own pixel frame using that homography, then crop to the reference's logo-lockup bounding box (found once, via green-pixel connected components — see `src/logo_mask.py`). Candidate and reference are now pixel-aligned.
3. **Colour-correct** — gray-world illuminant correction (computed from the full warped frame, applied to the crop), so a white-balance shift isn't read as a recolour tamper.
4. **Score**:
   - `ssim` / `ssim_local` — global-mean and localized (trimmed 20th-percentile) SSIM (Wang et al. 2004) between corrected crop and reference.
   - `color_dE` — CIE76 ΔE between the corrected crop's dominant green ink colour and the reference's.
   - `anisotropy` — SVD of the homography's local Jacobian at the logo centre; how far the local scaling is from uniform (rotation + isotropic scale). **Important caveat, documented in `src/geometry.py`**: a homography from a single isolated planar patch cannot, on its own, distinguish "genuine but photographed at an angle" from "pre-stretched but photographed head-on" — this score is only meaningful once calibrated against a plausible-camera-angle distribution (below), not as a standalone proof.

The 3 per-reference scores are aggregated with the **median** (robust to one reference registering worse than the others).

## Calibration (`src/calibrate.py`, `src/synth_asset.py`, `src/synth_scene.py`)

There are no labels for the real 50 images, so a small synthetic benchmark
is generated to see whether these scores separate genuine from tampered
*at all*, before trusting them on real data:

- A clean logo "stamp" is cut out of `ref_1.jpg` (green ink → alpha).
- 4 known, labelled tampers are applied to copies of it: anisotropic stretch of the wordmark, hue-shifted recolour, icon/wordmark re-spacing, and a mirrored-glyph stand-in for a letterform swap (we don't have an alternate font, so this is a deliberate simplification — documented in `synth_asset.py`).
- Each variant (plus untampered copies) is composited onto a **procedurally generated** background (gradient + directional streak + sensor noise — not copied from the real dataset, to keep calibration independent) with a random plausible camera-like homography, optional blur, and JPEG recompression at a random quality.
- The full pipeline above is run on all of these, and genuine-vs-tampered separation is reported as a z-score per metric per tamper type.

**What this found, plainly:** separation is real but modest — `color_dE`
separates recolour by +0.82 genuine-population std; `ssim_local` separates
letterform-swap/recolour/respace by ~0.5-0.6 std; the anisotropy score
barely moves for the stretch tamper (+0.11 std). The reason, worked out
after looking at it: `tamper_stretch` only stretches the wordmark, not the
whole lockup, and a single RANSAC-fit homography over the whole logo tends
to treat that inconsistency as outlier noise rather than folding it into
the recovered transform — so a *partial* stretch shows up as local
misalignment (which `ssim_local` catches, weakly) rather than as global
anisotropy. A logo-wide uniform stretch would likely show up in
`anisotropy` instead, but is also the case where the identifiability
caveat above is sharpest.

**A second, important finding:** real images' absolute scores sit far
above the synthetic-genuine baseline (e.g. real `ssim_local` median ≈0.62
vs synthetic-genuine mean ≈0.16 — a +2.1σ offset). `run.py` prints this
offset explicitly every run rather than hiding it. This is almost certainly
a synthetic/real registration-quality gap (the procedural backgrounds and
harsher random homographies make synthetic registration systematically
noisier), not evidence that the real photos are unusually clean. Because of
this, **real images are ranked against each other** (median/MAD of the real
population), not against the synthetic thresholds — the synthetic
benchmark is used only to say which signals separate which tamper types at
all, not to set an absolute cutoff.

## A concrete example of an honest failure mode (and a partial fix)

The two most "flagged" real images (`009.jpg`, `041.jpg`) were inspected by
eye against their rectified crops. Both show a black wedge at one edge —
`cv2.warpPerspective` pulling in content from outside the candidate photo's
actual frame, because those photos are framed more tightly around the logo
than the references are. A low-flag image (`017.jpg`) showed no such
artifact and rectified cleanly.

This is now checked directly rather than left as a visual impression:
`localize.py` warps a same-size all-white mask through the identical
homography and measures what fraction of the reference bbox is actually
backed by real candidate pixels (`coverage`). Images below 90% coverage are
excluded from the ranking as `inconclusive` rather than scored as if fully
registered (`src/report.py`, `config.MIN_COVERAGE`).

Running it: **7 of 50 images** fall below 90% coverage, including
`041.jpg` (89.5%) — confirming that diagnosis quantitatively — and, as a
bonus neither of us had inspected before, `004.jpg` (90.0%), `013.jpg`
(82%), `016.jpg` (78%), `021.jpg` (87%), `037.jpg` (83%), `040.jpg` (89%).

**But `009.jpg` measured 95.3% coverage** — above the threshold — and
stayed in the ranked, flagged set. So the black wedge I saw in its crop was
real but smaller than it looked (a thin strip can look visually prominent
in a wide, short crop while covering only a few percent of the area), and
is evidently not the main driver of its anomaly score. That's an honest,
useful result in itself: the coverage guard fixed the artifact it was built
for (and found more instances of it than manual inspection had), while
correctly *not* explaining away an image that isn't actually explained by
it. `003.jpg` (100% coverage, still 3/4 flags after the fix) is now the
most genuinely interesting candidate for closer inspection, having no known
registration excuse.

## Licences

- OpenCV (`opencv-python-headless`): Apache 2.0
- NumPy: BSD-3-Clause

## A note on a dependency that was dropped

`scikit-image`/`scipy` were used for SSIM in an earlier draft, but a scipy
binary in this environment (PROPACK, pulled in transitively through
`skimage.metrics`) failed to `dlopen` on this macOS/arm64 + Python 3.10
combination. Rather than depend on that fragile import chain for something
this well-specified, SSIM is implemented directly from the Wang et al.
(2004) equations using `cv2.GaussianBlur` for the local statistics (see
`src/structural.py`) — one fewer dependency, and matches the paper exactly
rather than a library's specific implementation choices.
