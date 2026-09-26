# Logo Tampering Detection — Research Write-up

## 1. Problem framing

The task: decide, for ~50 phone photos of NORVIA packaging, which show the
genuine logo and which show a subtly altered one — stretched, recoloured,
re-spaced, or with a letterform swapped for a look-alike — with no labels
and only 3 clean reference images to define "genuine."

**What makes this hard is not any single distortion, but that three
independent sources of variation move the same pixels as the one thing
we're trying to detect:**

- **Camera geometry.** Every photo is a different angle/distance/rotation
  of a flat label. A homography can fully explain any combination of
  rotation, scale, shear and anisotropic ("stretched-looking") distortion
  between two views of one rigid plane — so a genuinely stretched logo and
  a genuine logo photographed at a skewed angle can produce numerically
  indistinguishable geometry from the logo's own keypoints alone. A real
  identifiability limit, not an engineering inconvenience
  (`code/src/geometry.py`).
- **Camera/scene photometry.** White balance, exposure, and the
  packaging's own background colour all shift the *apparent* green-ink
  colour in ways that can look exactly like a recolour tamper unless
  corrected for first.
- **Compression and focus.** JPEG requantises colour and blurs letterform
  detail — both mimic what a genuine small tamper would do to the same
  signals.

Underneath that is a **data-provenance sub-problem I didn't expect going
in**: the dataset carries essentially no EXIF metadata, and its 50 images
fall into 18 distinct JPEG-quantization clusters of 2–5 images each (the 3
references form their own separate cluster) — the signature of a script
batch-rendering images at varied quality settings, not 50 independent
phone photographs. I take this as reasonably strong evidence the photos
are synthetic composites, not photos of a physically altered printed
object. That rules out an entire family of otherwise-plausible approaches:
there's no in-image splice boundary for copy-move/JPEG-ghost forensics to
find, since nothing was pasted into an otherwise-real photo — any
tampering lives in how the logo asset was rendered, before compositing.

Two further sub-problems are easy to under-weight. **No localisation is
given** — the logo must be found before anything about it can be measured,
and a naive global colour threshold fails immediately since several
backgrounds carry their own greenish tint (verified: a crude bounding-box
probe produced 20–90% deviations that turned out to be pose noise, not
tampering — rectification must happen *before* comparison, not after).
And **the four tamper types aren't equally visible to the same signal** —
a global structural-similarity score dilutes a small localised edit (one
respaced gap, one swapped letter) sitting inside an otherwise-unchanged
crop, and a homography fit robustly across the whole logo can vote a
partial, localised stretch down as outlier noise rather than fold it into
the recovered transform (confirmed in Section 3).

**Assumptions, and why they're reasonable:**

1. *One verdict per photo*, even though multiple tamper types could
   coexist — the brief describes single alterations per example, and this
   keeps the experiment tractable.
2. *"Genuine" is defined relative to the 3 references'* shared geometry/
   colour, not an external brand spec (none was provided) — a real
   limitation given n=3, stated rather than hidden.
3. *Deviation explained by perspective, white balance, or JPEG artifacts is
   not tampering* — only residual deviation after correcting for those
   counts as signal. This is the load-bearing assumption of the whole
   approach.
4. *The dataset is synthetically rendered* — argued from the evidence
   above; justifies ruling out splice-forensics, and is treated as
   falsifiable, not certain.
5. *Some images may be entirely genuine* — the tampered count is unknown,
   so results are continuous, ranked scores, not a forced binary split.

*(See Section 2 for the approaches considered, Section 3 for what was
actually measured, and Section 4 for what the identifiability and
dilution problems above imply about what to try next.)*

## 2. Approach exploration

Six approaches were considered. All citations are peer-reviewed journal
articles unless noted; DOIs are given so they're independently checkable.

**A. Classical alignment + structural/colour/geometric scoring (the one
run).** ORB + Lowe ratio test + RANSAC homography registers each photo
against each reference (transform-recovery machinery from Amerini et al.
2011, IEEE TIFS, [DOI](https://doi.org/10.1109/TIFS.2011.2129512); planar
model from Zhang 2000, IEEE TPAMI,
[DOI](https://doi.org/10.1109/34.888718)); the rectified crop is then
compared via SSIM (Wang et al. 2004, IEEE TIP,
[DOI](https://doi.org/10.1109/TIP.2003.819861)), a gray-world-corrected
Delta-E (Gijsenij et al. 2011, IEEE TIP,
[DOI](https://doi.org/10.1109/TIP.2011.2118224)), and a homography-
anisotropy score. **Good**: interpretable, needs no training data,
failure modes are individually diagnosable. **Bad**: partial tampering
dilutes in whole-crop aggregation; geometry alone can't separate "tampered"
from "photographed at an angle." **Pursued in full** — cheap, doesn't need
labels to run, and its weaknesses can be named precisely rather than
hidden inside a black box.

**B. Copy-move / JPEG-ghost forensics.** Finds a *splice boundary* —
pasted-region statistics disagreeing with the rest of the photo
(Christlein et al. 2012, IEEE TIFS,
[DOI](https://doi.org/10.1109/TIFS.2012.2218597); Farid 2009, IEEE TIFS,
[DOI](https://doi.org/10.1109/TIFS.2008.2012215)). **Not pursued**: Section
1's evidence (no EXIF, 18 tight JPEG-quantization clusters) says these
photos are synthetic composites rendered whole, not real photos with a
pasted edit — there is no splice boundary to find. Ruled out for a
specific, falsifiable reason, not by default.

**C. Learned / few-shot embedding similarity.** Compare a learned
feature distance instead of raw pixels — closest precedent is Zhou et
al.'s few-shot counterfeit-packaging detector (2026, Expert Systems with
Applications, [DOI](https://doi.org/10.1016/j.eswa.2025.129456)), which is
structurally close to "3 references, no labels." **Good**: more robust to
camera noise. **Bad**: a similarity score, not an explanation of *which*
region is wrong; needs more labelled data to validate honestly than 3
references give. **Not pursued in depth** — the strongest two-more-weeks
candidate (Section 4), but judged too costly to validate honestly now.

**D. Font/letterform forensics.** Bertrand et al.'s CRF model scores
characters against a font-property knowledge base to flag copy/paste or
look-alike-glyph forgery (ICDAR 2015,
[DOI](https://doi.org/10.1109/ICDAR.2015.7333827) — a conference paper;
the journal literature on letterform-swap detection specifically turned
out to be thin, worth saying rather than stretching a citation to cover
it). **Good**: targets per-glyph shape directly. **Bad**: built for
scanned documents, not phone photos of curved packaging — character
segmentation is a real risk. **Not pursued as a full pipeline**, but its
logic is exactly what Section 4 recommends building next.

**E. A trained/fine-tuned logo detector.** Per the logo-detection
literature (Hou et al. 2023 survey, ACM TOMM,
[DOI](https://doi.org/10.1145/3611309); Yousaf et al. 2021 Patch-CNN,
[DOI](https://doi.org/10.3233/JIFS-190660)). **Not pursued** — these need
thousands of labelled instances per class, and wasn't needed anyway:
Approach A's classical localizer registered all 50 real images
successfully on its own.

**F. Industrial one-class anomaly detection.** SLSG trains a self-
supervised one-class classifier against *simulated* anomalies (Yang et
al. 2024, Pattern Recognition,
[DOI](https://doi.org/10.1016/j.patcog.2024.110862)) — the closest
published analog to this exact situation. **Not pursued as a full
pipeline** (needs far more "normal" images than 50 to train), but its core
idea — calibrate against simulated anomalies, one-class not two-class —
was adopted directly into Approach A's calibration step (Section 3)
rather than left as an unused citation.

Approach A was run because it needed no training data, its failure modes
could be named and checked, and it fit the time budget. B/D/F's ideas were
absorbed into A rather than discarded; C and E need more data than this
task provides to validate honestly.

## 3. One experiment, actually run

Code: `code/`, runs end-to-end with `python run.py` (~30s, verified in a
clean virtualenv, OpenCV + NumPy only). Full per-image numbers in
`code/outputs/real_scores.csv` and `synthetic_calibration.csv`.

### 3.1 Registration held up

All 50 real photos registered against all 3 references (localize →
rectify, 2/A). **0 of 50 failed completely**; one (`012.jpg`) failed
against a single reference (too few RANSAC inliers) but succeeded against
the other two. Given the perspective/lighting range in this set, that's
the first real result: classical feature-matching wasn't the bottleneck.

### 3.2 The calibration measurement

With no ground truth, the core measurement is **does any of the four
scores separate known-genuine from known-tampered at all**, on a
synthetic benchmark built to test exactly that (mechanics in
`synth_asset.py`, `synth_scene.py`, `calibrate.py`): 40 genuine + 20-per-
type tampered (stretch/recolor/respace/letterform-swap) scenes —
procedural backgrounds, randomized camera-like pose, optional blur, random
JPEG recompression — scored with the identical real-image pipeline.
Separation from genuine, in genuine-population standard deviations:

| tamper type | strongest signal | shift | verdict |
|---|---|---|---|
| recolor | `color_dE` | **+0.82σ** | separable |
| letterform_swap | `ssim_local` | −0.62σ | weakly separable |
| respace | `ssim_local` | −0.54σ | weakly separable |
| stretch | `ssim_local` | −0.15σ / `anisotropy` +0.11σ | **not separable** |

This took one real correction to get right: the first version of
`ssim_local` (5th percentile of the SSIM map) saturated near zero for
genuine *and* tampered images alike — the map's own border pixels are
unreliable regardless of content — and only carried signal after switching
to a border-trimmed 20th percentile.

**Stretch not separating is informative, not just null.** `tamper_stretch`
only stretches the wordmark, not the whole lockup — and a single
RANSAC-fit homography over the *whole* logo tends to vote down an
inconsistent sub-region as outlier noise rather than fold it into the
recovered transform. So a partial stretch leaks into local misalignment
(`ssim_local`, weakly) rather than global anisotropy, which is what that
score was built to catch. A lockup-wide uniform stretch would likely move
`anisotropy` instead — but that's exactly where Section 1's
identifiability limit is sharpest: a uniform stretch and a viewing-angle
change are genuinely indistinguishable from the logo's own geometry alone.

### 3.3 Real images don't sit where synthetic ones do

Before trusting any 3.2 threshold on the real 50, real scores were checked
against the synthetic-genuine baseline directly:

| metric | real median | synthetic-genuine mean ± std | offset |
|---|---|---|---|
| ssim | 0.688 | 0.520 ± 0.062 | +2.7σ |
| ssim_local | 0.620 | 0.159 ± 0.222 | +2.1σ |
| color_dE | 10.27 | 8.95 ± 5.25 | +0.3σ |
| anisotropy | 1.321 | 1.139 ± 0.094 | +1.9σ |

Real images sit far above synthetic-genuine on 3 of 4 metrics — almost
certainly because the synthetic scenes' harsher poses/backgrounds make
*registration* noisier there, not because real photos are unusually clean.
A direct z-score transfer would flag nearly the whole real set from this
offset alone, which the data doesn't support. Real images are therefore
ranked **against each other** (population median/MAD), not against
synthetic absolutes — `run.py` prints this offset every run so it can't be
silently ignored later.

### 3.4 Applying it to the real 50 — a fix that worked, and its limit

Ranking by how many of 4 scores exceed 1.5 population-MAD (anomalous
direction), the top two were `009.jpg` and `041.jpg` (3/4 flags each).
Their rectified crops showed a black wedge at one edge —
`cv2.warpPerspective` pulling in content from outside the photo's actual
frame, because those photos are framed tighter than the references.
`017.jpg` (low-flag) showed no such artifact.

Checked directly rather than left as an impression: an all-white mask
warped through the identical homography measures what fraction of the
reference bbox is backed by real candidate pixels ("coverage"). Below 90%
is excluded as `inconclusive` rather than scored as fully registered.

**7 of 50 fall below 90%** — `041.jpg` at 89.5% (confirming the visual
read), plus six more (`004`, `013`, `016`, `021`, `037`, `040`) manual
inspection had missed. **But `009.jpg` measured 95.3%** — above threshold —
and stayed flagged: its wedge was real but small enough not to be the main
driver. That's the useful part — the fix resolved what it was built for
without over-correcting the one case it doesn't explain.

Top of the ranked list, 7 inconclusive images excluded:

| image | flags | ssim | ssim_local | color_dE | anisotropy |
|---|---|---|---|---|---|
| `003.jpg` | 3/4 | 0.644 | 0.529 | 20.56 | 1.324 |
| `009.jpg` | 3/4 | 0.623 | 0.521 | 14.29 | 1.432 |
| `011.jpg` | 2/4 | 0.652 | 0.554 | 17.36 | 1.288 |
| `019.jpg` | 2/4 | 0.644 | 0.552 | 4.46 | 1.374 |
| `029.jpg` | 2/4 | 0.593 | 0.248 | 5.72 | 1.373 |

`003.jpg` (100% coverage, no registration excuse) is the single most
genuinely interesting candidate this experiment produced — not tampered,
just unexplained.

### 3.5 What this does and doesn't establish

"Worked on a few images" would be pointing at `011.jpg` or `017.jpg`'s
clean crops and calling it a win. The 3.2 table is not that — it's a
controlled measurement with known labels, showing each signal's actual
(modest, uneven) separating power, including where and why it fails.

**Established:** registration is reliable; recolor is the one tamper type
separable with real (if not strong) evidence; a specific registration
artifact was found, quantified, and partially — not fully — corrected;
`003.jpg` is a defensible, evidence-ranked top candidate for review.

**Not established:** that any specific image is tampered. 35 of 50 carry
zero flags — given weak stretch-detection and a small calibration sample,
that could mean genuine, or could mean the pipeline can't see their tamper
type yet. Those are different claims this experiment can't yet separate.
No image here can be called tampered with confidence on this evidence
alone.

## 4. What I would do next, with two more weeks

Ranked by expected leverage, with what I'd measure to know each one
actually worked rather than just looked better.

**1. Close the synthetic/real domain gap.** The single highest-leverage
fix, because it's the reason the calibration in 3.2 can't be used as an
absolute threshold on real data at all right now. Instead of procedural
backgrounds, mask out just the logo region of real candidate photos and
composite synthetic genuine/tampered stamps into those *real* backgrounds
at each photo's own recovered pose — everything except the controlled
variable (tampered or not) then matches real data almost exactly.
*Measure*: rerun the Section 3.3 real-vs-synthetic offset table; success
means that offset collapsing from 1.9–2.7σ to under, say, 0.5σ on all four
metrics. Only once that holds would I trust an absolute threshold transfer.

**2. Per-letter / per-glyph decomposition.** Segment each character in the
rectified crop (connected components, matched positionally against the
reference's own per-letter boxes) and score each one individually —
position, aspect ratio, stroke width, small-window SSIM — instead of one
whole-lockup or icon/wordmark-split score. This directly targets the two
tamper types the current signals dilute away: respacing (gap between
specific letters) and letterform swap (one glyph's shape). *Measure*:
rerun the Section 3.2 calibration table with per-letter scores substituted
in; success means stretch and respace moving from ~0.15–0.5σ separation to
something closer to recolor's 0.82σ.

**3. Extend the frame-coverage fix into the scoring itself.** The coverage
guard (3.4) currently excludes low-coverage images outright. A finer
version would mask the SSIM computation to only the pixels backed by real
candidate data, so a photo with 80% coverage still contributes a genuine
partial score instead of being dropped entirely. *Measure*: check whether
any of the 7 currently-excluded images then produce a score consistent
with their neighbours' distribution (suggesting they were fine all along)
versus a genuinely different one (suggesting the coverage problem was
masking a real signal).

**4. Use inter-reference agreement as a confidence dial, not just a
median.** All three references currently collapse to one median per
metric per image, throwing away whether they agreed. If all three flag an
image, that's stronger evidence than one flagging it and two not.
*Measure*: for the current top candidates (`003.jpg`, `009.jpg`), report
the plain three-way spread per metric rather than the aggregated value,
and check whether `003.jpg`'s flagged status holds up per-reference or is
driven by one reference disagreeing with the other two.

**5. A calibrated classifier, once 1 and 2 are in place.** A small model
(even logistic regression on the per-letter + global features) trained on
a much larger, domain-matched synthetic set could output an actual
probability rather than a raw flag count. *Measure*: precision/recall on a
held-out slice of the same synthetic benchmark — explicitly *not* claimed
as validated on real tampering, since no real labels exist to check
against; the write-up would need to say so as plainly as this one does.

**What I would not expect two more weeks to fully close**: the
identifiability limit in Section 1 (geometry alone can't separate
"tampered" from "photographed at an angle" for an isolated planar patch)
is a property of the problem, not the implementation — closing it needs
either more reference viewpoints or an independent, non-logo coplanar
feature to cross-check against, neither of which exists in the current
data. The realistic end state for this class of problem is a ranked,
confidence-scored shortlist for human review, not a fully automated
verdict — which is also how production anti-counterfeiting systems are
generally designed, not as an oracle but as triage.

## 5. AI tool log

I used Claude (Anthropic, via Claude Code) throughout: literature search,
full-dataset visual inspection (all 50 images, not a sample), EXIF/
quantization analysis, and the entire implementation — including, when
asked, a second pass to critique its own earlier output.

**Where it helped:** finding and independently verifying 12 citations (11
journal + 1 clearly-labelled conference paper) against ≥2 independent
sources each before use — including correctly flagging that the best
letterform-forensics and an early counterfeit-detection paper were
conference-only rather than stretching them into journal citations;
building and iterating the full calibration pipeline in one session; and,
when asked "how do we get an actually confident answer," producing a
ranked, concrete next-steps list rather than a generic "collect more data."

**Where it was wrong, and how that got caught:**

- *A path-quoting bug nearly merged two unrelated projects.* An unquoted
  space in a folder name sent a `mkdir` into a different, pre-existing
  directory holding an unrelated earlier attempt at this same assignment.
  Caught by treating the unexpected listing as something to investigate,
  not overwrite — but it took me saying "that's an earlier attempt, use
  the real folder" before paths were re-verified explicitly going forward.
- *A dataset hypothesis was wrong.* Two images showed overlapping small-
  print text, first read as a possible compositing seam (relevant to
  Approach B). A third image where the same labels *didn't* overlap
  showed it was just random placement colliding — caught only by
  deliberately looking for a counter-example.
- *An EDA "finding" was a measurement artifact.* A crude bounding-box
  probe showed 20-90% deviations that briefly looked like tampering
  before re-checking showed it was pose noise — now the write-up's own
  evidence for why rectification must precede comparison.
- *The reporting logic's first draft would have been actively misleading.*
  It scored real images against the synthetic-genuine baseline directly;
  given the domain gap (3.3), that would have flagged most real images as
  "anomalous" from the offset alone, not from the logos. Caught only by
  printing that offset explicitly before trusting any threshold — the
  clearest single case here of AI output needing a sanity check rather
  than being taken at face value.
- *A visual diagnosis was half right.* Both top-flagged images were
  assumed to share one explanation (a rendering artifact) from a visual
  look. Building the actual coverage measurement (3.4) confirmed it for
  one and not the other (`009.jpg`, 95.3% coverage) — the fix was to
  report the second as still unexplained, not force-fit it.
- Smaller bugs (an alpha-compositing step pasting background pixels
  instead of blending; a localized-SSIM metric saturating at zero for
  every class) were only caught by inspecting intermediate debug images,
  not by the code running without errors.

**Net assessment:** most useful for breadth and speed — citations,
full-dataset inspection, a working pipeline in one sitting. Not reliable
unchecked at any single step: every claim that survived into this
write-up was one re-verified by an independent check, which is the
standard held throughout, not just in this section.
