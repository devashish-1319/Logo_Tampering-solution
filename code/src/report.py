"""Combine real-image scores with the synthetic-genuine calibration into a
per-image z-score / flag count -- an approximate confidence signal, not a
verdict. See write-up for the honesty caveats on what this can and can't
claim."""
import csv
import numpy as np

METRICS = ["ssim", "ssim_local", "color_dE", "anisotropy"]
# direction: +1 means "higher = more anomalous", -1 means "lower = more anomalous"
DIRECTION = {"ssim": -1, "ssim_local": -1, "color_dE": +1, "anisotropy": +1}
FLAG_Z = 1.5


def genuine_stats(synthetic_summary):
    g = synthetic_summary["genuine"]
    return {m: (g[m]["mean"], g[m]["std"]) for m in METRICS if m in g}


def compare_domains(real_rows, genuine_ref_stats):
    """Report the real-vs-synthetic gap explicitly rather than silently
    using synthetic thresholds on real data. In this run the gap turned out
    to be large (e.g. real ssim_local sits ~0.5-0.66 vs a synthetic-genuine
    mean of ~0.16) -- almost certainly because the synthetic scenes'
    registration is systematically noisier than the real photos', not
    because the real photos are unusually clean. A direct z-score transfer
    would flag most of the real set as 'anomalous' purely from that offset,
    which is not a claim the data supports. This function exists so that
    gap is visible in every run's output, not just this note."""
    lines = []
    for m in METRICS:
        if m not in genuine_ref_stats:
            continue
        vals = np.array([r[m] for r in real_rows if r.get(m) is not None])
        if len(vals) == 0:
            continue
        gmean, gstd = genuine_ref_stats[m]
        gap_sigma = (np.median(vals) - gmean) / gstd if gstd > 1e-9 else float("nan")
        lines.append(f"  {m:12s} real median={np.median(vals):8.3f}  "
                      f"synthetic-genuine mean={gmean:8.3f} std={gstd:6.3f}  "
                      f"-> offset = {gap_sigma:+.1f} synthetic-sigma")
    return lines


def score_real_images_within_population(real_rows):
    """Rank real images against the REAL population's own median/MAD, since
    the synthetic baseline's absolute scale does not transfer (see
    compare_domains). This only says 'unusual relative to the other 49
    photos', not 'tampered' -- some of that spread is genuine pose/lighting
    variation the pipeline doesn't fully normalise away.

    Images flagged `low_coverage` (a tight photo crop left part of the
    warped reference frame with no real pixel data -- see the 009/041
    write-up) are excluded from the baseline population stats, since their
    own scores are unreliable and would otherwise drag the "normal" band
    around for everyone else too. They are still scored and reported, just
    tagged `inconclusive` rather than ranked as anomalous -- a coverage
    problem is not evidence about the logo either way."""
    out = []
    stats = {}
    clean_rows = [r for r in real_rows if not r.get("low_coverage")]
    for m in METRICS:
        vals = np.array([r[m] for r in clean_rows if r.get(m) is not None])
        if len(vals) == 0:
            continue
        median = float(np.median(vals))
        mad = float(np.median(np.abs(vals - median))) * 1.4826  # ~ std for normal data
        stats[m] = (median, mad if mad > 1e-9 else float(np.std(vals)) or 1.0)

    for row in real_rows:
        if row.get("low_coverage"):
            out.append(dict(row, **{f"z_{m}": None for m in METRICS}, n_flags=0,
                             inconclusive=True))
            continue
        z = {}
        n_flags = 0
        for m in METRICS:
            if m not in stats or row.get(m) is None:
                continue
            median, mad = stats[m]
            zscore = (row[m] - median) / mad * DIRECTION[m]
            z[m] = zscore
            if zscore > FLAG_Z:
                n_flags += 1
        out.append(dict(row, **{f"z_{m}": z.get(m) for m in METRICS}, n_flags=n_flags,
                         inconclusive=False))
    return out, stats


def write_csv(rows, path, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
