"""Generate the synthetic genuine/tampered benchmark, run the real pipeline
on every example, and summarise how well each score separates known-genuine
from known-tampered. This calibration is used as a *relative* signal (how
far does a real image's score sit from the synthetic-genuine distribution,
in genuine-population standard deviations) rather than copying an absolute
threshold across -- the synthetic scenes are systematically easier/harder
than the real photos in ways that don't transfer 1:1 (see write-up).
"""
import csv
from dataclasses import dataclass
import numpy as np
import cv2

import config
from synth_asset import (extract_stamp, tamper_stretch, tamper_recolor,
                          tamper_respace, tamper_letterform_swap)
from synth_scene import composite
from pipeline import process_image

CANVAS_WH = (960, 1280)
N_GENUINE = 40
N_PER_TAMPER = 20

TAMPER_SPECS = {
    "stretch": dict(fn=tamper_stretch, params=lambda rng: dict(
        factor=float(rng.choice([rng.uniform(1.06, 1.16), rng.uniform(0.85, 0.94)])))),
    "recolor": dict(fn=tamper_recolor, params=lambda rng: dict(
        hue_shift_deg=float(rng.choice([rng.uniform(10, 25), rng.uniform(-25, -10)])))),
    "respace": dict(fn=tamper_respace, params=lambda rng: dict(
        gap_px=int(rng.choice([rng.integers(15, 35), -rng.integers(15, 30)])))),
    "letterform_swap": dict(fn=tamper_letterform_swap, params=lambda rng: dict(
        letter_index=int(rng.integers(0, 6)))),
}


def generate_and_score(refs, seed=0, out_csv=None, debug_dir=None):
    rng = np.random.default_rng(seed)
    ref_bgr = cv2.imread(str(config.REFERENCE_DIR / "ref_1.jpg"))
    stamp, split_x = extract_stamp(ref_bgr)

    rows = []

    def run_one(tag, stamp_variant, idx):
        img = composite(stamp_variant, CANVAS_WH, rng)
        if debug_dir is not None and idx == 0:
            cv2.imwrite(str(debug_dir / f"calib_{tag}.jpg"), img)
        res = process_image(f"{tag}_{idx}", img, refs)
        rows.append(dict(
            tag=tag, n_ok=res.n_refs_ok,
            ssim=res.ssim_median, ssim_local=res.ssim_local_median,
            color_dE=res.color_delta_e_median,
            anisotropy=res.anisotropy_median,
        ))

    for i in range(N_GENUINE):
        run_one("genuine", stamp, i)

    for tamper_name, spec in TAMPER_SPECS.items():
        for i in range(N_PER_TAMPER):
            params = spec["params"](rng)
            variant = spec["fn"](stamp, split_x, **params)
            run_one(tamper_name, variant, i)

    if out_csv is not None:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["tag", "n_ok", "ssim", "ssim_local", "color_dE", "anisotropy"])
            w.writeheader()
            w.writerows(rows)

    return rows


def summarize(rows):
    tags = sorted(set(r["tag"] for r in rows))
    summary = {}
    for tag in tags:
        sub = [r for r in rows if r["tag"] == tag and r["n_ok"] > 0]
        if not sub:
            continue
        summary[tag] = {}
        for metric in ("ssim", "ssim_local", "color_dE", "anisotropy"):
            vals = np.array([r[metric] for r in sub if r[metric] is not None])
            if len(vals) == 0:
                continue
            summary[tag][metric] = dict(mean=float(vals.mean()), std=float(vals.std()),
                                         n=len(vals))
    return summary


def print_summary(summary):
    genuine = summary.get("genuine", {})
    print(f"{'tag':18s} {'metric':10s} {'mean':>8s} {'std':>7s} {'z_vs_genuine':>13s}")
    for tag, metrics in summary.items():
        for metric, stats in metrics.items():
            z = ""
            if tag != "genuine" and metric in genuine and genuine[metric]["std"] > 1e-9:
                z = f"{(stats['mean'] - genuine[metric]['mean']) / genuine[metric]['std']:+.2f}"
            print(f"{tag:18s} {metric:10s} {stats['mean']:8.3f} {stats['std']:7.3f} {z:>13s}")
