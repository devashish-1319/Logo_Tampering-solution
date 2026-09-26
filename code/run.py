#!/usr/bin/env python3
"""Single entry point: run the logo-tampering detection pipeline on the real
50-image dataset, run the synthetic calibration benchmark, combine the two
into a per-image confidence report, and save everything under outputs/.

Usage:
    python run.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import cv2
import numpy as np
import config
from reference import load_references
from pipeline import prepare_reference_colors, process_image
import calibrate
import report as report_mod


def run_real_images(refs):
    rows = []
    crops = {}
    for path in sorted(config.IMAGES_DIR.glob("*.jpg")):
        bgr = cv2.imread(str(path))
        res = process_image(path.name, bgr, refs)
        rows.append(dict(
            name=path.name, n_ok=res.n_refs_ok, best_ref=res.best_ref,
            ssim=res.ssim_median, ssim_local=res.ssim_local_median,
            color_dE=res.color_delta_e_median, anisotropy=res.anisotropy_median,
            coverage_min=res.coverage_min, low_coverage=res.low_coverage,
        ))
        if res.low_coverage:
            print(f"  [low frame coverage: {res.coverage_min:.0%}] {path.name} -- "
                  f"scores below are unreliable, see README")
        if res.debug_crop_bgr is not None:
            crops[path.name] = res.debug_crop_bgr
        if res.n_refs_ok == 0:
            reasons = [(s.ref_name, s.reason) for s in res.per_reference]
            print(f"  [localization FAILED] {path.name}: {reasons}")
    return rows, crops


def main():
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    config.DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading references...")
    refs = load_references()
    prepare_reference_colors(refs)

    print("\n[1/3] Scoring the 50 real candidate images...")
    t0 = time.time()
    real_rows, real_crops = run_real_images(refs)
    print(f"  done in {time.time() - t0:.1f}s ({len(real_rows)} images)")

    print("\n[2/3] Generating + scoring the synthetic calibration benchmark...")
    t0 = time.time()
    synth_rows = calibrate.generate_and_score(
        refs, seed=0,
        out_csv=config.OUTPUTS_DIR / "synthetic_calibration.csv",
        debug_dir=config.DEBUG_DIR,
    )
    synth_summary = calibrate.summarize(synth_rows)
    print(f"  done in {time.time() - t0:.1f}s ({len(synth_rows)} synthetic examples)")
    print()
    calibrate.print_summary(synth_summary)

    print("\n[3/3] Comparing real-image scores to the synthetic-genuine baseline...")
    genuine_stats = report_mod.genuine_stats(synth_summary)
    print("  Real-vs-synthetic offset (sanity check before trusting any threshold transfer):")
    for line in report_mod.compare_domains(real_rows, genuine_stats):
        print(line)
    print("  -> see write-up: where this offset is large, it reflects a synthetic/real")
    print("     registration-quality gap, not real-image anomalousness. Real images are")
    print("     therefore ranked against EACH OTHER below, not against synthetic absolutes.")

    flagged, real_pop_stats = report_mod.score_real_images_within_population(real_rows)
    fieldnames = ["name", "n_ok", "best_ref", "coverage_min", "low_coverage",
                  "inconclusive", "ssim", "ssim_local", "color_dE",
                  "anisotropy", "z_ssim", "z_ssim_local", "z_color_dE",
                  "z_anisotropy", "n_flags"]
    report_mod.write_csv(flagged, config.OUTPUTS_DIR / "real_scores.csv", fieldnames)

    inconclusive = [r for r in flagged if r["inconclusive"]]
    ranked = [r for r in flagged if not r["inconclusive"]]
    if inconclusive:
        print(f"\n{len(inconclusive)} image(s) marked INCONCLUSIVE (frame coverage < "
              f"{config.MIN_COVERAGE:.0%} -- rectification pulled in content from outside "
              f"the photo, see README) and excluded from ranking below:")
        for r in inconclusive:
            print(f"  {r['name']:10s} coverage={r['coverage_min']:.0%}")

    flagged_sorted = sorted(ranked, key=lambda r: -r["n_flags"])
    print("\nTop 10 remaining real images by flag count (metrics >1.5 MAD from the REAL "
          "population's own median, in the anomalous direction -- i.e. unusual relative to "
          "the other clean-coverage photos, not a tamper verdict):")
    print(f"{'name':10s} {'flags':>6s} {'ssim':>7s} {'ssim_loc':>9s} {'dE':>7s} {'aniso':>7s}")
    for r in flagged_sorted[:10]:
        print(f"{r['name']:10s} {r['n_flags']:6d} {r['ssim']:7.3f} {r['ssim_local']:9.3f} "
              f"{r['color_dE']:7.2f} {r['anisotropy']:7.3f}")

    top_dir = config.DEBUG_DIR / "top_flagged"
    top_dir.mkdir(exist_ok=True)
    ref_by_name = {r.name: r for r in refs}
    for rank, row in enumerate(flagged_sorted[:8], start=1):
        name = row["name"]
        if name not in real_crops or row["best_ref"] not in ref_by_name:
            continue
        cand_crop = real_crops[name]
        ref_crop = ref_by_name[row["best_ref"]].corrected_crop
        h = max(cand_crop.shape[0], ref_crop.shape[0])
        pad = lambda im: cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, 0, cv2.BORDER_CONSTANT)
        side_by_side = cv2.hconcat([pad(ref_crop), np.full((h, 8, 3), 255, dtype="uint8"),
                                     pad(cand_crop)])
        out_path = top_dir / f"rank{rank}_{name.replace('.jpg', '')}_vs_{row['best_ref']}"
        cv2.imwrite(str(out_path.with_suffix(".png")), side_by_side)

    print(f"\nWrote:\n  {config.OUTPUTS_DIR / 'real_scores.csv'}\n"
          f"  {config.OUTPUTS_DIR / 'synthetic_calibration.csv'}\n"
          f"  top-flagged reference-vs-candidate crops in {top_dir}")


if __name__ == "__main__":
    main()
