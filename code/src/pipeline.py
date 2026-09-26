"""Per-image pipeline: localize -> rectify -> colour-correct -> score.

For each candidate photo we register against all 3 references independently
(each is an separate, valid "genuine" exemplar) and aggregate with the
median, which is more robust to one reference's registration being noisy
than either picking a single reference or averaging.
"""
from dataclasses import dataclass, field
import numpy as np

import config
from localize import register_against_reference
import color_correct as cc
import structural
import geometry


def prepare_reference_colors(refs):
    """Precompute each reference's own colour-corrected crop + dominant
    green Lab colour, once, so every candidate is compared against the same
    fixed target."""
    for ref in refs:
        x0, y0, x1, y1 = ref.bbox
        crop = ref.bgr[y0:y1, x0:x1]
        corrected = cc.correct(ref.bgr, crop)
        ref.corrected_crop = corrected
        ref.green_lab = cc.dominant_green_lab(corrected)
    return refs


@dataclass
class PerReferenceScore:
    ref_name: str
    ok: bool
    reason: str = ""
    n_inliers: int = 0
    inlier_ratio: float = 0.0
    coverage: float = None
    ssim: float = None
    ssim_local: float = None
    color_delta_e: float = None
    anisotropy: float = None
    scale_major: float = None
    scale_minor: float = None
    rotation_deg: float = None


@dataclass
class ImageResult:
    name: str
    n_refs_ok: int
    per_reference: list = field(default_factory=list)
    ssim_median: float = None
    ssim_local_median: float = None
    color_delta_e_median: float = None
    anisotropy_median: float = None
    coverage_min: float = None
    low_coverage: bool = False
    best_ref: str = None
    debug_crop_bgr: np.ndarray = None  # from the best-registered reference


def score_against_reference(cand_bgr, cand_gray, ref) -> PerReferenceScore:
    reg = register_against_reference(cand_bgr, cand_gray, ref)
    if not reg.ok:
        return PerReferenceScore(ref_name=ref.name, ok=False, reason=reg.reason,
                                  n_inliers=reg.n_inliers, inlier_ratio=reg.inlier_ratio)

    corrected_crop = cc.correct(reg.warped_bgr, reg.rectified_crop)
    ssim_score, ssim_local, _ = structural.compare(corrected_crop, ref.corrected_crop)

    cand_lab = cc.dominant_green_lab(corrected_crop)
    de = cc.color_delta_e(cand_lab, ref.green_lab)

    geo = geometry.anisotropy_from_homography(reg.H, ref.bbox)

    return PerReferenceScore(ref_name=ref.name, ok=True, n_inliers=reg.n_inliers,
                              inlier_ratio=reg.inlier_ratio, coverage=reg.coverage,
                              ssim=ssim_score,
                              ssim_local=ssim_local,
                              color_delta_e=de, anisotropy=geo["anisotropy"],
                              scale_major=geo["scale_major"], scale_minor=geo["scale_minor"],
                              rotation_deg=geo["rotation_deg"]), corrected_crop


def process_image(name: str, cand_bgr: np.ndarray, refs) -> ImageResult:
    import cv2
    cand_gray = cv2.cvtColor(cand_bgr, cv2.COLOR_BGR2GRAY)

    per_ref = []
    crops = {}
    for ref in refs:
        result = score_against_reference(cand_bgr, cand_gray, ref)
        if isinstance(result, tuple):
            score, crop = result
            crops[ref.name] = crop
        else:
            score = result
        per_ref.append(score)

    ok_scores = [s for s in per_ref if s.ok]
    res = ImageResult(name=name, n_refs_ok=len(ok_scores), per_reference=per_ref)

    if ok_scores:
        res.ssim_median = float(np.median([s.ssim for s in ok_scores]))
        res.ssim_local_median = float(np.median([s.ssim_local for s in ok_scores]))
        des = [s.color_delta_e for s in ok_scores if s.color_delta_e is not None]
        res.color_delta_e_median = float(np.median(des)) if des else None
        res.anisotropy_median = float(np.median([s.anisotropy for s in ok_scores]))
        cov_vals = [s.coverage for s in ok_scores if s.coverage is not None]
        res.coverage_min = float(min(cov_vals)) if cov_vals else None
        res.low_coverage = res.coverage_min is not None and res.coverage_min < config.MIN_COVERAGE
        best = max(ok_scores, key=lambda s: s.n_inliers)
        res.best_ref = best.ref_name
        res.debug_crop_bgr = crops.get(best.ref_name)

    return res
