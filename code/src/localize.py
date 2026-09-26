"""Register a candidate photo against one reference: ORB keypoint matching +
Lowe ratio test + RANSAC homography, then warp the candidate into the
reference's own pixel frame so the two become directly, pixel-for-pixel
comparable inside the reference's logo-lockup bbox.
"""
from dataclasses import dataclass
import cv2
import numpy as np

import config


@dataclass
class RegistrationResult:
    ok: bool
    reason: str = ""
    n_good_matches: int = 0
    n_inliers: int = 0
    inlier_ratio: float = 0.0
    H: np.ndarray = None            # candidate -> reference homography
    warped_bgr: np.ndarray = None   # candidate warped into reference's frame
    rectified_crop: np.ndarray = None  # warped_bgr cropped to reference.bbox
    coverage: float = 1.0            # fraction of the crop backed by real candidate
                                      # pixels rather than warpPerspective's black
                                      # fill (see the 009/041 false-positive writeup:
                                      # a tightly-framed photo can map part of the
                                      # reference's bbox to outside the candidate's
                                      # own frame, which looks like a structural
                                      # mismatch but is really a coverage problem)


_orb = cv2.ORB_create(nfeatures=config.ORB_N_FEATURES)
_bf = cv2.BFMatcher(cv2.NORM_HAMMING)


def register_against_reference(cand_bgr: np.ndarray, cand_gray: np.ndarray,
                                ref) -> RegistrationResult:
    kp_c, des_c = _orb.detectAndCompute(cand_gray, None)
    if des_c is None or len(kp_c) < config.MIN_MATCH_COUNT:
        return RegistrationResult(ok=False, reason="too few candidate keypoints")

    matches = _bf.knnMatch(ref.descriptors, des_c, k=2)
    good = [m for pair in matches if len(pair) == 2
            for m, n in [pair] if m.distance < config.LOWE_RATIO * n.distance]

    if len(good) < config.MIN_MATCH_COUNT:
        return RegistrationResult(ok=False, reason=f"too few good matches ({len(good)})",
                                   n_good_matches=len(good))

    src_ref = np.float32([ref.keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_cand = np.float32([kp_c[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(dst_cand, src_ref, cv2.RANSAC,
                                  config.RANSAC_REPROJ_THRESHOLD)
    if H is None or mask is None:
        return RegistrationResult(ok=False, reason="homography estimation failed",
                                   n_good_matches=len(good))

    n_inliers = int(mask.sum())
    inlier_ratio = n_inliers / len(good)
    if n_inliers < config.MIN_MATCH_COUNT:
        return RegistrationResult(ok=False, reason=f"too few RANSAC inliers ({n_inliers})",
                                   n_good_matches=len(good), n_inliers=n_inliers,
                                   inlier_ratio=inlier_ratio)

    ref_h, ref_w = ref.gray.shape
    warped = cv2.warpPerspective(cand_bgr, H, (ref_w, ref_h))
    x0, y0, x1, y1 = ref.bbox
    crop = warped[y0:y1, x0:x1]

    cand_h, cand_w = cand_gray.shape
    coverage_mask = np.ones((cand_h, cand_w), dtype=np.uint8) * 255
    warped_coverage = cv2.warpPerspective(coverage_mask, H, (ref_w, ref_h))
    crop_coverage = warped_coverage[y0:y1, x0:x1]
    coverage = float((crop_coverage > 127).mean()) if crop_coverage.size else 0.0

    return RegistrationResult(ok=True, n_good_matches=len(good), n_inliers=n_inliers,
                               inlier_ratio=inlier_ratio, H=H, warped_bgr=warped,
                               rectified_crop=crop, coverage=coverage)


def register_against_all(cand_bgr: np.ndarray, refs) -> list:
    cand_gray = cv2.cvtColor(cand_bgr, cv2.COLOR_BGR2GRAY)
    return [register_against_reference(cand_bgr, cand_gray, ref) for ref in refs]
