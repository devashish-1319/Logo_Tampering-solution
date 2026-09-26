"""Decompose the candidate->reference homography into an anisotropic-scale
("stretch") signal.

IMPORTANT identifiability caveat (documented here because it drives how this
score is used downstream, not just for the record): a homography has 8 DOF,
which is exactly enough to explain ANY combination of rotation, uniform
scale, anisotropic scale and shear between two views of a single flat
surface. That means a genuine logo photographed at a steep, skewed angle and
a genuinely pre-stretched logo photographed head-on can produce numerically
IDENTICAL homographies -- this signal cannot, by itself, tell those two
cases apart from the logo's own keypoints alone. What it *can* do is flag
"the projected shape deviates from the reference by more than plausible
camera-angle foreshortening would produce", which is why the raw anisotropy
ratio is only meaningful once calibrated against a distribution of
known-genuine images photographed at realistic angles (see synthetic_bench).
"""
import numpy as np


def _jacobian_at(H_inv: np.ndarray, point_xy, eps: float = 1.0) -> np.ndarray:
    """Finite-difference Jacobian of the homography map H_inv at point_xy
    (mapping reference coords -> candidate coords), i.e. how a small patch
    of reference-space is locally scaled/rotated/sheared in candidate space.
    """
    x, y = point_xy

    def apply(px, py):
        v = H_inv @ np.array([px, py, 1.0])
        return np.array([v[0] / v[2], v[1] / v[2]])

    fx = (apply(x + eps, y) - apply(x - eps, y)) / (2 * eps)
    fy = (apply(x, y + eps) - apply(x, y - eps)) / (2 * eps)
    return np.column_stack([fx, fy])  # 2x2: columns are d(out)/dx, d(out)/dy


def anisotropy_from_homography(H: np.ndarray, ref_bbox) -> dict:
    """H maps candidate -> reference. Evaluate local shape distortion at the
    centre of the reference's logo bbox."""
    x0, y0, x1, y1 = ref_bbox
    center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    H_inv = np.linalg.inv(H)
    J = _jacobian_at(H_inv, center)  # ref -> candidate, local linear map

    # Polar/SVD decomposition: J = U S V^T
    U, S, Vt = np.linalg.svd(J)
    s_major, s_minor = float(S[0]), float(S[1])
    anisotropy = s_major / s_minor if s_minor > 1e-9 else float("inf")

    rotation = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
    return dict(anisotropy=anisotropy, scale_major=s_major, scale_minor=s_minor,
                rotation_deg=float(rotation))
