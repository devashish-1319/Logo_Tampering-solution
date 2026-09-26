"""Gray-world illuminant correction (Gijsenij et al. 2011 survey, static
methods family) plus a colour-distance metric on the logo's own green ink,
so a camera white-balance shift isn't mistaken for a recolour tamper.

Gray-world assumes the average scene reflectance is neutral grey; that's a
weak assumption for a single tightly-cropped logo (mostly green ink + a
plain background), so correction is computed from the wider *warped full
frame* (background + logo + any print elements), where the assumption holds
much better, then the resulting per-channel gains are applied to the logo
crop specifically.
"""
import numpy as np
import cv2


def gray_world_gains(bgr: np.ndarray) -> np.ndarray:
    means = bgr.reshape(-1, 3).mean(axis=0).astype(np.float64)  # B,G,R
    means = np.clip(means, 1e-6, None)
    target = means.mean()
    return target / means  # gains per channel, B,G,R order


def apply_gains(bgr: np.ndarray, gains: np.ndarray) -> np.ndarray:
    out = bgr.astype(np.float64) * gains.reshape(1, 1, 3)
    return np.clip(out, 0, 255).astype(np.uint8)


def correct(full_frame_bgr: np.ndarray, crop_bgr: np.ndarray) -> np.ndarray:
    gains = gray_world_gains(full_frame_bgr)
    return apply_gains(crop_bgr, gains)


def dominant_green_lab(bgr: np.ndarray) -> np.ndarray:
    """Mean CIELAB colour of the green-ink pixels in a (colour-corrected)
    logo crop. Returns None if no green pixels are found."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.int16)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mask = (g > r + 12) & (g > b + 12) & (g > 40)
    if mask.sum() < 20:
        return None
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    return lab[mask].mean(axis=0)  # L, a, b


def color_delta_e(lab_a: np.ndarray, lab_b: np.ndarray) -> float:
    """Plain Euclidean CIE76 Delta-E between two mean Lab colours."""
    if lab_a is None or lab_b is None:
        return None
    return float(np.linalg.norm(lab_a - lab_b))
