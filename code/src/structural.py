"""SSIM (Wang, Bovik, Sheikh & Simoncelli, 2004), implemented directly from
the paper's own equations (11x11 circular-symmetric Gaussian window,
sigma=1.5, K1=0.01, K2=0.03) using cv2.GaussianBlur for the local
statistics. Implemented by hand rather than pulled from scikit-image: this
environment hit a broken scipy binary (PROPACK dlopen failure on this
macOS/arm64 + Python combination) that skimage.metrics pulls in transitively
through scipy.ndimage -- rather than depend on that fragile chain for
something this well-specified, it's reproduced directly against the paper,
which also keeps the dependency footprint to just OpenCV + numpy.

Only meaningful once the two images are pixel-aligned -- SSIM has no
tolerance for misregistration, which is why this always runs downstream of
localize.py.
"""
import cv2
import numpy as np


def _ssim_gray(a: np.ndarray, b: np.ndarray, win_size: int = 11, sigma: float = 1.5):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    L = 255.0
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    def blur(x):
        return cv2.GaussianBlur(x, (win_size, win_size), sigma)

    mu_a = blur(a)
    mu_b = blur(b)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b

    sigma_a2 = blur(a * a) - mu_a2
    sigma_b2 = blur(b * b) - mu_b2
    sigma_ab = blur(a * b) - mu_ab

    numerator = (2 * mu_ab + C1) * (2 * sigma_ab + C2)
    denominator = (mu_a2 + mu_b2 + C1) * (sigma_a2 + sigma_b2 + C2)
    ssim_map = numerator / denominator
    return float(ssim_map.mean()), ssim_map


def compare(crop_a_bgr: np.ndarray, crop_b_bgr: np.ndarray):
    """crop_a is resized onto crop_b's shape if they differ (can happen when
    two references have very slightly different bbox sizes after padding)."""
    h, w = crop_b_bgr.shape[:2]
    if crop_a_bgr.shape[:2] != (h, w):
        crop_a_bgr = cv2.resize(crop_a_bgr, (w, h), interpolation=cv2.INTER_LINEAR)

    gray_a = cv2.cvtColor(crop_a_bgr, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(crop_b_bgr, cv2.COLOR_BGR2GRAY)

    win_size = 11
    if min(gray_a.shape) < win_size:
        win_size = min(gray_a.shape) if min(gray_a.shape) % 2 == 1 else min(gray_a.shape) - 1
        win_size = max(3, win_size)

    mean_score, ssim_map = _ssim_gray(gray_a, gray_b, win_size=win_size)
    # A global mean dilutes a small localised anomaly (one respaced gap, one
    # swapped letter) across a crop that is mostly unchanged icon/background.
    # A low-percentile value of the map captures "how bad does it get
    # somewhere", which is far more sensitive to that failure mode -- but the
    # very lowest percentiles are dominated by border/window artefacts that
    # are near-zero even for a perfect genuine match, so (a) a margin is
    # trimmed off the map's edges first and (b) the 20th percentile is used
    # rather than an extreme one, trading some sensitivity for a floor that
    # genuine matches don't already sit on.
    mh, mw = ssim_map.shape
    my, mx = int(mh * 0.12), int(mw * 0.06)
    trimmed = ssim_map[my:mh - my, mx:mw - mx] if mh > 2 * my and mw > 2 * mx else ssim_map
    p20_score = float(np.percentile(trimmed, 20))
    return mean_score, p20_score, ssim_map
