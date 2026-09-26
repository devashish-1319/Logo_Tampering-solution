"""Build an isolated RGBA 'stamp' of the clean NORVIA lockup from a
reference image (green ink -> opaque, everything else in the bbox ->
transparent), and apply known, labelled perturbations to it: this is the
synthetic-tampering benchmark data described in the write-up (Christlein et
al.'s own benchmark-construction method; SLSG's 'simulated anomaly'
calibration logic) -- since the real 50 images have no ground truth, we
manufacture a small set where the tamper type and magnitude are known, to
see whether the pipeline's scores actually separate genuine from tampered
before trusting them on the real data.
"""
import numpy as np
import cv2

import config
from logo_mask import green_mask, logo_bbox_with_padding


def extract_stamp(ref_bgr: np.ndarray, pad_frac: float = 0.06):
    """Returns (stamp_bgra, split_x) where split_x is the local x-coordinate
    (within the stamp) separating the leaf icon from the wordmark block --
    reused later to simulate icon/wordmark respacing."""
    rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
    bbox = logo_bbox_with_padding(rgb, pad_frac)
    x0, y0, x1, y1 = bbox
    crop_bgr = ref_bgr[y0:y1, x0:x1]
    crop_rgb = rgb[y0:y1, x0:x1]
    mask = green_mask(crop_rgb)

    # small morphological close (numpy/opencv, no scipy) to avoid speckled
    # alpha edges from anti-aliasing
    mask_u8 = (mask.astype(np.uint8)) * 255
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    alpha = mask_u8

    bgra = np.dstack([crop_bgr, alpha])

    # icon occupies roughly the left ~22% of the bbox width (same heuristic
    # used during EDA / logo_mask windowing)
    split_x = int(bgra.shape[1] * 0.22)
    return bgra, split_x


def stamp_to_rgb_mask(stamp_bgra: np.ndarray):
    bgr = stamp_bgra[..., :3]
    alpha = stamp_bgra[..., 3] > 0
    return bgr, alpha


# ---------------------------------------------------------------- tampers --

def tamper_stretch(stamp_bgra: np.ndarray, split_x: int, factor: float):
    """Anisotropic horizontal stretch of the *wordmark* only (icon kept at
    its original proportions, as a logo-asset edit would plausibly do)."""
    h, w = stamp_bgra.shape[:2]
    icon = stamp_bgra[:, :split_x]
    word = stamp_bgra[:, split_x:]
    new_w = max(1, int(word.shape[1] * factor))
    word_stretched = cv2.resize(word, (new_w, h), interpolation=cv2.INTER_LINEAR)
    return np.concatenate([icon, word_stretched], axis=1)


def tamper_recolor(stamp_bgra: np.ndarray, split_x: int, hue_shift_deg: float):
    bgr = stamp_bgra[..., :3].copy()
    alpha = stamp_bgra[..., 3]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.int16)
    shift = int(hue_shift_deg / 2)  # opencv hue is 0-179
    hsv[..., 0] = (hsv[..., 0] + shift) % 180
    bgr2 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    out = stamp_bgra.copy()
    out[..., :3] = bgr2
    return out


def tamper_respace(stamp_bgra: np.ndarray, split_x: int, gap_px: int):
    """Opens/closes the gap between the icon and the wordmark block by
    gap_px (positive = more space, negative = tighter), independent of any
    stretch -- a pure re-spacing edit."""
    h, w = stamp_bgra.shape[:2]
    icon = stamp_bgra[:, :split_x]
    word = stamp_bgra[:, split_x:]
    if gap_px >= 0:
        filler = np.zeros((h, gap_px, 4), dtype=stamp_bgra.dtype)
        return np.concatenate([icon, filler, word], axis=1)
    else:
        cut = min(-gap_px, word.shape[1] - 5)
        return np.concatenate([icon, word[:, cut:]], axis=1)


def tamper_letterform_swap(stamp_bgra: np.ndarray, split_x: int, letter_index: int = 2,
                            n_letters: int = 6):
    """Approximates a look-alike-glyph swap by horizontally mirroring one
    letter's own glyph in place (plausible shape, wrong identity) -- we
    don't have access to actual alternate-font glyphs, so this is a
    deliberately simplified stand-in; documented as such in the write-up."""
    out = stamp_bgra.copy()
    word = out[:, split_x:]
    letter_w = word.shape[1] // n_letters
    lx0 = letter_index * letter_w
    lx1 = lx0 + letter_w if letter_index < n_letters - 1 else word.shape[1]
    word[:, lx0:lx1] = word[:, lx0:lx1][:, ::-1]
    return out
