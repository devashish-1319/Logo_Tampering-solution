"""Compose a stamp onto a procedurally-generated background with a random,
plausible camera pose and JPEG recompression, producing a candidate-like
image with fully known ground truth (tamper type + magnitude), for pipeline
calibration. Backgrounds are generated from scratch (gradient + noise +
directional streaks) rather than reused from the real dataset, so the
calibration set is methodologically independent of the 50 unlabeled photos
it will be used to interpret.
"""
import numpy as np
import cv2

import config

RNG = np.random.default_rng()


def make_background(size_wh, rng):
    w, h = size_wh
    base = rng.uniform(0.55, 0.92)
    tint = rng.uniform(-0.08, 0.08, size=3)
    # smooth linear gradient across a random direction
    angle = rng.uniform(0, 2 * np.pi)
    xs, ys = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    grad = xs * np.cos(angle) + ys * np.sin(angle)
    grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-9)
    shade = base + (grad - 0.5) * rng.uniform(0.05, 0.25)
    img = np.zeros((h, w, 3), dtype=np.float64)
    for c in range(3):
        img[..., c] = np.clip(shade + tint[c], 0.05, 0.98)
    img = (img * 255).astype(np.uint8)

    # a subtle "brushed metal" directional streak texture, like several of
    # the real photos show
    if rng.random() < 0.5:
        streaks = rng.normal(0, 8, size=(h, w)).astype(np.float64)
        streaks = cv2.GaussianBlur(streaks, (0, 0), sigmaX=6, sigmaY=0.4)
        img = np.clip(img.astype(np.float64) + streaks[..., None], 0, 255).astype(np.uint8)

    # mild sensor-noise
    noise = rng.normal(0, rng.uniform(2, 6), size=img.shape)
    img = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    return img


def random_homography(size_wh, rng, max_rotation_deg=18, max_tilt_deg=22, scale_range=(0.8, 1.15)):
    """A homography approximating a phone photographing a flat label from a
    plausible handheld angle: in-plane rotation + a perspective tilt around
    a random axis + isotropic scale. This is what the geometric-anomaly
    score is calibrated against as 'explainable by camera pose'."""
    w, h = size_wh
    cx, cy = w / 2, h / 2

    rot = np.radians(rng.uniform(-max_rotation_deg, max_rotation_deg))
    scale = rng.uniform(*scale_range)
    R = np.array([[np.cos(rot), -np.sin(rot), 0],
                  [np.sin(rot), np.cos(rot), 0],
                  [0, 0, 1]], dtype=np.float64)
    S = np.diag([scale, scale, 1.0])

    tilt = np.radians(rng.uniform(-max_tilt_deg, max_tilt_deg))
    tilt_axis = rng.uniform(0, 2 * np.pi)
    # simple synthetic perspective: push two opposite corners along the tilt axis
    strength = np.sin(tilt) * 0.35
    dx, dy = np.cos(tilt_axis), np.sin(tilt_axis)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    push = np.float32([
        [-dx, -dy], [dx, dy], [dx, dy], [-dx, -dy]
    ]) * strength * min(w, h)
    dst = (src + push).astype(np.float32)
    P = cv2.getPerspectiveTransform(src, dst)

    T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
    T2 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
    H = T2 @ P @ S @ R @ T1
    return H


def composite(stamp_bgra, canvas_wh, rng, position_frac=None):
    cw, ch = canvas_wh
    bg = make_background(canvas_wh, rng)

    sh, sw = stamp_bgra.shape[:2]
    if position_frac is None:
        position_frac = (rng.uniform(0.12, 0.35), rng.uniform(0.30, 0.55))
    px = int(position_frac[0] * cw)
    py = int(position_frac[1] * ch)

    canvas_bgra = np.zeros((ch, cw, 4), dtype=np.uint8)
    canvas_bgra[..., :3] = bg
    x1, y1 = min(px + sw, cw), min(py + sh, ch)
    sw_c, sh_c = x1 - px, y1 - py
    if sw_c <= 0 or sh_c <= 0:
        raise ValueError("stamp does not fit on canvas at requested position")

    stamp_region = stamp_bgra[:sh_c, :sw_c]
    alpha = (stamp_region[..., 3:4].astype(np.float64)) / 255.0
    bg_region = canvas_bgra[py:y1, px:x1, :3].astype(np.float64)
    blended = stamp_region[..., :3].astype(np.float64) * alpha + bg_region * (1 - alpha)
    canvas_bgra[py:y1, px:x1, :3] = blended.astype(np.uint8)
    canvas_bgra[py:y1, px:x1, 3] = stamp_region[..., 3]

    H = random_homography(canvas_wh, rng)
    warped = cv2.warpPerspective(canvas_bgra[..., :3], H, canvas_wh,
                                  borderMode=cv2.BORDER_REPLICATE)
    alpha_full = np.zeros((ch, cw), dtype=np.uint8)
    alpha_full[py:y1, px:x1] = canvas_bgra[py:y1, px:x1, 3]
    warped_alpha = cv2.warpPerspective(alpha_full, H, canvas_wh)

    # slight focus blur, like a handheld phone shot
    if rng.random() < 0.7:
        k = int(rng.choice([0, 3, 3, 5]))
        if k > 0:
            warped = cv2.GaussianBlur(warped, (k, k), 0)

    quality = int(rng.integers(70, 96))
    ok, enc = cv2.imencode(".jpg", warped, [cv2.IMWRITE_JPEG_QUALITY, quality])
    warped_jpeg = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return warped_jpeg
