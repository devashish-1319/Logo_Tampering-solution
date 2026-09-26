"""Green-pixel segmentation of the NORVIA wordmark/icon, used only on
near-frontal, single-plane images (references, and synthetic assets) where
there is no background clutter of a similar hue to confuse it. This is
deliberately NOT used on raw candidate photos (see write-up: a naive global
colour mask is swamped by tinted backgrounds and pose) -- it is only ever
applied inside a rectified frame where the logo's rough location is already
known from feature-based registration.
"""
import numpy as np


def green_mask(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    return (g > r + 12) & (g > b + 12) & (g > 50) & (g < 225)


def largest_component_bbox(mask: np.ndarray, min_pixels: int = 30):
    """Flood-fill connected components (pure numpy, no scipy dependency),
    return the bbox (x0, y0, x1, y1) of the largest one, or None."""
    H, W = mask.shape
    ys, xs = np.where(mask)
    coords = list(zip(ys.tolist(), xs.tolist()))
    coord_set = set(coords)
    best = None
    best_len = 0
    seen = set()
    for y, x in coords:
        if (y, x) in seen:
            continue
        stack = [(y, x)]
        seen.add((y, x))
        comp = [(y, x)]
        while stack:
            cy, cx = stack.pop()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = cy + dy, cx + dx
                if (ny, nx) in coord_set and (ny, nx) not in seen:
                    seen.add((ny, nx))
                    stack.append((ny, nx))
                    comp.append((ny, nx))
        if len(comp) > best_len:
            best_len = len(comp)
            best = comp
    if best is None or best_len < min_pixels:
        return None
    comp_arr = np.array(best)
    y0, x0 = comp_arr.min(axis=0)
    y1, x1 = comp_arr.max(axis=0)
    return int(x0), int(y0), int(x1), int(y1)


def all_components(mask: np.ndarray, min_pixels: int = 8):
    """Return bboxes of every connected component with >= min_pixels pixels."""
    ys, xs = np.where(mask)
    coord_set = set(zip(ys.tolist(), xs.tolist()))
    seen = set()
    boxes = []
    for y, x in coord_set:
        if (y, x) in seen:
            continue
        stack = [(y, x)]
        seen.add((y, x))
        comp = [(y, x)]
        while stack:
            cy, cx = stack.pop()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = cy + dy, cx + dx
                if (ny, nx) in coord_set and (ny, nx) not in seen:
                    seen.add((ny, nx))
                    stack.append((ny, nx))
                    comp.append((ny, nx))
        if len(comp) >= min_pixels:
            arr = np.array(comp)
            y0, x0 = arr.min(axis=0)
            y1, x1 = arr.max(axis=0)
            boxes.append((int(x0), int(y0), int(x1), int(y1), len(comp)))
    return boxes


def logo_lockup_bbox(rgb: np.ndarray):
    """The wordmark is made of disconnected letter glyphs, so the single
    largest connected green component is usually just the leaf icon (one
    contiguous blob), not the whole lockup. We use that icon as an anchor,
    then take the tight bbox of ALL green pixels within a generous window
    around it (icon + wordmark + tagline sit close together; anything green
    elsewhere in the frame is excluded by restricting to that window first).
    """
    mask = green_mask(rgb)
    anchor = largest_component_bbox(mask)
    if anchor is None:
        return None
    ax0, ay0, ax1, ay1 = anchor
    aw, ah = ax1 - ax0, ay1 - ay0
    icon_size = (ax1 - ax0) * (ay1 - ay0)
    H, W = rgb.shape[:2]
    # generous window: wordmark+tagline extend well to the right and below
    # the icon glyph; expand asymmetrically to capture that.
    wx0 = max(0, ax0 - aw)
    wx1 = min(W, ax1 + aw * 9)
    wy0 = max(0, ay0 - ah)
    wy1 = min(H, ay1 + ah * 3)
    window_mask = np.zeros_like(mask)
    window_mask[wy0:wy1, wx0:wx1] = mask[wy0:wy1, wx0:wx1]

    # Individual letters are separate components, much smaller than the
    # icon; stray anti-aliasing specks are smaller still. Keep components
    # that are at least a token fraction of the icon's pixel count so a
    # handful of noise pixels elsewhere in the window can't blow the bbox
    # out to the whole window.
    min_component_pixels = max(15, int(0.01 * mask[ay0:ay1, ax0:ax1].sum()))
    boxes = all_components(window_mask, min_pixels=min_component_pixels)
    if not boxes:
        return anchor
    xs0 = min(b[0] for b in boxes)
    ys0 = min(b[1] for b in boxes)
    xs1 = max(b[2] for b in boxes)
    ys1 = max(b[3] for b in boxes)
    return xs0, ys0, xs1, ys1


def logo_bbox_with_padding(rgb: np.ndarray, pad_frac: float):
    bbox = logo_lockup_bbox(rgb)
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    px, py = int(w * pad_frac), int(h * pad_frac)
    H, W = rgb.shape[:2]
    x0 = max(0, x0 - px)
    y0 = max(0, y0 - py)
    x1 = min(W - 1, x1 + px)
    y1 = min(H - 1, y1 + py)
    return x0, y0, x1, y1
