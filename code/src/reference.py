"""Load the 3 reference images once, precompute their ORB keypoints and each
one's logo-lockup bbox (in its own pixel coordinates)."""
from dataclasses import dataclass
import cv2
import numpy as np

import config
from logo_mask import logo_bbox_with_padding


@dataclass
class ReferenceEntry:
    name: str
    bgr: np.ndarray
    gray: np.ndarray
    keypoints: list
    descriptors: np.ndarray
    bbox: tuple  # (x0, y0, x1, y1) padded logo-lockup bbox, this ref's own frame


def load_references(n_features: int = config.ORB_N_FEATURES):
    orb = cv2.ORB_create(nfeatures=n_features)
    refs = []
    for name in config.REFERENCE_FILES:
        path = config.REFERENCE_DIR / name
        bgr = cv2.imread(str(path))
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        bbox = logo_bbox_with_padding(rgb, config.LOGO_PAD_FRAC)
        refs.append(ReferenceEntry(name=name, bgr=bgr, gray=gray, keypoints=kp,
                                    descriptors=des, bbox=bbox))
    return refs
