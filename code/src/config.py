"""Paths and shared constants."""
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = CODE_DIR.parent
DATA_DIR = ROOT_DIR / "candidate_package"
IMAGES_DIR = DATA_DIR / "images"
REFERENCE_DIR = DATA_DIR / "reference"
OUTPUTS_DIR = CODE_DIR / "outputs"
DEBUG_DIR = OUTPUTS_DIR / "debug"

REFERENCE_FILES = ["ref_1.jpg", "ref_2.jpg", "ref_3.jpg"]

# Canonical frame every candidate gets rectified into (pixels).
CANON_SIZE = (900, 525)  # (width, height), ~ the reference images' own aspect ratio

# Padding (fraction of bbox size) kept around the green-mask logo bbox when
# defining the "logo region" used for scoring.
LOGO_PAD_FRAC = 0.18

# Below this fraction of the reference bbox being backed by real candidate
# pixels (rest is warpPerspective's black fill from a too-tight photo crop),
# treat the registration as unreliable rather than scoring it normally.
MIN_COVERAGE = 0.90

ORB_N_FEATURES = 4000
RANSAC_REPROJ_THRESHOLD = 4.0
MIN_MATCH_COUNT = 12
LOWE_RATIO = 0.75
