# --- color_score.py  (LAB-based) ---
#
# Why LAB?
#   • L* is lightness — we can ignore it and compare only a*b* (chroma).
#   • a*b* distance is perceptually meaningful: similar colours cluster
#     tightly regardless of the lighting level in the photo.
#   • No hand-tuned hue-tolerance constants needed; a single Euclidean
#     distance threshold does the job.

import cv2
import numpy as np
from dataclasses import dataclass, field

# Pixels whose LAB chroma (sqrt(a²+b²)) is below this are considered
# achromatic (white / grey / black) and excluded from the profile.
CHROMA_THRESH = 12.0

# When scoring a crop, only pixels whose chroma exceeds this threshold
# are used. Keeps glare / neutral background from polluting the score.
SCORE_CHROMA_THRESH = 10.0

# Maximum a*b* Euclidean distance to count a pixel as "matching" the
# template colour profile.  ~25 covers typical photo-vs-sprite variation.
LAB_MATCH_RADIUS = 30.0

# Minimum coloured pixels required to build / use a profile.
MIN_CHROMA_PIXELS = 10


@dataclass
class ColorProfile:
    label: str
    clusters: list = field(default_factory=list)   # [(a, b, weight), …]


# ── profile builder ───────────────────────────────────────────────────────────
def _chroma_pixels_lab(img_bgr, alpha_mask=None):
    """Return (a*, b*) arrays for every sufficiently chromatic pixel."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    # OpenCV encodes Lab as: L in [0,255], a/b in [0,255] with 128 = neutral
    a = lab[:, :, 1] - 128.0
    b = lab[:, :, 2] - 128.0

    chroma = np.sqrt(a ** 2 + b ** 2)

    if alpha_mask is not None:
        mask = alpha_mask & (chroma > CHROMA_THRESH)
    else:
        not_black = np.any(img_bgr > 20, axis=2)
        mask = not_black & (chroma > CHROMA_THRESH)

    return a[mask], b[mask]


def build_color_profiles(template_dict):
    """
    Build a LAB colour profile for every template.

    Strategy: k-means (k=3) on the a*b* pixels, then store the cluster
    centres + their relative weights.  Falls back to a single weighted
    centroid if there are too few pixels for k-means.
    """
    profiles = {}

    for label, path in template_dict.items():
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        if img.ndim == 3 and img.shape[2] == 4:
            alpha_mask = img[:, :, 3] > 10
            img_bgr = img[:, :, :3]
        else:
            alpha_mask = None
            img_bgr = img

        a_vals, b_vals = _chroma_pixels_lab(img_bgr, alpha_mask)

        if len(a_vals) < MIN_CHROMA_PIXELS:
            continue

        pts = np.column_stack([a_vals, b_vals]).astype(np.float32)

        k = min(3, len(pts))
        if k >= 2:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
            _, labels, centres = cv2.kmeans(
                pts, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS
            )
            counts = np.bincount(labels.flatten(), minlength=k)
            total  = counts.sum()
            clusters = [
                (float(centres[i, 0]), float(centres[i, 1]), counts[i] / total)
                for i in range(k)
            ]
        else:
            # Single centroid fallback
            clusters = [(float(a_vals.mean()), float(b_vals.mean()), 1.0)]

        profiles[label] = ColorProfile(label, clusters)

    return profiles


# ── perspective-aligned crop (unchanged API) ──────────────────────────────────

def extract_aligned_patch(frame_bgr, quad, size=(240, 240)):
    dst = np.float32([
        [0,        0       ],
        [size[0],  0       ],
        [size[0],  size[1] ],
        [0,        size[1] ],
    ])
    H = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(frame_bgr, H, size)


# ── per-detection scorer ──────────────────────────────────────────────────────

def _lab_match_score(crop_bgr, profile):
    """
    Fraction of chromatic pixels in *crop* whose a*b* colour falls within
    LAB_MATCH_RADIUS of at least one cluster centre in *profile*.

    Returns a float in [0, 1].
    """
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    a = lab[:, :, 1] - 128.0
    b = lab[:, :, 2] - 128.0

    chroma = np.sqrt(a ** 2 + b ** 2)
    mask   = chroma > SCORE_CHROMA_THRESH

    if mask.sum() < MIN_CHROMA_PIXELS:
        return 0.0

    a_c = a[mask]
    b_c = b[mask]

    # A pixel "matches" if it's close to ANY cluster centre.
    matched = np.zeros(a_c.shape, dtype=bool)
    for ca, cb, _w in profile.clusters:
        dist = np.sqrt((a_c - ca) ** 2 + (b_c - cb) ** 2)
        matched |= dist < LAB_MATCH_RADIUS

    return float(matched.sum() / mask.sum())


def color_score(frame_bgr, quad, profile):
    """Public API: score one detection quad against a colour profile."""
    patch = extract_aligned_patch(frame_bgr, quad)
    return _lab_match_score(patch, profile)


# ── conflict resolver (same API as before) ────────────────────────────────────

def resolve_conflicts_color(detections, frame_bgr, profiles):
    """
    For each group of detections that share a base label (e.g. 'meditite'
    and 'meditite_shiny'), keep only the one with the highest LAB colour
    score.
    """
    groups = {}
    for det in detections:
        base = det.label.replace("_shiny", "")
        groups.setdefault(base, []).append(det)

    winners = []

    for base, candidates in groups.items():
        if len(candidates) == 1:
            winners.append(candidates[0])
            continue

        scores = {}
        for det in candidates:
            profile = profiles.get(det.label)
            if profile is None:
                scores[det.label] = 0.0
            else:
                scores[det.label] = color_score(frame_bgr, det.quad, profile)

        best = max(candidates, key=lambda d: scores[d.label])

        for det in candidates:
            print(
                f"  [LAB] '{det.label}'  orb={det.score:.2f}"
                f"  lab={scores[det.label]:.3f}"
                f"{'  ✓' if det is best else ''}"
            )

        winners.append(best)

    return winners
