"""
Pokemon Sprite Matcher v3 — ORB + Homography + LAB colour scoring
------------------------------------------------------------------
Replaces SIFT (float descriptors, KD-tree FLANN) with ORB (binary
descriptors, LSH FLANN) and the bespoke HSV colour pass with a
perceptually uniform CIE-LAB scorer.

Why ORB?
  • Patent-free, ships with every OpenCV build.
  • Binary (BRIEF) descriptors → Hamming distance → 4–8× faster matching
    than SIFT's float L2 distance.
  • Oriented & scale-invariant keypoints (FAST corners + Harris score).
  • Works well on upscaled pixel art with the sharpening pre-pass.

Why LAB?
  • L* (lightness) is separated from a* / b* (chroma channels).
  • We compare only a*b*, so brightness differences from camera/screen
    don't affect the colour decision.
  • Perceptually uniform: Euclidean distance in a*b* ≈ perceived colour
    difference — no hand-tuned hue-tolerance constants.

Usage:
    pip install opencv-python numpy
    python main.py --video  gameplay.mp4
    python main.py --webcam
"""

import cv2
import numpy as np
import argparse
import threading
import time
from dataclasses import dataclass, field
from color_score import build_color_profiles, resolve_conflicts_color

# ── Tuning knobs ──────────────────────────────────────────────────────────────

# Size to upscale templates to before extracting features.
TEMPLATE_UPSCALE_PX = 240

# ORB feature count per template.  More → better recall, slower startup.
ORB_N_FEATURES = 2000

# Lowe's ratio test threshold for ORB.
# ORB distances are integers (Hamming), so 0.75 is a safe starting point.
LOWE_RATIO = 0.75

# Minimum good matches needed to attempt homography.
MIN_MATCH_COUNT = 12

# RANSAC reprojection threshold (pixels).
RANSAC_THRESHOLD = 5.0

# Reject homographies whose quad area deviates too far from the template area.
AREA_RATIO_RANGE = (0.1, 10.0)

# Draw colours per label (BGR).
# Shiny labels get a gold tint; regular labels cycle through a palette.
DEFAULT_COLOR = (0, 255, 100)
SHINY_COLOR   = (0, 215, 255)   # gold — visually distinct for shinies

_PALETTE = [
    ( 50, 180, 255),   # amber
    (255, 100,  50),   # blue
    (100, 255, 150),   # mint
    (200,  80, 255),   # purple
    (255, 200,  50),   # sky blue
    ( 80, 255, 220),   # yellow-green
]

def label_color(label: str, index: int) -> tuple:
    """Return a BGR colour: gold for shinies, cycling palette for regular."""
    if label.endswith("_shiny"):
        return SHINY_COLOR
    return _PALETTE[index % len(_PALETTE)]


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Template:
    label:       str
    image:       np.ndarray       # upscaled BGR
    keypoints:   list
    descriptors: np.ndarray       # uint8, shape (N, 32) — ORB/BRIEF
    corners:     np.ndarray       # (4,1,2) float32
    color:       tuple = field(default_factory=lambda: DEFAULT_COLOR)


@dataclass
class Detection:
    label:  str
    quad:   np.ndarray    # (4,1,2) int32
    score:  float         # inlier ratio (0–1)
    color:  tuple = field(default_factory=lambda: (0, 255, 100))


# ── Template preparation ──────────────────────────────────────────────────────

def build_templates(template_dict: dict) -> tuple[list[Template], cv2.ORB]:
    """Load images, upscale, extract ORB features."""
    orb = cv2.ORB_create(
        nfeatures=ORB_N_FEATURES,
        scaleFactor=1.2,      # pyramid scale — 1.2 gives finer scale steps
        nlevels=10,           # more pyramid levels → better scale coverage
        edgeThreshold=15,
        firstLevel=0,
        WTA_K=2,              # 2 = standard BRIEF (Hamming distance)
        scoreType=cv2.ORB_HARRIS_SCORE,   # Harris > FAST for stability
        patchSize=31,
        fastThreshold=10,     # lower → more keypoints on low-contrast art
    )

    templates = []
    for index, (label, path) in enumerate(template_dict.items()):
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] Cannot load: {path}")
            continue

        # Upscale with Lanczos — preserves edges better for pixel art
        up = cv2.resize(
            img, (TEMPLATE_UPSCALE_PX, TEMPLATE_UPSCALE_PX),
            interpolation=cv2.INTER_LANCZOS4,
        )

        gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)

        # Sharpening helps ORB's FAST corner detector on soft pixel art edges
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        gray = cv2.filter2D(gray, -1, kernel)

        kp, des = orb.detectAndCompute(gray, None)
        if des is None or len(kp) < MIN_MATCH_COUNT:
            print(f"[WARN] Too few keypoints for '{label}': {len(kp) if kp else 0}")
            continue

        h, w = up.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

        templates.append(Template(label, up, kp, des, corners, label_color(label, index)))
        print(f"[INFO] '{label}' — {len(kp)} ORB keypoints at {w}×{h}px")

    return templates, orb


# ── FLANN-LSH matcher (binary descriptors) ───────────────────────────────────

def build_matcher() -> cv2.FlannBasedMatcher:
    """
    FLANN with LSH index — the correct index for binary (Hamming) descriptors.
    KD-tree does NOT work for ORB; LSH does.

    table_number / key_size / multi_probe_level are the standard OpenCV
    defaults for ORB; tweak multi_probe_level upward for better recall at
    the cost of speed.
    """
    index_params = dict(
        algorithm=6,          # FLANN_INDEX_LSH
        table_number=12,      # number of hash tables
        key_size=20,          # key bits per table
        multi_probe_level=2,  # adjacent bucket probes (0 = exact LSH)
    )
    search_params = dict(checks=50)
    return cv2.FlannBasedMatcher(index_params, search_params)


# ── Per-frame detection ───────────────────────────────────────────────────────

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    CLAHE on greyscale boosts local contrast, critical for screen glare
    and dim-room photos.  Identical to the SIFT version — preprocessing
    is detector-agnostic.
    """
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def is_valid_quad(quad: np.ndarray, template_area: float) -> bool:
    """Reject degenerate or wildly wrong-size homography quads."""
    pts  = quad.reshape(4, 2).astype(float)
    n    = len(pts)
    area = abs(sum(
        pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]
        for i in range(n)
    )) / 2.0

    if area < 1:
        return False

    ratio = area / template_area
    return AREA_RATIO_RANGE[0] <= ratio <= AREA_RATIO_RANGE[1]


def match_one(
    frame_gray:    np.ndarray,
    frame_orb_kp:  list,
    frame_orb_des: np.ndarray,
    template:      Template,
    matcher:       cv2.FlannBasedMatcher,
) -> Detection | None:
    """Try to find one template in the frame. Returns Detection or None."""

    if frame_orb_des is None:
        return None

    # ORB descriptors must be uint8 for Hamming distance in FLANN-LSH
    # (detectAndCompute already returns uint8, but guard against edge cases)
    t_des = template.descriptors
    f_des = frame_orb_des
    if t_des.dtype != np.uint8:
        t_des = t_des.astype(np.uint8)
    if f_des.dtype != np.uint8:
        f_des = f_des.astype(np.uint8)

    # kNN k=2 for Lowe's ratio test
    try:
        raw_matches = matcher.knnMatch(t_des, f_des, k=2)
    except cv2.error:
        # FLANN-LSH can raise if the frame has too few features
        return None

    # Lowe's ratio test — same logic as SIFT, different distance scale
    # (Hamming integers typically 0–256; a ratio of 0.75 is still sensible)
    good = [
        m for pair in raw_matches
        if len(pair) == 2
        for m, n in [pair]
        if m.distance < LOWE_RATIO * n.distance
    ]

    if len(good) < MIN_MATCH_COUNT:
        return None

    src_pts = np.float32([template.keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([frame_orb_kp[m.trainIdx].pt       for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_THRESHOLD)

    if H is None:
        return None

    inliers      = int(mask.sum())
    inlier_ratio = inliers / len(good)

    if inliers < MIN_MATCH_COUNT:
        return None

    quad = cv2.perspectiveTransform(template.corners, H)

    h, w = template.image.shape[:2]
    if not is_valid_quad(quad, w * h):
        return None

    return Detection(template.label, quad.astype(int), inlier_ratio, template.color)


def detect(frame, templates, orb, matcher, profiles) -> np.ndarray:
    """Run detection on one frame, return annotated copy."""
    output     = frame.copy()
    frame_gray = preprocess_frame(frame)

    kp, des = orb.detectAndCompute(frame_gray, None)

    detections = []
    for tmpl in templates:
        det = match_one(frame_gray, kp, des, tmpl, matcher)
        if det:
            detections.append(det)
            print(f"  ✓ '{det.label}'  inlier_ratio={det.score:.2f}  inliers≥{MIN_MATCH_COUNT}")

    detections = resolve_conflicts_color(detections, frame, profiles)

    for det in detections:
        cv2.polylines(output, [det.quad], isClosed=True, color=det.color, thickness=2)

        x, y  = det.quad[0][0]
        text  = f"{det.label}  {det.score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(output, (x, y - th - 6), (x + tw + 4, y), det.color, -1)
        cv2.putText(output, text, (x + 2, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    return output


# ── Video / webcam runner ─────────────────────────────────────────────────────

def run_on_video(source, templates, orb, matcher, profiles):
    if isinstance(source, str) and source.startswith("http"):
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {source}")

    latest_frame  = None
    latest_result = None
    frame_lock    = threading.Lock()
    result_lock   = threading.Lock()
    stop_event    = threading.Event()

    def capture_loop():
        nonlocal latest_frame
        while not stop_event.is_set():
            if not cap.grab():
                time.sleep(0.001)
                continue
            ret, frame = cap.retrieve()
            if not ret or frame is None or frame.mean() < 1.0:
                continue
            with frame_lock:
                latest_frame = frame

    def detection_loop():
        nonlocal latest_result
        last_processed = None
        while not stop_event.is_set():
            with frame_lock:
                frame = latest_frame
            if frame is None or frame is last_processed:
                time.sleep(0.001)
                continue
            last_processed = frame
            result = detect(frame, templates, orb, matcher, profiles)
            with result_lock:
                latest_result = result

    t_capture   = threading.Thread(target=capture_loop,   daemon=True)
    t_detection = threading.Thread(target=detection_loop, daemon=True)
    t_capture.start()
    t_detection.start()

    print("[Video] Running — Q to quit")

    while True:
        with result_lock:
            display = latest_result
        if display is not None:
            cv2.imshow("Pokemon Matcher", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    stop_event.set()
    cap.release()
    cv2.destroyAllWindows()


# ── Template registry ─────────────────────────────────────────────────────────

def build_template_registry(
    regular_dir: str = "./images/regular",
    shiny_dir:   str = "./images/shiny",
) -> dict:
    """
    Scan regular_dir and shiny_dir for PNG files whose stems are Pokédex
    numbers and return a flat label->path dict, e.g.:
        { "0122": "./images/regular/0122.png",
          "0122_shiny": "./images/shiny/0122.png", ... }
    """
    import glob
    import os

    registry = {}
    for path in sorted(glob.glob(os.path.join(regular_dir, "*.png"))):
        dex = os.path.splitext(os.path.basename(path))[0]
        registry[dex] = path

    for path in sorted(glob.glob(os.path.join(shiny_dir, "*.png"))):
        dex = os.path.splitext(os.path.basename(path))[0]
        registry[f"{dex}_shiny"] = path

    if not registry:
        print(f"[WARN] No PNGs found in '{regular_dir}' or '{shiny_dir}'")
    else:
        regular_count = sum(1 for k in registry if not k.endswith("_shiny"))
        shiny_count   = sum(1 for k in registry if k.endswith("_shiny"))
        print(f"[INFO] Loaded {regular_count} regular + {shiny_count} shiny template(s)")

    return registry


TEMPLATES = build_template_registry()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pokemon sprite matcher v3 (ORB + LAB)")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video",  metavar="PATH")
    group.add_argument("--webcam", action="store_true")
    parser.add_argument(
        "--target",
        metavar="DEX_NUM",
        required=True,
        help="Pokédex number to match (e.g. 0122 or 122). Loads both regular and shiny variants.",
    )
    args = parser.parse_args()

    target = args.target
    filtered = {
        label: path
        for label, path in TEMPLATES.items()
        if label == target or label == f"{target}_shiny"
    }
    if not filtered:
        raise RuntimeError(
            f"No templates found for Pokédex number '{target}'. "
            f"Expected '{target}.png' in regular/shiny dirs."
        )
    print(f"[INFO] Target: {target} — loading {list(filtered.keys())}")

    templates, orb = build_templates(filtered)
    print(f"[DEBUG] TEMPLATES paths: {filtered}")
    profiles = build_color_profiles(filtered)

    if not templates:
        raise RuntimeError("No templates loaded.")

    matcher = build_matcher()

    if args.video:
        run_on_video(args.video, templates, orb, matcher, profiles)
    elif args.webcam:
        run_on_video(0, templates, orb, matcher, profiles)


if __name__ == "__main__":
    main()
