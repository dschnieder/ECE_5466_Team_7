"""
Pokemon Sprite Matcher — LAB Colour Space Visualiser
-----------------------------------------------------
Drop-in alternative to main.py that adds a live side panel showing the
CIE a*b* plane for every loaded colour profile AND the chromatic pixels
from the current detection crop.

Usage:
    python main_lab_viz.py --video gameplay.mp4 --target 0122
    python main_lab_viz.py --webcam              --target 0122

Panel layout (right of the video frame):
  ┌──────────────────────────────────────┐
  │  a*b* colour space                   │
  │  · coloured dots = profile clusters  │
  │    (size ∝ cluster weight)           │
  │  · white scatter = detection pixels  │
  │  · dashed circle = LAB_MATCH_RADIUS  │
  │  · colour wheel hue ring (reference) │
  └──────────────────────────────────────┘
  ┌──────────────────────────────────────┐
  │  Per-label LAB score bar chart       │
  └──────────────────────────────────────┘
"""

import cv2
import numpy as np
import argparse
import threading
import time
import math
from dataclasses import dataclass, field

from color_score import (
    build_color_profiles,
    resolve_conflicts_color,
    color_score,
    extract_aligned_patch,
    LAB_MATCH_RADIUS,
    SCORE_CHROMA_THRESH,
    MIN_CHROMA_PIXELS,
    ColorProfile,
)

# ── Re-use tuning knobs from main ────────────────────────────────────────────

TEMPLATE_UPSCALE_PX = 240
ORB_N_FEATURES      = 2000
LOWE_RATIO          = 0.75
MIN_MATCH_COUNT     = 12
RANSAC_THRESHOLD    = 5.0
AREA_RATIO_RANGE    = (0.1, 10.0)
DEFAULT_COLOR       = (0, 255, 100)
SHINY_COLOR         = (0, 215, 255)

_PALETTE = [
    ( 50, 180, 255),
    (255, 100,  50),
    (100, 255, 150),
    (200,  80, 255),
    (255, 200,  50),
    ( 80, 255, 220),
]

def label_color(label: str, index: int) -> tuple:
    if label.endswith("_shiny"):
        return SHINY_COLOR
    return _PALETTE[index % len(_PALETTE)]


# ── Data structures (identical to main.py) ───────────────────────────────────

@dataclass
class Template:
    label:       str
    image:       np.ndarray
    keypoints:   list
    descriptors: np.ndarray
    corners:     np.ndarray
    color:       tuple = field(default_factory=lambda: DEFAULT_COLOR)


@dataclass
class Detection:
    label:  str
    quad:   np.ndarray
    score:  float
    color:  tuple = field(default_factory=lambda: (0, 255, 100))


# ── Template / matcher builders (identical to main.py) ───────────────────────

def build_templates(template_dict):
    orb = cv2.ORB_create(
        nfeatures=ORB_N_FEATURES, scaleFactor=1.2, nlevels=10,
        edgeThreshold=15, firstLevel=0, WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=10,
    )
    templates = []
    for index, (label, path) in enumerate(template_dict.items()):
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] Cannot load: {path}")
            continue
        up   = cv2.resize(img, (TEMPLATE_UPSCALE_PX, TEMPLATE_UPSCALE_PX),
                          interpolation=cv2.INTER_LANCZOS4)
        gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        gray = cv2.filter2D(gray, -1, kernel)
        kp, des = orb.detectAndCompute(gray, None)
        if des is None or len(kp) < MIN_MATCH_COUNT:
            print(f"[WARN] Too few keypoints for '{label}'")
            continue
        h, w = up.shape[:2]
        corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        templates.append(Template(label, up, kp, des, corners, label_color(label, index)))
        print(f"[INFO] '{label}' — {len(kp)} ORB keypoints")
    return templates, orb


def build_matcher():
    index_params  = dict(algorithm=6, table_number=12, key_size=20, multi_probe_level=2)
    search_params = dict(checks=50)
    return cv2.FlannBasedMatcher(index_params, search_params)


# ── Frame helpers (identical to main.py) ─────────────────────────────────────

def preprocess_frame(frame):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def is_valid_quad(quad, template_area):
    pts  = quad.reshape(4, 2).astype(float)
    n    = len(pts)
    area = abs(sum(
        pts[i][0]*pts[(i+1)%n][1] - pts[(i+1)%n][0]*pts[i][1]
        for i in range(n)
    )) / 2.0
    if area < 1:
        return False
    ratio = area / template_area
    return AREA_RATIO_RANGE[0] <= ratio <= AREA_RATIO_RANGE[1]


def match_one(frame_gray, frame_orb_kp, frame_orb_des, template, matcher):
    if frame_orb_des is None:
        return None
    t_des = template.descriptors.astype(np.uint8)
    f_des = frame_orb_des.astype(np.uint8)
    try:
        raw_matches = matcher.knnMatch(t_des, f_des, k=2)
    except cv2.error:
        return None
    good = [m for pair in raw_matches if len(pair)==2
            for m, n in [pair] if m.distance < LOWE_RATIO * n.distance]
    if len(good) < MIN_MATCH_COUNT:
        return None
    src_pts = np.float32([template.keypoints[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([frame_orb_kp[m.trainIdx].pt       for m in good]).reshape(-1,1,2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_THRESHOLD)
    if H is None:
        return None
    inliers = int(mask.sum())
    if inliers < MIN_MATCH_COUNT:
        return None
    quad = cv2.perspectiveTransform(template.corners, H)
    h, w = template.image.shape[:2]
    if not is_valid_quad(quad, w * h):
        return None
    return Detection(template.label, quad.astype(int), inliers / len(good), template.color)


# ── LAB visualiser helpers ────────────────────────────────────────────────────

PANEL_SIZE   = 400          # square panel (pixels)
AB_RANGE     = 128.0        # a* and b* both span ±128 in OpenCV
PANEL_PAD    = 20           # margin around the plot area
PLOT_SIZE    = PANEL_SIZE - 2 * PANEL_PAD

# Map a*b* → panel pixel coordinates
def ab_to_xy(a, b):
    """Convert a*, b* values to integer (x, y) in the panel."""
    x = int(PANEL_PAD + (a + AB_RANGE) / (2 * AB_RANGE) * PLOT_SIZE)
    y = int(PANEL_PAD + (AB_RANGE - b) / (2 * AB_RANGE) * PLOT_SIZE)
    return x, y


def draw_ab_axes(panel):
    """Draw the a*b* axis cross, grid lines, labels, and hue reference ring."""
    cx, cy = ab_to_xy(0, 0)

    # --- faint concentric chroma rings ---
    for chroma in (25, 50, 75, 100):
        r = int(chroma / (2 * AB_RANGE) * PLOT_SIZE)
        cv2.circle(panel, (cx, cy), r, (50, 50, 50), 1, cv2.LINE_AA)
        cv2.putText(panel, str(chroma), (cx + r + 2, cy - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (70, 70, 70), 1)

    # --- hue colour ring (thin rainbow rim at chroma=110) ---
    ring_r = int(110 / (2 * AB_RANGE) * PLOT_SIZE)
    for angle_deg in range(0, 360, 2):
        rad   = math.radians(angle_deg)
        a_val = 110.0 * math.cos(rad)
        b_val = 110.0 * math.sin(rad)
        hx, hy = ab_to_xy(a_val, b_val)
        # Convert this a*b* point (L=60) to BGR for a saturated swatch
        lab_px = np.uint8([[[60, int(a_val + 128), int(b_val + 128)]]])
        bgr = cv2.cvtColor(lab_px, cv2.COLOR_Lab2BGR)[0][0]
        cv2.circle(panel, (hx, hy), 3, (int(bgr[0]), int(bgr[1]), int(bgr[2])), -1)

    # --- axes ---
    cv2.line(panel, (PANEL_PAD, cy), (PANEL_SIZE - PANEL_PAD, cy), (120,120,120), 1)
    cv2.line(panel, (cx, PANEL_PAD), (cx, PANEL_SIZE - PANEL_PAD), (120,120,120), 1)

    # --- axis labels ---
    cv2.putText(panel, "a* (+red/-green)", (PANEL_PAD, cy - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160,160,160), 1)
    cv2.putText(panel, "b*", (cx + 4, PANEL_PAD + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160,160,160), 1)
    cv2.putText(panel, "(+yellow/-blue)", (cx + 4, PANEL_PAD + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160,160,160), 1)


def draw_detection_pixels(panel, crop_bgr):
    """
    Scatter the chromatic a*b* pixels from the detection crop onto the panel.
    Uses a small semi-transparent dot per pixel (subsampled for speed).
    """
    if crop_bgr is None:
        return
    lab  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    a    = (lab[:, :, 1] - 128.0).flatten()
    b    = (lab[:, :, 2] - 128.0).flatten()
    chroma = np.sqrt(a**2 + b**2)
    mask = chroma > SCORE_CHROMA_THRESH
    a_c, b_c = a[mask], b[mask]

    if len(a_c) < MIN_CHROMA_PIXELS:
        return

    # Subsample to at most 2000 points so the panel stays readable
    step = max(1, len(a_c) // 2000)
    for av, bv in zip(a_c[::step], b_c[::step]):
        px, py = ab_to_xy(av, bv)
        if 0 <= px < PANEL_SIZE and 0 <= py < PANEL_SIZE:
            cv2.circle(panel, (px, py), 2, (220, 220, 220), -1, cv2.LINE_AA)


def draw_profile_clusters(panel, profiles, label_bgr_map):
    """
    Plot each profile's k-means cluster centres as filled circles.
    Circle radius encodes relative weight; match-radius ring shown dashed.
    """
    for label, profile in profiles.items():
        bgr = label_bgr_map.get(label, (200, 200, 200))
        for ca, cb, weight in profile.clusters:
            px, py = ab_to_xy(ca, cb)
            r = max(6, int(weight * 28))   # weight 1.0 → r=28, 0.33 → r~9

            # dashed match-radius ring
            match_r = int(LAB_MATCH_RADIUS / (2 * AB_RANGE) * PLOT_SIZE)
            _draw_dashed_circle(panel, (px, py), match_r, bgr, dash=8)

            # filled cluster dot
            cv2.circle(panel, (px, py), r,    bgr, -1, cv2.LINE_AA)
            cv2.circle(panel, (px, py), r + 1, (0,0,0), 1, cv2.LINE_AA)

            # centroid crosshair
            cv2.line(panel, (px - r, py), (px + r, py), (0,0,0), 1)
            cv2.line(panel, (px, py - r), (px, py + r), (0,0,0), 1)


def _draw_dashed_circle(img, centre, radius, color, dash=10):
    """Draw a dashed circle (approximate via arc segments)."""
    cx, cy = centre
    total_steps = max(1, int(2 * math.pi * radius))
    on = True
    count = 0
    prev = None
    for step in range(total_steps + 1):
        angle = 2 * math.pi * step / total_steps
        x = int(cx + radius * math.cos(angle))
        y = int(cy + radius * math.sin(angle))
        if prev is not None and on:
            cv2.line(img, prev, (x, y), color, 1, cv2.LINE_AA)
        prev = (x, y)
        count += 1
        if count >= dash:
            count = 0
            on = not on


def draw_score_bars(bar_panel, scores, label_bgr_map, panel_w=400):
    """
    Horizontal bar chart of LAB colour scores for each detected label.
    bar_panel is drawn in-place.
    """
    bar_panel[:] = 18   # dark background
    if not scores:
        cv2.putText(bar_panel, "No detections", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120,120,120), 1)
        return

    bar_h   = 28
    gap     = 10
    max_bar = panel_w - 160
    y       = 14

    cv2.putText(bar_panel, "LAB colour match scores",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
    y += 20

    for label, score in sorted(scores.items()):
        bgr   = label_bgr_map.get(label, (200, 200, 200))
        bar_w = int(score * max_bar)

        cv2.rectangle(bar_panel, (130, y), (130 + bar_w, y + bar_h), bgr, -1)
        cv2.rectangle(bar_panel, (130, y), (130 + max_bar, y + bar_h), (60,60,60), 1)

        cv2.putText(bar_panel, label[:18], (6, y + bar_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, bgr, 1)
        cv2.putText(bar_panel, f"{score:.3f}", (136 + bar_w, y + bar_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220,220,220), 1)
        y += bar_h + gap

    if y > bar_panel.shape[0]:
        # silently clipped — panel is fixed size
        pass


# ── Detection with visualiser state ──────────────────────────────────────────

def detect_and_viz(frame, templates, orb, matcher, profiles,
                   label_bgr_map, viz_state: dict) -> np.ndarray:
    """
    Run detection, update viz_state with current crop + scores,
    return annotated frame.
    """
    output     = frame.copy()
    frame_gray = preprocess_frame(frame)
    kp, des    = orb.detectAndCompute(frame_gray, None)

    detections = []
    for tmpl in templates:
        det = match_one(frame_gray, kp, des, tmpl, matcher)
        if det:
            detections.append(det)

    detections = resolve_conflicts_color(detections, frame, profiles)

    # Compute LAB scores for every profile (not just winning detections)
    scores = {}
    latest_crop = None
    for det in detections:
        for label, profile in profiles.items():
            scores[label] = color_score(frame, det.quad, profile)
        # Use the first (or only) detection crop for scatter plot
        if latest_crop is None:
            latest_crop = extract_aligned_patch(frame, det.quad)

    viz_state["crop"]   = latest_crop
    viz_state["scores"] = scores

    for det in detections:
        cv2.polylines(output, [det.quad], isClosed=True, color=det.color, thickness=2)
        x, y  = det.quad[0][0]
        text  = f"{det.label}  {det.score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(output, (x, y - th - 6), (x + tw + 4, y), det.color, -1)
        cv2.putText(output, text, (x + 2, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    return output


# ── Side panel composer ───────────────────────────────────────────────────────

def build_side_panel(profiles, label_bgr_map, viz_state: dict,
                     frame_h: int) -> np.ndarray:
    """
    Compose a vertical side panel of width PANEL_SIZE that fits frame_h.
    Top section: a*b* scatter + profile clusters.
    Bottom section: score bar chart.
    """
    ab_panel  = np.full((PANEL_SIZE, PANEL_SIZE, 3), 18, dtype=np.uint8)
    draw_ab_axes(ab_panel)
    draw_detection_pixels(ab_panel, viz_state.get("crop"))
    draw_profile_clusters(ab_panel, profiles, label_bgr_map)

    # Legend
    ly = PANEL_PAD
    for label, bgr in label_bgr_map.items():
        cv2.circle(ab_panel, (PANEL_SIZE - 90, ly), 5, bgr, -1)
        cv2.putText(ab_panel, label[:14], (PANEL_SIZE - 82, ly + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, bgr, 1)
        ly += 14

    # Score bars take remaining height
    bar_h   = max(60, frame_h - PANEL_SIZE)
    bar_panel = np.full((bar_h, PANEL_SIZE, 3), 18, dtype=np.uint8)
    draw_score_bars(bar_panel, viz_state.get("scores", {}), label_bgr_map, PANEL_SIZE)

    return np.vstack([ab_panel, bar_panel])


# ── Video runner ──────────────────────────────────────────────────────────────

def run_on_video(source, templates, orb, matcher, profiles, label_bgr_map):
    if isinstance(source, str) and source.startswith("http"):
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {source}")

    latest_frame  = None
    latest_result = None
    frame_lock    = threading.Lock()
    result_lock   = threading.Lock()
    stop_event    = threading.Event()
    viz_state     = {"crop": None, "scores": {}}

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
            result = detect_and_viz(
                frame, templates, orb, matcher, profiles,
                label_bgr_map, viz_state,
            )
            with result_lock:
                latest_result = (result, frame.shape[0])

    t1 = threading.Thread(target=capture_loop,  daemon=True)
    t2 = threading.Thread(target=detection_loop, daemon=True)
    t1.start(); t2.start()

    print("[Video] Running — Q to quit")

    while True:
        with result_lock:
            data = latest_result

        if data is not None:
            annotated, fh = data
            side  = build_side_panel(profiles, label_bgr_map, viz_state, fh)

            # Pad side panel height to match frame if necessary
            if side.shape[0] < fh:
                pad = np.full((fh - side.shape[0], PANEL_SIZE, 3), 18, dtype=np.uint8)
                side = np.vstack([side, pad])
            elif side.shape[0] > fh:
                side = side[:fh]

            display = np.hstack([annotated, side])
            cv2.imshow("Pokemon Matcher — LAB Visualiser", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    stop_event.set()
    cap.release()
    cv2.destroyAllWindows()


# ── Template registry (same as main.py) ──────────────────────────────────────

def build_template_registry(regular_dir="./images/regular", shiny_dir="./images/shiny"):
    import glob, os
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
        r = sum(1 for k in registry if not k.endswith("_shiny"))
        s = sum(1 for k in registry if     k.endswith("_shiny"))
        print(f"[INFO] Loaded {r} regular + {s} shiny template(s)")
    return registry


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pokemon sprite matcher v3 — with live LAB colour-space visualiser"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video",  metavar="PATH")
    group.add_argument("--webcam", action="store_true")
    parser.add_argument(
        "--target", metavar="DEX_NUM", required=True,
        help="Pokédex number (e.g. 0122). Loads regular + shiny variants.",
    )
    parser.add_argument(
        "--regular-dir", default="./images/regular",
        help="Directory of regular-form PNGs  (default: ./images/regular)",
    )
    parser.add_argument(
        "--shiny-dir",   default="./images/shiny",
        help="Directory of shiny-form PNGs     (default: ./images/shiny)",
    )
    args = parser.parse_args()

    TEMPLATES = build_template_registry(args.regular_dir, args.shiny_dir)

    target   = args.target
    filtered = {
        label: path for label, path in TEMPLATES.items()
        if label == target or label == f"{target}_shiny"
    }
    if not filtered:
        raise RuntimeError(
            f"No templates found for '{target}'. "
            f"Expected '{target}.png' in regular/shiny dirs."
        )
    print(f"[INFO] Target: {target} — loading {list(filtered.keys())}")

    templates, orb = build_templates(filtered)
    profiles       = build_color_profiles(filtered)

    if not templates:
        raise RuntimeError("No templates loaded.")

    # Build a label→BGR colour map for the visualiser
    label_bgr_map = {tmpl.label: tmpl.color for tmpl in templates}

    matcher = build_matcher()

    source = args.video if args.video else 0
    run_on_video(source, templates, orb, matcher, profiles, label_bgr_map)


if __name__ == "__main__":
    main()
