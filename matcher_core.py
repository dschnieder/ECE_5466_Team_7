"""
Core matching primitives shared by runtime entry points.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np


TEMPLATE_UPSCALE_PX = 240
ORB_N_FEATURES = 2000
LOWE_RATIO = 0.75
MIN_MATCH_COUNT = 12
RANSAC_THRESHOLD = 5.0
AREA_RATIO_RANGE = (0.1, 10.0)

DEFAULT_COLOR = (0, 255, 100)
SHINY_COLOR = (0, 215, 255)

_PALETTE = [
    (50, 180, 255),
    (255, 100, 50),
    (100, 255, 150),
    (200, 80, 255),
    (255, 200, 50),
    (80, 255, 220),
]


def label_color(label: str, index: int) -> tuple:
    if label.endswith("_shiny"):
        return SHINY_COLOR
    return _PALETTE[index % len(_PALETTE)]


@dataclass
class Template:
    label: str
    image: np.ndarray
    keypoints: list
    descriptors: np.ndarray
    corners: np.ndarray
    color: tuple = field(default_factory=lambda: DEFAULT_COLOR)


@dataclass
class Detection:
    label: str
    quad: np.ndarray
    score: float
    color: tuple = field(default_factory=lambda: DEFAULT_COLOR)


def build_templates(template_dict: dict) -> tuple[list[Template], cv2.ORB]:
    orb = cv2.ORB_create(
        nfeatures=ORB_N_FEATURES,
        scaleFactor=1.2,
        nlevels=10,
        edgeThreshold=15,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31,
        fastThreshold=10,
    )

    templates = []
    for index, (label, path) in enumerate(template_dict.items()):
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] Cannot load: {path}")
            continue

        up = cv2.resize(
            img,
            (TEMPLATE_UPSCALE_PX, TEMPLATE_UPSCALE_PX),
            interpolation=cv2.INTER_LANCZOS4,
        )
        gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)

        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        gray = cv2.filter2D(gray, -1, kernel)

        kp, des = orb.detectAndCompute(gray, None)
        if des is None or len(kp) < MIN_MATCH_COUNT:
            print(f"[WARN] Too few keypoints for '{label}': {len(kp) if kp else 0}")
            continue

        h, w = up.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        templates.append(
            Template(label, up, kp, des, corners, label_color(label, index))
        )
        print(f"[INFO] '{label}' - {len(kp)} ORB keypoints at {w}x{h}px")

    return templates, orb


def build_matcher() -> cv2.FlannBasedMatcher:
    index_params = dict(
        algorithm=6,
        table_number=12,
        key_size=20,
        multi_probe_level=2,
    )
    search_params = dict(checks=50)
    return cv2.FlannBasedMatcher(index_params, search_params)


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def is_valid_quad(quad: np.ndarray, template_area: float) -> bool:
    pts = quad.reshape(4, 2).astype(float)
    n = len(pts)
    area = (
        abs(
            sum(
                pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]
                for i in range(n)
            )
        )
        / 2.0
    )

    if area < 1:
        return False

    ratio = area / template_area
    return AREA_RATIO_RANGE[0] <= ratio <= AREA_RATIO_RANGE[1]


def match_one(
    frame_gray: np.ndarray,
    frame_orb_kp: list,
    frame_orb_des: np.ndarray,
    template: Template,
    matcher: cv2.FlannBasedMatcher,
) -> Detection | None:
    if frame_orb_des is None:
        return None

    t_des = template.descriptors
    f_des = frame_orb_des
    if t_des.dtype != np.uint8:
        t_des = t_des.astype(np.uint8)
    if f_des.dtype != np.uint8:
        f_des = f_des.astype(np.uint8)

    try:
        raw_matches = matcher.knnMatch(t_des, f_des, k=2)
    except cv2.error:
        return None

    good = [
        m
        for pair in raw_matches
        if len(pair) == 2
        for m, n in [pair]
        if m.distance < LOWE_RATIO * n.distance
    ]
    if len(good) < MIN_MATCH_COUNT:
        return None

    src_pts = np.float32([template.keypoints[m.queryIdx].pt for m in good]).reshape(
        -1, 1, 2
    )
    dst_pts = np.float32([frame_orb_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    h_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_THRESHOLD)
    if h_matrix is None:
        return None

    inliers = int(mask.sum())
    inlier_ratio = inliers / len(good)
    if inliers < MIN_MATCH_COUNT:
        return None

    quad = cv2.perspectiveTransform(template.corners, h_matrix)
    h, w = template.image.shape[:2]
    if not is_valid_quad(quad, w * h):
        return None

    return Detection(template.label, quad.astype(int), inlier_ratio, template.color)


def build_template_registry(
    regular_dir: str = "./images/regular",
    shiny_dir: str = "./images/shiny",
) -> dict:
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
        shiny_count = sum(1 for k in registry if k.endswith("_shiny"))
        print(
            f"[INFO] Loaded {regular_count} regular + {shiny_count} shiny template(s)"
        )

    return registry

