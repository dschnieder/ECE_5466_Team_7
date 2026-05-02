"""
Pokemon Shiny Hunter — WebSocket Server Entry Point
----------------------------------------------------
Runs in webcam (PiCamera2) mode by default, or video mode when `--video`
is set. An Android app connects via WebSocket to:
  • Send which Pokémon to hunt  (set_targets)
  • Receive shiny-detected alerts in real time

Servo behaviour:
  • GPIO 17 (RESET_SERVO_PIN)  — actuated on confident non-shiny detection
    to soft-reset the game.
  • GPIO 27 (A_BUTTON_PIN)     — pressed repeatedly when no detection has
    occurred for NO_DETECT_TIMEOUT_S seconds, to skip menus and start the
    next encounter.

Protocol (JSON over WebSocket, port 8765 by default):

  App  → Server:
    {"action": "set_targets", "targets": ["122", "25"]}
    {"action": "get_targets"}
    {"action": "ping"}

  Server → App:
    {"event": "shiny_detected",   "label": "122_shiny", "score": 0.94, "ts": ...}
    {"event": "non_shiny_reset",  "label": "122",       "score": 0.91, "ts": ...}
    {"event": "a_button_press",   "count": 3,                          "ts": ...}
    {"event": "targets_updated",  "targets": ["122", "25"]}
    {"event": "targets_current",  "targets": ["122"]}
    {"event": "pong"}
    {"event": "error",            "message": "..."}

Run:
    pip install opencv-python numpy websockets gpiozero
    python server_main.py [--port 8765] [--webcam-index 0]
    python server_main.py --video gameplay.mp4 [--port 8765]
"""

import asyncio
import json
import threading
import time
import argparse
import logging

import cv2
import numpy as np
import websockets
from websockets.server import WebSocketServerProtocol

from matcher_core import (
    build_template_registry,
    build_templates,
    build_matcher,
    preprocess_frame,
    match_one,
)
from color_score import build_color_profiles, resolve_conflicts_color

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("shiny-hunter")

# ── Servo / GPIO config ───────────────────────────────────────────────────────

RESET_SERVO_PIN  = 17    # GPIO pin for soft-reset servo
A_BUTTON_PIN     = 27    # GPIO pin for A-button servo

# Servo pulse widths in microseconds — tune to your hardware.
# Common SG90/MG90 range: 500–2500 µs. 1500 µs = centre / neutral.
#
# Neutral (resting) positions — set per servo so each can be tuned independently.
RESET_SERVO_NEUTRAL_PW    = 800  # µs — resting position for reset servo
A_BUTTON_SERVO_NEUTRAL_PW = 1000  # µs — resting position for A-button servo
#
# Press offsets: how far (in µs) to move from neutral to actuate each button.
RESET_SERVO_PRESS_OFFSET_PW    = 400  # µs — how far reset servo moves to actuate
A_BUTTON_SERVO_PRESS_OFFSET_PW = 500  # µs — how far A-button servo moves to actuate
#
SERVO_PRESS_S   = 0.15  # seconds to hold the press
SERVO_RELEASE_S = 0.10  # seconds to pause after releasing before next action

# ── Detection thresholds ──────────────────────────────────────────────────────

# Minimum inlier ratio (0–1) to act on a detection.
SHINY_MIN_SCORE     = 0.50   # confident shiny    → notify app
NON_SHINY_MIN_SCORE = 0.55   # confident non-shiny → actuate reset servo

# Cooldowns: don't re-trigger the same action within N seconds.
SHINY_COOLDOWN_S     = 10.0
NON_SHINY_COOLDOWN_S = 30.0  # soft-reset takes time; give the game a moment

# If no detection arrives for this long, assume we're between battles and
# start pressing A to advance menus / trigger the next encounter.
NO_DETECT_TIMEOUT_S = 8.0

# Gap between repeated A presses while in idle / menu mode.
A_PRESS_INTERVAL_S = 1.5

# ── PiCamera config ───────────────────────────────────────────────────────────

PICAM_WIDTH  = 640
PICAM_HEIGHT = 480

# ── Servo controller ──────────────────────────────────────────────────────────


class ServoController:
    """
    Thin wrapper around gpiozero's Servo.

    gpiozero is the easiest servo library on Pi OS — ships by default,
    no raw PWM maths, cleans up on exit automatically.

    If GPIO is not available (e.g. running on a dev machine) it falls back
    to logging-only mode so the rest of the server still runs.
    """

    def __init__(self, pin: int, name: str, neutral_pw: int = 1500):
        self._name = name
        self._neutral_pw = neutral_pw
        self._lock = threading.Lock()
        try:
            from gpiozero import Servo

            # Try pigpio pin factory for smoother PWM; fall back silently.
            factory = None
            try:
                from gpiozero.pins.pigpio import PiGPIOFactory
                factory = PiGPIOFactory()
            except Exception:
                pass

            kwargs = dict(
                pin=pin,
                min_pulse_width=500  / 1_000_000,  # 500 µs → seconds
                max_pulse_width=2500 / 1_000_000,  # 2500 µs → seconds
                frame_width=20       / 1_000,       # 20 ms period (50 Hz)
            )
            if factory:
                kwargs["pin_factory"] = factory

            self._servo = Servo(**kwargs)
            self._servo.value = self._pw_to_value(self._neutral_pw)  # move to neutral on startup
            self._available = True
            log.info("[Servo:%s] Initialised on GPIO %d, neutral=%dµs", name, pin, neutral_pw)

        except Exception as exc:
            self._servo = None
            self._available = False
            log.warning(
                "[Servo:%s] GPIO not available (%s) — running in simulation mode",
                name, exc,
            )

    def _pw_to_value(self, pulse_us: int) -> float:
        """Map a pulse width in µs to gpiozero's -1…+1 value range."""
        lo, hi = 500, 2500
        return (pulse_us - lo) / (hi - lo) * 2 - 1

    def press(self, offset_pw: int):
        """Move to press position, hold briefly, return to neutral."""
        with self._lock:
            if not self._available:
                log.info("[Servo:%s] (simulated) press", self._name)
                return
            try:
                press_pw = self._neutral_pw + offset_pw
                self._servo.value = self._pw_to_value(press_pw)
                time.sleep(SERVO_PRESS_S)
                self._servo.value = self._pw_to_value(self._neutral_pw)
                time.sleep(SERVO_RELEASE_S)
            except Exception as exc:
                log.warning("[Servo:%s] press failed: %s", self._name, exc)

    def close(self):
        if self._servo:
            try:
                self._servo.value = self._pw_to_value(self._neutral_pw)
                self._servo.close()
            except Exception:
                pass


# ── Shared state (thread-safe) ────────────────────────────────────────────────


class HunterState:
    """Shared mutable state between the detection thread and WS coroutines."""

    def __init__(self):
        self._lock = threading.Lock()
        self._targets: set[str] = set()
        self._event_queue: list[dict] = []

    # ── target management ────────────────────────────────────────────────────

    def set_targets(self, dex_numbers: list[str]) -> list[str]:
        cleaned = [str(d).strip() for d in dex_numbers if str(d).strip()]
        with self._lock:
            self._targets = set(cleaned)
        log.info("Targets updated: %s", cleaned)
        return cleaned

    def get_targets(self) -> list[str]:
        with self._lock:
            return sorted(self._targets)

    def is_target(self, label: str) -> bool:
        base = label.replace("_shiny", "")
        with self._lock:
            return base in self._targets

    def has_targets(self) -> bool:
        with self._lock:
            return bool(self._targets)

    # ── event queue ──────────────────────────────────────────────────────────

    def push_event(self, event: dict):
        with self._lock:
            self._event_queue.append(event)

    def drain_events(self) -> list[dict]:
        with self._lock:
            events, self._event_queue = self._event_queue, []
        return events


STATE = HunterState()

# ── Template cache (rebuilt when targets change) ──────────────────────────────


class TemplateCache:
    """Keeps compiled ORB templates + color profiles for the active targets."""

    def __init__(self, all_templates: dict):
        self._all = all_templates
        self._lock = threading.Lock()
        self._current_keys: frozenset = frozenset()
        self._templates = []
        self._orb = None
        self._matcher = None
        self._profiles = {}

    def refresh_if_needed(self):
        wanted = frozenset(
            label for label, _path in self._all.items() if STATE.is_target(label)
        )
        with self._lock:
            if wanted == self._current_keys:
                return
            if not wanted:
                self._templates = []
                self._profiles = {}
                self._current_keys = wanted
                return
            log.info("Rebuilding templates for: %s", sorted(wanted))
            subset = {k: self._all[k] for k in wanted if k in self._all}
            templates, orb = build_templates(subset)
            profiles = build_color_profiles(subset)
            matcher = build_matcher()
            self._templates = templates
            self._orb = orb
            self._matcher = matcher
            self._profiles = profiles
            self._current_keys = wanted

    def get(self):
        with self._lock:
            return self._templates, self._orb, self._matcher, self._profiles


# ── Camera source abstraction ─────────────────────────────────────────────────


def _open_capture(video_source: int | str):
    """
    Return a (read_fn, close_fn) pair abstracting PiCamera2 vs VideoCapture.

    read_fn()  → np.ndarray | None
    close_fn() → None
    """
    if isinstance(video_source, int):
        from picamera2 import Picamera2
        from libcamera import controls as lc

        picam2 = Picamera2()
        picam2.configure(picam2.create_video_configuration(
            main={"format": "RGB888", "size": (PICAM_WIDTH, PICAM_HEIGHT)}
        ))
        picam2.start()
        # Must be called after start() or it is silently ignored
        picam2.set_controls({
            "AfMode": lc.AfModeEnum.Continuous,
            "AfSpeed": lc.AfSpeedEnum.Fast,
        })
        log.info(
            "PiCamera2 started — %dx%d RGB888, continuous AF enabled",
            PICAM_WIDTH, PICAM_HEIGHT,
        )

        def read_fn():
            return picam2.capture_array()

        def close_fn():
            picam2.stop()
            picam2.close()
            log.info("PiCamera2 closed.")

    else:
        if video_source.startswith("http"):
            cap = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        else:
            cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {video_source}")

        log.info("Video source opened: %s", video_source)

        def read_fn():
            ret, frame = cap.read()
            return frame if (ret and frame is not None) else None

        def close_fn():
            cap.release()
            log.info("VideoCapture released.")

    return read_fn, close_fn


# ── Detection thread ──────────────────────────────────────────────────────────


def detection_thread(
    cache: TemplateCache,
    video_source: int | str,
    reset_servo: ServoController,
    a_button_servo: ServoController,
    stop_event: threading.Event,
):
    """
    Main detection loop. Responsibilities:

      1. Pull frames from PiCam or video file.
      2. Run ORB matcher against active templates.
      3. Confident SHINY    → push alert to app, then stop the program.
      4. Confident NON-SHINY → actuate reset servo + notify app.
      5. No detection for NO_DETECT_TIMEOUT_S → press A repeatedly to
         advance menus and trigger the next encounter.

    No servos are actuated until at least one target Pokémon has been set.

    Servo presses are dispatched to daemon threads so a slow press never
    blocks the camera read loop.
    """
    try:
        read_frame, close_capture = _open_capture(video_source)
    except RuntimeError as exc:
        log.error("%s", exc)
        stop_event.set()
        return

    def _press_async(servo: ServoController, offset_pw: int):
        """Fire a servo press in a short-lived daemon thread."""
        threading.Thread(target=servo.press, args=(offset_pw,), daemon=True).start()

    last_detection_time  = time.time()
    last_non_shiny_reset: dict[str, float] = {}
    last_shiny_alert:     dict[str, float] = {}
    a_press_count = 0
    in_idle_mode  = False

    try:
        while not stop_event.is_set():
            cache.refresh_if_needed()
            templates, orb, matcher, profiles = cache.get()

            frame = read_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            now = time.time()

            # ── No targets set → wait silently, do not touch any servos ──────
            if not STATE.has_targets():
                log.debug("[Idle] Waiting for targets to be set — servos inactive")
                time.sleep(0.5)
                # Reset the idle clock so the A-press timeout doesn't fire
                # the moment targets are eventually configured.
                last_detection_time = now
                in_idle_mode = False
                a_press_count = 0
                continue

            # ── No templates loaded yet (targets set but not compiled) ────────
            if not templates or orb is None:
                time.sleep(0.1)
                continue

            # ── Run ORB detection ─────────────────────────────────────────────
            frame_gray = preprocess_frame(frame)
            kp, des = orb.detectAndCompute(frame_gray, None)

            detections = []
            for tmpl in templates:
                det = match_one(frame_gray, kp, des, tmpl, matcher)
                if det:
                    detections.append(det)

            detections = resolve_conflicts_color(detections, frame, profiles)

            if detections:
                last_detection_time = now
                in_idle_mode  = False
                a_press_count = 0   # back in battle — stop counting A presses

            # ── Act on each detection ─────────────────────────────────────────
            for det in detections:
                is_shiny = det.label.endswith("_shiny")

                if is_shiny and det.score >= SHINY_MIN_SCORE:
                    last = last_shiny_alert.get(det.label, 0.0)
                    if now - last < SHINY_COOLDOWN_S:
                        continue
                    last_shiny_alert[det.label] = now

                    log.info("🌟 SHINY: %s (score=%.2f) — notifying and stopping", det.label, det.score)
                    STATE.push_event({
                        "event": "shiny_detected",
                        "label": det.label,
                        "score": round(det.score, 4),
                        "ts":    round(now, 3),
                    })
                    # Give the event broadcaster a moment to flush the
                    # notification to connected clients before we shut down.
                    time.sleep(1.0)
                    stop_event.set()
                    return

                elif not is_shiny and det.score >= NON_SHINY_MIN_SCORE:
                    last = last_non_shiny_reset.get(det.label, 0.0)
                    if now - last < NON_SHINY_COOLDOWN_S:
                        continue
                    last_non_shiny_reset[det.label] = now

                    log.info(
                        "↩️  Non-shiny '%s' (score=%.2f) — actuating reset servo",
                        det.label, det.score,
                    )
                    _press_async(reset_servo, RESET_SERVO_PRESS_OFFSET_PW)
                    STATE.push_event({
                        "event": "non_shiny_reset",
                        "label": det.label,
                        "score": round(det.score, 4),
                        "ts":    round(now, 3),
                    })

            # ── Idle / no-detection logic ─────────────────────────────────────
            if now - last_detection_time >= NO_DETECT_TIMEOUT_S:
                if not in_idle_mode:
                    in_idle_mode = True
                    log.info(
                        "[Idle] No detection for %.1fs — pressing A to advance",
                        now - last_detection_time,
                    )
                _press_async(a_button_servo, A_BUTTON_SERVO_PRESS_OFFSET_PW)
                a_press_count += 1
                STATE.push_event({
                    "event": "a_button_press",
                    "count": a_press_count,
                    "ts":    round(now, 3),
                })
                # Sleep so we don't spam A faster than the game responds.
                time.sleep(A_PRESS_INTERVAL_S)

    finally:
        close_capture()
        reset_servo.close()
        a_button_servo.close()
        log.info("Detection thread stopped.")


# ── WebSocket server ──────────────────────────────────────────────────────────

CONNECTED: set[WebSocketServerProtocol] = set()
CONNECTED_LOCK = asyncio.Lock()


async def broadcast(message: dict):
    """Send a JSON message to every connected client."""
    if not CONNECTED:
        return
    payload = json.dumps(message)
    async with CONNECTED_LOCK:
        targets = list(CONNECTED)
    for ws in targets:
        try:
            await ws.send(payload)
        except Exception:
            pass


async def event_broadcaster():
    """Periodically drain the event queue and broadcast to all clients."""
    while True:
        await asyncio.sleep(0.25)
        for event in STATE.drain_events():
            await broadcast(event)


async def ws_handler(websocket: WebSocketServerProtocol):
    """Handle one connected Android client."""
    addr = websocket.remote_address
    log.info("Client connected: %s", addr)

    async with CONNECTED_LOCK:
        CONNECTED.add(websocket)

    try:
        async for raw in websocket:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send(
                    json.dumps({"event": "error", "message": "Invalid JSON"})
                )
                continue

            action = msg.get("action", "")

            if action == "set_targets":
                raw_targets = msg.get("targets", [])
                if not isinstance(raw_targets, list):
                    await websocket.send(
                        json.dumps({"event": "error", "message": "'targets' must be a list"})
                    )
                    continue
                new_targets = STATE.set_targets(raw_targets)
                await broadcast({"event": "targets_updated", "targets": new_targets})

            elif action == "get_targets":
                await websocket.send(json.dumps({
                    "event": "targets_current",
                    "targets": STATE.get_targets(),
                }))

            elif action == "ping":
                await websocket.send(json.dumps({"event": "pong"}))

            else:
                await websocket.send(json.dumps({
                    "event": "error",
                    "message": f"Unknown action: '{action}'",
                }))

    except websockets.exceptions.ConnectionClosedOK:
        pass
    except websockets.exceptions.ConnectionClosedError as e:
        log.warning("Client %s dropped: %s", addr, e)
    finally:
        async with CONNECTED_LOCK:
            CONNECTED.discard(websocket)
        log.info("Client disconnected: %s", addr)


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Pokemon Shiny Hunter — WebSocket server")
    parser.add_argument("--port",         type=int, default=8765,           help="WebSocket port (default 8765)")
    parser.add_argument("--webcam-index", type=int, default=0,              help="PiCam/webcam index (default 0)")
    parser.add_argument("--video",        type=str, default=None,           help="Video file path or HTTP stream URL")
    parser.add_argument("--host",         type=str, default="0.0.0.0",      help="Bind host (default 0.0.0.0)")
    parser.add_argument("--reset-pin",    type=int, default=RESET_SERVO_PIN, help=f"GPIO pin for reset servo (default {RESET_SERVO_PIN})")
    parser.add_argument("--a-button-pin", type=int, default=A_BUTTON_PIN,   help=f"GPIO pin for A-button servo (default {A_BUTTON_PIN})")
    args = parser.parse_args()

    source: int | str = args.video if args.video else args.webcam_index

    reset_servo    = ServoController(args.reset_pin,    name="reset",    neutral_pw=RESET_SERVO_NEUTRAL_PW)
    a_button_servo = ServoController(args.a_button_pin, name="A-button", neutral_pw=A_BUTTON_SERVO_NEUTRAL_PW)

    all_templates = build_template_registry()
    cache = TemplateCache(all_templates)
    stop_event = threading.Event()

    det_thread = threading.Thread(
        target=detection_thread,
        args=(cache, source, reset_servo, a_button_servo, stop_event),
        daemon=True,
        name="detection",
    )
    det_thread.start()

    async def serve():
        async with websockets.serve(ws_handler, args.host, args.port):
            log.info("WebSocket server listening on ws://%s:%d", args.host, args.port)
            log.info("Connect your Android app to ws://<THIS_PI_IP>:%d", args.port)
            broadcaster = asyncio.create_task(event_broadcaster())
            try:
                # Poll stop_event so the server exits when the detection
                # thread signals a shiny was found.
                while not stop_event.is_set():
                    await asyncio.sleep(0.5)
            finally:
                broadcaster.cancel()

    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        log.info("Shutting down…")
    finally:
        stop_event.set()
        det_thread.join(timeout=3)
        log.info("Server stopped.")


if __name__ == "__main__":
    main()
