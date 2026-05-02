# Pokémon Shiny Hunter

An automated Raspberry Pi-based system that watches a Pokémon game screen in real time, detects whether the encountered Pokémon is shiny, and physically presses game buttons via servo motors to soft-reset until a shiny is found. An Android app connects over WebSocket for live notifications and target management.

---

## How It Works

```
Game Screen
    │
    ▼
Raspberry Pi Camera (CSI)
    │  captures live video
    ▼
main.py / server_main.py
    │  ORB feature matching + homography (matcher_core.py)
    │  CIE-LAB colour scoring (color_score.py)
    ▼
Detection decision
    ├─ Shiny found?     → WebSocket alert to Android app → stop
    ├─ Non-shiny found? → GPIO 17 servo presses RESET button
    └─ No detection?    → GPIO 27 servo presses A button (advances menus)
```

---

## Hardware

### 1. Raspberry Pi 5 (recommended) or Pi 4B

The compute unit that runs all Python code, hosts the WebSocket server, and drives the GPIO servos.

- **Purchase:** [raspberrypi.com/products/raspberry-pi-5](https://www.raspberrypi.com/products/raspberry-pi-5/) — ~$60 USD (4 GB) or ~$80 USD (8 GB)
- Also available from [Adafruit](https://www.adafruit.com), [PiShop.us](https://www.pishop.us), [Amazon](https://amazon.com)
- You'll need a microSD card (≥16 GB, Class 10 / A1), a USB-C power supply (≥27 W for Pi 5), and a case

### 2. Raspberry Pi Camera Module 3 (with autofocus)

The code uses Picamera2's continuous autofocus (`AfMode: Continuous`, `AfSpeed: Fast`). Arducam IMX519 PDAF&CDAF Autofocus Camera Module is strongly recommended.

- **Purchase:** [Arducam](https://www.uctronics.com/arducam-imx519-autofocus-camera-module-for-raspberry-pi.html) — ~$25 USD

### 3. SG90 or MG90S Micro Servo Motors (×2)

Two servos are used:
- **GPIO 17** (`RESET_SERVO_PIN`) — physically presses the game's reset / soft-reset button
- **GPIO 27** (`A_BUTTON_PIN`) — presses the A button to advance menus between encounters

Pulse range configured in code: 500 µs – 2500 µs at 50 Hz.

- **SG90 Purchase:** [Amazon](https://www.amazon.com/s?k=sg90+servo) — ~$2–5 USD each; sold in packs of 5 or 10
- Power the servo rail from a separate 5 V source if the Pi's 5 V GPIO rail causes brownouts

### 5. Android Smartphone

Used as the control & notification interface. Connects to the Pi's WebSocket server over the local Wi-Fi network.

- Any Android phone running Android 8.0+ works
- The app communicates on port `8765` (configurable)
- Needs to be on the same Wi-Fi network as the Pi

---

## Software

### Operating System

**Raspberry Pi OS (Bookworm, 64-bit)**
- Required for Picamera2 and libcamera support
- **Download:** [raspberrypi.com/software](https://www.raspberrypi.com/software/)
- Flash to microSD with **Raspberry Pi Imager**: [raspberrypi.com/software](https://www.raspberrypi.com/software/)

### Python 3.11+

Comes pre-installed with Raspberry Pi OS Bookworm. Verify with:

```bash
python3 --version
```

### Python Libraries

#### OpenCV (`opencv-python`)
Computer vision library for ORB feature detection, FLANN-LSH matching, homography, CLAHE preprocessing, and LAB colour space conversion.

```bash
pip install opencv-python
```

- **Docs:** [docs.opencv.org](https://docs.opencv.org/4.x/)
- **PyPI:** [pypi.org/project/opencv-python](https://pypi.org/project/opencv-python/)

#### NumPy (`numpy`)
Array math underpinning all image and descriptor operations.

```bash
pip install numpy
```

- **Docs:** [numpy.org](https://numpy.org/doc/)
- **PyPI:** [pypi.org/project/numpy](https://pypi.org/project/numpy/)

#### Picamera2 (`picamera2`)
The official Python library for libcamera-based Raspberry Pi cameras. **Do not** use the legacy `picamera` package — it is incompatible with Pi OS Bookworm.

```bash
# Recommended: system package (already present on Bookworm)
sudo apt install python3-picamera2

# Alternative via pip (if not present)
pip install picamera2
```

- **Docs:** [datasheets.raspberrypi.com/camera/picamera2-manual.pdf](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf)
- **GitHub:** [github.com/raspberrypi/picamera2](https://github.com/raspberrypi/picamera2)

#### websockets (`websockets`)
Async WebSocket server library. The Pi runs a server on port `8765`; the Android app connects as a client.

```bash
pip install websockets
```

- **Docs:** [websockets.readthedocs.io](https://websockets.readthedocs.io/)
- **PyPI:** [pypi.org/project/websockets](https://pypi.org/project/websockets/)

#### gpiozero (`gpiozero`)
High-level GPIO library for controlling the servo motors. Ships by default on Raspberry Pi OS.

```bash
# Usually already installed; if not:
sudo apt install python3-gpiozero
# or:
pip install gpiozero
```

- **Docs:** [gpiozero.readthedocs.io](https://gpiozero.readthedocs.io/)

#### pigpio (`pigpio`) — Optional but Recommended

Hardware PWM backend for gpiozero. Produces smoother servo motion than the software PWM default. The code automatically uses it if available and falls back silently if not.

```bash
sudo apt install pigpio python3-pigpio
sudo systemctl enable pigpiod
sudo systemctl start pigpiod
```

- **Docs:** [abyz.me.uk/rpi/pigpio](http://abyz.me.uk/rpi/pigpio/)

---

## Project Structure

```
shiny-hunter/
├── main.py              # Standalone mode: video file or PiCam, visual output
├── server_main.py       # WebSocket server mode: headless, Android app control
├── matcher_core.py      # ORB template building, FLANN-LSH matching primitives
├── color_score.py       # CIE-LAB colour profile building and scoring
├── main_lab_viz.py      # Development/debug visualiser for LAB colour scores
├── cam_test.py          # Camera sanity-check: live preview with autofocus
└── images/
    ├── regular/         # PNG sprites named by Pokédex number, e.g. 0025.png
    └── shiny/           # Shiny variants with identical filenames
```

---

## Setup: Step-by-Step

### Step 1 — Flash Raspberry Pi OS

1. Download and install **Raspberry Pi Imager** from [raspberrypi.com/software](https://www.raspberrypi.com/software/)
2. Insert your microSD card
3. In Imager: choose **Raspberry Pi OS (64-bit)** → your SD card → Write
4. In Imager's advanced settings (⚙️), pre-configure your Wi-Fi SSID/password and enable SSH

### Step 2 — First Boot and Camera Enable

```bash
# SSH into the Pi or open a terminal
sudo raspi-config
# → Interface Options → Camera → Enable
# Reboot when prompted
sudo reboot
```

### Step 3 — Connect the Camera

Attach the Camera Module 3 via the CSI ribbon cable before powering on. Blue side of the ribbon faces toward the USB ports on Pi 4/5.

### Step 4 — Wire the Servos

| Servo | Signal wire | Pi GPIO pin |
|-------|-------------|-------------|
| Reset servo | Data | GPIO 17 (Pin 11) |
| A-button servo | Data | GPIO 27 (Pin 13) |

Connect all servo GND wires to any Pi GND pin (e.g. Pin 6 or Pin 9). Connect all servo VCC wires to Pi 5 V (Pin 2 or Pin 4), or to an external 5 V supply if you have more than one servo.

### Step 5 — Install Dependencies

```bash
# System packages
sudo apt update && sudo apt install -y \
    python3-picamera2 \
    python3-gpiozero \
    pigpio python3-pigpio \
    libcamera-apps

# Enable pigpio daemon (for smooth PWM)
sudo systemctl enable pigpiod && sudo systemctl start pigpiod

# Python packages
pip install opencv-python numpy websockets
```

### Step 6 — Add Sprite Images

Place Pokémon sprite PNGs named by zero-padded Pokédex number into the image directories:

```
images/regular/0025.png   ← Pikachu regular
images/shiny/0025.png     ← Pikachu shiny
```

Sprites can be downloaded from [veekun.com/dex/downloads](https://veekun.com/dex/downloads) or extracted from a game dump. Transparent-background PNGs work best.

### Step 7 — Test the Camera

```bash
python cam_test.py
# Press Q to quit. Confirms autofocus is working before running the full matcher.
```

### Step 8 — Run

**Standalone mode** (on-screen display, no Android app needed):

```bash
# Test with a video file
python main.py --video gameplay.mp4 --target 25

# Run live with PiCam
python main.py --webcam --target 25
```

**WebSocket server mode** (headless, Android app connects):

```bash
# Default port 8765
python server_main.py

# Custom port or video file for testing
python server_main.py --port 9000
python server_main.py --video gameplay.mp4 --port 8765

# Override GPIO pins
python server_main.py --reset-pin 17 --a-button-pin 27
```

The server will print your Pi's IP address. In your Android app, connect to:

```
ws://<PI_IP_ADDRESS>:8765
```

---

## Android App WebSocket Protocol

The server speaks JSON over WebSocket on port `8765`.

**App → Server:**

```json
{ "action": "set_targets", "targets": ["25", "122"] }
{ "action": "get_targets" }
{ "action": "ping" }
```

**Server → App:**

```json
{ "event": "shiny_detected",  "label": "25_shiny", "score": 0.94, "ts": 1234567890.123 }
{ "event": "non_shiny_reset", "label": "25",        "score": 0.91, "ts": ... }
{ "event": "a_button_press",  "count": 3,                           "ts": ... }
{ "event": "targets_updated", "targets": ["25", "122"] }
{ "event": "pong" }
```

---

## Tuning

Key constants you may want to adjust live in `server_main.py` and `matcher_core.py`:

| Constant | Default | Description |
|---|---|---|
| `SHINY_MIN_SCORE` | `0.50` | Minimum ORB inlier ratio to alert on a shiny |
| `NON_SHINY_MIN_SCORE` | `0.55` | Minimum inlier ratio to trigger a reset |
| `MIN_MATCH_COUNT` | `12` | Minimum good ORB matches to attempt homography |
| `RESET_SERVO_NEUTRAL_PW` | `800 µs` | Resting pulse width of reset servo |
| `A_BUTTON_SERVO_NEUTRAL_PW` | `1000 µs` | Resting pulse width of A-button servo |
| `NO_DETECT_TIMEOUT_S` | `8.0 s` | Seconds without detection before A is pressed |
| `LAB_MATCH_RADIUS` | `30.0` | CIE-LAB a*b* distance threshold for colour match |

---

## Troubleshooting

**Camera not found / `Picamera2` import error**
Make sure the camera is connected, the interface is enabled via `raspi-config`, and you installed `python3-picamera2` via `apt`, not just `pip`.

**Servo jitter**
Install `pigpio`, enable the daemon with `sudo systemctl start pigpiod`, and re-run. Hardware PWM via pigpio is much smoother than software PWM.

**Too many false positives / negatives**
Adjust `SHINY_MIN_SCORE` / `NON_SHINY_MIN_SCORE` up or down. Also try tuning `LAB_MATCH_RADIUS` in `color_score.py` — a lower value makes colour matching stricter.

**No templates loaded**
Verify PNG files exist in `./images/regular/` and `./images/shiny/` and are named with the correct Pokédex number (e.g. `0025.png`).

**Android app can't connect**
Confirm the phone and Pi are on the same Wi-Fi network. Check that port `8765` is not blocked by a firewall (`sudo ufw allow 8765` if ufw is active).

---
