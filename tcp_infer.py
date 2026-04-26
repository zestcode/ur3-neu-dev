"""
tcp_infer.py — UR3 inference using an openpi LIBERO-format policy.

Pairs with `tcp_record.py`. The policy server is expected to consume the
LIBERO observation schema and emit LIBERO-format actions:

  observation
    observation/image        (H, W, 3) uint8 — third-person (D435i)
    observation/wrist_image  (H, W, 3) uint8 — wrist
    observation/state        (11,) float32  — [eef_x, eef_y, eef_z,
                                                r1x, r1y, r1z,
                                                r2x, r2y, r2z,
                                                gripper, gripper]
                              Rotation is 6D continuous (Zhou et al. 2019):
                              first two columns of the rotation matrix,
                              computed from RTDE axis-angle via Rodrigues.
                              Matches lerobot_ur3_record.py / tcp_record.py.
    prompt                   str
  action (7,) float32
    [sm_dx, sm_dy, sm_dz, sm_drx, sm_dry, sm_drz, gripper]
    The first 6 dims are the same SpaceMouse-style 6-DoF velocity command in
    the robot base frame that was recorded as the action during teleop, so
    they are forwarded directly to `rtde_c.speedL(...)`. action[6] is the
    gripper command in [0, 1] and is thresholded to open/close.

Architecture (mirrors ur3_inference_display.py)
------------
- Main thread       : tkinter live preview + stats overlay
- InferenceThread   : robot + policy server, runs the control loop
- CameraThread      : D435i (base) + OpenCV (wrist), grab/retrieve, no recording

Run
---
1. Start the policy server (LIBERO-format checkpoint trained on tcp_record.py):
       cd /home/robotics/Desktop/Project_UR3/openpi
       uv run scripts/serve_policy.py policy:checkpoint \\
           --policy.config <your_libero_config> --policy.dir <ckpt_dir>

2. Run this script:
       conda activate spacemouse-ur
       python3 tcp_infer.py
"""

import signal
import time
import serial
import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk
from collections import deque
from threading import Thread, Event, Lock

import websockets.sync.client
import pyrealsense2 as rs

from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from openpi_client import websocket_client_policy, action_chunk_broker, msgpack_numpy

# ---------------------------------------------------------------------------
# Rotation utility — must match lerobot_ur3_record.py / tcp_record.py exactly
# so the state distribution at inference matches the training distribution.
# ---------------------------------------------------------------------------


def axis_angle_to_rotation_6d(axis_angle) -> np.ndarray:
    """Convert axis-angle vector to 6D continuous rotation (Zhou et al. 2019).

    Returns the first two columns of the rotation matrix, flattened in
    column-major order: [r1.x, r1.y, r1.z, r2.x, r2.y, r2.z].

    The third column is recoverable as r1 × r2, so no information is lost.
    Avoids the θ=π discontinuity inherent in axis-angle representations.
    """
    aa = np.asarray(axis_angle, dtype=np.float64)
    theta = float(np.linalg.norm(aa))
    if theta < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    axis = aa / theta
    K = np.array([
        [    0.0, -axis[2],  axis[1]],
        [ axis[2],     0.0, -axis[0]],
        [-axis[1],  axis[0],     0.0],
    ])
    R = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
    return R[:, :2].T.flatten()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROBOT_HOST  = "192.168.0.2"
SERVER_HOST = "localhost"
SERVER_PORT = 8000

# LIBERO / tcp_record convention: 256×256 images.
IMAGE_SIZE   = (256, 256)
DISPLAY_SIZE = (640, 480)

TASK_PROMPT = "pick up the pink cylinder, hold it, and place it into the orange box"

# Inference cadence — match the dataset fps so the deltas the policy emits
# are played out at the same dt they were recorded at. Current datasets are
# collected by lerobot_ur3_record.py at fps=30 (its default).
CONTROL_HZ   = 30
CONTROL_DT   = 1.0 / CONTROL_HZ
SPEEDL_ACCEL = 0.25  # gentler than the recording's 0.5 to reduce jerk
SPEEDL_TIME  = 0.01   # safety timeout — robot decelerates if no cmd in this window

ACTION_HORIZON = 10  # pi0 / pi0.5 default

# ---------------------------------------------------------------------------
# Per-axis sign flips applied to the model's 6-DoF output before speedL.
# Use this if a specific axis is inverted at runtime — typically a
# coordinate-frame mismatch between training and inference (e.g. base camera
# yaw differs, or you fine-tuned on top of a checkpoint that was trained with
# a different world-axis convention).
#
# Order: [vx, vy, vz, wx, wy, wz]. Diagnostic: read the on-screen `TCP` and
# `vL(m/s)` lines while the robot is running. If commanding `vL[i] > 0`
# decreases `TCP[i]`, set `ACTION_SIGN[i] = -1`.
#
# Default flips X because the policy currently sends the robot backward on
# +X (per observed runtime behavior). Flip more axes only if needed.
ACTION_SIGN = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

# Global slow-down on the 6-DoF velocity (gripper unaffected). 1.0 = the
# raw model output (~0.1 m/s peak from training scale). 0.5 = half speed.
ACTION_SPEED_SCALE = 0.6

# Action magnitude clamps applied AFTER sign-flip + scale. Recording uses
# SCALE_FACTOR=0.1 on a [-1, 1] SpaceMouse signal, so post-scale actions are
# bounded by ~0.1; the limits below are a safety net for outliers.
ACTION_LIN_MAX = 0.20   # |m/s|
ACTION_ROT_MAX = 0.80   # |rad/s|
GRIPPER_THRESHOLD = 0.6

# After a gripper open/close transition, lock out opposite gripper commands
# AND freeze the robot (zero velocity) for this many seconds. This handles
# two failure modes at once:
#   1. The gripper has long mechanical response time; without a pause the
#      robot keeps moving before the jaws have actually closed on the object.
#   2. The model's gripper output is bimodal but occasionally flips (mostly
#      1s with isolated -1 spikes). The lockout debounces those spikes so a
#      single bad sample can't reopen a freshly-closed gripper.
# Tune up if you see partial closes; tune down for snappier behavior.
GRIPPER_FREEZE_TIME = 1   # seconds

# Sliding-window majority filter on the gripper sign. Each tick we push the
# current sample (1 if action[6] > 0 else 0) into a deque of this length, and
# the *committed* gripper command is the majority vote over the window. A
# stray spike has to last more than half the window to flip the state — so a
# single -1 in a stream of +1s is simply outvoted and ignored.
#
# Sizing rule of thumb: window ≈ (max isolated burst length) × 2 + 1.
# At CONTROL_HZ=30 a window of 5 covers ~167 ms of votes. Increase to 7 or
# 9 if you still see flips; decrease to 3 if responses feel too sluggish.
GRIPPER_FILTER_WINDOW = 30

# D435i (base / "observation/image")
D435I_WIDTH  = 640
D435I_HEIGHT = 480
D435I_FPS    = 30

# Wrist OpenCV camera ("observation/wrist_image")
WRIST_CAM_INDEX  = 1
WRIST_CAM_WIDTH  = 640
WRIST_CAM_HEIGHT = 480

STATS_WINDOW = 75

# Gripper RS485
GRIPPER_PORT          = "/dev/ttyUSB0"
GRIPPER_BAUDRATE      = 115200
GRIPPER_TIMEOUT       = 1
GRIPPER_COMMAND_DELAY = 0.2
GRIPPER_CNT           = 0
GRIPPER_WINDOWS       = 30
GRIPPER_HISTORY       = 0

COMMANDS = {
    "clamp_min":     "01 FB 00 01 F4 00 00 2A 94 01 00 6B",
    "clamp_max":     "01 FB 01 01 F4 00 00 00 00 01 00 6B",
    "motor_enable":  "01 F3 AB 01 00 6B",
    "release_block": "01 0E 52 6B",
}

# ---------------------------------------------------------------------------
# Gripper controller (same RS485 protocol as tcp_record.py)
# ---------------------------------------------------------------------------

class GripperController:
    def __init__(self, port, baudrate, timeout, command_delay):
        self.command_delay = command_delay
        self._lock   = Lock()
        self._worker = None
        self.ser = serial.Serial(
            port=port, baudrate=baudrate,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            timeout=timeout,
        )
        if not self.ser.is_open:
            self.ser.open()
        print(f"[Gripper] Connected on {port}")

    def _send(self, command_name):
        hex_str = COMMANDS.get(command_name)
        if hex_str is None:
            print(f"[Gripper] Unknown command: {command_name}")
            return
        data = bytes.fromhex(hex_str.replace(" ", ""))
        with self._lock:
            self.ser.write(data)
            self.ser.flush()
        time.sleep(self.command_delay)

    def send_async(self, command_name):
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = Thread(target=self._send, args=(command_name,), daemon=True)
        self._worker.start()

    def enable(self):
        self._send("motor_enable")

    def close(self):
        if self.ser.is_open:
            self.ser.close()
        print("[Gripper] Serial port closed")

# ---------------------------------------------------------------------------
# Inference stats — control thread writes, display thread reads
# ---------------------------------------------------------------------------

class InferenceStats:
    def __init__(self, window=STATS_WINDOW):
        self._lock = Lock()
        self._infer_times = deque(maxlen=window)
        self._loop_times  = deque(maxlen=window)
        self._chunk_count = 0
        self._last_status   = ""
        self._last_tcp      = None
        self._last_action   = None
        self._last_gripper  = None

    def record(self, infer_ms, loop_ms, status, tcp, action, gripper):
        with self._lock:
            self._infer_times.append(infer_ms)
            self._loop_times.append(loop_ms)
            self._chunk_count += 1
            self._last_status  = status
            self._last_tcp     = tcp
            self._last_action  = action
            self._last_gripper = gripper

    def snapshot(self):
        with self._lock:
            if not self._loop_times:
                return None
            infer = np.array(self._infer_times)
            loop  = np.array(self._loop_times)
            return {
                "chunk_count": self._chunk_count,
                "infer_ms":    float(infer.mean()),
                "loop_ms":     float(loop.mean()),
                "loop_fps":    1000.0 / float(loop.mean()) if loop.mean() > 0 else 0,
                "infer_fps":   1000.0 / float(infer.mean()) if infer.mean() > 0 else 0,
                "infer_max":   float(infer.max()),
                "status":      self._last_status,
                "tcp":         self._last_tcp,
                "action":      self._last_action,
                "gripper":     self._last_gripper,
            }

# ---------------------------------------------------------------------------
# Camera thread — D435i (base) + OpenCV (wrist), no recording
# ---------------------------------------------------------------------------

class CameraThread(Thread):
    """Continuously grabs base + wrist frames; exposes the latest of each at
    model size (256×256) and a combined display frame."""

    def __init__(self, wrist_index, d435i_width, d435i_height, d435i_fps,
                 wrist_width, wrist_height):
        super().__init__(daemon=True, name="CameraThread")
        self._wrist_index = wrist_index
        self._d435i_w = d435i_width
        self._d435i_h = d435i_height
        self._d435i_fps = d435i_fps
        self._wrist_w = wrist_width
        self._wrist_h = wrist_height
        self._lock = Lock()
        self._stop = Event()
        self._base_frame    = None  # (256,256,3) RGB for model
        self._wrist_frame   = None  # (256,256,3) RGB for model
        self._display_frame = None  # combined RGB for display

    def get_model_frames(self):
        with self._lock:
            return self._base_frame, self._wrist_frame

    def get_display_frame(self):
        with self._lock:
            return self._display_frame

    def stop(self):
        self._stop.set()

    def _start_d435i(self):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self._d435i_w, self._d435i_h,
                             rs.format.bgr8, self._d435i_fps)
        pipeline.start(config)
        # Warmup
        for _ in range(5):
            pipeline.wait_for_frames(timeout_ms=5000)
        return pipeline

    def run(self):
        # ---- D435i ----
        try:
            pipeline = self._start_d435i()
            print(f"[Camera] D435i started ({self._d435i_w}x{self._d435i_h}@{self._d435i_fps})")
        except Exception as exc:
            print(f"[Camera] D435i failed: {exc}")
            return

        # ---- Wrist OpenCV ----
        cap_wrist = cv2.VideoCapture(self._wrist_index, cv2.CAP_V4L2)
        cap_wrist.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap_wrist.set(cv2.CAP_PROP_FRAME_WIDTH, self._wrist_w)
        cap_wrist.set(cv2.CAP_PROP_FRAME_HEIGHT, self._wrist_h)
        if not cap_wrist.isOpened():
            print(f"[Camera] Cannot open wrist cam at index {self._wrist_index}")
            pipeline.stop()
            return
        print(f"[Camera] Wrist cam opened (index={self._wrist_index})")

        try:
            while not self._stop.is_set():
                # ---- D435i frame ----
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=200)
                except Exception:
                    continue
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                bgr_b = np.asanyarray(color_frame.get_data())
                rgb_b = cv2.cvtColor(bgr_b, cv2.COLOR_BGR2RGB)
                model_b = cv2.resize(rgb_b, IMAGE_SIZE)
                disp_b  = cv2.resize(rgb_b, DISPLAY_SIZE)
                cv2.putText(disp_b, "base (D435i)", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # ---- Wrist frame ----
                cap_wrist.grab()
                ret, frame_w = cap_wrist.retrieve()
                if not ret:
                    continue
                rgb_w = cv2.cvtColor(frame_w, cv2.COLOR_BGR2RGB)
                model_w = cv2.resize(rgb_w, IMAGE_SIZE)
                disp_w  = cv2.resize(rgb_w, DISPLAY_SIZE)
                cv2.putText(disp_w, "wrist", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                combined = np.hstack([disp_b, disp_w])

                with self._lock:
                    self._base_frame    = model_b
                    self._wrist_frame   = model_w
                    self._display_frame = combined
        finally:
            try:
                pipeline.stop()
            except Exception:
                pass
            cap_wrist.release()
            print("[Camera] Released.")

# ---------------------------------------------------------------------------
# Action safety — TCP-velocity space (no joint-space safety filter)
# ---------------------------------------------------------------------------

def clamp_action(a6: np.ndarray) -> tuple[np.ndarray, str]:
    """Clamp a 6-DoF Cartesian velocity command to the configured magnitude
    limits. Returns (clamped, reason). Returns ("nan",) reason if non-finite.
    """
    if not np.all(np.isfinite(a6)):
        return a6, "nan"

    out = a6.copy()
    lin_norm = np.linalg.norm(out[:3])
    rot_norm = np.linalg.norm(out[3:])
    reason = "ok"
    if lin_norm > ACTION_LIN_MAX:
        out[:3] *= ACTION_LIN_MAX / lin_norm
        reason = f"lin_clamp {lin_norm:.3f}->{ACTION_LIN_MAX:.3f}"
    if rot_norm > ACTION_ROT_MAX:
        out[3:] *= ACTION_ROT_MAX / rot_norm
        reason = (reason + " ; " if reason != "ok" else "") + \
                 f"rot_clamp {rot_norm:.3f}->{ACTION_ROT_MAX:.3f}"
    return out, reason

# ---------------------------------------------------------------------------
# Inference loop — background thread
# ---------------------------------------------------------------------------

def _inference_loop(stop_event: Event, cam: CameraThread, stats: InferenceStats) -> None:
    # ---- Robot ----
    try:
        rtde_r = RTDEReceiveInterface(ROBOT_HOST)
        rtde_c = RTDEControlInterface(ROBOT_HOST)
        print("[Control] Robot connected.")
    except Exception as exc:
        print(f"[Control] Robot connection failed: {exc}")
        stop_event.wait()
        return

    # ---- Gripper ----
    gripper = None
    try:
        gripper = GripperController(
            port=GRIPPER_PORT, baudrate=GRIPPER_BAUDRATE,
            timeout=GRIPPER_TIMEOUT, command_delay=GRIPPER_COMMAND_DELAY,
        )
        gripper.enable()
    except Exception as exc:
        print(f"[Control] Gripper not available: {exc}")

    # ---- Policy server ----
    # Retry the websocket connect: serve_policy.py can take 30-90 s to finish
    # loading the checkpoint and initialize the GPU before it accepts the
    # handshake. The library default open_timeout (10 s) is too short.
    print(f"[Control] Connecting to ws://{SERVER_HOST}:{SERVER_PORT} ...")
    uri = f"ws://{SERVER_HOST}:{SERVER_PORT}"
    ws = None
    deadline = time.perf_counter() + 180.0   # give up after 3 min
    while ws is None:
        if stop_event.is_set():
            return
        try:
            ws = websockets.sync.client.connect(
                uri, compression=None, max_size=None,
                ping_interval=None, ping_timeout=None, close_timeout=600,
                open_timeout=120,
            )
        except (TimeoutError, ConnectionRefusedError, OSError) as exc:
            if time.perf_counter() > deadline:
                print(f"[Control] Could not reach {uri} in 180 s: {exc}")
                print("          Is `serve_policy.py` running? Check with: "
                      "ss -ltnp | grep 8000")
                return
            print(f"[Control] {type(exc).__name__}: {exc} — retrying in 2 s...")
            stop_event.wait(2.0)
    metadata = msgpack_numpy.unpackb(ws.recv())
    raw_policy = websocket_client_policy.WebsocketClientPolicy.__new__(
        websocket_client_policy.WebsocketClientPolicy
    )
    raw_policy._ws = ws
    raw_policy._packer = msgpack_numpy.Packer()
    raw_policy._uri = uri
    raw_policy._api_key = None
    raw_policy._server_metadata = metadata
    policy = action_chunk_broker.ActionChunkBroker(
        raw_policy, action_horizon=ACTION_HORIZON
    )
    print("[Control] Connected. Inference loop started (Ctrl+C to stop).")

    gripper_state = 0.0
    # Class A1: freeze + lock + policy chunk reset RE-ENABLED to break the
    # mid-actuation feedback loop. Filter (A2) stays disabled.
    gripper_lock_until = 0.0   # perf_counter timestamp; while now < this, freeze
    # gripper_samples = deque(maxlen=GRIPPER_FILTER_WINDOW)  # filter (A2) — disabled
    first_infer = True
    global GRIPPER_CNT
    global GRIPPER_HISTORY
    global GRIPPER_WINDOWS

    try:
        while not stop_event.is_set():
            loop_start = time.perf_counter()

            if rtde_r.getRobotMode() != 7:
                print("[Control] Robot not ready (mode != 7), standby ...")
                stop_event.wait(timeout=1)
                policy.reset()
                first_infer = True
                continue

            # ---- 1. Read observation ----
            tcp_pose = rtde_r.getActualTCPPose()  # [x, y, z, rx, ry, rz]
            # 11-dim state: [pos(3), 6D rotation, gripper, gripper] —
            # matches lerobot_ur3_record.py / tcp_record.py training schema.
            rot_6d = axis_angle_to_rotation_6d(tcp_pose[3:6])
            state = np.array(
                list(tcp_pose[:3]) + list(rot_6d) + [gripper_state, gripper_state],
                dtype=np.float32,
            )

            base_img, wrist_img = cam.get_model_frames()
            if base_img is None or wrist_img is None:
                print("[Control] Waiting for camera frames...")
                time.sleep(0.1)
                continue

            # ---- 2. Build observation in LIBERO schema ----
            obs = {
                "observation/state":       state,
                "observation/image":       base_img,
                "observation/wrist_image": wrist_img,
                "prompt":                  TASK_PROMPT,
            }

            # ---- 3. Inference ----
            if first_infer:
                print("[Control] Sending first inference request "
                      "(server warmup: ~5-30 s for PyTorch, "
                      "2-5 min for JAX JIT)...", flush=True)
            t_infer = time.perf_counter()
            result = policy.infer(obs)
            infer_ms = (time.perf_counter() - t_infer) * 1000
            if first_infer:
                print(f"[Control] First inference done ({infer_ms:.0f} ms)")
                first_infer = False
            action = np.asarray(result["actions"], dtype=np.float64)
           
            if action.shape[-1] < 7:
                print(f"[Control] Unexpected action shape: {action.shape}")
                continue

            # ---- 4. Sign flip + slow-down + safety clamp ----
            cart_vel = action[:6] * ACTION_SIGN * ACTION_SPEED_SCALE
            cart_vel, reason = clamp_action(cart_vel)
            if reason == "nan":
                print("[Safety] NaN/Inf in action, skipped")
                stats.record(infer_ms, (time.perf_counter() - loop_start) * 1000,
                             "nan", tcp_pose[:3], action[:6].tolist(), float(action[6]))
                continue

            # ---- 4b. Gripper freeze (A1) — break mid-actuation OOD loop ----
            # While gripper is mid-actuation we DON'T trust the model's
            # gripper output (it sees a half-closed/half-open jaw image
            # which is OOD for training data with instant transitions).
            # We zero out arm motion and skip gripper transitions until
            # the lock expires.
            now = time.perf_counter()
            gripper_locked = now < gripper_lock_until
            if gripper_locked:
                cart_vel = np.zeros(6)
                reason = f"gripper_freeze {gripper_lock_until - now:.2f}s"

            # ---- 5. Execute Cartesian velocity ----
            rtde_c.speedL(cart_vel.tolist(), acceleration=SPEEDL_ACCEL, time=SPEEDL_TIME)

            # ---- 6. Gripper (raw threshold + A1 lock; filter A2 disabled) ----
            target_gripper = float(action[6])

            # --- Filter (A2) DISABLED ---
            # gripper_samples.append(1 if target_gripper > 0.9 else 0)
            # if len(gripper_samples) < GRIPPER_FILTER_WINDOW:
            #     filtered_target = gripper_state
            # else:
            #     votes_close = sum(gripper_samples)
            #     filtered_target = 1.0 if votes_close * 2 > GRIPPER_FILTER_WINDOW else 0.0

            print(f"[Gripper] raw={target_gripper:+.3f}  t={now:.2f}  "
                  f"locked={gripper_locked}")

            # Skip gripper transitions while locked — gives the gripper time to
            # physically reach the commanded state before we let the model
            # influence anything (its observations are OOD during actuation).
            if gripper is not None and not gripper_locked:
                # Convention (matches recorder): action[6] / gripper_state
                #   1.0 → OPEN  (release_block, then clamp_max)
                #   0.0 → CLOSE (clamp_min)
                new_state = 1.0 if target_gripper > GRIPPER_THRESHOLD else 0.0
                if new_state != gripper_state:
                    if new_state == 1.0:
                        gripper._send("release_block")
                        gripper.send_async("clamp_max")
                        print(f"[Gripper] Open  (raw={target_gripper:+.3f}) "
                              f"— freeze {GRIPPER_FREEZE_TIME:.2f}s")
                    else:
                        gripper.send_async("clamp_min")
                        print(f"[Gripper] Close (raw={target_gripper:+.3f}) "
                              f"— freeze {GRIPPER_FREEZE_TIME:.2f}s")
                    gripper_state = new_state
                    gripper_lock_until = now + GRIPPER_FREEZE_TIME
                    # Drop any queued actions so the next tick fetches a fresh
                    # chunk planned from the post-actuation observation.
                    policy.reset()

            loop_ms = (time.perf_counter() - loop_start) * 1000
            stats.record(infer_ms, loop_ms, reason,
                         tcp_pose[:3], cart_vel.tolist(), target_gripper)

            # ---- Pacing ----
            elapsed = time.perf_counter() - loop_start
            sleep_t = CONTROL_DT - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    finally:
        try:
            rtde_c.speedStop()
            rtde_c.stopScript()
        except Exception as e:
            print(f"[Control] stopScript: {e}")
        if gripper is not None:
            gripper.close()
        print("[Control] Stopped.")

# ---------------------------------------------------------------------------
# Display — tkinter (main thread)
# ---------------------------------------------------------------------------

def _run_display(stop_event: Event, cam: CameraThread, stats: InferenceStats) -> None:
    root = tk.Tk()
    root.title("UR3 LIBERO Inference — TCP velocity")
    root.protocol("WM_DELETE_WINDOW", lambda: (stop_event.set(), root.destroy()))
    signal.signal(signal.SIGINT, lambda *_: (stop_event.set(), root.destroy()))

    label = tk.Label(root)
    label.pack()

    stats_label = tk.Label(
        root, text="Waiting for inference ...", font=("Courier", 11),
        justify=tk.LEFT, anchor="w", bg="black", fg="lime",
    )
    stats_label.pack(fill=tk.X)

    def update_frame():
        frame = cam.get_display_frame()
        if frame is not None:
            img = ImageTk.PhotoImage(Image.fromarray(frame))
            label.imgtk = img
            label.configure(image=img)

        snap = stats.snapshot()
        if snap is not None:
            tcp = snap["tcp"]
            act = snap["action"]
            grip = snap["gripper"]
            lines = [
                f"Chunks: {snap['chunk_count']}   "
                f"Loop: {snap['loop_fps']:.1f} FPS ({snap['loop_ms']:.1f} ms)   "
                f"Infer: {snap['infer_ms']:.1f} ms ({snap['infer_fps']:.1f}/s, max {snap['infer_max']:.0f})   "
                f"Status: {snap['status']}",
            ]
            if tcp is not None:
                lines.append(
                    f"TCP: [{tcp[0]:+.4f}, {tcp[1]:+.4f}, {tcp[2]:+.4f}] m"
                )
            if act is not None:
                a = act
                lines.append(
                    f"vL(m/s): [{a[0]:+.3f}, {a[1]:+.3f}, {a[2]:+.3f}]   "
                    f"vR(rad/s): [{a[3]:+.3f}, {a[4]:+.3f}, {a[5]:+.3f}]"
                )
            if grip is not None:
                lines.append(f"gripper: {grip:+.3f}  (>{GRIPPER_THRESHOLD:.2f} = close)")
            stats_label.configure(text="\n".join(lines))

        root.after(33, update_frame)

    def check_stop():
        if stop_event.is_set():
            root.destroy()
            return
        root.after(200, check_stop)

    update_frame()
    check_stop()
    root.mainloop()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    stop_event = Event()
    stats = InferenceStats()

    cam = CameraThread(
        wrist_index=WRIST_CAM_INDEX,
        d435i_width=D435I_WIDTH, d435i_height=D435I_HEIGHT, d435i_fps=D435I_FPS,
        wrist_width=WRIST_CAM_WIDTH, wrist_height=WRIST_CAM_HEIGHT,
    )
    cam.start()

    control = Thread(
        target=_inference_loop, args=(stop_event, cam, stats),
        daemon=True, name="InferenceThread",
    )
    control.start()

    _run_display(stop_event, cam, stats)
    stop_event.set()
    print("\nShutting down ...")
    control.join(timeout=3)
    cam.stop()
    print("Done.")


if __name__ == "__main__":
    main()
