"""
tcp_infer_no_gripper_lock.py — UR3 inference WITHOUT gripper freeze/lock/filter.

Same as tcp_infer.py but all gripper lockout, freeze, and majority-vote filter
logic has been removed. Gripper transitions happen immediately based on the raw
model output thresholded against GRIPPER_THRESHOLD — no debounce, no arm freeze,
no policy reset on transition.

Use this version when training data has ramped gripper state (via
ramp_gripper_state.py) so the model has learned smooth transitions and the
mid-actuation OOD problem is mitigated by the ramp.

Run
---
1. Start the policy server:
       cd /home/robotics/Desktop/Project_UR3/openpi
       uv run scripts/serve_policy.py policy:checkpoint \\
           --policy.config <your_libero_config> --policy.dir <ckpt_dir>

2. Run this script:
       conda activate spacemouse-ur
       python3 tcp_infer_no_gripper_lock.py
"""

import argparse
import os
import pickle
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
from openpi_client import websocket_client_policy, msgpack_numpy

# ---------------------------------------------------------------------------
# Rotation utility — must match lerobot_ur3_record.py / tcp_record.py exactly
# ---------------------------------------------------------------------------


def axis_angle_to_rotation_6d(axis_angle) -> np.ndarray:
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
SERVER_HOST = "129.10.45.118"
SERVER_PORT = 8000

IMAGE_SIZE   = (256, 256)
DISPLAY_SIZE = (640, 480)

TASK_PROMPT = "open the pot by removing its lid"

# {"task_index": 0, "task": "pick up the pink cylinder and place it in the orange box"}
# {"task_index": 1, "task": "pick up the white glass and put on a brown coaster"}
# {"task_index": 2, "task": "Remove cup from nested cups"}
# {"task_index": 3, "task": "open the pot by removing its lid"}
# {"task_index": 4, "task": "Single-finger push to blue marker"}


CONTROL_HZ   = 30
CONTROL_DT   = 1.0 / CONTROL_HZ
SPEEDL_ACCEL = 0.25
SPEEDL_TIME  = 0.08  # ~2.5 control periods — robot holds velocity between commands

ACTION_HORIZON = 10

ACTION_SIGN = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
ACTION_SPEED_SCALE = 0.7

ACTION_LIN_MAX = 0.20
ACTION_ROT_MAX = 0.80
GRIPPER_CLOSE_THRESH = 0.1
GRIPPER_OPEN_THRESH  = 0.9

# D435i (base)
D435I_WIDTH  = 640
D435I_HEIGHT = 480
D435I_FPS    = 30

# Wrist OpenCV camera
WRIST_CAM_INDEX  = 6
WRIST_CAM_WIDTH  = 640
WRIST_CAM_HEIGHT = 480
    
STATS_WINDOW = 75

# Gripper RS485
GRIPPER_PORT          = "/dev/ttyUSB0"
GRIPPER_BAUDRATE      = 115200
GRIPPER_TIMEOUT       = 1
GRIPPER_COMMAND_DELAY = 0.2

COMMANDS = {
    "clamp_min": "01 FB 00 01 F4 00 00 2A 94 01 00 6B",  # Close gripper
    "clamp_max": "01 FB 01 01 F4 00 00 00 00 01 00 6B",  # Open to maximum
    "motor_enable": "01 F3 AB 01 00 6B",                 # Enable motor
    "release_block": "01 0E 52 6B",                      # Release stall/block
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
# Inference stats
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
# Camera thread
# ---------------------------------------------------------------------------

class CameraThread(Thread):
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
        self._base_frame    = None
        self._wrist_frame   = None
        self._display_frame = None

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
        for _ in range(5):
            pipeline.wait_for_frames(timeout_ms=5000)
        return pipeline

    def run(self):
        try:
            pipeline = self._start_d435i()
            print(f"[Camera] D435i started ({self._d435i_w}x{self._d435i_h}@{self._d435i_fps})")
        except Exception as exc:
            print(f"[Camera] D435i failed: {exc}")
            return

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
# Action safety
# ---------------------------------------------------------------------------

def clamp_action(a6: np.ndarray) -> tuple[np.ndarray, str]:
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
# Prefetch broker — fetches next action chunk in background while current
# chunk is executing, eliminating the stall at every chunk boundary.
# ---------------------------------------------------------------------------

class PrefetchBroker:
    def __init__(self, raw_policy, action_horizon, prefetch_steps=3):
        self._policy        = raw_policy
        self._horizon       = action_horizon
        self._prefetch_steps = prefetch_steps
        self._chunk         = None
        self._cur_step      = 0
        self._last_results  = None  # kept for API compatibility
        self._lock          = Lock()
        self._prefetch_thread  = None
        self._prefetch_result  = None

    def reset(self):
        if self._prefetch_thread is not None:
            self._prefetch_thread.join(timeout=1)
        self._chunk = None
        self._cur_step = 0
        self._last_results = None
        self._prefetch_thread = None
        self._prefetch_result = None

    def _do_fetch(self, obs):
        try:
            result = self._policy.infer(obs)
            chunk = np.asarray(result["actions"], dtype=np.float64)
            with self._lock:
                self._prefetch_result = chunk
        except Exception as e:
            print(f"[Prefetch] Fetch failed: {e}")

    def _start_prefetch(self, obs):
        if self._prefetch_thread is None or not self._prefetch_thread.is_alive():
            with self._lock:
                self._prefetch_result = None
            self._prefetch_thread = Thread(
                target=self._do_fetch, args=(obs,), daemon=True, name="PrefetchThread"
            )
            self._prefetch_thread.start()

    def infer(self, obs):
        is_new_chunk = self._last_results is None  # matches ActionChunkBroker semantics

        if is_new_chunk:
            if self._prefetch_thread is not None:
                self._prefetch_thread.join()  # wait if still in flight (should be brief)
            with self._lock:
                prefetched = self._prefetch_result
                self._prefetch_result = None
            self._prefetch_thread = None

            if prefetched is not None:
                self._chunk = prefetched
                print("[Prefetch] Used prefetched chunk (zero stall)")
            else:
                t0 = time.perf_counter()
                result = self._policy.infer(obs)
                self._chunk = np.asarray(result["actions"], dtype=np.float64)
                print(f"[Prefetch] Blocking fetch ({(time.perf_counter()-t0)*1000:.0f} ms)")
            self._cur_step = 0
            self._last_results = self._chunk

        action = self._chunk[self._cur_step]
        self._cur_step += 1

        # Signal chunk exhaustion so next tick sees is_new_chunk=True
        if self._cur_step >= self._horizon:
            self._last_results = None

        # Start prefetch when PREFETCH_STEPS steps remain in this chunk
        steps_left = self._horizon - self._cur_step
        if steps_left <= self._prefetch_steps and self._prefetch_thread is None:
            self._start_prefetch(obs)

        return {"actions": action}


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

def _inference_loop(stop_event: Event, cam: CameraThread, stats: InferenceStats,
                    chunk_barrier: bool = True) -> None:
    try:
        rtde_r = RTDEReceiveInterface(ROBOT_HOST)
        rtde_c = RTDEControlInterface(ROBOT_HOST)
        print("[Control] Robot connected.")
    except Exception as exc:
        print(f"[Control] Robot connection failed: {exc}")
        stop_event.wait()
        return

    gripper = None
    try:
        gripper = GripperController(
            port=GRIPPER_PORT, baudrate=GRIPPER_BAUDRATE,
            timeout=GRIPPER_TIMEOUT, command_delay=GRIPPER_COMMAND_DELAY,
        )
        gripper.enable()
    except Exception as exc:
        print(f"[Control] Gripper not available: {exc}")

    print(f"[Control] Connecting to ws://{SERVER_HOST}:{SERVER_PORT} ...")
    uri = f"ws://{SERVER_HOST}:{SERVER_PORT}"
    ws = None
    deadline = time.perf_counter() + 180.0
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
    policy = PrefetchBroker(raw_policy, action_horizon=ACTION_HORIZON)
    print("[Control] Connected. Inference loop started (Ctrl+C to stop).")

    gripper_state = 1.0
    first_infer = True
    chunk_num = 0
    tick_n = 0
    SLOW_TICK_MS = 50.0
    DEBUG_EVERY = 30
    robot_mode = 7  # cached; re-checked every MODE_CHECK_INTERVAL ticks
    MODE_CHECK_INTERVAL = 10

    try:
        while not stop_event.is_set():
            loop_start = time.perf_counter()

            t_mode_start = time.perf_counter()
            if tick_n % MODE_CHECK_INTERVAL == 0:
                robot_mode = rtde_r.getRobotMode()
            if robot_mode != 7:
                print("[Control] Robot not ready (mode != 7), standby ...")
                stop_event.wait(timeout=1)
                policy.reset()
                first_infer = True
                continue
            t_mode_ms = (time.perf_counter() - t_mode_start) * 1000

            # ---- 1. Read observation ----
            t_obs_start = time.perf_counter()
            tcp_pose = rtde_r.getActualTCPPose()
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
            t_obs_ms = (time.perf_counter() - t_obs_start) * 1000

            # ---- 2. Build observation ----
            obs = {
                "observation/state":       state,
                "observation/image":       base_img,
                "observation/wrist_image": wrist_img,
                "prompt":                  TASK_PROMPT,
            }

            # ---- 3. Inference ----
            if first_infer:
                print("[Control] Sending first inference request ...", flush=True)
                dump_dir = "/tmp/tcp_infer_obs_dump"
                os.makedirs(dump_dir, exist_ok=True)
                with open(os.path.join(dump_dir, "obs_first.pkl"), "wb") as _f:
                    pickle.dump(obs, _f)
                print(f"[Control] First obs dumped to {dump_dir}/obs_first.pkl", flush=True)
            t_infer = time.perf_counter()
            is_new_chunk = policy._last_results is None
            result = policy.infer(obs)
            infer_ms = (time.perf_counter() - t_infer) * 1000
            if is_new_chunk:
                chunk_num += 1
            step_in_chunk = policy._cur_step
            if first_infer:
                print(f"[Control] First inference done ({infer_ms:.0f} ms)")
                first_infer = False
            action = np.asarray(result["actions"], dtype=np.float64)

            if action.shape[-1] < 7:
                print(f"[Control] Unexpected action shape: {action.shape}")
                continue

            # ---- 4. Sign flip + slow-down + safety clamp ----
            t_act_start = time.perf_counter()
            cart_vel_raw = action[:6] * ACTION_SIGN * ACTION_SPEED_SCALE
            lin_raw = float(np.linalg.norm(cart_vel_raw[:3]))
            rot_raw = float(np.linalg.norm(cart_vel_raw[3:]))
            cart_vel, reason = clamp_action(cart_vel_raw)
            if reason == "nan":
                print("[Safety] NaN/Inf in action, skipped")
                stats.record(infer_ms, (time.perf_counter() - loop_start) * 1000,
                             "nan", tcp_pose[:3], action[:6].tolist(), float(action[6]))
                continue
            t_act_ms = (time.perf_counter() - t_act_start) * 1000

            # ---- 5. Execute Cartesian velocity ----
            t_speedL_start = time.perf_counter()
            rtde_c.speedL(cart_vel.tolist(), acceleration=SPEEDL_ACCEL, time=SPEEDL_TIME)
            t_speedL_ms = (time.perf_counter() - t_speedL_start) * 1000

            # ---- 6. Gripper ----
            t_grip_start = time.perf_counter()
            target_gripper = float(action[6])
            chunk_tag = f"C{chunk_num}:{step_in_chunk}/{ACTION_HORIZON}"
            eval_gripper = is_new_chunk or not chunk_barrier
            if is_new_chunk:
                print(f"[Gripper] raw={target_gripper:+.3f}  state={gripper_state:.1f}  <<< {chunk_tag} NEW >>>")
            elif not chunk_barrier:
                print(f"[Gripper] raw={target_gripper:+.3f}  state={gripper_state:.1f}      {chunk_tag}")
            else:
                print(f"[Gripper] raw={target_gripper:+.3f}  state={gripper_state:.1f}      {chunk_tag} (hold)")

            if gripper is not None and eval_gripper:
                if target_gripper < GRIPPER_CLOSE_THRESH:
                    new_state = 0.0
                elif target_gripper > GRIPPER_OPEN_THRESH:
                    new_state = 1.0
                else:
                    new_state = gripper_state

                if new_state != gripper_state:
                    if new_state == 1.0:
                        gripper._send("release_block")
                        gripper.send_async("clamp_max")
                        print(f"[Gripper] Open  (raw={target_gripper:+.3f})")
                    else:
                        gripper.send_async("clamp_min")
                        print(f"[Gripper] Close (raw={target_gripper:+.3f})")
                    gripper_state = new_state
            t_grip_ms = (time.perf_counter() - t_grip_start) * 1000

            loop_ms = (time.perf_counter() - loop_start) * 1000
            stats.record(infer_ms, loop_ms, reason,
                         tcp_pose[:3], cart_vel.tolist(), target_gripper)

            # ---- Per-tick breakdown print ----
            tick_n += 1
            slow = loop_ms > SLOW_TICK_MS
            periodic = tick_n % DEBUG_EVERY == 0
            if slow or periodic or is_new_chunk:
                tag = "SLOW" if slow else ("CHK " if is_new_chunk else "    ")
                print(
                    f"[tick {tick_n:5d} {tag}] total={loop_ms:6.1f}ms  "
                    f"mode={t_mode_ms:4.1f}  obs={t_obs_ms:5.1f}  "
                    f"infer={infer_ms:6.1f}  act={t_act_ms:4.1f}  "
                    f"speedL={t_speedL_ms:5.1f}  grip={t_grip_ms:5.1f}  | "
                    f"lin={lin_raw:.3f}m/s rot={rot_raw:.3f}rad/s "
                    f"a6={target_gripper:+.2f} g={gripper_state:.1f} "
                    f"clamp={reason}"
                )

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
    root.title("UR3 Inference — no gripper lock")
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
                lines.append(f"gripper: {grip:+.3f}  (<{GRIPPER_CLOSE_THRESH} close, >{GRIPPER_OPEN_THRESH} open)")
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

def _reset_to_home():
    import math
    home = [0, -math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, 0]
    print("[Reset] Moving arm to home position ...")
    try:
        rtde_c = RTDEControlInterface(ROBOT_HOST)
        rtde_c.moveJ(home, speed=0.5, acceleration=0.5)
        rtde_c.stopScript()
        print("[Reset] Arm at home.")
    except Exception as e:
        print(f"[Reset] Arm move failed: {e}")
    print("[Reset] Opening gripper ...")
    try:
        ser = serial.Serial(GRIPPER_PORT, GRIPPER_BAUDRATE, timeout=1)
        for cmd in ("motor_enable", "release_block", "clamp_max"):
            ser.write(bytes.fromhex(COMMANDS[cmd].replace(" ", "")))
            time.sleep(0.3)
        ser.close()
        print("[Reset] Gripper open.")
    except Exception as e:
        print(f"[Reset] Gripper open failed: {e}")


def main():
    global SERVER_HOST, SERVER_PORT
    parser = argparse.ArgumentParser(description="UR3 inference — no gripper lock")
    parser.add_argument("--no-barrier", action="store_true",
                        help="evaluate gripper every step instead of chunk boundary only")
    parser.add_argument("--server-host", default=SERVER_HOST, help="Policy server host (default: %(default)s)")
    parser.add_argument("--server-port", type=int, default=SERVER_PORT, help="Policy server port (default: %(default)s)")
    parser.add_argument("--reset-after", action="store_true",
                        help="move arm to home and open gripper after shutdown")
    args = parser.parse_args()

    SERVER_HOST = args.server_host
    SERVER_PORT = args.server_port
    chunk_barrier = not args.no_barrier
    print(f"[Config] chunk_barrier={chunk_barrier}  (--no-barrier {'ON' if args.no_barrier else 'off'})")

    stop_event = Event()
    stats = InferenceStats()

    cam = CameraThread(
        wrist_index=WRIST_CAM_INDEX,
        d435i_width=D435I_WIDTH, d435i_height=D435I_HEIGHT, d435i_fps=D435I_FPS,
        wrist_width=WRIST_CAM_WIDTH, wrist_height=WRIST_CAM_HEIGHT,
    )
    cam.start()

    control = Thread(
        target=_inference_loop, args=(stop_event, cam, stats, chunk_barrier),
        daemon=True, name="InferenceThread",
    )
    control.start()

    _run_display(stop_event, cam, stats)
    stop_event.set()
    print("\nShutting down ...")
    control.join(timeout=10)
    if control.is_alive():
        print("[Warn] Inference thread did not stop cleanly.")
    cam.stop()
    if args.reset_after:
        time.sleep(1.0)  # let RTDE connection from inference thread fully close
        _reset_to_home()
    print("Done.")


if __name__ == "__main__":
    main()
