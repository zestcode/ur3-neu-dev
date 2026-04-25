"""
teleop_vision_record_depth.py — UR3 SpaceMouse teleoperation with D435i + wrist camera.

Multi-episode recording: press Space to end current episode and start the next.
Close window or Ctrl+C to stop entirely.

Threads
-------
- Main thread      : tkinter live preview + keyboard events
- ControlThread    : 100 Hz teleoperation loop; records joint state
- D435iRecorder    : RealSense D435i RGB capture + MP4 recording (per episode)
- CamRecorder      : wrist OpenCV camera capture + MP4 recording (per episode)

Output — LeRobot v2.1 format
-----
  recordings/<timestamp>/
    data/chunk-000/episode_000000.parquet, episode_000001.parquet, ...
    videos/chunk-000/observation.images.cam_high/episode_000000.mp4, ...
    videos/chunk-000/observation.images.cam_wrist/episode_000000.mp4, ...
    meta/info.json, episodes.jsonl, tasks.jsonl, episodes_stats.jsonl

Usage
-----
    conda activate spacemouse-ur
    python3 teleop_vision_record_depth.py
"""

import signal
import tkinter as tk
from PIL import Image, ImageTk

from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from spnav import spnav_open, spnav_poll_event, spnav_close, SpnavMotionEvent, SpnavButtonEvent
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.configs import ColorMode
from threading import Thread, Event, Lock
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import pyrealsense2 as rs
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import json
import serial
import time
import cv2


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROBOT_HOST = "192.168.0.2"
SCALE_FACTOR = 0.1

GRIPPER_PORT = "/dev/ttyUSB0"
GRIPPER_BAUDRATE = 115200
GRIPPER_TIMEOUT = 1
GRIPPER_COMMAND_DELAY = 0.2

D435I_WIDTH  = 640
D435I_HEIGHT = 480
D435I_FPS    = 30    # D435i hardware only supports 6/15/30; rate-limited to LEROBOT_FPS when writing

WRIST_CAM_INDEX  = 1
WRIST_CAM_WIDTH  = 640
WRIST_CAM_HEIGHT = 480
WRIST_CAM_FPS    = 25

OPENPI_IMAGE_SIZE = (224, 224)

LEROBOT_FPS = 25
LEROBOT_CAM_NAMES = [
    "observation.images.cam_high",
    "observation.images.cam_wrist",
]
TASK_INSTRUCTION_DEFAULT = "pick up the pink cylinder and place it into the orange box"

COMMANDS = {
    "clamp_min":    "01 FB 00 01 F4 00 00 2A 94 01 00 6B",
    "clamp_max":    "01 FB 01 01 F4 00 00 00 00 01 00 6B",
    "motor_enable": "01 F3 AB 01 00 6B",
    "release_block": "01 0E 52 6B",
}

CHUNK = "chunk-000"

# ---------------------------------------------------------------------------
# SpaceMouse thread
# ---------------------------------------------------------------------------

class Spacemouse(Thread):
    def __init__(self, max_value=300, deadzone=(0, 0, 0, 0, 0, 0), dtype=np.float32):
        if np.issubdtype(type(deadzone), np.number):
            deadzone = np.full(6, fill_value=deadzone, dtype=dtype)
        else:
            deadzone = np.array(deadzone, dtype=dtype)
        assert (deadzone >= 0).all()
        super().__init__(daemon=True)
        self.stop_event = Event()
        self.max_value = max_value
        self.dtype = dtype
        self.deadzone = deadzone
        self.motion_event = SpnavMotionEvent([0, 0, 0], [0, 0, 0], 0)
        self.button_state = defaultdict(lambda: False)
        self.tx_zup_spnav = np.array([
            [0,  0, -1],
            [1,  0,  0],
            [0,  1,  0],
        ], dtype=dtype)

    def get_motion_state(self):
        me = self.motion_event
        state = np.array(me.translation + me.rotation, dtype=self.dtype) / self.max_value
        is_dead = (-self.deadzone < state) & (state < self.deadzone)
        state[is_dead] = 0
        return state

    def get_motion_state_transformed(self):
        state = self.get_motion_state()
        tf_state = np.zeros_like(state)
        tf_state[:3] = self.tx_zup_spnav @ state[:3]
        tf_state[3:] = self.tx_zup_spnav @ state[3:]
        return tf_state * SCALE_FACTOR

    def is_button_pressed(self, button_id):
        return self.button_state[button_id]

    def stop(self):
        self.stop_event.set()
        self.join()

    def run(self):
        spnav_open()
        try:
            while not self.stop_event.is_set():
                event = spnav_poll_event()
                if isinstance(event, SpnavMotionEvent):
                    self.motion_event = event
                elif isinstance(event, SpnavButtonEvent):
                    self.button_state[event.bnum] = event.press
                else:
                    time.sleep(1 / 200)
        finally:
            spnav_close()

# ---------------------------------------------------------------------------
# Gripper controller
# ---------------------------------------------------------------------------

class GripperController:
    def __init__(self, port, baudrate, timeout, command_delay):
        self.command_delay = command_delay
        self._lock = Lock()
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
        print(f"Gripper connected on {port}")

    def _send(self, command_name):
        hex_str = COMMANDS.get(command_name)
        if hex_str is None:
            return
        data = bytes.fromhex(hex_str.replace(" ", ""))
        with self._lock:
            self.ser.write(data)
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
        print("Gripper serial port closed")

# ---------------------------------------------------------------------------
# D435i recorder thread
# ---------------------------------------------------------------------------

class D435iRecorderThread(Thread):
    """D435i capture + recording with separate grab/write threads.

    A background grab thread polls the RealSense pipeline as fast as possible
    and updates the latest frame for display.  The main thread (run) pulls
    the latest frame at a steady 30 FPS cadence and writes it to the video,
    so disk I/O never blocks the display update.
    """

    def __init__(self, output_path: Path, width: int, height: int,
                 fps: int, stop_event: Event):
        super().__init__(daemon=True, name="D435iRecorder")
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.stop_event = stop_event
        self.frame_count = 0
        self.error: Exception | None = None
        self._lock = Lock()
        self._display_frame = None   # full-res RGB for tkinter preview
        self._latest_rgb = None      # full-res RGB for video writer

    def get_display_frame(self):
        with self._lock:
            return self._display_frame

    def _grab_loop(self, pipeline):
        """Background: grab frames from D435i, update display + latest."""
        while not self.stop_event.is_set():
            try:
                frames = pipeline.wait_for_frames(timeout_ms=200)
            except Exception:
                continue
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            bgr = np.asanyarray(color_frame.get_data())
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            with self._lock:
                self._display_frame = rgb.copy()
                self._latest_rgb = rgb

    def run(self):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.width, self.height,
                             rs.format.bgr8, self.fps)
        try:
            pipeline.start(config)
            for _ in range(5):
                pipeline.wait_for_frames(timeout_ms=5000)
        except Exception as exc:
            self.error = exc
            print(f"[D435i] Failed to start: {exc}")
            return

        # Start background grab thread
        grabber = Thread(target=self._grab_loop, args=(pipeline,),
                         daemon=True, name="D435iGrab")
        grabber.start()

        out_w, out_h = OPENPI_IMAGE_SIZE
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self.output_path), fourcc, LEROBOT_FPS, (out_w, out_h))
        if not writer.isOpened():
            self.error = RuntimeError(f"VideoWriter failed for {self.output_path}")
            self.stop_event.set()
            grabber.join(timeout=3)
            pipeline.stop()
            return

        print(f"[D435i] Recording -> {self.output_path}")
        frame_interval = 1.0 / LEROBOT_FPS
        next_write_time = time.perf_counter()
        try:
            while not self.stop_event.is_set():
                now = time.perf_counter()
                sleep_dur = next_write_time - now
                if sleep_dur > 0:
                    time.sleep(sleep_dur)

                with self._lock:
                    rgb = self._latest_rgb
                if rgb is None:
                    time.sleep(0.001)
                    continue

                small = cv2.resize(rgb, OPENPI_IMAGE_SIZE)
                writer.write(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
                self.frame_count += 1
                next_write_time += frame_interval
                if next_write_time < time.perf_counter():
                    next_write_time = time.perf_counter() + frame_interval
        except Exception as exc:
            self.error = exc
        finally:
            self.stop_event.set()
            grabber.join(timeout=3)
            writer.release()
            pipeline.stop()
            print(f"[D435i] Stopped. Frames: {self.frame_count}")

# ---------------------------------------------------------------------------
# Wrist camera recorder thread
# ---------------------------------------------------------------------------

class CameraRecorderThread(Thread):
    def __init__(self, cam_index: int, output_path: Path,
                 fps: int, width: int, height: int,
                 stop_event: Event):
        super().__init__(daemon=True, name=f"CamRecorder-{cam_index}")
        self.cam_index = cam_index
        self.output_path = output_path
        self.fps = fps
        self.width = width
        self.height = height
        self.stop_event = stop_event
        self.frame_count = 0
        self.error: Exception | None = None
        self._display_lock = Lock()
        self._display_frame = None

    def get_display_frame(self):
        with self._display_lock:
            return self._display_frame

    def run(self):
        cfg = OpenCVCameraConfig(
            index_or_path=self.cam_index,
            fps=self.fps, width=self.width, height=self.height,
            color_mode=ColorMode.RGB,
        )
        out_w, out_h = OPENPI_IMAGE_SIZE
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, (out_w, out_h))
        if not writer.isOpened():
            self.error = RuntimeError(f"VideoWriter failed")
            return

        print(f"[CamRecorder-{self.cam_index}] Recording -> {self.output_path}")
        frame_interval = 1.0 / self.fps
        next_write_time = time.perf_counter()
        try:
            with OpenCVCamera(cfg) as cam:
                while not self.stop_event.is_set():
                    try:
                        frame = cam.async_read(timeout_ms=200)
                    except TimeoutError:
                        continue
                    frame = cv2.resize(frame, (self.width, self.height))
                    with self._display_lock:
                        self._display_frame = frame.copy()
                    now = time.perf_counter()
                    if now >= next_write_time:
                        small = cv2.resize(frame, OPENPI_IMAGE_SIZE)
                        writer.write(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
                        self.frame_count += 1
                        next_write_time += frame_interval
                        # 相机帧率低于目标时，防止 next_write_time 越落越远
                        if next_write_time < now:
                            next_write_time = now + frame_interval
        except Exception as exc:
            self.error = exc
        finally:
            writer.release()
            print(f"[CamRecorder-{self.cam_index}] Stopped. Frames: {self.frame_count}")

# ---------------------------------------------------------------------------
# Save one episode (parquet + per-episode stats line)
# ---------------------------------------------------------------------------

def trim_video(video_path: Path, n_frames: int):
    """Trim video to exactly n_frames. Replaces the file in-place."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= n_frames:
        cap.release()
        return  # already short enough

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    tmp_path = video_path.with_suffix(".tmp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (w, h))

    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)

    cap.release()
    writer.release()
    tmp_path.replace(video_path)
    print(f"[Trim] {video_path.name}: {total} -> {n_frames} frames")


def save_episode_data(
    out_dir: Path,
    log_t: list,
    log_joints: list,
    log_gripper: list,
    episode_index: int,
    task_index: int = 0,
) -> int:
    """Save parquet for one episode. Trims videos to match. Returns frame count."""
    ep_str = f"episode_{episode_index:06d}"

    # Determine frame counts from all videos
    vid_frame_counts = []
    for cam_name in LEROBOT_CAM_NAMES:
        vid_path = out_dir / "videos" / CHUNK / cam_name / f"{ep_str}.mp4"
        cap = cv2.VideoCapture(str(vid_path))
        vid_frame_counts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        cap.release()

    N_video = min(vid_frame_counts) if vid_frame_counts else 0
    if N_video == 0:
        print(f"[Save] Episode {episode_index}: video has 0 frames, skipped.")
        return 0

    t_arr       = np.array(log_t,       dtype=np.float64)
    joints_arr  = np.array(log_joints,  dtype=np.float32)
    gripper_arr = np.array(log_gripper, dtype=np.float32)

    # N_frames from robot data duration
    N_robot = int(t_arr[-1] * LEROBOT_FPS)

    # Use the shorter of video and robot data
    N_frames = min(N_video, N_robot)
    if N_frames == 0:
        print(f"[Save] Episode {episode_index}: 0 aligned frames, skipped.")
        return 0

    target_t = np.arange(N_frames, dtype=np.float64) / LEROBOT_FPS
    idx      = np.clip(np.searchsorted(t_arr, target_t, side="left"), 0, len(t_arr) - 1)
    joints_ds  = joints_arr[idx]
    gripper_ds = gripper_arr[idx]

    obs_state = np.concatenate([joints_ds, gripper_ds[:, None]], axis=1).astype(np.float32)

    actions_ds = np.empty_like(obs_state)
    actions_ds[:-1] = obs_state[1:]
    actions_ds[-1]  = obs_state[-1]

    # Trim all videos to exactly N_frames
    for cam_name in LEROBOT_CAM_NAMES:
        vid_path = out_dir / "videos" / CHUNK / cam_name / f"{ep_str}.mp4"
        trim_video(vid_path, N_frames)

    data_dir = out_dir / "data" / CHUNK
    data_dir.mkdir(parents=True, exist_ok=True)

    table = pa.table({
        "observation.state": pa.array(obs_state.tolist(),  type=pa.list_(pa.float32())),
        "action":            pa.array(actions_ds.tolist(), type=pa.list_(pa.float32())),
        "timestamp":         pa.array(target_t.astype(np.float32), type=pa.float32()),
        "episode_index":     pa.array(np.full(N_frames, episode_index, dtype=np.int64)),
        "frame_index":       pa.array(np.arange(N_frames, dtype=np.int64)),
        "index":             pa.array(np.arange(N_frames, dtype=np.int64)),
        "next.done":         pa.array(np.append(np.zeros(N_frames - 1, dtype=bool), True)),
        "task_index":        pa.array(np.full(N_frames, task_index, dtype=np.int64)),
    })
    pq.write_table(table, data_dir / f"{ep_str}.parquet")

    # Append episode stats
    ep_stats: dict = {}
    for key, arr in [("observation.state", obs_state), ("action", actions_ds)]:
        ep_stats[key] = {
            "mean": arr.mean(axis=0).tolist(),
            "std":  (arr.std(axis=0) + 1e-8).tolist(),
            "min":  arr.min(axis=0).tolist(),
            "max":  arr.max(axis=0).tolist(),
            "count": [N_frames],
        }

    meta_dir = out_dir / "meta"
    meta_dir.mkdir(exist_ok=True)
    with open(meta_dir / "episodes_stats.jsonl", "a") as f:
        f.write(json.dumps({"episode_index": episode_index, "stats": ep_stats}) + "\n")
    with open(meta_dir / "episodes.jsonl", "a") as f:
        f.write(json.dumps({"episode_index": episode_index, "task_index": task_index, "length": N_frames}) + "\n")

    print(f"[Save] Episode {episode_index}: {N_frames} frames saved.")
    return N_frames


def save_final_metadata(out_dir: Path, total_episodes: int, total_frames: int,
                        task_instruction: str):
    """Write info.json and tasks.jsonl after all episodes are done."""
    meta_dir = out_dir / "meta"
    meta_dir.mkdir(exist_ok=True)

    joint_names = ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"]
    features: dict = {
        "observation.state": {"dtype": "float32", "shape": [7], "names": joint_names},
        "action":            {"dtype": "float32", "shape": [7], "names": joint_names},
        "timestamp":         {"dtype": "float32", "shape": [1]},
        "episode_index":     {"dtype": "int64",   "shape": [1]},
        "frame_index":       {"dtype": "int64",   "shape": [1]},
        "index":             {"dtype": "int64",   "shape": [1]},
        "next.done":         {"dtype": "bool",    "shape": [1]},
        "task_index":        {"dtype": "int64",   "shape": [1]},
    }
    h, w = OPENPI_IMAGE_SIZE[1], OPENPI_IMAGE_SIZE[0]
    for cam_name in LEROBOT_CAM_NAMES:
        features[cam_name] = {
            "dtype": "video", "shape": [h, w, 3],
            "video_info": {
                "video.fps": float(LEROBOT_FPS), "video.codec": "mp4v",
                "video.pix_fmt": "yuv420p", "video.is_depth_map": False, "has_audio": False,
            },
        }

    info = {
        "codebase_version": "v2.1",
        "fps": LEROBOT_FPS,
        "robot_type": "ur3",
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_chunks": 1,
        "chunks_size": 1000,
        "features": features,
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2))
    (meta_dir / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": task_instruction}) + "\n"
    )
    print(f"\nDataset saved -> {out_dir.resolve()}")
    print(f"  {total_episodes} episodes, {total_frames} total frames")


# ---------------------------------------------------------------------------
# Start camera threads for one episode
# ---------------------------------------------------------------------------

def start_cam_threads(out_dir: Path, episode_index: int, ep_stop: Event):
    ep_str = f"episode_{episode_index:06d}"

    d435i_dir = out_dir / "videos" / CHUNK / LEROBOT_CAM_NAMES[0]
    d435i_dir.mkdir(parents=True, exist_ok=True)
    d435i = D435iRecorderThread(
        output_path=d435i_dir / f"{ep_str}.mp4",
        width=D435I_WIDTH, height=D435I_HEIGHT, fps=D435I_FPS,
        stop_event=ep_stop,
    )

    wrist_dir = out_dir / "videos" / CHUNK / LEROBOT_CAM_NAMES[1]
    wrist_dir.mkdir(parents=True, exist_ok=True)
    wrist = CameraRecorderThread(
        cam_index=WRIST_CAM_INDEX,
        output_path=wrist_dir / f"{ep_str}.mp4",
        fps=WRIST_CAM_FPS, width=WRIST_CAM_WIDTH, height=WRIST_CAM_HEIGHT,
        stop_event=ep_stop,
    )

    d435i.start()
    wrist.start()
    return [d435i, wrist]


# ---------------------------------------------------------------------------
# Control loop — background thread, manages episodes
# ---------------------------------------------------------------------------

def _control_loop(stop_event: Event, next_ep_event: Event, start_ep_event: Event,
                  out_dir: Path, task_instruction: str,
                  cam_threads_ref: list, status_var_ref: list):
    sm = Spacemouse(deadzone=0.2)
    sm.start()

    def set_status(msg):
        if status_var_ref:
            status_var_ref[0].set(msg)

    try:
        rtde_c = RTDEControlInterface(ROBOT_HOST)
        rtde_r = RTDEReceiveInterface(ROBOT_HOST)
        print("[Control] Robot connected.")
    except Exception as exc:
        print(f"[Control] Robot connection failed: {exc}. Display-only mode.")
        sm.stop()
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

    episode_index = 0
    total_frames = 0
    gripper_state = 0.0
    prev_btn = [False, False]

    try:
        while not stop_event.is_set():
            # ---- Start new episode ----
            ep_stop = Event()
            cam_threads = start_cam_threads(out_dir, episode_index, ep_stop)
            cam_threads_ref.clear()
            cam_threads_ref.extend(cam_threads)

            log_t:       list[float] = []
            log_joints:  list[list]  = []
            log_gripper: list[float] = []
            t0 = time.perf_counter()

            next_ep_event.clear()
            start_ep_event.clear()
            print(f"\n{'='*50}")
            print(f"  Recording Episode {episode_index}  (Space = end episode)")
            print(f"{'='*50}")
            set_status(f"● Recording Episode {episode_index}  [Space = end]")

            # ---- Record loop for this episode ----
            status_update_counter = 0
            while not stop_event.is_set() and not next_ep_event.is_set():
                if rtde_r.getRobotMode() == 7:
                    t_now = time.perf_counter() - t0

                    motion_state = sm.get_motion_state_transformed()
                    rtde_c.speedL(motion_state, acceleration=0.5, time=0.1)
                    joint_pos = rtde_r.getActualQ()

                    if gripper is not None:
                        btn0 = sm.is_button_pressed(0)
                        if btn0 and not prev_btn[0]:
                            gripper.send_async("clamp_min")
                            gripper_state = 1.0
                        prev_btn[0] = btn0

                        btn1 = sm.is_button_pressed(1)
                        if btn1 and not prev_btn[1]:
                            gripper._send("release_block")
                            gripper.send_async("clamp_max")
                            gripper_state = 0.0
                        prev_btn[1] = btn1

                    log_t.append(t_now)
                    log_joints.append(joint_pos)
                    log_gripper.append(gripper_state)

                    # Update status bar every ~0.5s (every 50 iterations at 100Hz)
                    status_update_counter += 1
                    if status_update_counter % 50 == 0:
                        elapsed = int(t_now)
                        est_frames = int(t_now * LEROBOT_FPS)
                        set_status(f"● Ep {episode_index}  {elapsed}s  ~{est_frames} frames  [Space = end]")

                    time.sleep(1 / 100)
                else:
                    time.sleep(1)

            # ---- End episode: stop cameras, save data ----
            ep_stop.set()
            for t in cam_threads:
                t.join(timeout=5)

            if log_t:
                n = save_episode_data(out_dir, log_t, log_joints, log_gripper, episode_index)
                total_frames += n
                episode_index += 1

            if stop_event.is_set():
                break

            # ---- Homing phase: robot still controllable, no data recorded ----
            print(f"  Move robot to start position, then press Space to begin Episode {episode_index}")
            set_status(f"■ Homing ... press Space to start Episode {episode_index}")

            while not stop_event.is_set() and not start_ep_event.is_set():
                if rtde_r.getRobotMode() == 7:
                    motion_state = sm.get_motion_state_transformed()
                    rtde_c.speedL(motion_state, acceleration=0.5, time=0.1)

                    if gripper is not None:
                        btn0 = sm.is_button_pressed(0)
                        if btn0 and not prev_btn[0]:
                            gripper.send_async("clamp_min")
                            gripper_state = 1.0
                        prev_btn[0] = btn0

                        btn1 = sm.is_button_pressed(1)
                        if btn1 and not prev_btn[1]:
                            gripper._send("release_block")
                            gripper.send_async("clamp_max")
                            gripper_state = 0.0
                        prev_btn[1] = btn1

                    time.sleep(1 / 100)
                else:
                    time.sleep(1)

    finally:
        try:
            rtde_c.stopScript()
        except Exception:
            pass
        sm.stop()
        if gripper is not None:
            gripper.close()

        # Save final metadata
        if episode_index > 0:
            save_final_metadata(out_dir, episode_index, total_frames, task_instruction)

        print("[Control] Stopped.")


# ---------------------------------------------------------------------------
# Display — tkinter on main thread
# ---------------------------------------------------------------------------

def _run_display(stop_event: Event, next_ep_event: Event, start_ep_event: Event,
                 cam_threads_ref: list, status_var_ref: list):
    root = tk.Tk()
    root.title("UR3 Teleop Record (D435i) — Space=control, Close=quit")
    root.protocol("WM_DELETE_WINDOW", lambda: (stop_event.set(), root.destroy()))
    signal.signal(signal.SIGINT, lambda *_: (stop_event.set(), root.destroy()))

    label = tk.Label(root)
    label.pack()

    status_var = tk.StringVar(value="● Recording Episode 0  [Space = end]")
    status_label = tk.Label(root, textvariable=status_var, font=("monospace", 14),
                            fg="white", bg="green", padx=10, pady=5)
    status_label.pack(pady=5, fill=tk.X)

    status_var_ref.append(status_var)

    # State: "recording" or "homing"
    phase = ["recording"]

    def on_space(event):
        if phase[0] == "recording":
            phase[0] = "homing"
            next_ep_event.set()
            status_label.configure(bg="orange")
        elif phase[0] == "homing":
            phase[0] = "recording"
            start_ep_event.set()
            status_label.configure(bg="green")

    root.bind("<space>", on_space)

    def update_frame():
        frames = []
        for ct in cam_threads_ref:
            f = ct.get_display_frame()
            if f is not None:
                f = cv2.resize(f, (D435I_WIDTH, D435I_HEIGHT))
                frames.append(f)
        if frames:
            combined = np.hstack(frames) if len(frames) > 1 else frames[0]
            img = ImageTk.PhotoImage(Image.fromarray(combined))
            label.imgtk = img
            label.configure(image=img)
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
    user_prompt = input(f"Task prompt [{TASK_INSTRUCTION_DEFAULT}]: ").strip()
    task_instruction = user_prompt if user_prompt else TASK_INSTRUCTION_DEFAULT
    print(f"Task: \"{task_instruction}\"")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("recordings") / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {out_dir.resolve()}")

    stop_event    = Event()
    next_ep_event = Event()
    start_ep_event = Event()
    cam_threads_ref: list = []
    status_var_ref: list = []

    control = Thread(
        target=_control_loop,
        args=(stop_event, next_ep_event, start_ep_event,
              out_dir, task_instruction, cam_threads_ref, status_var_ref),
        daemon=True, name="ControlThread",
    )
    control.start()

    _run_display(stop_event, next_ep_event, start_ep_event,
                 cam_threads_ref, status_var_ref)
    stop_event.set()
    print("\nShutting down ...")
    control.join(timeout=5)
    print("Done.")


if __name__ == "__main__":
    main()
