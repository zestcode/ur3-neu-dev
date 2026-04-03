"""
record_teleoperation.py  —  UR3 teleoperation + LeRobot dataset recording.

Three concurrent threads, all sharing a single clock anchor t0:

  T1  Spacemouse     — polls 3D mouse at 200 Hz                    (input only)
  T2  CameraCapture  — queues (t-t0, RGB frame) at camera native FPS
  T3  RobotControl   — sends speedL to UR3 at 100 Hz, caches latest action

Main writer thread (driven by camera frames):
  For each dequeued camera frame, reads robot state + cached action, then calls
  dataset.add_frame().  All timestamps = time.time() - t0 (unified clock).

Output: LeRobot-format dataset at <output>/
  data/chunk-000/episode_000000.parquet   — state + action tabular data
  videos/chunk-000/observation.images.camera/episode_000000.mp4

Usage:
    python3 record_teleoperation.py [--camera 0] [--output recordings/YYYYMMDD_HHMMSS]
                                    [--repo-id lab/ur3_teleop] [--task "teleoperation"]

Stop with Ctrl+C — dataset is saved and finalized on exit.
"""

import argparse
import queue
import time
from collections import defaultdict
from threading import Event, Lock, Thread

import cv2
import numpy as np
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from spnav import (SpnavButtonEvent, SpnavMotionEvent,
                   spnav_close, spnav_open, spnav_poll_event)

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ── Robot ──────────────────────────────────────────────────────────────────────
ROBOT_HOST   = "192.168.0.2"
SCALE_FACTOR = 0.1
ACCELERATION = 0.5
SPEEDL_TIME  = 0.1     # safety timeout [s]: robot decelerates if no new cmd arrives
CONTROL_HZ   = 100

# ── SpaceMouse ─────────────────────────────────────────────────────────────────
SM_MAX_VALUE = 300     # 300 wired / 500 wireless
SM_DEADZONE  = 0.2


# ══════════════════════════════════════════════════════════════════════════════
# T1: SpaceMouse input  (copied verbatim from 3DConnexion_UR3_Teleop.py)
# ══════════════════════════════════════════════════════════════════════════════

class Spacemouse(Thread):
    def __init__(self, max_value=300, deadzone=(0, 0, 0, 0, 0, 0), dtype=np.float32):
        if np.issubdtype(type(deadzone), np.number):
            deadzone = np.full(6, fill_value=deadzone, dtype=dtype)
        else:
            deadzone = np.array(deadzone, dtype=dtype)
        assert (deadzone >= 0).all()
        super().__init__(daemon=True)
        self.stop_event   = Event()
        self.max_value    = max_value
        self.dtype        = dtype
        self.deadzone     = deadzone
        self.motion_event = SpnavMotionEvent([0, 0, 0], [0, 0, 0], 0)
        self.button_state = defaultdict(lambda: False)
        self.tx_zup_spnav = np.array([
            [0, 0, -1],
            [1, 0,  0],
            [0, 1,  0],
        ], dtype=dtype)

    def get_motion_state(self):
        me    = self.motion_event
        state = np.array(me.translation + me.rotation, dtype=self.dtype) / self.max_value
        is_dead = (-self.deadzone < state) & (state < self.deadzone)
        state[is_dead] = 0
        return state

    def get_motion_state_transformed(self):
        """Robot-frame 6D velocity, scaled — same transform as teleop scripts."""
        state    = self.get_motion_state()
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


# ══════════════════════════════════════════════════════════════════════════════
# T2: Camera capture
# ══════════════════════════════════════════════════════════════════════════════

class CameraCapture(Thread):
    """Captures frames and queues (timestamp, RGB_frame) pairs.

    timestamp = time.time() - t0  (unified clock, in seconds since episode start).
    Oldest frame is silently dropped when queue is full (writer falling behind).
    """

    def __init__(self, camera_index: int, t0: float, stop_event: Event, maxsize: int = 120):
        super().__init__(daemon=True)
        self.camera_index = camera_index
        self.t0           = t0
        self.stop_event   = stop_event
        self.q            = queue.Queue(maxsize=maxsize)
        self.width: int | None  = None
        self.height: int | None = None
        self.fps: float | None  = None
        self.ready = Event()
        self.error: str | None  = None

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self.error = f"Cannot open camera index {self.camera_index}"
            self.ready.set()
            return

        self.width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        raw_fps     = cap.get(cv2.CAP_PROP_FPS)
        self.fps    = raw_fps if raw_fps > 0 else 30.0
        self.ready.set()
        print(f"[Camera] {self.width}×{self.height} @ {self.fps:.1f} fps")

        try:
            while not self.stop_event.is_set():
                ret, bgr = cap.read()
                if not ret:
                    time.sleep(0.005)
                    continue
                t   = time.time() - self.t0          # unified clock
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if self.q.full():
                    try:
                        self.q.get_nowait()          # evict oldest
                    except queue.Empty:
                        pass
                self.q.put_nowait((t, rgb))
        finally:
            cap.release()
            print("[Camera] Stopped.")


# ══════════════════════════════════════════════════════════════════════════════
# T3: Robot control — 100 Hz, caches latest action for the writer
# ══════════════════════════════════════════════════════════════════════════════

class RobotController(Thread):
    """Sends SpaceMouse-derived speedL commands to UR3 at CONTROL_HZ.

    action = [tx, ty, tz, rx, ry, rz, btn0, btn1]  shape (8,) float32
      [0:6]  — 6-D TCP velocity sent to speedL (robot frame, m/s and rad/s)
      [6:8]  — SpaceMouse button states (0.0 / 1.0)

    Thread-safe: get_latest_action() snapshots the last sent action.
    """

    def __init__(self, rtde_c: RTDEControlInterface, sm: Spacemouse, stop_event: Event):
        super().__init__(daemon=True)
        self.rtde_c     = rtde_c
        self.sm         = sm
        self.stop_event = stop_event
        self._lock      = Lock()
        self._action    = np.zeros(8, dtype=np.float32)

    def get_latest_action(self) -> np.ndarray:
        with self._lock:
            return self._action.copy()

    def run(self):
        interval = 1.0 / CONTROL_HZ
        print(f"[Robot] Control loop at {CONTROL_HZ} Hz")
        while not self.stop_event.is_set():
            motion = self.sm.get_motion_state_transformed()   # shape (6,)
            btn0   = float(self.sm.is_button_pressed(0))
            btn1   = float(self.sm.is_button_pressed(1))
            self.rtde_c.speedL(motion, acceleration=ACCELERATION, time=SPEEDL_TIME)
            with self._lock:
                self._action[:6] = motion
                self._action[6]  = btn0
                self._action[7]  = btn1
            time.sleep(interval)
        print("[Robot] Control loop stopped.")


# ══════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    ts = time.strftime("%Y%m%d_%H%M%S")
    p  = argparse.ArgumentParser(description="UR3 teleoperation recorder — LeRobot format")
    p.add_argument("--camera",  type=int, default=0,
                   help="Camera device index (default: 0)")
    p.add_argument("--output",  type=str, default=f"recordings/{ts}",
                   help="Root directory for the LeRobot dataset")
    p.add_argument("--repo-id", type=str, default="lab/ur3_teleop",
                   help="Dataset repo_id (default: lab/ur3_teleop)")
    p.add_argument("--task",    type=str, default="teleoperation",
                   help="Task description string (default: teleoperation)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── 1. Probe camera for W×H (needed before creating feature schema) ────────
    print(f"Probing camera {args.camera} ...")
    probe = cv2.VideoCapture(args.camera)
    if not probe.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")
    cam_w   = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h   = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_fps = probe.get(cv2.CAP_PROP_FPS)
    probe.release()
    record_fps = max(1, int(round(raw_fps))) if raw_fps > 0 else 30
    print(f"Camera: {cam_w}×{cam_h} @ {record_fps} fps")

    # ── 2. Build LeRobot feature schema ────────────────────────────────────────
    #
    #  observation.state  (12,)  — q0..q5 [rad], qd0..qd5 [rad/s]
    #  observation.images.camera (H, W, 3) — RGB video
    #  action             (8,)   — 6-D TCP velocity + 2 buttons
    #
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (12,),
            "names": ["q0", "q1", "q2", "q3", "q4", "q5",
                      "qd0", "qd1", "qd2", "qd3", "qd4", "qd5"],
        },
        "observation.images.camera": {
            "dtype": "video",
            "shape": (cam_h, cam_w, 3),
            "names": ["height", "width", "channels"],
        },
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": ["tx", "ty", "tz", "rx", "ry", "rz", "btn0", "btn1"],
        },
    }

    # ── 3. Create LeRobot dataset (local, no HF push required) ────────────────
    print(f"Creating dataset at: {args.output}")
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=record_fps,
        features=features,
        root=args.output,
        robot_type="ur3",
        use_videos=True,
    )

    # ── 4. Unified clock anchor — set BEFORE any thread starts ─────────────────
    #    All timestamps = time.time() - t0  (relative seconds since episode start)
    t0 = time.time()

    # ── 5. Connect to robot ────────────────────────────────────────────────────
    print(f"Connecting to UR3 at {ROBOT_HOST} ...")
    rtde_c = RTDEControlInterface(ROBOT_HOST)
    rtde_r = RTDEReceiveInterface(ROBOT_HOST)
    print("Robot connected.")

    stop_event = Event()

    # ── 6. Start the three threads ─────────────────────────────────────────────
    sm = Spacemouse(max_value=SM_MAX_VALUE, deadzone=SM_DEADZONE)
    sm.start()                                   # T1

    cam = CameraCapture(args.camera, t0, stop_event)
    cam.start()                                  # T2
    if not cam.ready.wait(timeout=5.0) or cam.error:
        raise RuntimeError(cam.error or "Camera init timeout")

    ctrl = RobotController(rtde_c, sm, stop_event)
    ctrl.start()                                 # T3

    print(f"\n[Recording] task='{args.task}' — Ctrl+C to stop.\n")
    frame_count = 0

    # ── 7. Main writer loop: camera frame = one dataset step ──────────────────
    #    For each frame dequeued from T2, read robot state and cached action,
    #    then write one row to the dataset.  Timestamp comes from the frame
    #    itself (time.time()-t0 at capture moment) → all modalities share t0.
    try:
        while True:
            try:
                t_frame, rgb = cam.q.get(timeout=1.0)
            except queue.Empty:
                continue

            # Robot state at this instant (RTDEReceiveInterface is thread-safe)
            q  = np.array(rtde_r.getActualQ(),  dtype=np.float32)   # joint pos [rad]
            qd = np.array(rtde_r.getActualQd(), dtype=np.float32)   # joint vel [rad/s]

            # Latest action sent by T3 (≤ 10 ms old at 100 Hz control rate)
            action = ctrl.get_latest_action()

            dataset.add_frame(
                frame={
                    "observation.state": np.concatenate([q, qd]),
                    "observation.images.camera": rgb,     # (H, W, 3) uint8 RGB
                    "action": action,
                },
                task=args.task,
                timestamp=t_frame,     # seconds since t0, from camera capture
            )
            frame_count += 1
            if frame_count % record_fps == 0:
                elapsed = t_frame
                print(f"\r[Recording] {elapsed:.1f} s  |  {frame_count} frames", end="", flush=True)

    except KeyboardInterrupt:
        print(f"\n\nStopping — {frame_count} frames recorded.")

    # ── 8. Shutdown all threads ────────────────────────────────────────────────
    stop_event.set()
    rtde_c.stopScript()
    cam.join(timeout=3.0)
    ctrl.join(timeout=3.0)
    sm.stop()

    # ── 9. Finalize LeRobot dataset ───────────────────────────────────────────
    if frame_count > 0:
        print("Saving episode ...")
        dataset.save_episode()
        dataset.finalize()          # writes parquet footer + meta/info.json
        print(f"\nDataset saved to: {dataset.root}")
        print(f"  data/    — {frame_count} rows (parquet)")
        print(f"  videos/  — episode_000000.mp4")
    else:
        print("No frames recorded — dataset not saved.")


if __name__ == "__main__":
    main()
