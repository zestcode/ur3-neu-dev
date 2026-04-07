"""
teleop_vision_record.py — UR3 SpaceMouse teleoperation with dual-camera recording.

Threads
-------
- Main thread     : 100 Hz teleoperation loop; records joint state and action.
- CameraThread×2  : Each thread captures one OpenCV camera via lerobot OpenCVCamera
                    and writes frames to an MP4.

Output (per run)
----------------
  recordings/<timestamp>/
    cam0.mp4         — base camera, RGB 224×224
    cam1.mp4         — wrist camera, RGB 224×224
                       (stored as standard BGR mp4; convert BGR→RGB when reading)
    robot_log.npz    — timestep-aligned arrays (openpi UR5 field names):
                          t          : float64 (N,)   seconds since episode start
                          joints     : float32 (N,6)  actual joint angles rad  ← observation.state[:6]
                          gripper    : float32 (N,1)  0.0=open 1.0=closed      ← observation.state[6]
                          actions    : float32 (N,7)  joints+gripper at each step
                                       (openpi DeltaActions computes q[t+1]-q[t] at training time)

Usage
-----
    conda activate spacemouse-ur
    pip install lerobot pyserial
    python3 teleop_vision_record.py

Stop with Ctrl+C — triggers graceful shutdown of all threads.
"""

from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from spnav import spnav_open, spnav_poll_event, spnav_close, SpnavMotionEvent, SpnavButtonEvent
from threading import Thread, Event, Lock
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np
import serial
import time
import cv2

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.configs import ColorMode

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROBOT_HOST = "192.168.0.2"
SCALE_FACTOR = 0.1

GRIPPER_PORT = "/dev/ttyUSB0"
GRIPPER_BAUDRATE = 115200
GRIPPER_TIMEOUT = 1
GRIPPER_COMMAND_DELAY = 0.2

# Capture at camera-native resolution; frames are resized to OPENPI_IMAGE_SIZE
# before writing so the stored MP4 matches openpi's expected input dimensions.
CAMERA_CONFIGS = [
    dict(index=0, fps=30, width=640, height=480),
    dict(index=2, fps=30, width=640, height=480),
]
OPENPI_IMAGE_SIZE = (224, 224)  # (width, height) required by openpi

COMMANDS = {
    "clamp_min":    "01 FB 00 01 F4 00 00 2A 94 01 00 6B",
    "clamp_max":    "01 FB 01 01 F4 00 00 00 00 01 00 6B",
    "motor_enable": "01 F3 AB 01 00 6B",
    "release_block": "01 0E 52 6B",
}

# ---------------------------------------------------------------------------
# SpaceMouse thread (unchanged from 3DConnexion_UR3_Teleop_Gripper.py)
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
# Gripper controller (unchanged from 3DConnexion_UR3_Teleop_Gripper.py)
# ---------------------------------------------------------------------------

class GripperController:
    def __init__(self, port, baudrate, timeout, command_delay):
        self.command_delay = command_delay
        self._lock = Lock()
        self._worker = None
        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate,
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
            print(f"Unknown gripper command: {command_name}")
            return
        data = bytes.fromhex(hex_str.replace(" ", ""))
        with self._lock:
            self.ser.write(data)
        time.sleep(self.command_delay)
        print(f"Gripper: {command_name}")

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
# Camera recorder thread (one per camera)
# ---------------------------------------------------------------------------

class CameraRecorderThread(Thread):
    """
    Captures frames from one camera using lerobot's OpenCVCamera and writes
    them to an MP4 video file.  Runs until stop_event is set.

    Parameters
    ----------
    cam_index : int
        OpenCV device index (0, 1, 2, …)
    output_path : Path
        Destination MP4 file path.
    fps : int
        Target capture/output frame rate.
    width, height : int
        Capture resolution.
    stop_event : threading.Event
        Shared stop signal; set this from the main thread to end recording.
    """

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

    def run(self):
        cfg = OpenCVCameraConfig(
            index_or_path=self.cam_index,
            fps=self.fps,
            width=self.width,
            height=self.height,
            color_mode=ColorMode.RGB,   # openpi expects RGB
        )
        out_w, out_h = OPENPI_IMAGE_SIZE
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(self.output_path), fourcc, self.fps, (out_w, out_h)
        )
        if not writer.isOpened():
            self.error = RuntimeError(f"VideoWriter failed for {self.output_path}")
            print(f"[CamRecorder-{self.cam_index}] ERROR: {self.error}")
            return

        print(f"[CamRecorder-{self.cam_index}] Recording {self.width}×{self.height}"
              f" → resized to {out_w}×{out_h} → {self.output_path}")
        try:
            with OpenCVCamera(cfg) as cam:
                while not self.stop_event.is_set():
                    try:
                        frame = cam.async_read(timeout_ms=200)  # (H, W, 3) RGB, native res
                    except TimeoutError:
                        continue  # no new frame yet, keep looping
                    # Resize to openpi input size, then convert RGB→BGR for VideoWriter
                    frame = cv2.resize(frame, OPENPI_IMAGE_SIZE)
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    self.frame_count += 1
        except Exception as exc:
            self.error = exc
            print(f"[CamRecorder-{self.cam_index}] ERROR: {exc}")
        finally:
            writer.release()
            print(f"[CamRecorder-{self.cam_index}] Stopped. "
                  f"Frames captured: {self.frame_count}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ---- output directory ------------------------------------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("recordings") / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {out_dir.resolve()}")

    # ---- shared stop event -----------------------------------------------
    stop_event = Event()

    # ---- camera threads --------------------------------------------------
    cam_threads: list[CameraRecorderThread] = []
    for i, cfg in enumerate(CAMERA_CONFIGS):
        t = CameraRecorderThread(
            cam_index=cfg["index"],
            output_path=out_dir / f"cam{i}.mp4",
            fps=cfg["fps"],
            width=cfg["width"],
            height=cfg["height"],
            stop_event=stop_event,
        )
        cam_threads.append(t)
        t.start()

    # ---- robot & spacemouse ----------------------------------------------
    sm = Spacemouse(deadzone=0.2)
    sm.start()

    rtde_c = RTDEControlInterface(ROBOT_HOST)
    rtde_r = RTDEReceiveInterface(ROBOT_HOST)

    gripper = GripperController(
        port=GRIPPER_PORT,
        baudrate=GRIPPER_BAUDRATE,
        timeout=GRIPPER_TIMEOUT,
        command_delay=GRIPPER_COMMAND_DELAY,
    )
    print("Enabling gripper motor...")
    gripper.enable()

    # ---- data log buffers (openpi UR5 field names) -----------------------
    log_t:       list[float]      = []
    log_joints:  list[list]       = []   # getActualQ()  (6,) rad
    log_gripper: list[float]      = []   # 0.0=open 1.0=closed
    log_actions: list[np.ndarray] = []   # joints + gripper  (7,) rad

    # Gripper state inferred from last command (open-loop gripper has no feedback)
    gripper_state = 0.0   # start assuming open

    prev_btn = [False, False]
    t0 = time.perf_counter()

    # ---- teleoperation loop (main thread, 100 Hz) ------------------------
    try:
        while True:
            if rtde_r.getRobotMode() == 7:
                t_now = time.perf_counter() - t0

                motion_state = sm.get_motion_state_transformed()
                rtde_c.speedL(motion_state, acceleration=0.5, time=0.1)

                # observation.state: actual joint angles from robot
                joint_pos = rtde_r.getActualQ()   # (6,) rad
                print(f"t={t_now:.2f}s  q: {[round(v,3) for v in joint_pos]}")

                # gripper edge detection + state tracking
                btn0 = sm.is_button_pressed(0)
                if btn0 and not prev_btn[0]:
                    gripper.send_async("clamp_min")
                    gripper_state = 1.0   # closed
                prev_btn[0] = btn0

                btn1 = sm.is_button_pressed(1)
                if btn1 and not prev_btn[1]:
                    gripper._send("release_block")   # blocking: freezes TCP during release
                    gripper.send_async("clamp_max")
                    gripper_state = 0.0   # open
                prev_btn[1] = btn1

                # record timestep — joints(6) + gripper(1) = action(7)
                action = np.array(joint_pos + [gripper_state], dtype=np.float32)
                log_t.append(t_now)
                log_joints.append(joint_pos)
                log_gripper.append(gripper_state)
                log_actions.append(action)

                time.sleep(1 / 100)
            else:
                print("Robot not ready (mode ≠ 7).")
                time.sleep(1)

    except KeyboardInterrupt:
        print("\nCtrl+C — shutting down …")

    finally:
        # ---- stop all threads --------------------------------------------
        stop_event.set()
        try:
            rtde_c.stopScript()
        except Exception as e:
            print(f"stopScript failed (connection may already be lost): {e}")
        sm.stop()
        gripper.close()

        for t in cam_threads:
            t.join(timeout=5)
            if t.error:
                print(f"[{t.name}] finished with error: {t.error}")

        # ---- save robot log ----------------------------------------------
        if log_t:
            npz_path = out_dir / "robot_log.npz"
            np.savez(
                npz_path,
                t=np.array(log_t, dtype=np.float64),          # (N,)
                joints=np.array(log_joints, dtype=np.float32), # (N,6) rad
                gripper=np.array(log_gripper, dtype=np.float32), # (N,1) 0.0/1.0
                actions=np.array(log_actions, dtype=np.float32), # (N,7)
            )
            print(f"Robot log saved → {npz_path}  ({len(log_t)} timesteps)")
        else:
            print("No robot data recorded.")

        print("Done.")


if __name__ == "__main__":
    main()
