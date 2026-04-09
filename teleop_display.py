"""
teleop_display.py — UR3 SpaceMouse teleoperation with live dual-camera display.

Thread design
-------------
- Main thread   : robot connection + 100 Hz control loop (most important).
                  If robot not ready → standby. If control error → raise.
- DisplayThread : tkinter + PIL camera display (daemon, best-effort).
                  Dies automatically when main thread exits.

Usage
-----
    conda activate spacemouse-ur
    pip install pillow
    python3 teleop_display.py

Stop with Ctrl+C.
"""

import signal
import tkinter as tk
from PIL import Image, ImageTk

import cv2
import numpy as np
import serial
import time
from collections import defaultdict
from threading import Thread, Event, Lock

from spnav import spnav_open, spnav_poll_event, spnav_close, SpnavMotionEvent, SpnavButtonEvent
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from rtde_io import RTDEIOInterface as RTDEIO

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROBOT_HOST   = "192.168.0.2"
SCALE_FACTOR = 0.1

GRIPPER_PORT          = "/dev/ttyUSB0"
GRIPPER_BAUDRATE      = 115200
GRIPPER_TIMEOUT       = 1
GRIPPER_COMMAND_DELAY = 0.2

CAMERA_CONFIGS = [
    dict(index=1, width=640, height=480),
    dict(index=3, width=640, height=480),
]

COMMANDS = {
    "clamp_min":     "01 FB 00 01 F4 00 00 2A 94 01 00 6B",
    "clamp_max":     "01 FB 01 01 F4 00 00 00 00 01 00 6B",
    "motor_enable":  "01 F3 AB 01 00 6B",
    "release_block": "01 0E 52 6B",
}

# ---------------------------------------------------------------------------
# SpaceMouse thread
# ---------------------------------------------------------------------------

class Spacemouse(Thread):
    def __init__(self, max_value=300, deadzone=(0,0,0,0,0,0), dtype=np.float32):
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
        self.motion_event = SpnavMotionEvent([0,0,0], [0,0,0], 0)
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
        print(f"Gripper connected on {port}")

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
        print(f"[Gripper] Sent: {command_name}")

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
# Robot control loop — runs in a background daemon thread
# ---------------------------------------------------------------------------

def _robot_control_loop(stop_event: Event) -> None:
    """Robot connection + 100 Hz control loop.

    Runs in a daemon thread so tkinter can own the main thread (Linux Tk
    is not thread-safe and must run on the main thread).  If the robot is
    unreachable the thread logs a warning and waits for stop_event; the
    display keeps running normally (display-only mode).
    """
    # ---- spacemouse (start before RTDE, matching Gripper.py order) -------
    sm = Spacemouse(deadzone=0.2)
    sm.start()

    # ---- robot connection ------------------------------------------------
    try:
        rtde_c = RTDEControlInterface(ROBOT_HOST)
        rtde_r = RTDEReceiveInterface(ROBOT_HOST)
        rtde_io = RTDEIO(ROBOT_HOST)
        print("[Control] Robot connected.")
    except Exception as exc:
        print(f"[Control] Robot connection failed ({exc}). Running display-only mode.")
        sm.stop()
        stop_event.wait()
        return

    # ---- gripper connection ----------------------------------------------
    gripper = None
    try:
        gripper = GripperController(
            port=GRIPPER_PORT,
            baudrate=GRIPPER_BAUDRATE,
            timeout=GRIPPER_TIMEOUT,
            command_delay=GRIPPER_COMMAND_DELAY,
        )
        gripper.enable()
    except Exception as exc:
        print(f"[Control] Gripper not available: {exc}")

    prev_btn = [False, False]

    # ---- 100 Hz control loop ---------------------------------------------
    try:
        while not stop_event.is_set():
            if rtde_r.getRobotMode() == 7:
                motion_state = sm.get_motion_state_transformed()
                rtde_c.speedL(motion_state, acceleration=0.5, time=0.1)

                tcp_vel = rtde_r.getActualTCPSpeed()
                tcp_vel_f = [0 if abs(v) < 0.01 else v for v in tcp_vel]
                print("TCP vel:", tcp_vel_f)

                btn0 = sm.is_button_pressed(0)
                btn1 = sm.is_button_pressed(1)

                if btn0 and not prev_btn[0]:
                    print("[Control] Button 0 pressed (close)")
                    if gripper is not None:
                        gripper.send_async("clamp_min")
                    else:
                        print("[Control] WARNING: gripper not connected, skipping command")
                prev_btn[0] = btn0

                if btn1 and not prev_btn[1]:
                    print("[Control] Button 1 pressed (open)")
                    if gripper is not None:
                        gripper._send("release_block")  # blocking: freezes TCP during release
                        gripper.send_async("clamp_max")
                    else:
                        print("[Control] WARNING: gripper not connected, skipping command")
                prev_btn[1] = btn1

                time.sleep(1 / 100)
            else:
                print("[Control] Robot not ready (mode ≠ 7), standby …")
                stop_event.wait(timeout=1)
    finally:
        try:
            rtde_c.stopScript()
        except Exception:
            pass
        sm.stop()
        if gripper is not None:
            gripper.close()


# ---------------------------------------------------------------------------
# Camera thread — reads frames in background so cap.read() never blocks tkinter
# ---------------------------------------------------------------------------

class CameraThread(Thread):
    """Continuously captures frames from all cameras in a background daemon thread.
    The main thread calls get_frame() to retrieve the latest composed image
    without blocking.
    """
    def __init__(self, configs):
        super().__init__(daemon=True, name="CameraThread")
        self._configs   = configs
        self._lock      = Lock()
        self._stop      = Event()
        self._frame     = None   # latest RGB numpy array, None until first frame

    def get_frame(self):
        """Return the latest composed frame (RGB ndarray) or None."""
        with self._lock:
            return self._frame

    def stop(self):
        self._stop.set()

    def run(self):
        caps = []
        for cfg in self._configs:
            cap = cv2.VideoCapture(cfg["index"], cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg["width"])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["height"])
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep only the latest frame in buffer
            if not cap.isOpened():
                print(f"[Camera] Cannot open camera {cfg['index']}, skipping.")
                continue
            caps.append((cfg["index"], cap))
            print(f"[Camera] Camera {cfg['index']} opened.")

        if not caps:
            print("[Camera] No cameras available.")
            return

        try:
            while not self._stop.is_set():
                # Grab from all cameras before decoding any — minimises inter-camera delay
                grabbed = [cap.grab() for _, cap in caps]

                frames = []
                for i, ((idx, cap), ok) in enumerate(zip(caps, grabbed)):
                    if not ok:
                        continue
                    ret, frame = cap.retrieve()
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        cv2.putText(frame, f"cam{i} ({idx})", (10, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        frames.append(frame)
                if frames:
                    target_w = self._configs[0]["width"]
                    target_h = self._configs[0]["height"]
                    frames = [cv2.resize(f, (target_w, target_h)) for f in frames]
                    combined = np.hstack(frames) if len(frames) > 1 else frames[0]
                    with self._lock:
                        self._frame = combined
        finally:
            for _, cap in caps:
                cap.release()
            print("[Camera] Cameras released.")


# ---------------------------------------------------------------------------
# Display — tkinter on the main thread; reads latest frame from CameraThread
# ---------------------------------------------------------------------------

def _run_display(stop_event: Event) -> None:
    """Tkinter window on the main thread. Never calls cap.read() directly."""
    cam = CameraThread(CAMERA_CONFIGS)
    cam.start()

    root = tk.Tk()
    root.title("UR3 Teleop — Camera View")
    root.protocol("WM_DELETE_WINDOW", lambda: (stop_event.set(), root.destroy()))

    # Ctrl+C: SIGINT is swallowed by tkinter mainloop on Linux without this
    signal.signal(signal.SIGINT, lambda *_: (stop_event.set(), root.destroy()))

    label = tk.Label(root)
    label.pack()

    def update_frame():
        frame = cam.get_frame()          # non-blocking — just reads latest
        if frame is not None:
            img = ImageTk.PhotoImage(Image.fromarray(frame))
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

    cam.stop()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    stop_event = Event()

    # Robot control runs in a background thread so tkinter can own the main thread
    control = Thread(
        target=_robot_control_loop, args=(stop_event,),
        daemon=True, name="ControlThread",
    )
    control.start()

    _run_display(stop_event)   # tkinter must run on the main thread; handles Ctrl+C internally
    stop_event.set()
    print("\nShutting down …")
    control.join(timeout=3)


if __name__ == "__main__":
    main()
