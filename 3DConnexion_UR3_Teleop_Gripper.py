from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from rtde_io import RTDEIOInterface as RTDEIO
from spnav import spnav_open, spnav_poll_event, spnav_close, SpnavMotionEvent, SpnavButtonEvent
from threading import Thread, Event, Lock
from collections import defaultdict
import numpy as np
import serial
import time


class Spacemouse(Thread):
    def __init__(self, max_value=300, deadzone=(0,0,0,0,0,0), dtype=np.float32):
        """
        Continuously listen to 3DConnexion SpaceMouse events and update the latest state.

        max_value: 300 for wired SpaceMouse, 500 for wireless
        deadzone: [0,1], number or tuple, axis with value lower than this will stay at 0

        front
        z
        ^   _
        |  (O) space mouse
        |
        *----->x right
        y
        """
        if np.issubdtype(type(deadzone), np.number):
            deadzone = np.full(6, fill_value=deadzone, dtype=dtype)
        else:
            deadzone = np.array(deadzone, dtype=dtype)
        assert (deadzone >= 0).all()

        super().__init__()
        self.stop_event = Event()
        self.max_value = max_value
        self.dtype = dtype
        self.deadzone = deadzone
        self.motion_event = SpnavMotionEvent([0,0,0], [0,0,0], 0)
        self.button_state = defaultdict(lambda: False)
        self.tx_zup_spnav = np.array([
            [0,0,-1],
            [1,0,0],
            [0,1,0]
        ], dtype=dtype)

    def get_motion_state(self):
        me = self.motion_event
        state = np.array(me.translation + me.rotation,
            dtype=self.dtype) / self.max_value
        is_dead = (-self.deadzone < state) & (state < self.deadzone)
        state[is_dead] = 0
        return state

    def get_motion_state_transformed(self):
        """
        Return in right-handed coordinate:
        z
        *------>y right
        |   _
        |  (O) space mouse
        v
        x back
        """
        state = self.get_motion_state()
        tf_state = np.zeros_like(state)
        tf_state[:3] = self.tx_zup_spnav @ state[:3]
        tf_state[3:] = self.tx_zup_spnav @ state[3:]
        tf_state = tf_state * SCALE_FACTOR
        return tf_state

    def is_button_pressed(self, button_id):
        return self.button_state[button_id]

    def stop(self):
        self.stop_event.set()
        self.join()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

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
                    time.sleep(1/200)
        finally:
            spnav_close()


# UR3 robot parameters
ROBOT_HOST = "192.168.0.2"
SCALE_FACTOR = 0.1

# RS485 gripper parameters
GRIPPER_PORT = "/dev/ttyUSB0"
GRIPPER_BAUDRATE = 115200
GRIPPER_TIMEOUT = 1
GRIPPER_COMMAND_DELAY = 0.2

COMMANDS = {
    "clamp_min":     "01 FB 00 01 F4 00 00 2A 94 01 00 6B",  # Close gripper
    "clamp_max":     "01 FB 01 01 F4 00 00 00 00 01 00 6B",  # Open gripper
    "motor_enable":  "01 F3 AB 01 00 6B",                    # Enable motor
    "release_block":  "01 0E 52 6B",
}


class GripperController:
    """Non-blocking RS485 gripper controller. Gripper commands run in a background
    thread so they never stall the 100 Hz robot control loop."""

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
        """Send a command in a background thread; ignores the call if a command
        is already in flight (prevents button-hold flooding)."""
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


def main():
    sm = Spacemouse(deadzone=0.2)
    sm.start()

    rtde_c = RTDEControlInterface(ROBOT_HOST)
    rtde_r = RTDEReceiveInterface(ROBOT_HOST)
    rtde_io = RTDEIO(ROBOT_HOST)

    gripper = GripperController(
        port=GRIPPER_PORT,
        baudrate=GRIPPER_BAUDRATE,
        timeout=GRIPPER_TIMEOUT,
        command_delay=GRIPPER_COMMAND_DELAY,
    )
    print("Enabling gripper motor...")
    gripper.enable()

    # Track previous button states for edge detection (trigger on press, not hold)
    prev_btn = [False, False]

    try:
        while True:
            if rtde_r.getRobotMode() == 7:
                motion_state = sm.get_motion_state_transformed()
                rtde_c.speedL(motion_state, acceleration=0.5, time=0.1)

                actual_velocity = rtde_r.getActualTCPSpeed()
                actual_velocity = [0 if abs(x) < 0.01 else x for x in actual_velocity]
                print("TCP velocity:", actual_velocity)

                # Button 0 (left)  — close gripper
                btn0 = sm.is_button_pressed(0)
                if btn0 and not prev_btn[0]:
                    gripper.send_async("clamp_min")
                prev_btn[0] = btn0

                # Button 1 (right) — open gripper
                btn1 = sm.is_button_pressed(1)
                if btn1 and not prev_btn[1]:
                    gripper._send("release_block")       
                    gripper.send_async("clamp_max")
                prev_btn[1] = btn1

                time.sleep(1/100)
            else:
                print("Robot is not ready.")
                time.sleep(1)

    except KeyboardInterrupt:
        print("Stopping robot")
        rtde_c.stopScript()
        sm.stop()
        gripper.close()


if __name__ == "__main__":
    main()
