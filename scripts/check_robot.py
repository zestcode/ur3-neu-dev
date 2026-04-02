#!/usr/bin/env python3
"""Check UR3 connection and status. If ready, optionally move to home position."""

import sys
import math
from rtde_receive import RTDEReceiveInterface
from rtde_control import RTDEControlInterface

ROBOT_HOST = "192.168.0.2"

# Home position in joint space (radians)
# [0°, -90°, 90°, -90°, -90°, 0°]
HOME_JOINTS = [0, -math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, 0]

ROBOT_MODES = {
    -1: "NO_CONTROLLER",
     0: "DISCONNECTED",
     1: "CONFIRM_SAFETY",
     2: "BOOTING",
     3: "POWER_OFF",
     4: "POWER_ON",
     5: "IDLE",
     6: "BACKDRIVE",
     7: "RUNNING",
}

print(f"Connecting to UR3 at {ROBOT_HOST}...")
try:
    rtde_r = RTDEReceiveInterface(ROBOT_HOST)
except Exception as e:
    print(f"Connection failed: {e}")
    sys.exit(1)

print("Connected.\n")

mode = rtde_r.getRobotMode()
mode_name = ROBOT_MODES.get(mode, "UNKNOWN")
ready = "OK — ready for teleoperation" if mode == 7 else "NOT ready (mode must be 7 / RUNNING)"

print(f"Robot mode:    {mode} ({mode_name})  →  {ready}")
print(f"TCP position:  {[round(v, 4) for v in rtde_r.getActualTCPPose()]}")
print(f"Joint angles:  {[round(v, 4) for v in rtde_r.getActualQ()]}")
print(f"TCP speed:     {[round(v, 4) for v in rtde_r.getActualTCPSpeed()]}")

if mode == 7:
    answer = input("\nMove to home position? [y/N] ").strip().lower()
    if answer == "y":
        rtde_c = RTDEControlInterface(ROBOT_HOST)
        print("Moving to home position...")
        rtde_c.moveJ(HOME_JOINTS, speed=0.5, acceleration=0.5)
        print("Done.")
        rtde_c.stopScript()
