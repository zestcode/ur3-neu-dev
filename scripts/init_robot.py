#!/usr/bin/env python3
"""Move UR3 to home position. Run after check_robot.py confirms the robot is ready."""

import sys
import math
from rtde_receive import RTDEReceiveInterface
from rtde_control import RTDEControlInterface

ROBOT_HOST = "192.168.0.2"

# Home position in joint space (radians)
# [0°, -90°, 90°, -90°, -90°, 0°]
HOME_JOINTS = [0, -math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, 0]

print(f"Connecting to UR3 at {ROBOT_HOST}...")
try:
    rtde_r = RTDEReceiveInterface(ROBOT_HOST)
    rtde_c = RTDEControlInterface(ROBOT_HOST)
except Exception as e:
    print(f"Connection failed: {e}")
    sys.exit(1)

mode = rtde_r.getRobotMode()
if mode != 7:
    print(f"Robot is not ready (mode={mode}). Enable Remote Control and ensure robot is running.")
    sys.exit(1)

print(f"Current joints: {[round(v, 4) for v in rtde_r.getActualQ()]}")
print(f"Target  joints: {[round(v, 4) for v in HOME_JOINTS]}")
answer = input("\nMove to home position? [y/N] ").strip().lower()
if answer == "y":
    print("Moving...")
    rtde_c.moveJ(HOME_JOINTS, speed=0.5, acceleration=0.5)
    print("Done.")

rtde_c.stopScript()
