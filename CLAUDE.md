# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

Install system dependencies:
```bash
sudo apt install libspnav-dev spacenavd
sudo systemctl start spacenavd
```

Create and activate the project conda environment:
```bash
conda create -n spacemouse-ur python=3.12
conda activate spacemouse-ur
pip install ur_rtde
pip install spnav --no-build-isolation
pip install numpy
```

**Known spnav issue:** `PyCObject_AsVoidPtr` is deprecated. After installing spnav, fix it:
```bash
# Find the installed file
SPNAV_PATH=$(python -c "import spnav; import os; print(os.path.dirname(spnav.__file__))")/\_\_init\_\_.py
sed -i 's/PyCObject_AsVoidPtr/PyCapsule_GetPointer/g' $SPNAV_PATH
```
On this machine the fix has already been applied to both `base` and `spacemouse-ur` environments.

## Running

```bash
conda activate spacemouse-ur

# UR3 teleoperation without gripper
python3 3DConnexion_UR3_Teleop.py

# UR3 teleoperation with RS485 gripper (SpaceMouse buttons control open/close)
python3 3DConnexion_UR3_Teleop_Gripper.py
```

Stop with `Ctrl+C` — this triggers graceful shutdown (stops RTDE script and SpaceMouse thread).

Reference scripts for UR5 and Robotiq gripper control are in `reference/`.

### Diagnostic scripts (`scripts/`)

```bash
# Raw spnav values only — minimal, no processing
python3 scripts/check_spacemouse_raw.py

# Raw + processed (deadzone-filtered, coordinate-transformed) — matches teleop behavior
python3 scripts/check_spacemouse.py
```

Both scripts print at 10Hz max and stop printing when the device is idle.

```bash
# Check UR3 connection and status; if ready, optionally move to home position
python3 scripts/check_robot.py

# Move to home position directly (run after check passes)
python3 scripts/init_robot.py
```

Home position: `[0°, -90°, 90°, -90°, -90°, 0°]` — all joints at multiples of 90°.

## Architecture

The system has three layers:

**SpaceMouse input** (`Spacemouse` class, a `Thread`): Polls `spnav` events at 200Hz and stores the latest `SpnavMotionEvent` and button states. The `get_motion_state_transformed()` method applies a coordinate frame rotation (`tx_zup_spnav`) to convert from SpaceMouse frame to robot frame and scales by `SCALE_FACTOR`. Deadzone is configured via the `deadzone=` constructor parameter (unified in one place); the hardcoded per-axis deadzone in `get_motion_state_transformed()` has been removed.

**Robot control** (RTDE): Uses `ur_rtde` to send Cartesian velocity commands (`speedL`) at 100Hz to the robot. Requires robot to be in mode 7 (running). Robot IP is hardcoded as `ROBOT_HOST = "192.168.0.2"`.

**Gripper control** (`GripperController` class in `3DConnexion_UR3_Teleop_Gripper.py`): Sends RS485 commands over a USB-to-RS485 serial adapter (`/dev/ttyUSB0`). SpaceMouse left button closes the gripper (`clamp_min`), right button opens it (`clamp_max`). Commands run in a background thread so they never block the 100Hz control loop; edge detection ensures a held button only triggers once. Motor is enabled automatically on startup. The Robotiq HAND-E TCP-based driver in `reference/robotiq_gripper.py` is not in use.

## Key Parameters

| Parameter | Location | Value |
|-----------|----------|-------|
| `ROBOT_HOST` | `3DConnexion_UR3_Teleop.py` | `192.168.0.2` |
| `SCALE_FACTOR` | `3DConnexion_UR3_Teleop.py` | `0.1` |
| `acceleration` | `3DConnexion_UR3_Teleop.py` | `0.5` |
| `max_value` | `Spacemouse.__init__` | `300` (wired); use `500` for wireless SpaceMouse |
| `deadzone` | `Spacemouse.__init__` (`deadzone=`) | `0.2` — scalar applies to all 6 axes; pass a 6-tuple for per-axis control |
| `speedL time` | `main()` | `0.1` s safety timeout: robot decelerates and stops if no new command arrives within this window |
| Control loop rate | `main()` | 100Hz (`time.sleep(1/100)`) |

## Lab Hardware Configuration

- **Robot**: Universal Robots UR3
- **Input device**: 3DConnexion SpaceMouse (wired, `max_value=300`)
- **Gripper**: RS485 industrial gripper via USB-to-RS485 adapter on `/dev/ttyUSB0`; controlled by `3DConnexion_UR3_Teleop_Gripper.py`
- **Robot IP**: `192.168.0.2`
- **Network**: Workstation and UR3 must be on the same subnet

Before running, verify connectivity and enable Remote Control on the UR3 teach pendant.

## RS485 Gripper

The gripper integration lives in `3DConnexion_UR3_Teleop_Gripper.py`. The original standalone test script and command definitions are in `gripper/gripper_test.py` (see `gripper/CLAUDE.md`).

Requires `pip install pyserial` and `sudo chmod 666 /dev/ttyUSB0` before running. Gripper port is configurable via `GRIPPER_PORT` at the top of the file (default `/dev/ttyUSB0`).

The `GripperController` in the main teleop script exposes only `clamp_min`, `clamp_max`, and `motor_enable`. The standalone `gripper/gripper_test.py` defines additional commands (`release_block`, `angle_clear`, `current_read`) useful for debugging hardware issues.

RS485 frame format: `[Address][Function Code][Direction/Sub-code][Speed×10][Position×10][Relative/Absolute][Sync Flag][Checksum]`

## Code Duplication

The `Spacemouse` class is copied verbatim into both `3DConnexion_UR3_Teleop.py` and `3DConnexion_UR3_Teleop_Gripper.py`. Any changes to SpaceMouse behavior must be applied to both files.

## Robot Mode Reference

`getRobotMode()` return values (from `scripts/check_robot.py`):

| Mode | Name |
|------|------|
| -1 | NO_CONTROLLER |
| 0 | DISCONNECTED |
| 3 | POWER_OFF |
| 5 | IDLE |
| 7 | RUNNING ← required for teleoperation |

## Import Quirk

The main script filename (`3DConnexion_UR3_Teleop.py`) starts with a digit, so Python cannot import it with a normal `import` statement. The diagnostic scripts in `scripts/` work around this using `importlib.util.spec_from_file_location` to load the `Spacemouse` class directly from the file path.
