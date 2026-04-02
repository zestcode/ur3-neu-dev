# UR3 SpaceMouse Teleoperation

Real-time Cartesian velocity teleoperation of a Universal Robots UR3 using a 3DConnexion SpaceMouse, with optional RS485 industrial gripper control.

## Hardware

| Device | Details |
|--------|---------|
| Robot arm | Universal Robots UR3 (`192.168.0.2`) |
| Input device | 3DConnexion SpaceMouse (wired, `max_value=300`) |
| Gripper | RS485 industrial gripper via USB-to-RS485 adapter on `/dev/ttyUSB0` |

Before running, ensure the workstation and UR3 are on the same subnet and **Remote Control** is enabled on the teach pendant.

## Setup

```bash
# System dependencies
sudo apt install libspnav-dev spacenavd
sudo systemctl start spacenavd

# Conda environment
conda create -n spacemouse-ur python=3.12
conda activate spacemouse-ur
pip install ur_rtde spnav --no-build-isolation numpy pyserial
```

**spnav compatibility fix** (`PyCObject_AsVoidPtr` is deprecated):

```bash
SPNAV_PATH=$(python -c "import spnav, os; print(os.path.dirname(spnav.__file__))")/__init__.py
sed -i 's/PyCObject_AsVoidPtr/PyCapsule_GetPointer/g' $SPNAV_PATH
```

> This fix has already been applied on this machine for both `base` and `spacemouse-ur` environments.

## Running

```bash
conda activate spacemouse-ur

# Without gripper
python3 3DConnexion_UR3_Teleop.py

# With RS485 gripper (authorize serial port first)
sudo chmod 666 /dev/ttyUSB0
python3 3DConnexion_UR3_Teleop_Gripper.py
```

Gripper control: SpaceMouse **left button** closes, **right button** opens. Stop with `Ctrl+C` for graceful shutdown.

## Diagnostics

```bash
# Raw SpaceMouse input only
python3 scripts/check_spacemouse_raw.py

# Raw + processed input (deadzone-filtered, coordinate-transformed — matches teleop behavior)
python3 scripts/check_spacemouse.py

# Check UR3 connection and status; optionally move to home position
python3 scripts/check_robot.py

# Move directly to home position [0°, -90°, 90°, -90°, -90°, 0°]
python3 scripts/init_robot.py
```

## References

- UR RTDE documentation: https://sdurobotics.gitlab.io/ur_rtde/index.html
- UR5 reference scripts and Robotiq gripper driver: `reference/`
- RS485 gripper standalone test: `gripper/gripper_test.py`
