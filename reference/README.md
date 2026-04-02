## Download spacemouse dependencies

```bash
sudo apt install libspnav-dev spacenavd; sudo systemctl start spacenavd
pip install spnav
```

## Check if spacemouse is connected to workstation
```bash
lsusb
```

## Download RTDE library
```bash
pip install --user ur_rtde
```

## Run spacemouse script (UR3)
```bash
python3 3DConnexion_UR3_Teleop.py
```

## Reference scripts
UR5 scripts and Robotiq gripper driver are in the `reference/` folder.

## Note:
In the spnav library, PyCObject_AsVoidPtr is deprecated

find . -name "spnav" on terminal to find spnav folder

Replace all instances of PyCObject_AsVoidPtr with PyCapsule_GetPointer in __init__.py

## To include more RTDE functionalities
https://sdurobotics.gitlab.io/ur_rtde/index.html
