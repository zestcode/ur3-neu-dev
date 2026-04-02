#!/usr/bin/env python3
"""Check SpaceMouse connection by printing live motion events. Move the device to verify input.

Prints two lines per update:
  raw   — direct spnav values, no processing
  proc  — normalized, deadzone-filtered, coordinate-transformed (matches teleop behavior)
"""

import importlib.util
import time
from pathlib import Path

# Load Spacemouse class from main teleop script (filename starts with digit, can't use import)
_spec = importlib.util.spec_from_file_location(
    "teleop", Path(__file__).parent.parent / "3DConnexion_UR3_Teleop.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
Spacemouse = _mod.Spacemouse

PRINT_HZ = 10   # max print rate for motion events
TIMEOUT = 10    # seconds to wait for first event before warning

sm = Spacemouse(deadzone=0.2)
sm.start()
print("SpaceMouse connection open. Move the device (Ctrl+C to quit)...\n")

event_count = 0
start = time.time()
last_raw = None

try:
    while True:
        me = sm.motion_event
        raw = list(me.translation) + list(me.rotation)
        if raw != last_raw and any(v != 0 for v in raw):
            event_count += 1
            proc = sm.get_motion_state_transformed().tolist()
            print(f"raw   | {raw}")
            print(f"proc  | {[round(v, 4) for v in proc]}\n")
        elif event_count == 0 and time.time() - start > TIMEOUT:
            print(f"No events received in {TIMEOUT}s — check spacenavd: systemctl status spacenavd")
            start = time.time()
        last_raw = raw
        time.sleep(1 / PRINT_HZ)
except KeyboardInterrupt:
    pass
finally:
    sm.stop()
    print(f"\nDone. Total non-zero events received: {event_count}")
