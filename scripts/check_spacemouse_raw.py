#!/usr/bin/env python3
"""Check SpaceMouse connection by printing raw spnav events. Move the device to verify input."""

from spnav import spnav_open, spnav_poll_event, spnav_close, SpnavMotionEvent, SpnavButtonEvent
import time

PRINT_HZ = 10   # max print rate for motion events
TIMEOUT = 10    # seconds to wait for first event before warning

spnav_open()
print("SpaceMouse connection open. Move the device (Ctrl+C to quit)...")

event_count = 0
start = time.time()
last_print = 0.0

try:
    while True:
        e = spnav_poll_event()
        if isinstance(e, SpnavMotionEvent):
            event_count += 1
            now = time.time()
            if now - last_print >= 1 / PRINT_HZ:
                print(f"Motion  | translation: {list(e.translation)}  rotation: {list(e.rotation)}")
                last_print = now
        elif isinstance(e, SpnavButtonEvent):
            event_count += 1
            print(f"Button  | id={e.bnum}  pressed={e.press}")
        else:
            if event_count == 0 and time.time() - start > TIMEOUT:
                print(f"No events received in {TIMEOUT}s — check spacenavd: systemctl status spacenavd")
                start = time.time()
            time.sleep(0.005)
except KeyboardInterrupt:
    pass
finally:
    spnav_close()
    print(f"\nDone. Total events received: {event_count}")
