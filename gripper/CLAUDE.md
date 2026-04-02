# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Python scripts for controlling an RS485-based industrial gripper connected to a Universal Robots UR3 (CB3). Communication uses a USB-to-RS485 adapter on Ubuntu.

## Hardware Requirements

- UR3 (CB3 series) robot
- RS485 Industrial Gripper
- USB-to-RS485 adapter on `/dev/ttyUSB0` (or `/dev/ttyUSB1`)
- External 24V DC power supply
- Ubuntu 22.04/24.04 host

## Setup

```bash
pip install pyserial
sudo chmod 666 /dev/ttyUSB0
```

## Running

```bash
python3 gripper_test.py
```

This runs an Enable → Close → Open sequence as a functional test.

## Code Architecture

`gripper_test.py` is the single script containing:

- **Configuration** (`PORT`, `BAUDRATE`, `TIMEOUT`, `COMMAND_DELAY`) — edit these at the top of the file to match your hardware
- **`COMMANDS` dict** — hex strings representing RS485 command frames sent to the gripper. Format per frame: `[Address][Function Code][Direction/Sub-code][Speed×10][Position×10][Relative/Absolute][Sync Flag][Checksum]`
- **`initialize_serial()`** — opens the serial port; stores the handle in the global `ser`
- **`send_command(name)`** — looks up a command by name, converts hex string to bytes, writes to serial
- **Action helpers** (`enable_action`, `grab_action`, `release_action`) — wrap `send_command` with appropriate `time.sleep` delays for motor settling

The script uses a global `ser` variable for the serial port handle. All commands are fire-and-forget (no response parsing beyond what `current_read` could provide).
