# UR3 Gripper Control (RS485)

This repository contains Python scripts for controlling an RS485-based industrial gripper integrated with a Universal Robots UR3 (CB3).

## 🛠 Hardware Setup

### 1. Requirements
* **Robot:** UR3 (CB3 series)
* **Gripper:** RS485 Industrial Gripper
* **Communication:** USB-to-RS485 Adapter 
* **Power:** External 24V DC Power Supply
* **Host:** Ubuntu 22.04/24.04

### 2. Wiring Diagram

* **RS485 A (+):** Connect to Adapter A+
* **RS485 B (-):** Connect to Adapter B-
* **VCC (24V):** External Power +24V

---
## 🚀 Software Setup
1. Install the required Python library:
```bash
pip install pyserial
```
2. Port Permissions
Ubuntu restricts access to serial ports by default. Grant permissions to your device (usually /dev/ttyUSB0):
```
sudo chmod 666 /dev/ttyUSB0
```
3. Basic Functional Test
Run the main script to perform a standard "Enable -> Close -> Open" sequence:
```
python3 gripper_test.py
```
