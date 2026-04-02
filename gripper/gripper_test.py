import serial
import time

# ====== Configuration Parameters ======
# On Ubuntu, the port is usually /dev/ttyUSB0 or /dev/ttyUSB1
PORT = "/dev/ttyUSB0" 
BAUDRATE = 115200
TIMEOUT = 1
COMMAND_DELAY = 0.2

# ====== Command Definitions ======
# Command format (Hex): 
# [0]: Address, [1]: Function Code, [2]: Direction/Sub-code, [3-4]: Speed*10, 
# [5-8]: Position*10, [9]: 0-Relative/1-Absolute, [10]: Sync Flag, [11]: Checksum
COMMANDS = {
    "clamp_min": "01 FB 00 01 F4 00 00 2A 94 01 00 6B",  # Close gripper
    "clamp_max": "01 FB 01 01 F4 00 00 00 00 01 00 6B",  # Open to maximum
    "motor_enable": "01 F3 AB 01 00 6B",                 # Enable motor
    "release_block": "01 0E 52 6B",                      # Release stall/block
    "angle_clear": "01 0A 6D 6B",                        # Reset angle to zero
    "current_read": "01 27 6B"                           # Read phase current
}

ser = None

def initialize_serial():
    """Initialize the serial connection"""
    global ser
    try:
        ser = serial.Serial(
            port=PORT,
            baudrate=BAUDRATE,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            timeout=TIMEOUT
        )
        print(f"[{time.strftime('%H:%M:%S')}] Successfully connected to {PORT}")
        if not ser.is_open:
            ser.open()
            print(f"[{time.strftime('%H:%M:%S')}] Serial port opened")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] Serial initialization failed: {str(e)}")

def send_command(command_name):
    """Send a specific command by name"""
    global ser
    if ser is None or not ser.is_open:
        print(f"[{time.strftime('%H:%M:%S')}] Serial not connected, command failed")
        return False

    command_hex = COMMANDS.get(command_name)
    if not command_hex:
        print(f"[{time.strftime('%H:%M:%S')}] Command not found: {command_name}")
        return False

    try:
        data = bytes.fromhex(command_hex)
        ser.write(data)
        print(f"[{time.strftime('%H:%M:%S')}] Successfully sent: {command_name}")
        time.sleep(COMMAND_DELAY)
        return True
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] Send failed: {str(e)}")
        return False

def grab_action():
    send_command("clamp_min")
    time.sleep(5)

def release_action():
    send_command("clamp_max")
    time.sleep(5)

def enable_action():
    send_command("motor_enable")
    time.sleep(0.3)

# ====== Execution Block ======
if __name__ == "__main__":
    initialize_serial()
    if ser and ser.is_open:
        print("Starting Test: Enabling motor...")
        enable_action()
        
        print("Action: Closing gripper...")
        grab_action()
        
        print("Action: Opening gripper...")
        release_action()
        
        print("Test complete. Closing serial port.")
        ser.close()
