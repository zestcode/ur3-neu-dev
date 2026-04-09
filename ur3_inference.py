"""
ur3_inference.py — UR3 openpi 推理控制脚本

架构
----
本脚本（客户端）← WebSocket → serve_policy.py（服务端，加载 pi0_ur3 checkpoint）

运行方式
--------
1. 启动推理服务端（在有 GPU 的机器上，进入 openpi 目录）：
       cd /home/robotics/Desktop/Project_UR3/openpi
       uv run scripts/serve_policy.py \
           --policy.config pi0_ur3 \
           --policy.dir checkpoints/pi0_ur3/<exp_name>/<step>

2. 启动本脚本（机器人控制机器）：
       conda activate spacemouse-ur
       pip install openpi-client   # 或: pip install -e openpi/packages/openpi-client
       python3 ur3_inference.py

Action chunking
---------------
Pi0 每次推理返回 action_horizon=50 步动作序列。ActionChunkBroker 将序列
逐步返回，缓冲耗尽后才重新查询服务器，既保证平滑执行又减少推理延迟。
"""

import time
import serial
import numpy as np
import cv2
from threading import Thread, Lock

from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from openpi_client import websocket_client_policy, action_chunk_broker

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

ROBOT_HOST   = "192.168.0.2"
SERVER_HOST  = "localhost"   # 服务端与本机在同一台机器时用 localhost；否则填 IP
SERVER_PORT  = 8000

CAMERA_BASE_INDEX  = 0   # base/overhead 摄像头 OpenCV 设备号
CAMERA_WRIST_INDEX = 2   # wrist 摄像头 OpenCV 设备号
IMAGE_SIZE         = (224, 224)  # (width, height)

TASK_PROMPT = "robot teleoperation"

# Pi0 action_horizon=50；ActionChunkBroker 每步返回一个动作
ACTION_HORIZON = 50

# servoJ 参数 — 与摄像头 FPS 对齐，30Hz 控制循环
SERVO_STEP_TIME  = 1 / 30   # 每步执行时间 (s)
SERVO_LOOKAHEAD  = 0.1      # 平滑时间窗口 (s)，越大越平滑但响应越慢
SERVO_GAIN       = 300      # 比例增益，越大越刚硬

# 夹爪阈值：模型输出 action[6] > 阈值 → 关闭，否则打开
GRIPPER_THRESHOLD    = 0.5
GRIPPER_PORT         = "/dev/ttyUSB0"
GRIPPER_BAUDRATE     = 115200
GRIPPER_TIMEOUT      = 1
GRIPPER_COMMAND_DELAY = 0.2

COMMANDS = {
    "clamp_min":     "01 FB 00 01 F4 00 00 2A 94 01 00 6B",
    "clamp_max":     "01 FB 01 01 F4 00 00 00 00 01 00 6B",
    "motor_enable":  "01 F3 AB 01 00 6B",
    "release_block": "01 0E 52 6B",
}

# ---------------------------------------------------------------------------
# 夹爪控制（与 teleop_vision_record.py 相同）
# ---------------------------------------------------------------------------

class GripperController:
    def __init__(self, port, baudrate, timeout, command_delay):
        self.command_delay = command_delay
        self._lock = Lock()
        self._worker = None
        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            timeout=timeout,
        )
        if not self.ser.is_open:
            self.ser.open()
        print(f"Gripper connected on {port}")

    def _send(self, command_name):
        hex_str = COMMANDS.get(command_name)
        if hex_str is None:
            print(f"Unknown gripper command: {command_name}")
            return
        data = bytes.fromhex(hex_str.replace(" ", ""))
        with self._lock:
            self.ser.write(data)
        time.sleep(self.command_delay)

    def send_async(self, command_name):
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = Thread(target=self._send, args=(command_name,), daemon=True)
        self._worker.start()

    def enable(self):
        self._send("motor_enable")

    def close(self):
        if self.ser.is_open:
            self.ser.close()
        print("Gripper serial port closed")


# ---------------------------------------------------------------------------
# 图像读取
# ---------------------------------------------------------------------------

def read_frame(cap: cv2.VideoCapture) -> np.ndarray:
    """从摄像头读取一帧，调整到 IMAGE_SIZE，转换为 RGB uint8。"""
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Camera read failed")
    frame = cv2.resize(frame, IMAGE_SIZE)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR→RGB（openpi 期望 RGB）


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    # ---- 摄像头 ----
    cap_base  = cv2.VideoCapture(CAMERA_BASE_INDEX)
    cap_wrist = cv2.VideoCapture(CAMERA_WRIST_INDEX)
    if not cap_base.isOpened() or not cap_wrist.isOpened():
        raise RuntimeError("无法打开摄像头，请检查设备号")

    # ---- 机器人 ----
    rtde_r = RTDEReceiveInterface(ROBOT_HOST)
    rtde_c = RTDEControlInterface(ROBOT_HOST)

    # ---- 夹爪 ----
    gripper = GripperController(
        port=GRIPPER_PORT,
        baudrate=GRIPPER_BAUDRATE,
        timeout=GRIPPER_TIMEOUT,
        command_delay=GRIPPER_COMMAND_DELAY,
    )
    gripper.enable()

    # ---- 连接推理服务器 ----
    print(f"连接推理服务器 ws://{SERVER_HOST}:{SERVER_PORT} ...")
    raw_policy = websocket_client_policy.WebsocketClientPolicy(
        host=SERVER_HOST, port=SERVER_PORT
    )
    # ActionChunkBroker：缓存 50 步动作序列，每步返回一个，缓冲耗尽时重新推理
    policy = action_chunk_broker.ActionChunkBroker(raw_policy, action_horizon=ACTION_HORIZON)
    print("已连接，开始推理控制循环（Ctrl+C 退出）")

    # 夹爪状态追踪（开环，按模型输出阈值判断）
    gripper_state = 0.0   # 0.0=开, 1.0=关

    try:
        while True:
            loop_start = time.perf_counter()

            if rtde_r.getRobotMode() != 7:
                print("机器人未就绪 (mode ≠ 7)，等待...")
                time.sleep(1.0)
                policy.reset()   # 清空动作缓冲，避免执行过期动作
                continue

            # ---- 1. 读取观测 ----
            joint_pos = rtde_r.getActualQ()   # list[6], rad
            state = np.array(joint_pos + [gripper_state], dtype=np.float32)  # (7,)

            base_img  = read_frame(cap_base)    # (224,224,3) RGB uint8
            wrist_img = read_frame(cap_wrist)   # (224,224,3) RGB uint8

            # ---- 2. 组装 observation ----
            # 键名为 UR3Inputs 期望的名称（推理时不经过 RepackTransform）
            obs = {
                "observation/state":       state,
                "observation/image":       base_img,
                "observation/wrist_image": wrist_img,
                "prompt":                  TASK_PROMPT,
            }

            # ---- 3. 推理（ActionChunkBroker 自动管理 chunk 缓冲） ----
            result  = policy.infer(obs)
            action  = result["actions"]   # (7,)：joints(6) 绝对值 + gripper(1)

            # ---- 4. 执行关节动作 ----
            target_joints = action[:6].tolist()
            rtde_c.servoJ(
                target_joints,
                speed=0.5,
                acceleration=0.5,
                time=SERVO_STEP_TIME,
                lookahead_time=SERVO_LOOKAHEAD,
                gain=SERVO_GAIN,
            )

            # ---- 5. 执行夹爪动作（阈值触发，边缘检测避免重复发送） ----
            target_gripper = float(action[6])
            new_gripper_state = 1.0 if target_gripper > GRIPPER_THRESHOLD else 0.0
            if new_gripper_state != gripper_state:
                if new_gripper_state == 1.0:
                    gripper.send_async("clamp_min")   # 关闭
                    print(f"夹爪 → 关闭 (action[6]={target_gripper:.3f})")
                else:
                    gripper._send("release_block")    # 阻塞释放
                    gripper.send_async("clamp_max")   # 打开
                    print(f"夹爪 → 打开  (action[6]={target_gripper:.3f})")
                gripper_state = new_gripper_state

            # ---- 控制循环频率 ----
            elapsed = time.perf_counter() - loop_start
            sleep_t = SERVO_STEP_TIME - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\nCtrl+C — 停止推理")

    finally:
        try:
            rtde_c.servoStop()
            rtde_c.stopScript()
        except Exception as e:
            print(f"stopScript: {e}")
        cap_base.release()
        cap_wrist.release()
        gripper.close()
        print("完成。")


if __name__ == "__main__":
    main()
