"""
ur3_inference_display.py — UR3 Pi0.5 推理控制 + 实时画面显示

架构
----
- 主线程    : tkinter 画面显示（Linux Tk 必须在主线程）
- 控制线程  : 连接机器人 + 推理服务器，30Hz 控制循环
- 摄像头线程: grab/retrieve 方式读取双摄像头，最小化帧间延迟

运行方式
--------
1. 启动推理服务端（pi0.5 base 模型）：
       cd /home/robotics/Desktop/Project_UR3/openpi
       uv run scripts/serve_policy.py policy:checkpoint \
           --policy.config pi05_ur3 \
           --policy.dir gs://openpi-assets/checkpoints/pi05_base

2. 启动本脚本：
       conda activate spacemouse-ur
       python3 ur3_inference_display.py

Ctrl+C 或关闭窗口停止。
"""

import signal
import tkinter as tk
from PIL import Image, ImageTk

import time
import serial
import numpy as np
import cv2
from collections import deque
from threading import Thread, Event, Lock

import websockets.sync.client

from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from openpi_client import websocket_client_policy, action_chunk_broker, msgpack_numpy

from safe_filter_local import UR3SafetyFilter, SafetyConfig

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

ROBOT_HOST   = "192.168.0.2"
SERVER_HOST  = "localhost"
SERVER_PORT  = 8000

CAMERA_BASE_INDEX  = 8
CAMERA_WRIST_INDEX = 1
IMAGE_SIZE         = (224, 224)   # 送入模型的尺寸
DISPLAY_SIZE       = (640, 480)   # 显示窗口的尺寸

TASK_PROMPT = "sort the pink cylinder into the orange box"

ACTION_HORIZON = 10   # Pi0.5

SERVO_STEP_TIME  = 1 / 30
SERVO_LOOKAHEAD  = 0.1
SERVO_GAIN       = 300

# ---------------------------------------------------------------------------
# 安全限制
# ---------------------------------------------------------------------------

# 每步最大关节变化量 (rad)。超过此值时等比缩放整个 action。
# 0.05 rad/step × 30 Hz ≈ 1.5 rad/s ≈ 86°/s
MAX_JOINT_DELTA = 0.05

# UR3 各关节位置软限制 (rad)。[-2π, 2π] 是硬件极限，这里收紧。
JOINT_LIMITS_LOW  = [-2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi]
JOINT_LIMITS_HIGH = [ 2*np.pi,  2*np.pi,  2*np.pi,  2*np.pi,  2*np.pi,  2*np.pi]

# 连续超限次数达到此值时触发紧急停止
MAX_CONSECUTIVE_VIOLATIONS = 10

STATS_WINDOW = 30  # number of recent chunks to average over

GRIPPER_THRESHOLD    = 0.3
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
# 夹爪控制
# ---------------------------------------------------------------------------

class GripperController:
    def __init__(self, port, baudrate, timeout, command_delay):
        self.command_delay = command_delay
        self._lock   = Lock()
        self._worker = None
        self.ser = serial.Serial(
            port=port, baudrate=baudrate,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            timeout=timeout,
        )
        if not self.ser.is_open:
            self.ser.open()
        print(f"[Gripper] Connected on {port}")

    def _send(self, command_name):
        hex_str = COMMANDS.get(command_name)
        if hex_str is None:
            print(f"[Gripper] Unknown command: {command_name}")
            return
        data = bytes.fromhex(hex_str.replace(" ", ""))
        with self._lock:
            self.ser.write(data)
            self.ser.flush()
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
        print("[Gripper] Serial port closed")

# ---------------------------------------------------------------------------
# 推理统计 — 控制线程写入，显示线程读取
# ---------------------------------------------------------------------------

class InferenceStats:
    def __init__(self, window=STATS_WINDOW):
        self._lock = Lock()
        self._infer_times = deque(maxlen=window)   # inference latency (ms)
        self._loop_times  = deque(maxlen=window)   # full loop period (ms)
        self._chunk_count = 0
        self._last_status = ""
        self._last_tcp    = None
        self._last_delta  = None

    def record(self, infer_ms, loop_ms, status, tcp, delta):
        with self._lock:
            self._infer_times.append(infer_ms)
            self._loop_times.append(loop_ms)
            self._chunk_count += 1
            self._last_status = status
            self._last_tcp    = tcp
            self._last_delta  = delta

    def snapshot(self):
        with self._lock:
            if not self._loop_times:
                return None
            infer = np.array(self._infer_times)
            loop  = np.array(self._loop_times)
            return {
                "chunk_count": self._chunk_count,
                "infer_ms":    float(infer.mean()),
                "loop_ms":     float(loop.mean()),
                "loop_fps":    1000.0 / float(loop.mean()) if loop.mean() > 0 else 0,
                "infer_fps":   1000.0 / float(infer.mean()) if infer.mean() > 0 else 0,
                "infer_max":   float(infer.max()),
                "status":      self._last_status,
                "tcp":         self._last_tcp,
                "delta":       self._last_delta,
            }

# ---------------------------------------------------------------------------
# 摄像头线程 — grab/retrieve 降低帧间延迟
# ---------------------------------------------------------------------------

class CameraThread(Thread):
    """后台持续抓帧，提供 get_frames() 返回最新的 RGB 帧（模型尺寸）和显示帧。"""

    def __init__(self, base_index, wrist_index):
        super().__init__(daemon=True, name="CameraThread")
        self._base_index  = base_index
        self._wrist_index = wrist_index
        self._lock  = Lock()
        self._stop  = Event()
        self._base_frame  = None   # (224,224,3) RGB for model
        self._wrist_frame = None   # (224,224,3) RGB for model
        self._display_frame = None # combined RGB for display

    def get_model_frames(self):
        with self._lock:
            return self._base_frame, self._wrist_frame

    def get_display_frame(self):
        with self._lock:
            return self._display_frame

    def stop(self):
        self._stop.set()

    def run(self):
        cap_base = cv2.VideoCapture(self._base_index, cv2.CAP_V4L2)
        cap_wrist = cv2.VideoCapture(self._wrist_index, cv2.CAP_V4L2)

        for cap in (cap_base, cap_wrist):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap_base.isOpened() or not cap_wrist.isOpened():
            print("[Camera] Cannot open cameras!")
            return

        print(f"[Camera] Opened base={self._base_index}, wrist={self._wrist_index}")

        try:
            while not self._stop.is_set():
                # grab 同时减少帧间延迟
                ok_b = cap_base.grab()
                ok_w = cap_wrist.grab()

                frames_display = []

                if ok_b:
                    ret, frame_b = cap_base.retrieve()
                    if ret:
                        rgb_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2RGB)
                        model_b = cv2.resize(rgb_b, IMAGE_SIZE)
                        disp_b = cv2.resize(rgb_b, DISPLAY_SIZE)
                        cv2.putText(disp_b, "base", (10, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        frames_display.append(disp_b)
                    else:
                        model_b = None
                else:
                    model_b = None

                if ok_w:
                    ret, frame_w = cap_wrist.retrieve()
                    if ret:
                        rgb_w = cv2.cvtColor(frame_w, cv2.COLOR_BGR2RGB)
                        model_w = cv2.resize(rgb_w, IMAGE_SIZE)
                        disp_w = cv2.resize(rgb_w, DISPLAY_SIZE)
                        cv2.putText(disp_w, "wrist", (10, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        frames_display.append(disp_w)
                    else:
                        model_w = None
                else:
                    model_w = None

                combined = np.hstack(frames_display) if len(frames_display) > 1 else (
                    frames_display[0] if frames_display else None
                )

                with self._lock:
                    self._base_frame  = model_b
                    self._wrist_frame = model_w
                    self._display_frame = combined
        finally:
            cap_base.release()
            cap_wrist.release()
            print("[Camera] Released.")



# ---------------------------------------------------------------------------
# 推理控制循环 — 后台线程
# ---------------------------------------------------------------------------

def _inference_loop(stop_event: Event, cam: CameraThread, stats: InferenceStats) -> None:
    # ---- 机器人 ----
    try:
        rtde_r = RTDEReceiveInterface(ROBOT_HOST)
        rtde_c = RTDEControlInterface(ROBOT_HOST)
        print("[Control] Robot connected.")
    except Exception as exc:
        print(f"[Control] Robot connection failed: {exc}")
        stop_event.wait()
        return

    # ---- 夹爪 ----
    gripper = None
    try:
        gripper = GripperController(
            port=GRIPPER_PORT, baudrate=GRIPPER_BAUDRATE,
            timeout=GRIPPER_TIMEOUT, command_delay=GRIPPER_COMMAND_DELAY,
        )
        gripper.enable()
    except Exception as exc:
        print(f"[Control] Gripper not available: {exc}")

    # ---- 安全过滤器 ----
    safety = UR3SafetyFilter(rtde_control=rtde_c, rtde_receive=rtde_r)
    print("[Safety] Filter initialized (local DH FK)")

    # ---- 推理服务器 ----
    print(f"[Control] Connecting to ws://{SERVER_HOST}:{SERVER_PORT} ...")
    uri = f"ws://{SERVER_HOST}:{SERVER_PORT}"
    ws = websockets.sync.client.connect(
        uri, compression=None, max_size=None,
        ping_interval=None, ping_timeout=None, close_timeout=600,
    )
    metadata = msgpack_numpy.unpackb(ws.recv())
    raw_policy = websocket_client_policy.WebsocketClientPolicy.__new__(
        websocket_client_policy.WebsocketClientPolicy
    )
    raw_policy._ws = ws
    raw_policy._packer = msgpack_numpy.Packer()
    raw_policy._uri = uri
    raw_policy._api_key = None
    raw_policy._server_metadata = metadata
    policy = action_chunk_broker.ActionChunkBroker(
        raw_policy, action_horizon=ACTION_HORIZON
    )
    print("[Control] Connected. Inference loop started (Ctrl+C to stop).")

    gripper_state = 0.0
    violation_count = 0
    first_infer = True

    try:
        while not stop_event.is_set():
            loop_start = time.perf_counter()

            if rtde_r.getRobotMode() != 7:
                print("[Control] Robot not ready (mode != 7), standby ...")
                stop_event.wait(timeout=1)
                policy.reset()
                violation_count = 0
                first_infer = True
                continue

            # ---- 1. 读取观测 ----
            joint_pos = rtde_r.getActualQ()
            state = np.array(joint_pos + [gripper_state], dtype=np.float32)

            base_img, wrist_img = cam.get_model_frames()
            if base_img is None or wrist_img is None:
                print("[Control] Waiting for camera frames...")
                time.sleep(0.1)
                continue

            # ---- 2. 组装 observation ----
            obs = {
                "observation/state":       state,
                "observation/image":       base_img,
                "observation/wrist_image": wrist_img,
                "prompt":                  TASK_PROMPT,
            }

            # ---- 3. 推理 ----
            if first_infer:
                print("[Control] Sending first inference request "
                      "(JAX JIT compilation may take 2-5 min)...", flush=True)
            t_infer = time.perf_counter()
            result = policy.infer(obs)
            infer_ms = (time.perf_counter() - t_infer) * 1000
            if first_infer:
                print(f"[Control] First inference done ({infer_ms:.0f} ms)")
                first_infer = False
            action = result["actions"]
            print(f"[Infer] gripper={float(action[6]):.4f}")
            if action[6]>0.1:
                print(f"[Infer] gripper!!!!!!!!!!!!!!!!!!!!!!!!!!!")

            # ---- 4. 安全检查 (safe_filter_local) ----
            raw_target = action[:6].astype(np.float64)

            if not np.all(np.isfinite(raw_target)):
                print("[Safety] NaN/Inf in action, skipped")
                # violation_count += 1
                if violation_count >= MAX_CONSECUTIVE_VIOLATIONS:
                    print(f"[Safety] {MAX_CONSECUTIVE_VIOLATIONS} consecutive violations — emergency stop!")
                    stop_event.set()
                    break
                continue

            q_now  = np.array(joint_pos, dtype=np.float64)
            qd_now = np.array(rtde_r.getActualQd(), dtype=np.float64)
            chunk  = raw_target.reshape(1, 6)

            safe_chunk, status = safety.filter(q_now, qd_now, chunk, dt=SERVO_STEP_TIME)

            delta = raw_target - q_now
            tcp_pose = rtde_r.getActualTCPPose()

            loop_ms = (time.perf_counter() - loop_start) * 1000
            stats.record(infer_ms, loop_ms, status.reason,
                         tcp_pose[:3], np.rad2deg(delta).tolist())
            
            if not status.ok:
                # violation_count += 1
                print(f"[Safety] BLOCKED ({violation_count}): {status.reason}  details={status.details}")
                if violation_count >= MAX_CONSECUTIVE_VIOLATIONS:
                    print(f"[Safety] {MAX_CONSECUTIVE_VIOLATIONS} consecutive violations — emergency stop!")
                    stop_event.set()
                    break
                continue

            violation_count = 0

            # ---- 5. 执行关节动作 ----
            rtde_c.servoJ(safe_chunk[0].tolist(), 0.5, 0.5, SERVO_STEP_TIME, SERVO_LOOKAHEAD, SERVO_GAIN)

            # ---- 6. 夹爪 ----
            if gripper is not None:
                target_gripper = float(action[6])
                new_state = 1.0 if target_gripper > GRIPPER_THRESHOLD else 0.0
                if new_state != gripper_state:
                    if new_state == 1.0:
                        gripper.send_async("clamp_min")
                        print(f"[Gripper] Close (action[6]={target_gripper:.3f})")
                    else:
                        gripper._send("release_block")
                        gripper.send_async("clamp_max")
                        print(f"[Gripper] Open  (action[6]={target_gripper:.3f})")
                    gripper_state = new_state

            # ---- 控制频率 ----
            elapsed = time.perf_counter() - loop_start
            sleep_t = SERVO_STEP_TIME - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    finally:
        try:
            rtde_c.servoStop()
            rtde_c.stopScript()
        except Exception as e:
            print(f"[Control] stopScript: {e}")
        if gripper is not None:
            gripper.close()
        print("[Control] Stopped.")

# ---------------------------------------------------------------------------
# 显示 — tkinter 主线程
# ---------------------------------------------------------------------------

def _run_display(stop_event: Event, cam: CameraThread, stats: InferenceStats) -> None:
    root = tk.Tk()
    root.title("UR3 Pi0.5 Inference — Camera View")
    root.protocol("WM_DELETE_WINDOW", lambda: (stop_event.set(), root.destroy()))
    signal.signal(signal.SIGINT, lambda *_: (stop_event.set(), root.destroy()))

    label = tk.Label(root)
    label.pack()

    stats_label = tk.Label(
        root, text="Waiting for inference ...", font=("Courier", 11),
        justify=tk.LEFT, anchor="w", bg="black", fg="lime",
    )
    stats_label.pack(fill=tk.X)

    def update_frame():
        frame = cam.get_display_frame()
        if frame is not None:
            img = ImageTk.PhotoImage(Image.fromarray(frame))
            label.imgtk = img
            label.configure(image=img)

        snap = stats.snapshot()
        if snap is not None:
            tcp = snap["tcp"]
            delta = snap["delta"]
            lines = [
                f"Chunks: {snap['chunk_count']}   "
                f"Loop: {snap['loop_fps']:.1f} FPS ({snap['loop_ms']:.1f} ms)   "
                f"Infer: {snap['infer_ms']:.1f} ms ({snap['infer_fps']:.1f}/s, max {snap['infer_max']:.0f})   "
                f"Status: {snap['status']}",
            ]
            if tcp is not None:
                lines.append(
                    f"TCP: [{tcp[0]:+.4f}, {tcp[1]:+.4f}, {tcp[2]:+.4f}]"
                )
            if delta is not None:
                d = delta
                lines.append(
                    f"dq(deg): [{d[0]:+.2f}, {d[1]:+.2f}, {d[2]:+.2f}, "
                    f"{d[3]:+.2f}, {d[4]:+.2f}, {d[5]:+.2f}]"
                )
            stats_label.configure(text="\n".join(lines))

        root.after(33, update_frame)

    def check_stop():
        if stop_event.is_set():
            root.destroy()
            return
        root.after(200, check_stop)

    update_frame()
    check_stop()
    root.mainloop()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    stop_event = Event()
    stats = InferenceStats()

    # 摄像头线程（被控制线程和显示线程共享）
    cam = CameraThread(CAMERA_BASE_INDEX, CAMERA_WRIST_INDEX)
    cam.start()

    # 推理控制在后台线程
    control = Thread(
        target=_inference_loop, args=(stop_event, cam, stats),
        daemon=True, name="InferenceThread",
    )
    control.start()

    # tkinter 显示在主线程
    _run_display(stop_event, cam, stats)
    stop_event.set()
    print("\nShutting down ...")
    control.join(timeout=3)
    cam.stop()
    print("Done.")


if __name__ == "__main__":
    main()
