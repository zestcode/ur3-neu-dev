# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
 
"""
Standalone UR3 + SpaceMouse + RS485 gripper data collection.
 
This script does NOT register a LeRobot Robot/Teleoperator — it talks to the
hardware directly (RTDE / spnav / pyserial) and reuses LeRobot's dataset and
recording utilities (LeRobotDataset, hw_to_dataset_features, build_dataset_frame,
busy_wait, init_keyboard_listener, VideoEncodingManager, cameras factory) to
get exactly the same frame-synchronisation behaviour as `record.py` without
modifying any library code.
 
State (8D, packed into observation.state):
    tcp_x.pos, tcp_y.pos, tcp_z.pos, tcp_rx.pos, tcp_ry.pos, tcp_rz.pos,
    gripper_left.pos, gripper_right.pos
  - 6D TCP pose from rtde_receive.getActualTCPPose() in robot base frame.
    Last three are an axis-angle rotation vector in radians (NOT Euler).
  - 2D mirrored binary gripper: both 0.0 if closed, both 1.0 if open. Sourced
    from the cached commanded state (no RS485 read-back during the loop).
 
Action (7D):
    tcp_x.vel, tcp_y.vel, tcp_z.vel, tcp_rx.vel, tcp_ry.vel, tcp_rz.vel,
    gripper.cmd
  - 6D TCP velocity in the base frame (m/s, rad/s) -> rtde_control.speedL.
  - 1D binary gripper command: 1.0 -> open, 0.0 -> closed. RS485 sequence
    fires only on transitions; release_block precedes the open clamp.
 
Example:
    python -m lerobot.record_ur3 \
        --robot_host=192.168.0.2 \
        --gripper_port=/dev/ttyUSB0 \
        --cameras='{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}' \
        --dataset.repo_id=${HF_USER}/ur3-spacemouse-test \
        --dataset.num_episodes=10 \
        --dataset.single_task="Pick the cube" \
        --display_data=true
"""
 
import logging
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from threading import Event, Lock, Thread
 
import numpy as np
 
from lerobot.cameras import (  # noqa: F401
    CameraConfig,
)
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.configs import parser
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import (
    init_logging,
    log_say,
)
from lerobot.utils.visualization_utils import log_rerun_data
 
logger = logging.getLogger(__name__)
 
 
# ============================================================================
# Hardware: SpaceMouse (spnav)
# ============================================================================
 
# SpaceMouse native -> robot base frame (right-hand, z-up).
# robot_x = -spnav_z, robot_y = spnav_x, robot_z = spnav_y. Same matrix is
# applied to translation and angular velocity.
_TX_ZUP_SPNAV = np.array(
    [
        [0, 0, -1],
        [1, 0, 0],
        [0, 1, 0],
    ],
    dtype=np.float32,
)
 
 
class Spacemouse(Thread):
    """Daemon thread that drains spnav events and exposes the latest motion +
    button state. Mirrors the reference implementation."""
 
    def __init__(self, max_value: int = 300, deadzone: float = 0.2, scale_factor: float = 0.1,
                 stale_timeout_s: float = 0.25):
        super().__init__(name="spacemouse-poll", daemon=True)
        # Lazy-imported types stored on the instance for use in the thread.
        from spnav import SpnavMotionEvent

        self.stop_event = Event()
        self.max_value = max_value
        self.deadzone = np.full(6, deadzone, dtype=np.float32)
        self.scale_factor = scale_factor
        self.motion_event = SpnavMotionEvent([0, 0, 0], [0, 0, 0], 0)
        self.button_state: dict[int, bool] = defaultdict(bool)
        self._lock = Lock()
        # Freshness guard for the cached motion event. spnav streams events at
        # ~100 Hz the entire time the puck is off-center; if the device's final
        # "zero" event is dropped (USB hiccup, kernel buffer overflow), the
        # cached value would otherwise replay forever and the robot would run
        # away. We stamp every motion event and treat the cache as zero once
        # it goes stale. 250 ms is far longer than the device's poll period
        # so steady holds never spuriously zero out.
        self._last_event_time = 0.0
        self._stale_timeout_s = stale_timeout_s

    def get_motion_state(self) -> np.ndarray:
        with self._lock:
            me = self.motion_event
            last_t = self._last_event_time
            t = list(me.translation)
            r = list(me.rotation)
        # If the spnav stream has stalled, the cached event is unsafe to
        # replay. Return zeros — the controller's speedL watchdog will then
        # decelerate the arm to rest within speedl_watchdog_s.
        if (time.perf_counter() - last_t) > self._stale_timeout_s:
            return np.zeros(6, dtype=np.float32)
        state = np.array(t + r, dtype=np.float32) / float(self.max_value)
        is_dead = (-self.deadzone < state) & (state < self.deadzone)
        state[is_dead] = 0.0
        return state
 
    def get_motion_state_transformed(self) -> np.ndarray:
        state = self.get_motion_state()
        out = np.empty_like(state)
        out[:3] = _TX_ZUP_SPNAV @ state[:3]
        out[3:] = _TX_ZUP_SPNAV @ state[3:]
        out *= self.scale_factor
        return out
 
    def is_button_pressed(self, button_id: int) -> bool:
        # .get() avoids mutating the defaultdict on read; the writer thread
        # touches the same dict so plain __getitem__ would race with resizes.
        return bool(self.button_state.get(button_id, False))
 
    def stop(self):
        self.stop_event.set()
        self.join(timeout=1.0)
 
    def run(self):
        from spnav import (
            SpnavButtonEvent,
            SpnavMotionEvent,
            spnav_close,
            spnav_open,
            spnav_poll_event,
        )
 
        spnav_open()
        try:
            while not self.stop_event.is_set():
                event = spnav_poll_event()
                if isinstance(event, SpnavMotionEvent):
                    with self._lock:
                        self.motion_event = event
                        self._last_event_time = time.perf_counter()
                elif isinstance(event, SpnavButtonEvent):
                    self.button_state[event.bnum] = bool(event.press)
                else:
                    time.sleep(1.0 / 200.0)
        finally:
            spnav_close()
 
 
# ============================================================================
# Hardware: RS485 gripper
# ============================================================================
 
GRIPPER_COMMANDS: dict[str, str] = {
    "clamp_min":     "01 FB 00 01 F4 00 00 2A 94 01 00 6B",  # close
    "clamp_max":     "01 FB 01 01 F4 00 00 00 00 01 00 6B",  # open
    "motor_enable":  "01 F3 AB 01 00 6B",
    "release_block": "01 0E 52 6B",
}
 
 
class GripperController:
    """Non-blocking RS485 gripper. Commands fire in a worker thread so the
    ~200 ms ack delay never stalls the recording loop."""
 
    def __init__(self, port: str, baudrate: int, timeout: float, command_delay: float):
        import serial
 
        self.command_delay = command_delay
        self._lock = Lock()
        self._worker: Thread | None = None
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
        logger.info(f"Gripper RS485 connected on {port}")
 
    def _write_blocking(self, command_name: str) -> None:
        hex_str = GRIPPER_COMMANDS.get(command_name)
        if hex_str is None:
            logger.warning(f"Unknown gripper command: {command_name}")
            return
        data = bytes.fromhex(hex_str.replace(" ", ""))
        with self._lock:
            self.ser.write(data)
        time.sleep(self.command_delay)
        logger.debug(f"Gripper sent: {command_name}")
 
    def send_async_seq(self, command_names: list[str]) -> None:
        # Drop new commands while a previous sequence is in flight; prevents
        # button-mash bus saturation. Matches the reference behaviour.
        if self._worker is not None and self._worker.is_alive():
            return
 
        def _run():
            for name in command_names:
                self._write_blocking(name)
 
        self._worker = Thread(target=_run, name="ur3-gripper-write", daemon=True)
        self._worker.start()
 
    def enable(self) -> None:
        # Synchronous — only ever called once at startup.
        self._write_blocking("motor_enable")
 
    def open_gripper(self) -> None:
        # release_block must precede the open clamp per the controller spec.
        self.send_async_seq(["release_block", "clamp_max"])
 
    def close_gripper(self) -> None:
        self.send_async_seq(["clamp_min"])
 
    def shutdown(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=2.0)
        if self.ser.is_open:
            self.ser.close()
        logger.info("Gripper serial port closed")
 
 
# ============================================================================
# Configuration
# ============================================================================
 
# UR robot mode 7 == RUNNING.
_UR_MODE_RUNNING = 7
 
# Hardware-side feature schema. These keys flow through hw_to_dataset_features
# unchanged and end up in observation.state / action one-to-one.
TCP_POSE_KEYS = ("tcp_x.pos", "tcp_y.pos", "tcp_z.pos", "tcp_rx.pos", "tcp_ry.pos", "tcp_rz.pos")
GRIPPER_OBS_KEYS = ("gripper_left.pos", "gripper_right.pos")
TCP_VEL_KEYS = ("tcp_x.vel", "tcp_y.vel", "tcp_z.vel", "tcp_rx.vel", "tcp_ry.vel", "tcp_rz.vel")
GRIPPER_ACTION_KEY = "gripper.cmd"
 
ROBOT_TYPE = "ur3"
 
 
@dataclass
class UR3DatasetConfig:
    repo_id: str
    single_task: str
    root: str | Path | None = None
    fps: int = 30
    episode_time_s: int | float = 60
    reset_time_s: int | float = 60
    num_episodes: int = 50
    video: bool = True
    push_to_hub: bool = True
    private: bool = False
    tags: list[str] | None = None
    num_image_writer_processes: int = 0
    num_image_writer_threads_per_camera: int = 4
    video_encoding_batch_size: int = 1
 
    def __post_init__(self):
        if self.single_task is None:
            raise ValueError("You need to provide a task as argument in `single_task`.")
 
 
@dataclass
class UR3RecordConfig:
    dataset: UR3DatasetConfig
    # ---- UR3 RTDE -----------------------------------------------------------
    robot_host: str = "192.168.0.2"
    # speedL acceleration limit (m/s^2 / rad/s^2 — UR uses one scalar).
    speedl_acceleration: float = 0.5
    # speedL watchdog window in seconds. UR3 auto-decelerates to 0 if no new
    # speedL is received within this window.
    speedl_watchdog_s: float = 0.1
    # Hard caps applied AFTER the SpaceMouse scale, as a last-line safety net.
    max_linear_speed: float = 0.25
    max_angular_speed: float = 1.0
    # Wait this many seconds at startup for the UR controller to reach mode 7.
    robot_ready_timeout_s: float = 30.0
 
    # ---- RS485 gripper ------------------------------------------------------
    gripper_port: str | None = "/dev/ttyUSB0"
    gripper_baudrate: int = 115200
    gripper_timeout_s: float = 1.0
    gripper_command_delay_s: float = 0.2
    # Initial latched gripper state. MUST match the physical gripper at start.
    initial_gripper_state: float = 1.0
 
    # ---- SpaceMouse ---------------------------------------------------------
    spacemouse_max_value: int = 300
    spacemouse_deadzone: float = 0.2
    spacemouse_scale_factor: float = 0.1
    close_button: int = 0
    open_button: int = 1
 
    # ---- Cameras ------------------------------------------------------------
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
 
    # ---- Recording UX -------------------------------------------------------
    display_data: bool = False
    play_sounds: bool = True
    resume: bool = False
 
 
# ============================================================================
# Hardware-side feature dicts (consumed by hw_to_dataset_features)
# ============================================================================
 
 
def build_hw_features(cfg: UR3RecordConfig) -> tuple[dict, dict]:
    obs_features_hw: dict[str, type | tuple] = {k: float for k in (*TCP_POSE_KEYS, *GRIPPER_OBS_KEYS)}
    for name, cam_cfg in cfg.cameras.items():
        obs_features_hw[name] = (cam_cfg.height, cam_cfg.width, 3)
    action_features_hw: dict[str, type] = {k: float for k in (*TCP_VEL_KEYS, GRIPPER_ACTION_KEY)}
    return obs_features_hw, action_features_hw
 
 
# ============================================================================
# Per-frame helpers
# ============================================================================
 
 
def read_observation(rtde_r, gripper_state: float, cameras: dict) -> dict:
    """Build the flat observation dict that build_dataset_frame consumes.
 
    Calls are ordered: cached RTDE pose (sub-ms), cached gripper command, then
    each camera in turn (camera latest-wins async_read, ~1-5 ms each).
    """
    pose = rtde_r.getActualTCPPose()
    obs: dict = {
        "tcp_x.pos": pose[0],
        "tcp_y.pos": pose[1],
        "tcp_z.pos": pose[2],
        "tcp_rx.pos": pose[3],
        "tcp_ry.pos": pose[4],
        "tcp_rz.pos": pose[5],
        "gripper_left.pos": gripper_state,
        "gripper_right.pos": gripper_state,
    }
    for cam_key, cam in cameras.items():
        obs[cam_key] = cam.async_read()
    return obs
 
 
def get_spacemouse_action(
    sm: Spacemouse, prev_btn: list[bool], gripper_state: float, cfg: UR3RecordConfig
) -> tuple[dict, list[bool], float]:
    """Read one tick of SpaceMouse state. Buttons are edge-triggered: each
    rising edge latches the gripper command to its target value."""
    motion = sm.get_motion_state_transformed()
 
    btn_close = sm.is_button_pressed(cfg.close_button)
    if btn_close and not prev_btn[0]:
        gripper_state = 0.0
 
    btn_open = sm.is_button_pressed(cfg.open_button)
    if btn_open and not prev_btn[1]:
        gripper_state = 1.0
 
    action = {
        "tcp_x.vel": float(motion[0]),
        "tcp_y.vel": float(motion[1]),
        "tcp_z.vel": float(motion[2]),
        "tcp_rx.vel": float(motion[3]),
        "tcp_ry.vel": float(motion[4]),
        "tcp_rz.vel": float(motion[5]),
        GRIPPER_ACTION_KEY: gripper_state,
    }
    return action, [btn_close, btn_open], gripper_state
 
 
def clip_speed(v: list[float], max_lin: float, max_ang: float) -> list[float]:
    lin = v[:3]
    ang = v[3:]
    lin_norm = float(np.linalg.norm(lin))
    ang_norm = float(np.linalg.norm(ang))
    if lin_norm > max_lin and lin_norm > 0:
        s = max_lin / lin_norm
        lin = [x * s for x in lin]
    if ang_norm > max_ang and ang_norm > 0:
        s = max_ang / ang_norm
        ang = [x * s for x in ang]
    return [*lin, *ang]
 
 
def send_action(
    rtde_c,
    gripper: GripperController | None,
    action: dict,
    prev_gripper_state: float,
    cfg: UR3RecordConfig,
) -> dict:
    """Forward TCP velocity to speedL and (on edge transition) drive the
    gripper. The TCP velocity command is fired EVERY frame so the controller's
    speedL watchdog (`time=cfg.speedl_watchdog_s`) is continually refreshed.
 
    `prev_gripper_state` MUST be the latched gripper command from the PREVIOUS
    iteration (i.e. before the current frame's button-edge update). Caller is
    responsible for snapshotting it before invoking get_spacemouse_action.
    """
    v = [float(action[k]) for k in TCP_VEL_KEYS]
    v = clip_speed(v, cfg.max_linear_speed, cfg.max_angular_speed)
    # speedL with time>0 IS the watchdog. URScript on the controller will
    # auto-decelerate to zero if no new speedL arrives within speedl_watchdog_s.
    rtde_c.speedL(v, cfg.speedl_acceleration, cfg.speedl_watchdog_s)
 
    target = 1.0 if float(action[GRIPPER_ACTION_KEY]) >= 0.5 else 0.0
    if gripper is not None and target != prev_gripper_state:
        if target >= 0.5:
            gripper.open_gripper()
        else:
            gripper.close_gripper()
 
    sent = {key: val for key, val in zip(TCP_VEL_KEYS, v, strict=True)}
    sent[GRIPPER_ACTION_KEY] = target
    return sent
 
 
def wait_until_running(rtde_r, timeout_s: float) -> None:
    """Block until UR controller mode is RUNNING (7) or timeout."""
    if timeout_s <= 0:
        mode = rtde_r.getRobotMode()
        if mode != _UR_MODE_RUNNING:
            raise RuntimeError(
                f"UR controller not in RUNNING mode (got {mode}); release the "
                "safety/brake before recording."
            )
        return
 
    deadline = time.perf_counter() + timeout_s
    last_mode = -1
    while time.perf_counter() < deadline:
        mode = rtde_r.getRobotMode()
        if mode == _UR_MODE_RUNNING:
            return
        if mode != last_mode:
            logger.info(f"Waiting for UR mode 7 (RUNNING)... current={mode}")
            last_mode = mode
        time.sleep(0.2)
    raise RuntimeError(
        f"UR controller did not reach RUNNING mode within {timeout_s}s "
        f"(last mode={rtde_r.getRobotMode()})."
    )
 
 
# ============================================================================
# Recording loop (mirrors record.py:record_loop)
# ============================================================================
 
 
@safe_stop_image_writer
def record_loop(
    *,
    rtde_r,
    rtde_c,
    sm: Spacemouse,
    gripper: GripperController | None,
    cameras: dict,
    events: dict,
    fps: int,
    control_time_s: int | float,
    cfg: UR3RecordConfig,
    initial_gripper_state: float,
    dataset: LeRobotDataset | None = None,
    single_task: str | None = None,
) -> float:
    """Run the recording or reset loop for `control_time_s` seconds.
 
    Returns the gripper state at exit so the next phase keeps consistency.
    """
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")
 
    timestamp = 0.0
    start_episode_t = time.perf_counter()
    prev_btn = [False, False]
    gripper_state = initial_gripper_state
 
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()
 
        if events["exit_early"]:
            events["exit_early"] = False
            break
 
        # 1. Observation: cached RTDE pose + cached gripper + cameras
        observation = read_observation(rtde_r, gripper_state, cameras)
        if dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, observation, prefix="observation")
 
        # 2. Action: SpaceMouse motion + edge-triggered gripper buttons.
        # Snapshot the latched state from the PREVIOUS frame before calling
        # get_spacemouse_action, since that call may flip gripper_state on a
        # button rising edge. send_action needs the pre-flip value to detect
        # the transition and fire the RS485 sequence.
        prev_gripper_state = gripper_state
        action, prev_btn, gripper_state = get_spacemouse_action(sm, prev_btn, gripper_state, cfg)
 
        # 3. Send: speedL refreshes the controller-side watchdog every frame;
        # the gripper RS485 sequence fires only on transitions vs prev_gripper_state.
        sent_action = send_action(rtde_c, gripper, action, prev_gripper_state, cfg)
 
        # 4. Pack frame into the dataset
        if dataset is not None:
            action_frame = build_dataset_frame(dataset.features, sent_action, prefix="action")
            frame = {**observation_frame, **action_frame}
            dataset.add_frame(frame, task=single_task)
 
        # 5. Visualisation
        if cfg.display_data:
            log_rerun_data(observation, action)
 
        # 6. Pace the loop to fps
        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1.0 / fps - dt_s)
        timestamp = time.perf_counter() - start_episode_t

    # Phase boundary: bring the arm to a controlled stop. Without this, if the
    # user is still holding the puck when the episode timer expires, the next
    # phase's first speedL would resume that velocity and the arm would move
    # straight through the boundary — perceived as a runaway. speedStop is
    # blocking until the controller has decelerated.
    try:
        rtde_c.speedStop()
    except Exception as e:
        logger.warning(f"speedStop at phase boundary failed: {e}")

    return gripper_state
 
 
# ============================================================================
# Main entry point
# ============================================================================
 
 
@parser.wrap()
def record(cfg: UR3RecordConfig) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        # Don't use _init_rerun: it doesn't send a default blueprint, so any
        # layout change you make in the viewer (e.g. closing a camera panel)
        # is persisted by rerun under the application_id "ur3_recording" and
        # reapplied on every subsequent launch — the camera view never comes
        # back. Build a default blueprint with one 2D view per camera and
        # mark it active+default so it overwrites the persisted state.
        import rerun as rr
        import rerun.blueprint as rrb
        os.environ["RERUN_FLUSH_NUM_BYTES"] = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
        rr.init("ur3_recording")
        cam_views = [
            rrb.Spatial2DView(origin=f"observation.{name}", name=name)
            for name in cfg.cameras.keys()
        ]
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Vertical(*cam_views) if cam_views else rrb.TextLogView(),
                rrb.TimeSeriesView(origin="/", name="signals"),
            ),
            rrb.BlueprintPanel(state="collapsed"),
            rrb.SelectionPanel(state="collapsed"),
        )
        rr.send_blueprint(blueprint, make_active=True, make_default=True)
        rr.spawn(memory_limit=os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%"))
 
    # ---- Build features schema ---------------------------------------------
    obs_features_hw, action_features_hw = build_hw_features(cfg)
    action_features = hw_to_dataset_features(action_features_hw, "action", cfg.dataset.video)
    obs_features = hw_to_dataset_features(obs_features_hw, "observation", cfg.dataset.video)
    dataset_features = {**action_features, **obs_features}
 
    # ---- Build cameras ------------------------------------------------------
    cameras = make_cameras_from_configs(cfg.cameras)
    n_cams = len(cameras)
 
    # ---- Connect hardware ---------------------------------------------------
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface
 
    log_say("Connecting UR3", cfg.play_sounds)
    rtde_r = RTDEReceiveInterface(cfg.robot_host)
    rtde_c = RTDEControlInterface(cfg.robot_host)
    wait_until_running(rtde_r, cfg.robot_ready_timeout_s)
 
    log_say("Starting SpaceMouse", cfg.play_sounds)
    sm = Spacemouse(
        max_value=cfg.spacemouse_max_value,
        deadzone=cfg.spacemouse_deadzone,
        scale_factor=cfg.spacemouse_scale_factor,
    )
    sm.start()
 
    gripper: GripperController | None = None
    if cfg.gripper_port is not None:
        log_say("Enabling gripper", cfg.play_sounds)
        gripper = GripperController(
            port=cfg.gripper_port,
            baudrate=cfg.gripper_baudrate,
            timeout=cfg.gripper_timeout_s,
            command_delay=cfg.gripper_command_delay_s,
        )
        gripper.enable()
 
    for cam in cameras.values():
        cam.connect()
 
    # ---- Build dataset ------------------------------------------------------
    if cfg.resume:
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )
        if n_cams > 0:
            dataset.start_image_writer(
                num_processes=cfg.dataset.num_image_writer_processes,
                num_threads=cfg.dataset.num_image_writer_threads_per_camera * n_cams,
            )
        sanity_check_dataset_robot_compatibility(dataset, _RobotShim(cfg), cfg.dataset.fps, dataset_features)
    else:
        sanity_check_dataset_name(cfg.dataset.repo_id, None)
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.dataset.fps,
            root=cfg.dataset.root,
            robot_type=ROBOT_TYPE,
            features=dataset_features,
            use_videos=cfg.dataset.video,
            image_writer_processes=cfg.dataset.num_image_writer_processes,
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * max(n_cams, 1),
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )
 
    listener, events = init_keyboard_listener()
 
    # Initial latched gripper state — must match the physical gripper.
    gripper_state = 1.0 if cfg.initial_gripper_state >= 0.5 else 0.0
 
    try:
        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
                gripper_state = record_loop(
                    rtde_r=rtde_r,
                    rtde_c=rtde_c,
                    sm=sm,
                    gripper=gripper,
                    cameras=cameras,
                    events=events,
                    fps=cfg.dataset.fps,
                    control_time_s=cfg.dataset.episode_time_s,
                    cfg=cfg,
                    initial_gripper_state=gripper_state,
                    dataset=dataset,
                    single_task=cfg.dataset.single_task,
                )
 
                # Reset phase: keep teleop active but drop frames.
                if not events["stop_recording"] and (
                    (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
                ):
                    log_say("Reset the environment", cfg.play_sounds)
                    gripper_state = record_loop(
                        rtde_r=rtde_r,
                        rtde_c=rtde_c,
                        sm=sm,
                        gripper=gripper,
                        cameras=cameras,
                        events=events,
                        fps=cfg.dataset.fps,
                        control_time_s=cfg.dataset.reset_time_s,
                        cfg=cfg,
                        initial_gripper_state=gripper_state,
                        dataset=None,
                        single_task=cfg.dataset.single_task,
                    )
 
                if events["rerecord_episode"]:
                    log_say("Re-record episode", cfg.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                # Guard: a stale exit_early carried over from the previous reset
                # phase can break the recording phase out before any add_frame
                # runs, leaving an empty buffer that save_episode rejects. Drop
                # such episodes rather than crash.
                buf = dataset.episode_buffer or {}
                if buf.get("size", 0) == 0:
                    log_say("Empty episode, skipping save", cfg.play_sounds)
                    dataset.clear_episode_buffer()
                    events["exit_early"] = False
                    continue

                dataset.save_episode()
                recorded_episodes += 1
 
        log_say("Stop recording", cfg.play_sounds, blocking=True)
    finally:
        # Tear down in safe order: stop motion FIRST, then cut sockets.
        try:
            rtde_c.speedStop()
        except Exception as e:
            logger.warning(f"speedStop() failed: {e}")
        try:
            rtde_c.stopScript()
        except Exception as e:
            logger.warning(f"stopScript() failed: {e}")
        try:
            rtde_c.disconnect()
        except Exception as e:
            logger.warning(f"rtde_c.disconnect() failed: {e}")
        try:
            rtde_r.disconnect()
        except Exception as e:
            logger.warning(f"rtde_r.disconnect() failed: {e}")
        try:
            sm.stop()
        except Exception as e:
            logger.warning(f"spacemouse.stop() failed: {e}")
        if gripper is not None:
            try:
                gripper.shutdown()
            except Exception as e:
                logger.warning(f"gripper.shutdown() failed: {e}")
        for cam in cameras.values():
            try:
                cam.disconnect()
            except Exception as e:
                logger.warning(f"camera.disconnect() failed: {e}")
        if not is_headless() and listener is not None:
            listener.stop()
 
    if cfg.dataset.push_to_hub:
        dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
 
    log_say("Exiting", cfg.play_sounds)
    return dataset
 
 
class _RobotShim:
    """Adapter so sanity_check_dataset_robot_compatibility can introspect this
    pseudo-robot. Only the fields that function actually touches are populated."""
 
    def __init__(self, cfg: UR3RecordConfig):
        self.robot_type = ROBOT_TYPE
        self.name = ROBOT_TYPE
        obs_hw, act_hw = build_hw_features(cfg)
        self.observation_features = obs_hw
        self.action_features = act_hw
 
 
def main():
    record()
 
 
if __name__ == "__main__":
    main()
 