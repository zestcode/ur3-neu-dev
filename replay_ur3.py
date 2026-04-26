"""
replay_ur3.py — Replay a recorded LeRobot dataset episode.

Two modes:

    HARDWARE replay (default)
        Drives the UR3 over RTDE: each frame's 6-DoF velocity goes to
        rtde_c.speedL, gripper edges go to RS485. Same I/O path as the
        recorder, so trajectories reproduce up to controller dynamics.

    SOFT replay (--soft)
        No hardware. State / action / images are logged to a rerun
        viewer at the recorded fps, so you can sanity-check a dataset
        visually. Use this BEFORE running the hardware replay if you
        want to confirm a recording looks reasonable.

The dataset must have been collected by lerobot_ur3_record.py:
    action schema = [tcp_x.vel, tcp_y.vel, tcp_z.vel,
                     tcp_rx.vel, tcp_ry.vel, tcp_rz.vel,
                     gripper.cmd]
    fps from meta/info.json

Usage:
    conda activate spacemouse-ur

    # Soft replay in rerun (no robot, no gripper, no risk):
    python3 replay_ur3.py \\
        --repo_id=user/ur3-pink-cylinder_6D \\
        --episode=0 --soft

    # Hardware replay (auto home, then drive arm + gripper):
    python3 replay_ur3.py \\
        --repo_id=user/ur3-pink-cylinder_6D \\
        --episode=0

    # Hardware replay without homing or gripper:
    python3 replay_ur3.py --repo_id=user/ur3-pink-cylinder_6D --episode=0 --no_home --no_gripper

Safety notes (hardware replay):
    - The arm WILL move. Make sure the workspace is clear and the
      teach pendant is in Remote Control with the URCap running (mode 7).
    - There is a 3-second countdown after homing before replay starts —
      Ctrl+C aborts cleanly during the countdown.
    - The script stops the arm (speedStop) on any error or exit.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

# Reuse helpers from the recorder so replay matches recording exactly.
from lerobot_ur3_record import (
    GripperController,
    HOME_JOINT_POSITIONS,
    _UR_MODE_RUNNING,
    move_to_home,
    wait_until_running,
)

logger = logging.getLogger(__name__)


def soft_replay(root: Path, info: dict, parquet: Path, fps: int) -> int:
    """Visual-only replay: log state, action, and camera frames to a rerun
    viewer at the recorded fps. Does NOT touch the robot or the gripper.
    Used to sanity-check a dataset before committing to a hardware replay.
    """
    import os
    import imageio.v3 as iio
    import rerun as rr
    import rerun.blueprint as rrb

    df = pd.read_parquet(parquet)
    state = np.stack(df["observation.state"].to_numpy()).astype(np.float64)   # (T, S)
    actions = np.stack(df["action"].to_numpy()).astype(np.float64)            # (T, A)
    T = len(df)
    dt = 1.0 / fps

    state_names = info["features"]["observation.state"]["names"]
    action_names = info["features"]["action"]["names"]
    cam_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]

    print(f"[soft]   T={T} frames  state={state.shape[1]}D  action={actions.shape[1]}D  "
          f"cameras={cam_keys}")

    # Open rerun with a clean blueprint (cameras + time-series), force-default
    # so any stale persisted layout doesn't hide the views.
    os.environ["RERUN_FLUSH_NUM_BYTES"] = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    rr.init(f"replay_{parquet.stem}")
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(*[rrb.Spatial2DView(origin=cam, name=cam.split(".")[-1])
                            for cam in cam_keys])
            if cam_keys else rrb.TextLogView(),
            rrb.TimeSeriesView(origin="/", name="signals"),
        ),
        rrb.BlueprintPanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
    )
    rr.send_blueprint(blueprint, make_active=True, make_default=True)
    rr.spawn(memory_limit=os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%"))

    # Pre-decode all camera frames via imageio (bundled FFmpeg handles AV1).
    # Episodes are short (~30s × 30fps × 480 × 640 × 3 ≈ 800 MB), so eager
    # decoding into memory avoids decoder seek issues at the cost of RAM.
    cam_buffers: dict[str, list] = {}
    for cam in cam_keys:
        vid_path = root / "videos/chunk-000" / cam / f"{parquet.stem}.mp4"
        if not vid_path.exists():
            print(f"[soft]   ⚠ video missing for {cam}: {vid_path}")
            continue
        try:
            print(f"[soft]   decoding {cam} ...")
            frames = list(iio.imiter(str(vid_path), plugin="FFMPEG"))
            cam_buffers[cam] = frames
            print(f"           {len(frames)} frames decoded")
        except Exception as e:
            print(f"[soft]   ⚠ failed to decode {cam}: {e}")

    print(f"[run]    Streaming to rerun at {fps} fps ...")
    t0 = time.perf_counter()
    for k in range(T):
        tick_start = time.perf_counter()

        # Set the rerun timeline to this frame's wall-clock offset.
        rr.set_time_seconds("frame_time", k * dt)

        # State scalars — picked up by TimeSeriesView
        for i, name in enumerate(state_names):
            rr.log(f"observation.{name}", rr.Scalar(float(state[k, i])))

        # Action scalars — picked up by TimeSeriesView
        for i, name in enumerate(action_names):
            rr.log(f"action.{name}", rr.Scalar(float(actions[k, i])))

        # Camera frames — picked up by Spatial2DView
        for cam, frames in cam_buffers.items():
            if k < len(frames):
                rr.log(cam, rr.Image(frames[k]))

        elapsed = time.perf_counter() - tick_start
        sleep_t = dt - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

        if k % (fps * 3) == 0:
            pct = 100.0 * k / max(1, T - 1)
            print(f"           frame {k:>5}/{T}  ({pct:5.1f}%)")

    wall = time.perf_counter() - t0
    print(f"[done]   Streamed {T} frames in {wall:.2f}s "
          f"(target {T*dt:.2f}s, drift {wall - T*dt:+.2f}s)")
    print(f"           rerun viewer is open. Close it to exit.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo_id", required=True,
                    help="Dataset repo id, e.g. user/ur3-pink-cylinder_6D")
    ap.add_argument("--episode", type=int, required=True,
                    help="Episode index to replay (0-based).")
    ap.add_argument("--root", default=None,
                    help="Local dataset root. Default: ~/.cache/huggingface/lerobot/<repo_id>")
    ap.add_argument("--soft", action="store_true",
                    help="Soft replay: log dataset to rerun viewer only. NO robot, "
                         "NO gripper, NO RTDE connection. Used for visual sanity-checks.")
    ap.add_argument("--robot_host", default="192.168.0.2")
    ap.add_argument("--gripper_port", default="/dev/ttyUSB0")
    ap.add_argument("--no_gripper", action="store_true",
                    help="Don't actually drive the RS485 gripper during replay")
    ap.add_argument("--no_home", action="store_true",
                    help="Skip moveJ-to-home before replay")
    ap.add_argument("--speedl_accel", type=float, default=0.5,
                    help="speedL acceleration limit (m/s^2 / rad/s^2)")
    ap.add_argument("--speedl_watchdog", type=float, default=0.1,
                    help="speedL watchdog window (s) — robot decelerates "
                         "if no new speedL arrives within this window")
    ap.add_argument("--countdown", type=float, default=3.0,
                    help="Seconds to wait after homing, before starting replay")
    args = ap.parse_args()

    # Resolve dataset path
    if args.root is None:
        root = Path("~/.cache/huggingface/lerobot").expanduser() / args.repo_id
    else:
        root = Path(args.root).expanduser()

    parquet = root / "data/chunk-000" / f"episode_{args.episode:06d}.parquet"
    info_path = root / "meta/info.json"
    if not parquet.exists():
        print(f"ERROR: episode parquet not found: {parquet}", file=sys.stderr)
        return 1
    if not info_path.exists():
        print(f"ERROR: info.json not found: {info_path}", file=sys.stderr)
        return 1

    info = json.load(open(info_path))
    fps = info["fps"]
    dt = 1.0 / fps
    action_names = info["features"]["action"]["names"]
    expected = ["tcp_x.vel", "tcp_y.vel", "tcp_z.vel",
                "tcp_rx.vel", "tcp_ry.vel", "tcp_rz.vel", "gripper.cmd"]
    if action_names != expected:
        print(f"ERROR: action schema mismatch.\n  expected: {expected}\n  got:      {action_names}",
              file=sys.stderr)
        return 1

    df = pd.read_parquet(parquet)
    actions = np.stack(df["action"].to_numpy()).astype(np.float64)  # (T, 7)
    T = len(actions)
    print(f"[load]   {parquet.name}: T={T} ({T*dt:.1f}s @ {fps} fps)")

    # ------ SOFT mode: visualize-only, no hardware ------
    if args.soft:
        return soft_replay(root, info, parquet, fps)

    # ------ Hardware connect ------
    print(f"[robot]  Connecting RTDE to {args.robot_host} ...")
    rtde_r = RTDEReceiveInterface(args.robot_host)
    rtde_c = RTDEControlInterface(args.robot_host)
    try:
        wait_until_running(rtde_r, 30.0)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        rtde_c.disconnect(); rtde_r.disconnect()
        return 1
    print(f"[robot]  mode={rtde_r.getRobotMode()} (7 = RUNNING)")

    gripper = None
    if not args.no_gripper:
        try:
            gripper = GripperController(
                port=args.gripper_port, baudrate=115200,
                timeout=1.0, command_delay=0.2,
            )
            gripper.enable()
            time.sleep(0.5)
        except Exception as e:
            print(f"WARN: gripper init failed: {e}. Continuing without gripper.", file=sys.stderr)
            gripper = None

    try:
        # ------ Move to home (unless --no_home) ------
        if not args.no_home:
            print(f"[home]   moveJ to home pose {[f'{j:+.3f}' for j in HOME_JOINT_POSITIONS]}")
            move_to_home(rtde_c, HOME_JOINT_POSITIONS, velocity=0.5, acceleration=0.5)
        else:
            print(f"[home]   skipped (--no_home)")

        # ------ Countdown ------
        print(f"[start]  Replay starts in {args.countdown:.0f}s. Ctrl+C to abort.")
        for s in range(int(args.countdown), 0, -1):
            print(f"           {s}...")
            time.sleep(1.0)

        # ------ Replay loop ------
        # Sync the gripper to the FIRST recorded action's gripper value before
        # the loop starts, so the first edge inside the loop only fires if the
        # action genuinely transitions during replay.
        prev_g = float(actions[0, 6])
        if gripper is not None:
            if prev_g >= 0.5:
                gripper.open_gripper()
            else:
                gripper.close_gripper()
            time.sleep(1.0)

        print(f"[run]    Replaying {T} frames at {fps} fps ...")
        t0 = time.perf_counter()
        for k in range(T):
            tick_start = time.perf_counter()

            v = actions[k, :6].tolist()
            g = float(actions[k, 6])

            rtde_c.speedL(v, args.speedl_accel, args.speedl_watchdog)

            if gripper is not None and g != prev_g:
                if g >= 0.5:
                    gripper.open_gripper()
                else:
                    gripper.close_gripper()
                prev_g = g

            elapsed = time.perf_counter() - tick_start
            sleep_t = dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

            # Periodic progress log (every ~3s)
            if k % (fps * 3) == 0:
                pct = 100.0 * k / max(1, T - 1)
                print(f"           frame {k:>5}/{T}  ({pct:5.1f}%)")

        wall = time.perf_counter() - t0
        print(f"[done]   Replayed {T} frames in {wall:.2f}s "
              f"(target {T*dt:.2f}s, drift {wall - T*dt:+.2f}s)")
        return 0

    except KeyboardInterrupt:
        print("\n[abort]  Ctrl+C — stopping arm and exiting.")
        return 130
    except Exception as e:
        print(f"\n[error]  {e}", file=sys.stderr)
        return 2
    finally:
        # Clean shutdown — same order as the recorder's finally
        try:
            rtde_c.speedStop()
        except Exception:
            pass
        try:
            rtde_c.stopScript()
        except Exception:
            pass
        try:
            rtde_c.disconnect()
        except Exception:
            pass
        try:
            rtde_r.disconnect()
        except Exception:
            pass
        if gripper is not None:
            try:
                gripper.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
