# UR3 LIBERO-Format Pipeline

End-to-end data collection and policy inference for a Universal Robots UR3 with a 3DConnexion SpaceMouse and RS485 industrial gripper. Datasets and inference observations conform to the LIBERO LeRobot schema, so a fine-tuned `openpi` `pi0` / `pi0.5` checkpoint can drive the arm directly.

```
SpaceMouse + cameras + gripper
        │
        ▼  (teleop)
[lerobot_ur3_record.py]  ─►  LeRobot v2.1 dataset
        │
        ▼  (fine-tune in openpi)
        │
        ▼
openpi policy server (WebSocket)
        ▲
        │  (observation → 7D action chunk)
        │
[tcp_infer.py]  ──drives──►  UR3 (RTDE speedL + RS485 gripper)
```

The two main scripts:

| Script | Purpose |
|---|---|
| **`lerobot_ur3_record.py`** | SpaceMouse teleop → LeRobot v2.1 dataset for fine-tuning |
| **`tcp_infer.py`** | Live inference: drive the UR3 from an openpi LIBERO-format policy |

Older standalone teleop scripts (`3DConnexion_UR3_Teleop.py`, etc.) are retained for ad-hoc use; see "Legacy" below.

## Hardware

| Device | Details |
|---|---|
| Robot arm | Universal Robots UR3, IP `192.168.0.2`, Remote Control enabled |
| Input device | 3DConnexion SpaceMouse (wired, `max_value=300`) |
| Cameras | Two USB cameras: third-person (`fixed`/`image`) + wrist-mounted (`cam_wrist`/`wrist_image`) |
| Gripper | RS485 industrial gripper via USB-to-RS485 adapter on `/dev/ttyUSB0` |

Before running, ensure the workstation and UR3 are on the same subnet and the **Remote Control** URCap script is running on the teach pendant.

## Setup

```bash
# System dependencies
sudo apt install libspnav-dev spacenavd
sudo systemctl start spacenavd

# Conda environment
conda create -n spacemouse-ur python=3.12
conda activate spacemouse-ur

# Core teleop deps (older scripts)
pip install ur_rtde spnav --no-build-isolation numpy pyserial

# LeRobot + openpi-client (for the new pipeline)
pip install lerobot opencv-python pyrealsense2 imageio[ffmpeg]
pip install -e <path-to-openpi>/packages/openpi-client
```

**spnav compatibility fix** (`PyCObject_AsVoidPtr` is deprecated):

```bash
SPNAV_PATH=$(python -c "import spnav, os; print(os.path.dirname(spnav.__file__))")/__init__.py
sed -i 's/PyCObject_AsVoidPtr/PyCapsule_GetPointer/g' $SPNAV_PATH
```

Authorize the gripper serial port (re-run after each reboot):

```bash
sudo chmod 666 /dev/ttyUSB0
```

---

## Data collection — `lerobot_ur3_record.py`

A standalone recorder that talks to RTDE / spnav / RS485 directly while reusing LeRobot's dataset, video-encoding and image-writer machinery so frame synchronisation matches upstream `record.py`.

### Schema

| Field | Shape | Layout |
|---|---|---|
| `observation.state` | `(11,)` float32 | `[eef_x, eef_y, eef_z, r1x, r1y, r1z, r2x, r2y, r2z, gripper, gripper]` |
| `action` | `(7,)` float32 | `[tcp_x.vel, tcp_y.vel, tcp_z.vel, tcp_rx.vel, tcp_ry.vel, tcp_rz.vel, gripper.cmd]` |
| `observation.images.<cam>` | `(H, W, 3)` video | One per camera passed via `--cameras` |

- **Position** is `rtde_r.getActualTCPPose()[0:3]` in robot base frame.
- **Rotation** is the 6-D continuous representation (Zhou et al. 2019): the first two columns of the rotation matrix, computed from RTDE axis-angle via Rodrigues. Avoids the θ=π wraparound that plagues axis-angle.
- **Gripper** is mirrored to two channels (LIBERO convention) and is binary (0=closed, 1=open).
- **Action** is the SpaceMouse 6-DoF velocity command (m/s, rad/s) sent to `rtde_c.speedL` plus a binary gripper command. Same value is recorded as the action.

### Run

```bash
conda activate spacemouse-ur

python3 lerobot_ur3_record.py \
    --dataset.repo_id=user/ur3-pink-cylinder \
    --dataset.single_task="pick up the pink cylinder and place it in the orange box" \
    --dataset.num_episodes=50 \
    --dataset.episode_time_s=50 \
    --dataset.reset_time_s=10 \
    --dataset.fps=30 \
    --dataset.push_to_hub=false \
    --display_data=true \
    --cameras='{image: {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30}, wrist_image: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}'
```

| Flag | Default | Notes |
|---|---|---|
| `--dataset.repo_id` | required | `<owner>/<name>`. Stored under `~/.cache/huggingface/lerobot/<repo_id>/`. |
| `--dataset.single_task` | required | Free-form prompt string. |
| `--dataset.num_episodes` | 50 | Number of NEW episodes (use with `--resume=true` to extend). |
| `--dataset.episode_time_s` | 60 | Hard timeout per episode. Right-arrow ends earlier. |
| `--dataset.reset_time_s` | 60 | Manual repositioning window between episodes. |
| `--dataset.fps` | 30 | Must match `tcp_infer.py CONTROL_HZ` and the inference fps. |
| `--dataset.push_to_hub` | true | Set `false` for local-only datasets. |
| `--resume` | false | Continue an existing dataset. Requires the schema match. |
| `--display_data` | false | Open rerun viewer with cameras + signals layout. |
| `--cameras` | `{}` | LeRobot camera-config dict. **Key names become column names** — use `image` and `wrist_image` for LIBERO compatibility. |
| `--home_joint_positions` | `[0, -π/2, π/2, -π/2, -π/2, 0]` | Used by the inter-episode home prompt. |

### Keyboard (during recording)

| Key | Action |
|---|---|
| `→` (right arrow) | End current episode early; saves the buffered frames. |
| `←` (left arrow) | End current episode AND mark for re-record (drops the buffer). |
| `Esc` | Stop the entire session cleanly (lets the in-flight save finish). |
| `Ctrl+C` | Hard interrupt — risks orphan parquet/MP4 mid-encoding. **Avoid during the SVT-AV1 logs.** |

### Inter-episode home prompt

After each recording phase, the script blocks on stdin:

```
Move to home? Press y or n.
=> Move robot to home position? [y/n]:
```

- `y` → `moveJ` to the configured home joint pose, then opens the gripper. Updates the cached gripper state so the next episode starts honest.
- `n` → continue without moving.

### Reliability features

| Mechanism | Why it exists |
|---|---|
| Initial gripper enforcement at startup | `gripper.enable()` only powers the motor; without an explicit open/close the physical jaws may not match `cfg.initial_gripper_state`, mislabeling early frames. |
| SpaceMouse 250 ms stale-event timeout | If spnav drops the final "zero" event when the puck recenters, the cached value would otherwise replay forever and the arm would run away. |
| `speedStop` on every `record_loop` exit | Prevents held-puck velocity from carrying across phase boundaries. |
| Orphan PNG-staging cleanup at startup | Removes `images/episode_NNN/` directories left behind by interrupted prior sessions before the new image_writer starts. |
| Empty-buffer guard before `save_episode` | A stale `exit_early` flag from the previous reset phase can break a recording phase before any frame is added. The guard skips save instead of crashing. |
| Custom rerun blueprint with `make_default=True` | Overrides any persisted layout that hides cameras. |

### Inspect a dataset

```bash
python3 check_data.py recordings/<your_dataset> --episode 0
```

Reports translation/rotation sync, action axis usage, frozen-state detection. See `check_data.py --help` for plot mode.

### Replay a recorded episode (sanity check)

```bash
# Soft replay (rerun viewer, no robot motion):
python3 replay_ur3.py --repo_id=user/ur3-pink-cylinder --episode=0 --soft

# Hardware replay (drives the arm):
python3 replay_ur3.py --repo_id=user/ur3-pink-cylinder --episode=0
```

---

## Inference — `tcp_infer.py`

A live policy client. Connects to an `openpi` policy server over WebSocket, sends 11-D state + camera images, receives 7-D action chunks, and forwards them to `rtde_c.speedL` + RS485 gripper at the configured control rate.

### Server-side: launching `openpi`

Train (or fine-tune from a `pi0_libero` checkpoint) on the recorded dataset, then serve:

```bash
cd /home/robotics/Desktop/Project_UR3/openpi
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config <your_libero_config> \
    --policy.dir <your_checkpoint_dir>
```

The server listens on TCP `0.0.0.0:8000` by default. First inference triggers JIT compilation (JAX: 2–5 min, PyTorch: 5–30 s).

### How `tcp_infer.py` fits the openpi config

openpi's LIBERO config uses a `RepackTransform` that maps the dataset's column names to internal model keys. The relevant mapping:

```
# openpi.training.config.LeRobotLiberoDataConfig._RepackTransform
{
    "observation/image":       "image"           # if dataset top-level keys
    "observation/wrist_image": "wrist_image"
    "observation/state":       "state"
    "actions":                 "actions"
    "prompt":                  "prompt"
}
```

`tcp_infer.py` sends the model-side keys directly so the server's repack works regardless of how the recorder named its columns:

| Sent over WebSocket | Type | Source |
|---|---|---|
| `observation/state` | `(11,)` float32 | TCP pose → 6D rotation + gripper × 2 |
| `observation/image` | `(256, 256, 3)` uint8 | D435i (or OpenCV) base camera, resized |
| `observation/wrist_image` | `(256, 256, 3)` uint8 | OpenCV wrist camera, resized |
| `prompt` | str | `TASK_PROMPT` constant |

The server replies with `{"actions": (action_horizon, 7) float32}`. The client wraps this in `ActionChunkBroker(action_horizon=10)` which slices one action per call and re-fetches every 10 ticks.

### Run

```bash
conda activate spacemouse-ur

# Start the policy server in another terminal first (above), then:
python3 tcp_infer.py
```

A tkinter window shows live camera frames + a stats overlay (FPS, inference latency, TCP, vL, vR, gripper). Close the window or `Ctrl+C` to stop. The arm decelerates via `speedStop` on shutdown.

### Configuration (top of `tcp_infer.py`)

| Constant | Default | Notes |
|---|---|---|
| `ROBOT_HOST` | `192.168.0.2` | UR3 IP |
| `SERVER_HOST`, `SERVER_PORT` | `localhost`, `8000` | openpi WebSocket endpoint |
| `IMAGE_SIZE` | `(256, 256)` | LIBERO convention |
| `TASK_PROMPT` | hard-coded | Prompt sent to the model. Edit per task. |
| `CONTROL_HZ` | `30` | Match the dataset fps. |
| `ACTION_HORIZON` | `10` | Client-side broker chunk window. |
| `ACTION_SIGN` | `[1,1,1,1,1,1]` | Per-axis polarity flips for fixing axis-frame mismatches without retraining. |
| `ACTION_SPEED_SCALE` | `0.6` | Global slow-down on the 6-DoF velocity. Lower for safer first runs. |
| `ACTION_LIN_MAX` | `0.20 m/s` | Hard clamp |
| `ACTION_ROT_MAX` | `0.80 rad/s` | Hard clamp |
| `GRIPPER_THRESHOLD` | `0.6` | `> threshold` → open, `≤ threshold` → close |
| `GRIPPER_FREEZE_TIME` | `1.0 s` | A1 freeze duration after each gripper transition |
| `WRIST_CAM_INDEX` | `1` | OpenCV index of wrist camera |

### Class A1 gripper mitigation (enabled by default)

During physical gripper actuation (~0.5–1 s), the wrist camera shows half-closed jaws — an out-of-distribution scene the model never saw during training (which has instant 0↔1 transitions). The model becomes uncertain in that window, and a threshold-based decoder amplifies the uncertainty into open/close oscillations.

`tcp_infer.py` breaks this feedback loop:

1. On every gripper transition, fire RS485 + start a 1-second lock.
2. While locked, **zero the arm velocity** (the arm holds in place during actuation) and **ignore further gripper output** from the model.
3. Drop the queued action chunk via `policy.reset()` so the next inference is a fresh chunk planned from the post-actuation observation.

If you still see oscillation **outside** the actuation window (the model can't decide whether to close in the first place), additional mitigations are commented in the source — Class A2 majority-vote filter and asymmetric hysteresis. Uncomment to escalate; see the labelled blocks in the inference loop.

### Convention reminders

- `gripper_state == 1.0` means **open**; `0.0` means **close**. Matches the recorder's RS485 sequences.
- The model's gripper output is in `[0, 1]`. Your fine-tuned checkpoint may need `GRIPPER_THRESHOLD` tuned (default 0.6).

---

## Utilities

| Script | What it does |
|---|---|
| `replay_ur3.py` | Replay a saved episode either in rerun viewer (`--soft`) or on the actual hardware. |
| `check_data.py` | Audit a recorded dataset: state/action distributions, sync correlation, frozen-state detection. |
| `verify_rtde_freeze.py` | Standalone diagnostic for `boost::asio` EOF in `ur_rtde`. |

## Diagnostics for the older teleop pipeline

```bash
# Raw SpaceMouse input only
python3 scripts/check_spacemouse_raw.py

# Raw + processed (deadzone-filtered, coordinate-transformed)
python3 scripts/check_spacemouse.py

# Check UR3 connection and status; optionally move to home position
python3 scripts/check_robot.py

# Move directly to home position [0°, -90°, 90°, -90°, -90°, 0°]
python3 scripts/init_robot.py
```

## Legacy (older standalone teleop)

Earlier scripts that pre-date the LIBERO pipeline. Useful as standalone teleop demos but not for ML data collection:

```bash
python3 3DConnexion_UR3_Teleop.py            # teleop without gripper
python3 3DConnexion_UR3_Teleop_Gripper.py    # teleop with RS485 gripper
```

Gripper control: SpaceMouse **left button** closes, **right button** opens. `Ctrl+C` stops gracefully.

## References

- UR RTDE documentation: https://sdurobotics.gitlab.io/ur_rtde/index.html
- openpi: `/home/robotics/Desktop/Project_UR3/openpi`
- LeRobot: `/home/robotics/Desktop/Project_UR3/lerobot`
- UR5 reference scripts and Robotiq gripper driver: `reference/`
- RS485 gripper standalone test: `gripper/gripper_test.py`
- 6D rotation representation: Zhou et al. 2019, *On the Continuity of Rotation Representations in Neural Networks*, https://arxiv.org/abs/1812.07035
