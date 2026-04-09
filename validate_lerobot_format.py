"""
validate_lerobot_format.py — 用 openpi 的 lerobot API 验证 UR3 数据集格式。

运行方式（从 openpi 项目根目录）：
    cd /home/robotics/Desktop/Project_UR3/openpi
    uv run python ../ur3-neu-dev/validate_lerobot_format.py [dataset_dir]

不传参数时默认使用最新的 recordings/<timestamp> 目录。
"""

import sys
import pathlib
import numpy as np
import einops

# ---- 路径 ------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).parent
RECORDINGS_DIR = REPO_ROOT / "recordings"


def latest_recording() -> pathlib.Path:
    dirs = sorted(RECORDINGS_DIR.iterdir())
    if not dirs:
        raise RuntimeError(f"recordings/ 目录为空: {RECORDINGS_DIR}")
    return dirs[-1]


# ---- openpi RepackTransform 映射（与 convert.py 一致）---------------------

REPACK_STRUCTURE = {
    "observation/image":       "observation.images.cam_high",
    "observation/wrist_image": "observation.images.cam_wrist",
    "observation/state":       "observation.state",
    "actions":                 "action",
}


def parse_image(img) -> np.ndarray:
    """LeRobot CHW float32 [0,1]  →  HWC uint8 [0,255]"""
    img = np.asarray(img)
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    if img.ndim == 3 and img.shape[0] == 3:
        img = einops.rearrange(img, "c h w -> h w c")
    return img


# ---- 验证主函数 ------------------------------------------------------------

def validate(dataset_dir: pathlib.Path) -> None:
    import torch
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from openpi import transforms

    print(f"验证数据集: {dataset_dir}\n")

    # 1. 用 openpi 的 lerobot 0.1.0 加载 v2.1 数据集
    ds = LeRobotDataset(repo_id="ur3_teleop", root=dataset_dir)
    print(f"[OK] 数据集加载: {len(ds)} 帧 @ {ds.fps} fps")
    print(f"     features: {list(ds.features.keys())}")

    # 2. RepackTransform
    repack = transforms.RepackTransform(REPACK_STRUCTURE)

    errors = []

    for frame_idx in [0, len(ds) // 2, len(ds) - 1]:
        raw = ds[frame_idx]
        raw_np = {k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in raw.items()}

        repacked = repack(raw_np)

        base_img  = parse_image(repacked["observation/image"])
        wrist_img = parse_image(repacked["observation/wrist_image"])
        state     = np.asarray(repacked["observation/state"], dtype=np.float32)
        action    = np.asarray(repacked["actions"],            dtype=np.float32)

        checks = [
            (state.shape  == (7,),          f"frame {frame_idx}: state shape {state.shape} != (7,)"),
            (action.shape == (7,),          f"frame {frame_idx}: action shape {action.shape} != (7,)"),
            (base_img.shape  == (224,224,3), f"frame {frame_idx}: cam_high shape {base_img.shape}"),
            (wrist_img.shape == (224,224,3), f"frame {frame_idx}: cam_wrist shape {wrist_img.shape}"),
            (base_img.dtype  == np.uint8,   f"frame {frame_idx}: cam_high dtype {base_img.dtype}"),
            (wrist_img.dtype == np.uint8,   f"frame {frame_idx}: cam_wrist dtype {wrist_img.dtype}"),
        ]
        for ok, msg in checks:
            if not ok:
                errors.append(msg)

    # 3. 元数据检查
    import json
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    meta_checks = [
        (info.get("codebase_version") == "v2.1",
         f"codebase_version={info.get('codebase_version')} (期望 v2.1)"),
        (info.get("fps") == 30,
         f"fps={info.get('fps')} (期望 30)"),
        ("observation.images.cam_high"  in info["features"],
         "缺少 observation.images.cam_high"),
        ("observation.images.cam_wrist" in info["features"],
         "缺少 observation.images.cam_wrist"),
        (info["features"]["observation.state"]["shape"] == [7],
         f"observation.state shape={info['features']['observation.state']['shape']}"),
        (info["features"]["action"]["shape"] == [7],
         f"action shape={info['features']['action']['shape']}"),
    ]
    for ok, msg in meta_checks:
        if not ok:
            errors.append(f"meta/info.json: {msg}")

    # 4. 结果
    print()
    if errors:
        print(f"[FAIL] 发现 {len(errors)} 个问题:")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    else:
        sample = ds[0]
        raw_np = {k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
        repacked = repack(raw_np)
        state = np.asarray(repacked["observation/state"], dtype=np.float32)
        action = np.asarray(repacked["actions"], dtype=np.float32)
        print("[OK] 元数据格式正确")
        print(f"[OK] state (7,):  {np.round(state, 4)}")
        print(f"[OK] action (7,): {np.round(action, 4)}")
        print(f"[OK] cam_high:  (224, 224, 3) uint8")
        print(f"[OK] cam_wrist: (224, 224, 3) uint8")
        print()
        print("=== 格式验证通过 — 与 openpi lerobot v2.1 完全兼容 ===")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        dataset_dir = pathlib.Path(sys.argv[1])
    else:
        dataset_dir = latest_recording()

    validate(dataset_dir)
