"""
Verify a LeRobot v2.1 dataset (especially after merge).

Checks:
  1. Format: info.json fields, file existence, path templates
  2. Timeseries: frame_index, index, timestamp continuity
  3. Video paths: parquet VideoFrame paths match actual files
  4. Source preservation: data values unchanged vs originals (optional)

Usage:
    python verify_dataset_v21.py --dataset zestcode5/ur3-multiple-task

    # With source comparison:
    python verify_dataset_v21.py \
        --dataset zestcode5/ur3-multiple-task \
        --source_a poweredshine/ur3-pot-lid-opening \
        --source_b zestcode5/ur3_pick_place
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HF_LEROBOT_HOME = Path.home() / ".cache" / "huggingface" / "lerobot"

# From lerobot/datasets/utils.py DEFAULT_FEATURES
REQUIRED_COLUMNS = {"timestamp", "frame_index", "episode_index", "index", "task_index"}

# Required info.json keys for v2.1
REQUIRED_INFO_KEYS = [
    "codebase_version", "robot_type", "total_episodes", "total_frames",
    "total_tasks", "total_videos", "total_chunks", "chunks_size",
    "fps", "splits", "data_path", "video_path", "features",
]


def resolve(repo_id, root=None):
    return Path(root) if root else HF_LEROBOT_HOME / repo_id


class C:
    """Check collector."""
    def __init__(self):
        self.ok = 0
        self.warn = []
        self.err = []

    def check(self, cond, msg, warn=False):
        if cond:
            self.ok += 1
        elif warn:
            self.warn.append(msg)
        else:
            self.err.append(msg)

    def report(self):
        print(f"\n{'='*60}")
        print(f"  PASSED:   {self.ok}")
        print(f"  WARNINGS: {len(self.warn)}")
        print(f"  ERRORS:   {len(self.err)}")
        print(f"{'='*60}")
        for w in self.warn:
            print(f"  ⚠  {w}")
        for e in self.err:
            print(f"  ✗  {e}")
        if not self.err:
            print(f"  ✓  All checks passed")
        return len(self.err) == 0


def check_format(root, c):
    """Check 1: v2.1 format compliance."""
    print("\n── 1. Format ──")

    info_p = root / "meta" / "info.json"
    c.check(info_p.exists(), "meta/info.json missing")
    if not info_p.exists():
        return None

    with open(info_p) as f:
        info = json.load(f)

    # codebase_version
    v = info.get("codebase_version")
    c.check(v == "v2.1", f"codebase_version='{v}', expected 'v2.1'")

    # Required keys
    for k in REQUIRED_INFO_KEYS:
        c.check(k in info, f"info.json missing key: '{k}'")

    # Path templates
    dp = info.get("data_path", "")
    c.check(
        "{episode_chunk" in dp and "{episode_index" in dp,
        f"data_path template invalid: '{dp}'"
    )
    vp = info.get("video_path", "")
    c.check(
        "{episode_chunk" in vp and "{video_key}" in vp and "{episode_index" in vp,
        f"video_path template invalid: '{vp}'"
    )

    n_ep = info["total_episodes"]
    cs = info.get("chunks_size", 1000)
    vkeys = [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]

    # tasks.jsonl
    tp = root / "meta" / "tasks.jsonl"
    c.check(tp.exists(), "meta/tasks.jsonl missing")
    tasks = {}
    if tp.exists():
        with open(tp) as f:
            for l in f:
                o = json.loads(l)
                tasks[o["task_index"]] = o["task"]
        c.check(len(tasks) == info.get("total_tasks", -1),
                f"tasks count: file={len(tasks)} vs info={info.get('total_tasks')}")

    # episodes.jsonl
    ep = root / "meta" / "episodes.jsonl"
    c.check(ep.exists(), "meta/episodes.jsonl missing")
    if ep.exists():
        with open(ep) as f:
            eps = [json.loads(l) for l in f]
        c.check(len(eps) == n_ep,
                f"episodes.jsonl count: {len(eps)} vs info={n_ep}")
        # Check episode indices contiguous
        indices = [e["episode_index"] for e in eps]
        c.check(indices == list(range(n_ep)),
                f"episodes.jsonl indices not 0..{n_ep-1}")

    # episodes_stats.jsonl
    sp = root / "meta" / "episodes_stats.jsonl"
    c.check(sp.exists(), "meta/episodes_stats.jsonl missing (v2.1 requires this)", warn=True)

    # splits
    splits = info.get("splits", {})
    expected_split = f"0:{n_ep}"
    c.check(
        splits.get("train") == expected_split,
        f"splits.train='{splits.get('train')}' expected '{expected_split}'"
    )

    # total_chunks
    expected_chunks = (n_ep - 1) // cs + 1 if n_ep > 0 else 0
    c.check(
        info.get("total_chunks") == expected_chunks,
        f"total_chunks={info.get('total_chunks')} expected {expected_chunks}"
    )

    # File existence for every episode
    missing_pq = 0
    missing_vid = 0
    for ei in range(n_ep):
        chunk = ei // cs
        pq = root / info["data_path"].format(episode_chunk=chunk, episode_index=ei)
        if not pq.exists():
            missing_pq += 1
        for vk in vkeys:
            vf = root / info["video_path"].format(episode_chunk=chunk, episode_index=ei, video_key=vk)
            if not vf.exists():
                missing_vid += 1

    c.check(missing_pq == 0, f"{missing_pq} parquet files missing")
    c.check(missing_vid == 0, f"{missing_vid} video files missing")

    # total_videos
    expected_vids = n_ep * len(vkeys)
    c.check(
        info.get("total_videos") == expected_vids,
        f"total_videos={info.get('total_videos')} expected {expected_vids}",
        warn=True,
    )

    print(f"  {n_ep} episodes, {len(vkeys)} video keys, chunks_size={cs}")
    print(f"  tasks: {tasks}")
    return info


def check_timeseries(root, info, c):
    """Check 2: Timeseries integrity."""
    print("\n── 2. Timeseries ──")

    n_ep = info["total_episodes"]
    total_fr = info["total_frames"]
    cs = info.get("chunks_size", 1000)
    vkeys = [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]

    # Load tasks for validation
    tasks = {}
    tp = root / "meta" / "tasks.jsonl"
    if tp.exists():
        with open(tp) as f:
            for l in f:
                o = json.loads(l)
                tasks[o["task_index"]] = o["task"]

    global_indices = []
    tasks_seen = set()

    for ei in range(n_ep):
        pq = root / info["data_path"].format(episode_chunk=ei // cs, episode_index=ei)
        if not pq.exists():
            continue
        df = pd.read_parquet(pq)

        # Required columns
        for col in REQUIRED_COLUMNS:
            c.check(col in df.columns, f"ep{ei}: missing column '{col}'")

        if not REQUIRED_COLUMNS.issubset(df.columns):
            continue

        # episode_index consistent
        uep = df["episode_index"].unique()
        c.check(len(uep) == 1 and int(uep[0]) == ei,
                f"ep{ei}: episode_index={uep.tolist()}, expected [{ei}]")

        # frame_index 0..N-1
        fi = df["frame_index"].values
        c.check(
            np.array_equal(fi, np.arange(len(df))),
            f"ep{ei}: frame_index not 0..{len(df)-1}"
        )

        # timestamp monotonic
        ts = df["timestamp"].values
        c.check(
            np.all(ts[1:] >= ts[:-1]),
            f"ep{ei}: timestamp not monotonic"
        )

        global_indices.extend(df["index"].values.tolist())
        tasks_seen.update(df["task_index"].unique().tolist())

        # Check video paths in parquet match files on disk
        for vk in vkeys:
            if vk not in df.columns:
                continue
            sample = df[vk].iloc[0] if len(df) > 0 else None
            if isinstance(sample, dict) and "path" in sample:
                path_val = df[vk].iloc[0]["path"]
                c.check(
                    (root / path_val).exists(),
                    f"ep{ei}: parquet video path '{path_val}' → file not found"
                )
                # Check path matches expected pattern for this episode
                expected_path = info["video_path"].format(
                    episode_chunk=ei // cs, episode_index=ei, video_key=vk
                )
                c.check(
                    path_val == expected_path,
                    f"ep{ei}: video path='{path_val}' expected '{expected_path}'"
                )

    # Global index contiguous
    gs = sorted(global_indices)
    c.check(len(gs) == total_fr,
            f"global index count={len(gs)} vs total_frames={total_fr}")
    if len(gs) == total_fr and total_fr > 0:
        c.check(
            gs == list(range(total_fr)),
            f"global index not contiguous 0..{total_fr-1}, range=[{min(gs)}..{max(gs)}]"
        )

    # Task indices valid
    invalid = tasks_seen - set(tasks.keys())
    c.check(len(invalid) == 0,
            f"parquet references unknown task_index: {invalid}")

    print(f"  Checked {n_ep} episodes, {len(global_indices)} frames")


def check_source(root, info, sa, sb, c):
    """Check 3: Source data preservation."""
    print("\n── 3. Source preservation ──")

    ia = json.load(open(sa / "meta" / "info.json"))
    ib = json.load(open(sb / "meta" / "info.json"))
    na = ia["total_episodes"]
    cs_a = ia.get("chunks_size", 1000)
    cs_b = ib.get("chunks_size", 1000)
    cs_m = info.get("chunks_size", 1000)

    vkeys = [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]
    data_cols = [k for k in info.get("features", {})
                 if k not in REQUIRED_COLUMNS
                 and info["features"][k].get("dtype") != "video"]

    # Check A
    ok_a = 0
    for ei in range(na):
        sp = sa / ia["data_path"].format(episode_chunk=ei // cs_a, episode_index=ei)
        dp = root / info["data_path"].format(episode_chunk=ei // cs_m, episode_index=ei)
        if not sp.exists() or not dp.exists():
            continue
        sdf = pd.read_parquet(sp)
        ddf = pd.read_parquet(dp)
        c.check(len(sdf) == len(ddf), f"A ep{ei}: frame count {len(sdf)} vs {len(ddf)}")
        for col in data_cols:
            if col in sdf.columns and col in ddf.columns:
                try:
                    sv = np.stack(sdf[col].values)
                    dv = np.stack(ddf[col].values)
                    eq = np.allclose(sv, dv, rtol=1e-6, atol=1e-8, equal_nan=True)
                except (ValueError, TypeError):
                    eq = sdf[col].equals(ddf[col])
                c.check(eq, f"A ep{ei} col '{col}': values changed")
        ok_a += 1

    # Check B
    nb = ib["total_episodes"]
    ok_b = 0
    for ei in range(nb):
        nei = ei + na
        sp = sb / ib["data_path"].format(episode_chunk=ei // cs_b, episode_index=ei)
        dp = root / info["data_path"].format(episode_chunk=nei // cs_m, episode_index=nei)
        if not sp.exists() or not dp.exists():
            continue
        sdf = pd.read_parquet(sp)
        ddf = pd.read_parquet(dp)
        c.check(len(sdf) == len(ddf), f"B ep{ei}→{nei}: frame count {len(sdf)} vs {len(ddf)}")
        for col in data_cols:
            if col in sdf.columns and col in ddf.columns:
                try:
                    sv = np.stack(sdf[col].values)
                    dv = np.stack(ddf[col].values)
                    eq = np.allclose(sv, dv, rtol=1e-6, atol=1e-8, equal_nan=True)
                except (ValueError, TypeError):
                    eq = sdf[col].equals(ddf[col])
                c.check(eq, f"B ep{ei}→{nei} col '{col}': values changed")
        ok_b += 1

    print(f"  A: verified {ok_a}/{na}  B: verified {ok_b}/{nb}")


def check_loadable(root, info, c):
    """Check 4: Try loading with lerobot 0.3.3."""
    print("\n── 4. LeRobot load test ──")
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        repo_id = info.get("repo_id", root.parent.name + "/" + root.name)
        ds = LeRobotDataset(repo_id, root=root)
        c.check(ds.num_episodes == info["total_episodes"],
                f"LeRobotDataset episodes={ds.num_episodes} vs {info['total_episodes']}")
        c.check(ds.num_frames == info["total_frames"],
                f"LeRobotDataset frames={ds.num_frames} vs {info['total_frames']}")
        # Sample a frame
        sample = ds[0]
        c.check("action" in sample or "observation.state" in sample,
                "Sample frame missing expected keys")
        print(f"  LeRobotDataset loaded OK: {ds.num_episodes}ep, {ds.num_frames}fr")
        print(f"  Sample keys: {list(sample.keys())[:8]}...")
    except ImportError:
        print("  Skipped (lerobot not importable in this env)")
    except Exception as e:
        c.check(False, f"LeRobotDataset load failed: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--root", default=None)
    p.add_argument("--source_a", default=None)
    p.add_argument("--source_b", default=None)
    p.add_argument("--root_a", default=None)
    p.add_argument("--root_b", default=None)
    p.add_argument("--skip_load", action="store_true", help="Skip LeRobotDataset load test")
    a = p.parse_args()

    root = resolve(a.dataset, a.root)
    print(f"Verifying: {root}\n")

    c = C()

    info = check_format(root, c)
    if info is None:
        c.report()
        sys.exit(1)

    check_timeseries(root, info, c)

    if a.source_a and a.source_b:
        check_source(root, info, resolve(a.source_a, a.root_a), resolve(a.source_b, a.root_b), c)

    if not a.skip_load:
        check_loadable(root, info, c)

    ok = c.report()

    if ok:
        print(f"\n── Upload ──")
        print(f"""python -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('{a.dataset}', repo_type='dataset', exist_ok=True)
api.upload_folder(folder_path='{root}', repo_id='{a.dataset}', repo_type='dataset')
api.create_tag('{a.dataset}', tag='v2.1', repo_type='dataset')
print('Done')
"
""")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
