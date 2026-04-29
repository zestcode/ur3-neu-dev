"""
Merge two LeRobot v2.1 datasets into one.

Core reindexing logic ported from lerobot v0.4.0 aggregate.py.
Adapted for v2.1 (one parquet + one video per episode).

Key points verified against real v2.1 info.json:
  - Chunk dir computed from episode_index // chunks_size
  - Video dir uses full feature key (e.g. observation.images.front)
  - Parquet video columns (VideoFrame dict with path+timestamp) rewritten
  - info.json: total_videos, total_chunks, data_path/video_path templates preserved

Usage:
    python merge_datasets_v21.py \
        --dataset_a poweredshine/ur3-pot-lid-opening \
        --dataset_b zestcode5/ur3_pick_place \
        --output zestcode5/ur3-multiple-task
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import pandas as pd

HF_LEROBOT_HOME = Path.home() / ".cache" / "huggingface" / "lerobot"

INFO_PATH  = "meta/info.json"
TASKS_PATH = "meta/tasks.jsonl"
EPISODES_PATH = "meta/episodes.jsonl"
STATS_PATH = "meta/episodes_stats.jsonl"
STATS_JSON = "meta/stats.json"


def resolve_root(repo_id, root=None):
    return Path(root) if root else HF_LEROBOT_HOME / repo_id

def load_json(p):
    with open(p) as f:
        return json.load(f)

def load_jsonl(p):
    if not Path(p).exists():
        return []
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]

def write_jsonl(p, items):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")

def get_video_keys(info):
    return [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]

def data_rel(info, ep_idx):
    cs = info.get("chunks_size", 1000)
    return info["data_path"].format(episode_chunk=ep_idx // cs, episode_index=ep_idx)

def video_rel(info, ep_idx, vk):
    cs = info.get("chunks_size", 1000)
    return info["video_path"].format(episode_chunk=ep_idx // cs, episode_index=ep_idx, video_key=vk)


# ── Validate (aggregate.py:45-78) ──────────────────────────────────────────

def validate(info_a, info_b):
    errs = []
    if info_a["fps"] != info_b["fps"]:
        errs.append(f"fps: {info_a['fps']} vs {info_b['fps']}")
    if info_a.get("robot_type") != info_b.get("robot_type"):
        errs.append(f"robot_type: {info_a.get('robot_type')} vs {info_b.get('robot_type')}")
    if info_a.get("data_path") != info_b.get("data_path"):
        errs.append(f"data_path template differs")
    if info_a.get("video_path") != info_b.get("video_path"):
        errs.append(f"video_path template differs")

    fa = info_a.get("features", {})
    fb = info_b.get("features", {})
    ka, kb = set(fa), set(fb)
    if ka != kb:
        errs.append(f"Feature keys differ. Only A: {ka-kb}, Only B: {kb-ka}")
    else:
        for k in ka:
            if fa[k] != fb[k]:
                errs.append(f"Feature '{k}' definition differs")
    if errs:
        raise ValueError("Validation failed:\n  " + "\n  ".join(errs))
    print("[validate] ✓")


# ── Tasks (aggregate.py:226-228) ───────────────────────────────────────────

def merge_tasks(ta, tb):
    merged, s2i, bremap = [], {}, {}
    for t in ta:
        if t["task"] not in s2i:
            s2i[t["task"]] = len(merged)
            merged.append({"task_index": len(merged), "task": t["task"]})
    for t in tb:
        if t["task"] not in s2i:
            s2i[t["task"]] = len(merged)
            merged.append({"task_index": len(merged), "task": t["task"]})
        bremap[t["task_index"]] = s2i[t["task"]]
    return merged, bremap


# ── Parquet reindex (aggregate.py:81-102) ──────────────────────────────────

def reindex_parquet(df, ep_offset, fr_offset, task_remap, video_keys, info):
    df = df.copy()
    df["episode_index"] = df["episode_index"] + ep_offset
    df["index"] = df["index"] + fr_offset

    if "task_index" in df.columns and task_remap:
        df["task_index"] = df["task_index"].map(task_remap)

    # Rewrite VideoFrame path+timestamp dicts inside parquet
    # v2.1 format: {"path": "videos/chunk-000/obs.img.front/episode_000003.mp4", "timestamp": 0.033}
    cs = info.get("chunks_size", 1000)
    for vk in video_keys:
        if vk not in df.columns:
            continue
        sample = df[vk].iloc[0] if len(df) > 0 else None
        if not isinstance(sample, dict) or "path" not in sample:
            continue
        new_col = []
        for row in df[vk]:
            row = dict(row)
            old_ep = int(re.search(r'episode_(\d+)', row["path"]).group(1))
            new_ep = old_ep + ep_offset
            new_chunk = new_ep // cs
            row["path"] = info["video_path"].format(
                episode_chunk=new_chunk, episode_index=new_ep, video_key=vk
            )
            new_col.append(row)
        df[vk] = new_col

    return df


# ── Main ───────────────────────────────────────────────────────────────────

def merge(repo_a, repo_b, repo_out, root_a=None, root_b=None, root_out=None):
    sa = resolve_root(repo_a, root_a)
    sb = resolve_root(repo_b, root_b)
    sd = resolve_root(repo_out, root_out)

    print(f"[A] {sa}\n[B] {sb}\n[→] {sd}")
    for label, p, r in [("A", sa, repo_a), ("B", sb, repo_b)]:
        if not (p / INFO_PATH).exists():
            raise FileNotFoundError(f"{label} not found: {p / INFO_PATH}")

    ia = load_json(sa / INFO_PATH)
    ib = load_json(sb / INFO_PATH)
    validate(ia, ib)

    if sd.exists():
        shutil.rmtree(sd)

    na, nb = ia["total_episodes"], ib["total_episodes"]
    fa, fb = ia["total_frames"], ib["total_frames"]
    vkeys = get_video_keys(ia)
    cs = ia.get("chunks_size", 1000)

    print(f"[info] A={na}ep/{fa}fr  B={nb}ep/{fb}fr  vkeys={vkeys}  cs={cs}")

    # Tasks
    ta = load_jsonl(sa / TASKS_PATH)
    tb = load_jsonl(sb / TASKS_PATH)
    mt, br = merge_tasks(ta, tb)
    print(f"[tasks] {[t['task'] for t in mt]}  B-remap={br}")

    # ── Copy A as-is ──
    print(f"\n[A] copying {na} episodes...")
    for ei in range(na):
        # parquet
        r = data_rel(ia, ei)
        s = sa / r
        if s.exists():
            d = sd / r; d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)
        # videos
        for vk in vkeys:
            r = video_rel(ia, ei, vk)
            s = sa / r
            if s.exists():
                d = sd / r; d.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(s, d)

    # ── Copy B reindexed ──
    print(f"[B] copying {nb} episodes (reindexing)...")
    for ei in range(nb):
        nei = ei + na
        # parquet
        src_r = data_rel(ib, ei)
        dst_r = data_rel(ia, nei)
        s = sb / src_r
        if s.exists():
            df = pd.read_parquet(s)
            df = reindex_parquet(df, na, fa, br, vkeys, ia)
            d = sd / dst_r; d.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(d)
        # videos
        for vk in vkeys:
            src_r = video_rel(ib, ei, vk)
            dst_r = video_rel(ia, nei, vk)
            s = sb / src_r
            if s.exists():
                d = sd / dst_r; d.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(s, d)

    # ── Metadata ──
    print("[meta] writing...")
    write_jsonl(sd / TASKS_PATH, mt)

    # episodes.jsonl
    ea = load_jsonl(sa / EPISODES_PATH)
    eb = load_jsonl(sb / EPISODES_PATH)
    me = list(ea)
    for e in eb:
        e = dict(e)
        e["episode_index"] += na
        if "task_index" in e and e["task_index"] in br:
            e["task_index"] = br[e["task_index"]]
        if "tasks" in e and isinstance(e["tasks"], list):
            e["tasks"] = [br.get(t, t) if isinstance(t, int) else t for t in e["tasks"]]
        me.append(e)
    write_jsonl(sd / EPISODES_PATH, me)

    # episodes_stats.jsonl
    sa_s = load_jsonl(sa / STATS_PATH)
    sb_s = load_jsonl(sb / STATS_PATH)
    ms = list(sa_s)
    for s in sb_s:
        s = dict(s)
        if "episode_index" in s:
            s["episode_index"] += na
        ms.append(s)
    write_jsonl(sd / STATS_PATH, ms)

    if (sa / STATS_JSON).exists():
        (sd / "meta").mkdir(parents=True, exist_ok=True)
        shutil.copy2(sa / STATS_JSON, sd / STATS_JSON)

    # info.json
    tot_ep = na + nb
    io = dict(ia)
    io["total_episodes"] = tot_ep
    io["total_frames"] = fa + fb
    io["total_tasks"] = len(mt)
    io["total_videos"] = ia.get("total_videos", 0) + ib.get("total_videos", 0)
    io["total_chunks"] = (tot_ep - 1) // cs + 1 if tot_ep > 0 else 0
    io["splits"] = {"train": f"0:{tot_ep}"}
    io["repo_id"] = repo_out
    with open(sd / INFO_PATH, "w") as f:
        json.dump(io, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Merge complete: {sd}")
    print(f"  Episodes: {na}+{nb}={tot_ep}")
    print(f"  Frames:   {fa}+{fb}={fa+fb}")
    print(f"  Videos:   {io['total_videos']}")
    print(f"  Tasks:    {[t['task'] for t in mt]}")
    print(f"{'='*60}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_a", required=True)
    p.add_argument("--dataset_b", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--root_a", default=None)
    p.add_argument("--root_b", default=None)
    p.add_argument("--output_root", default=None)
    a = p.parse_args()
    merge(a.dataset_a, a.dataset_b, a.output, a.root_a, a.root_b, a.output_root)
