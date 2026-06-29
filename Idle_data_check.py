"""
analyze_idle_frames.py
======================
分析 LeRobot 数据集中 idle（action ≈ 0）帧的占比，
并给出是否因 idle 过多导致模型 freeze 的诊断意见。
 
支持两种数据格式：
  1. LeRobot HuggingFace 格式（parquet / Arrow）
  2. 原始 HDF5 格式（.hdf5 / .h5）
 
用法:
  python analyze_idle_frames.py --dataset_path /path/to/dataset [options]
 
选项:
  --dataset_path   数据集根目录（包含 data/ 子目录）或单个 .hdf5 文件
  --threshold      判定 idle 的 L-inf 阈值（默认 0.01）
  --per_joint      是否打印每个 joint 维度的 idle 占比
  --per_episode    是否打印每条 episode 的 idle 占比
  --plot           生成可视化图表（需要 matplotlib）
  --output_dir     图表保存目录（默认与脚本同级）
"""
 
import argparse
import os
import sys
import glob
import json
from pathlib import Path
 
import numpy as np
 
# ──────────────────────────────────────────────────────────────
# 数据加载
# ──────────────────────────────────────────────────────────────
 
def load_parquet_dataset(dataset_path: Path):
    """加载 LeRobot 标准 parquet 格式数据集，返回 actions 数组和 episode 索引列表。"""
    try:
        import pandas as pd
    except ImportError:
        sys.exit("❌ 需要安装 pandas：pip install pandas pyarrow")
 
    parquet_files = sorted(glob.glob(str(dataset_path / "data" / "**" / "*.parquet"), recursive=True))
    if not parquet_files:
        parquet_files = sorted(glob.glob(str(dataset_path / "**" / "*.parquet"), recursive=True))
    if not parquet_files:
        return None, None
 
    print(f"📂 找到 {len(parquet_files)} 个 parquet 文件")
    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
 
    # 找 action 列
    action_col = None
    for candidate in ["action", "actions", "action_values"]:
        if candidate in df.columns:
            action_col = candidate
            break
    if action_col is None:
        action_cols = [c for c in df.columns if "action" in c.lower()]
        if not action_cols:
            sys.exit(f"❌ 找不到 action 列，现有列：{list(df.columns)}")
        action_col = action_cols[0]
 
    print(f"✅ 使用 action 列：'{action_col}'")
 
    # action 可能是 list/array 存储
    sample = df[action_col].iloc[0]
    if isinstance(sample, (list, np.ndarray)):
        actions = np.stack(df[action_col].values)
    else:
        actions = df[action_col].values.reshape(-1, 1)
 
    # episode 分组
    episode_col = None
    for candidate in ["episode_index", "episode_id", "episode"]:
        if candidate in df.columns:
            episode_col = candidate
            break
 
    if episode_col:
        episode_ids = df[episode_col].values
    else:
        episode_ids = np.zeros(len(actions), dtype=int)
 
    return actions, episode_ids
 
 
def load_hdf5_dataset(dataset_path: Path):
    """加载 HDF5 格式数据集。"""
    try:
        import h5py
    except ImportError:
        sys.exit("❌ 需要安装 h5py：pip install h5py")
 
    if dataset_path.is_file():
        hdf5_files = [dataset_path]
    else:
        hdf5_files = sorted(
            glob.glob(str(dataset_path / "**" / "*.hdf5"), recursive=True) +
            glob.glob(str(dataset_path / "**" / "*.h5"), recursive=True)
        )
 
    if not hdf5_files:
        return None, None
 
    print(f"📂 找到 {len(hdf5_files)} 个 HDF5 文件")
    all_actions = []
    all_episode_ids = []
 
    for ep_idx, fpath in enumerate(hdf5_files):
        with h5py.File(fpath, "r") as f:
            # 尝试常见路径
            action_key = None
            for candidate in ["action", "actions", "data/action", "obs/action"]:
                if candidate in f:
                    action_key = candidate
                    break
            if action_key is None:
                print(f"  ⚠️  {Path(fpath).name} 中找不到 action key，跳过")
                continue
            acts = f[action_key][:]  # shape: (T, D) or (T,)
            if acts.ndim == 1:
                acts = acts.reshape(-1, 1)
            all_actions.append(acts)
            all_episode_ids.append(np.full(len(acts), ep_idx, dtype=int))
 
    if not all_actions:
        return None, None
 
    actions = np.concatenate(all_actions, axis=0)
    episode_ids = np.concatenate(all_episode_ids, axis=0)
    return actions, episode_ids
 
 
def load_dataset(dataset_path: Path):
    actions, episode_ids = load_parquet_dataset(dataset_path)
    if actions is not None:
        return actions, episode_ids, "parquet"
 
    actions, episode_ids = load_hdf5_dataset(dataset_path)
    if actions is not None:
        return actions, episode_ids, "hdf5"
 
    sys.exit(f"❌ 在 {dataset_path} 中未找到支持的数据格式（parquet / hdf5）")
 
 
# ──────────────────────────────────────────────────────────────
# 统计核心
# ──────────────────────────────────────────────────────────────
 
def compute_idle_stats(actions: np.ndarray, threshold: float):
    """
    返回全局 idle 统计。
    idle 定义：该帧所有 action 维度的绝对值均 < threshold。
    """
    is_idle = np.all(np.abs(actions) < threshold, axis=1)   # (T,)
    total = len(is_idle)
    idle_count = is_idle.sum()
    idle_ratio = idle_count / total if total > 0 else 0.0
    return is_idle, idle_count, total, idle_ratio
 
 
def compute_per_joint_idle(actions: np.ndarray, threshold: float):
    """每个关节维度单独统计 idle 占比。"""
    n_dims = actions.shape[1]
    stats = []
    for d in range(n_dims):
        mask = np.abs(actions[:, d]) < threshold
        stats.append({
            "joint": d,
            "idle_count": int(mask.sum()),
            "idle_ratio": float(mask.mean()),
            "mean_abs": float(np.mean(np.abs(actions[:, d]))),
            "max_abs":  float(np.max(np.abs(actions[:, d]))),
            "std":      float(np.std(actions[:, d])),
        })
    return stats
 
 
def compute_per_episode_idle(actions: np.ndarray, episode_ids: np.ndarray,
                              threshold: float):
    """每条 episode 统计 idle 占比。"""
    unique_eps = np.unique(episode_ids)
    stats = []
    for ep in unique_eps:
        mask = episode_ids == ep
        ep_actions = actions[mask]
        is_idle = np.all(np.abs(ep_actions) < threshold, axis=1)
        stats.append({
            "episode": int(ep),
            "total_frames": int(len(ep_actions)),
            "idle_count":   int(is_idle.sum()),
            "idle_ratio":   float(is_idle.mean()),
        })
    return stats
 
 
# ──────────────────────────────────────────────────────────────
# 诊断逻辑
# ──────────────────────────────────────────────────────────────
 
DIAGNOSIS_THRESHOLDS = {
    "critical":  0.40,   # >= 40% → 高风险，极有可能导致 freeze
    "warning":   0.20,   # >= 20% → 中风险，需要清洗
    "mild":      0.10,   # >= 10% → 轻微，建议检查
}
 
def diagnose(idle_ratio: float, per_joint_stats, per_episode_stats,
             action_threshold: float) -> str:
    lines = []
    lines.append("\n" + "═" * 60)
    lines.append("  🔍  IDLE 数据诊断报告")
    lines.append("═" * 60)
 
    # 总体判断
    pct = idle_ratio * 100
    if idle_ratio >= DIAGNOSIS_THRESHOLDS["critical"]:
        level = "🔴 高风险"
        verdict = ("idle 帧占比 **严重偏高**，这是导致模型 freeze 的主要嫌疑。\n"
                   "  模型会学到'什么都不做'是安全的默认行为，\n"
                   "  在 inference 时倾向于输出全零或近零 action，\n"
                   "  尤其在低 loss checkpoint 中更为严重（模型过拟合了 idle 模式）。")
    elif idle_ratio >= DIAGNOSIS_THRESHOLDS["warning"]:
        level = "🟡 中风险"
        verdict = ("idle 帧占比偏高，**存在数据污染风险**。\n"
                   "  建议清洗后重新训练，观察 freeze 现象是否缓解。")
    elif idle_ratio >= DIAGNOSIS_THRESHOLDS["mild"]:
        level = "🟠 轻微"
        verdict = ("idle 帧占比在可接受范围内，但仍建议检查。\n"
                   "  如果 freeze 发生在特定阶段，可能是局部 episode 的问题。")
    else:
        level = "🟢 正常"
        verdict = ("idle 帧占比较低，**数据 idle 比例不是主要问题**。\n"
                   "  freeze 现象可能来自其他原因（chunk size、归一化、\n"
                   "  温度参数、inference 时的图像预处理差异等）。")
 
    lines.append(f"\n  总体水平：{level}  ({pct:.1f}% idle @ threshold={action_threshold})")
    lines.append(f"\n  分析：\n  {verdict}")
 
    # 找出 idle 最多的 episodes
    if per_episode_stats:
        heavy = [e for e in per_episode_stats if e["idle_ratio"] >= 0.5]
        if heavy:
            lines.append(f"\n  ⚠️  有 {len(heavy)} 条 episode 的 idle 占比 ≥ 50%：")
            for e in sorted(heavy, key=lambda x: -x["idle_ratio"])[:10]:
                lines.append(f"     ep {e['episode']:>4d}: {e['idle_ratio']*100:5.1f}%  "
                              f"({e['idle_count']}/{e['total_frames']} 帧)")
 
    # 找出 idle 最多的关节
    if per_joint_stats:
        heavy_joints = [j for j in per_joint_stats if j["idle_ratio"] >= 0.5]
        if heavy_joints:
            lines.append(f"\n  ⚠️  有 {len(heavy_joints)} 个关节维度 idle 占比 ≥ 50%：")
            for j in sorted(heavy_joints, key=lambda x: -x["idle_ratio"]):
                lines.append(f"     joint {j['joint']:>2d}: {j['idle_ratio']*100:5.1f}%  "
                              f"mean_abs={j['mean_abs']:.4f}  std={j['std']:.4f}")
 
    # 建议
    lines.append("\n  📌  建议措施：")
    if idle_ratio >= DIAGNOSIS_THRESHOLDS["warning"]:
        lines.append("  1. 删除 idle 帧（动作绝对值全部 < threshold 的连续段）")
        lines.append("  2. 对连续 idle 段做截断：保留最多 N 帧（如 5~10 帧）过渡帧")
        lines.append("  3. 检查数据采集流程，是否在抓取前/后有大量等待时间未剪辑")
        lines.append("  4. 对剩余数据重新做归一化（action mean/std 会因清洗后改变）")
        lines.append("  5. 重新训练，优先对比 medium-loss checkpoint 的 inference 效果")
    else:
        lines.append("  1. 检查 inference 时的图像预处理是否与训练一致")
        lines.append("  2. 检查 chunk_size / temporal_ensemble 参数")
        lines.append("  3. 用更高 threshold（如 0.05）重新运行本脚本确认")
 
    lines.append("\n" + "═" * 60)
    return "\n".join(lines)
 
 
# ──────────────────────────────────────────────────────────────
# 可视化
# ──────────────────────────────────────────────────────────────
 
def plot_stats(actions, is_idle, episode_ids, per_joint_stats,
               per_episode_stats, output_dir: Path, threshold: float):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("⚠️  matplotlib 未安装，跳过绘图。pip install matplotlib")
        return
 
    output_dir.mkdir(parents=True, exist_ok=True)
    n_dims = actions.shape[1]
 
    # ── 图1：全局 action 分布 + idle mask ──────────────────────
    fig, axes = plt.subplots(n_dims + 1, 1,
                             figsize=(16, 2.5 * (n_dims + 1)),
                             sharex=True)
    fig.suptitle(f"Action Distribution & Idle Frames  (threshold={threshold})",
                 fontsize=13, fontweight="bold")
 
    t = np.arange(len(actions))
    colors = plt.cm.tab10.colors
 
    for d in range(n_dims):
        ax = axes[d]
        ax.plot(t, actions[:, d], lw=0.4, alpha=0.8,
                color=colors[d % len(colors)], label=f"joint {d}")
        ax.axhline( threshold, color="red",   lw=0.6, ls="--", alpha=0.5)
        ax.axhline(-threshold, color="red",   lw=0.6, ls="--", alpha=0.5)
        ax.fill_between(t, -threshold, threshold,
                        where=is_idle, color="red", alpha=0.15)
        ax.set_ylabel(f"j{d}", fontsize=8)
        ax.tick_params(labelsize=7)
 
    # idle mask 单独一行
    ax = axes[-1]
    ax.fill_between(t, 0, is_idle.astype(float), color="red", alpha=0.6)
    ax.set_ylim(-0.05, 1.2)
    ax.set_ylabel("idle", fontsize=8)
    ax.set_xlabel("frame index", fontsize=9)
    ax.tick_params(labelsize=7)
 
    plt.tight_layout()
    p1 = output_dir / "idle_action_timeline.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾  保存: {p1}")
 
    # ── 图2：per-joint idle 柱状图 ──────────────────────────────
    fig, ax = plt.subplots(figsize=(max(6, n_dims * 0.8), 4))
    joints = [s["joint"] for s in per_joint_stats]
    ratios = [s["idle_ratio"] * 100 for s in per_joint_stats]
    bar_colors = ["#e74c3c" if r >= 40 else "#f39c12" if r >= 20 else "#2ecc71"
                  for r in ratios]
    ax.bar(joints, ratios, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax.axhline(40, color="red",    ls="--", lw=1, label="40% (critical)")
    ax.axhline(20, color="orange", ls="--", lw=1, label="20% (warning)")
    ax.set_xlabel("Joint dimension", fontsize=10)
    ax.set_ylabel("Idle ratio (%)", fontsize=10)
    ax.set_title("Per-Joint Idle Ratio", fontsize=11)
    ax.set_xticks(joints)
    ax.legend(fontsize=8)
    plt.tight_layout()
    p2 = output_dir / "idle_per_joint.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾  保存: {p2}")
 
    # ── 图3：per-episode idle 分布直方图 ───────────────────────
    if len(per_episode_stats) > 1:
        ep_ratios = [e["idle_ratio"] * 100 for e in per_episode_stats]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
 
        ax1.hist(ep_ratios, bins=min(30, len(ep_ratios)),
                 color="#3498db", edgecolor="white", linewidth=0.5)
        ax1.axvline(40, color="red",    ls="--", lw=1.2, label="40%")
        ax1.axvline(20, color="orange", ls="--", lw=1.2, label="20%")
        ax1.set_xlabel("Idle ratio per episode (%)", fontsize=10)
        ax1.set_ylabel("Count", fontsize=10)
        ax1.set_title("Distribution of Per-Episode Idle Ratio", fontsize=11)
        ax1.legend(fontsize=8)
 
        ep_ids = [e["episode"] for e in per_episode_stats]
        ax2.bar(range(len(ep_ids)), ep_ratios,
                color=["#e74c3c" if r >= 40 else "#f39c12" if r >= 20 else "#2ecc71"
                       for r in ep_ratios],
                width=1.0, edgecolor="none")
        ax2.axhline(40, color="red",    ls="--", lw=1, label="40% critical")
        ax2.axhline(20, color="orange", ls="--", lw=1, label="20% warning")
        ax2.set_xlabel("Episode index", fontsize=10)
        ax2.set_ylabel("Idle ratio (%)", fontsize=10)
        ax2.set_title("Per-Episode Idle Ratio", fontsize=11)
        ax2.legend(fontsize=8)
 
        plt.tight_layout()
        p3 = output_dir / "idle_per_episode.png"
        fig.savefig(p3, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  💾  保存: {p3}")
 
    # ── 图4：action 绝对值分布（log scale）──────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    all_abs = np.abs(actions).flatten()
    ax.hist(all_abs, bins=200, color="#9b59b6", edgecolor="none", alpha=0.8,
            log=True)
    ax.axvline(threshold, color="red", ls="--", lw=1.5,
               label=f"threshold={threshold}")
    ax.set_xlabel("|action|", fontsize=10)
    ax.set_ylabel("Count (log)", fontsize=10)
    ax.set_title("Distribution of |action| values", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    p4 = output_dir / "action_abs_distribution.png"
    fig.savefig(p4, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾  保存: {p4}")
 
 
# ──────────────────────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser(
        description="分析 LeRobot 数据集中 idle 帧占比，诊断模型 freeze 风险")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="数据集根目录或单个 .hdf5 文件路径")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="idle 判定阈值（L-inf，默认 0.01）")
    parser.add_argument("--per_joint",   action="store_true",
                        help="打印每个 joint 维度的统计")
    parser.add_argument("--per_episode", action="store_true",
                        help="打印每条 episode 的统计")
    parser.add_argument("--plot", action="store_true",
                        help="生成可视化图表")
    parser.add_argument("--output_dir", type=str, default="./idle_analysis_output",
                        help="图表保存目录")
    args = parser.parse_args()
 
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        sys.exit(f"❌ 路径不存在：{dataset_path}")
 
    print(f"\n🔄  加载数据集：{dataset_path}")
    actions, episode_ids, fmt = load_dataset(dataset_path)
    print(f"✅  数据格式：{fmt}  |  总帧数：{len(actions)}  |  action 维度：{actions.shape[1]}")
 
    # ── 全局统计 ────────────────────────────────────────────────
    is_idle, idle_count, total, idle_ratio = compute_idle_stats(actions, args.threshold)
 
    print(f"\n{'─'*50}")
    print(f"  📊  全局 Idle 统计  (threshold = {args.threshold})")
    print(f"{'─'*50}")
    print(f"  总帧数       : {total:>10,}")
    print(f"  Idle 帧数    : {idle_count:>10,}  ({idle_ratio*100:.2f}%)")
    print(f"  Active 帧数  : {total-idle_count:>10,}  ({(1-idle_ratio)*100:.2f}%)")
 
    # action 绝对值统计
    abs_actions = np.abs(actions)
    print(f"\n  Action |·| 统计（所有维度合并）：")
    print(f"    均值 : {abs_actions.mean():.5f}")
    print(f"    中位 : {np.median(abs_actions):.5f}")
    print(f"    95%  : {np.percentile(abs_actions, 95):.5f}")
    print(f"    最大 : {abs_actions.max():.5f}")
    pct_below = [(np.abs(actions) < t).all(axis=1).mean() * 100
                 for t in [0.001, 0.005, 0.01, 0.02, 0.05]]
    print(f"\n  各 threshold 下的 idle 占比：")
    for t, p in zip([0.001, 0.005, 0.01, 0.02, 0.05], pct_below):
        bar = "█" * int(p / 2)
        print(f"    {t:.3f} : {p:6.2f}%  {bar}")
 
    # ── Per-joint ───────────────────────────────────────────────
    per_joint_stats = compute_per_joint_idle(actions, args.threshold)
    if args.per_joint:
        print(f"\n  🦾  Per-Joint Idle 统计：")
        print(f"  {'joint':>6}  {'idle%':>7}  {'mean|a|':>9}  {'max|a|':>9}  {'std':>9}")
        for s in per_joint_stats:
            flag = " ⚠️" if s["idle_ratio"] >= 0.4 else ""
            print(f"  {s['joint']:>6}  {s['idle_ratio']*100:>6.1f}%"
                  f"  {s['mean_abs']:>9.5f}  {s['max_abs']:>9.5f}"
                  f"  {s['std']:>9.5f}{flag}")
    else:
        print(f"\n  💡  用 --per_joint 查看每个关节的详细统计")
 
    # ── Per-episode ──────────────────────────────────────────────
    per_episode_stats = compute_per_episode_idle(actions, episode_ids, args.threshold)
    n_episodes = len(per_episode_stats)
    ep_ratios = [e["idle_ratio"] for e in per_episode_stats]
    heavy_episodes = sum(1 for r in ep_ratios if r >= 0.5)
 
    print(f"\n  📹  Episode 统计：共 {n_episodes} 条")
    print(f"    idle ≥ 50% 的 episode：{heavy_episodes} 条  ({heavy_episodes/n_episodes*100:.1f}%)")
    print(f"    episode 级 idle 中位数：{np.median(ep_ratios)*100:.1f}%")
 
    if args.per_episode:
        print(f"\n  📋  Per-Episode 详情：")
        print(f"  {'ep':>5}  {'frames':>7}  {'idle':>6}  {'ratio':>7}")
        for e in per_episode_stats:
            flag = " 🔴" if e["idle_ratio"] >= 0.5 else " 🟡" if e["idle_ratio"] >= 0.2 else ""
            print(f"  {e['episode']:>5}  {e['total_frames']:>7}  "
                  f"{e['idle_count']:>6}  {e['idle_ratio']*100:>6.1f}%{flag}")
    else:
        print(f"  💡  用 --per_episode 查看每条 episode 的详细统计")
 
    # ── 诊断 ────────────────────────────────────────────────────
    print(diagnose(idle_ratio, per_joint_stats, per_episode_stats, args.threshold))
 
    # ── 保存 JSON 结果 ──────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "dataset_path": str(dataset_path),
        "threshold": args.threshold,
        "total_frames": int(total),
        "idle_frames": int(idle_count),
        "idle_ratio": float(idle_ratio),
        "action_dims": int(actions.shape[1]),
        "n_episodes": n_episodes,
        "heavy_episodes_over_50pct": int(heavy_episodes),
        "per_joint": per_joint_stats,
        "per_episode": per_episode_stats,
    }
    json_path = output_dir / "idle_analysis_result.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  💾  JSON 结果已保存：{json_path}")
 
    # ── 可视化 ──────────────────────────────────────────────────
    if args.plot:
        print(f"\n  📈  生成图表...")
        plot_stats(actions, is_idle, episode_ids,
                   per_joint_stats, per_episode_stats,
                   output_dir, args.threshold)
 
    print(f"\n✅  分析完成！\n")
 
 
if __name__ == "__main__":
    main()
