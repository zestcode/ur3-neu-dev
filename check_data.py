"""
check_data.py — diagnose a tcp_record / lerobot_ur3_record dataset.

Three checks per episode:

  1. Action axis usage — per-channel mean/std/quantiles. Flags channels that
     are stuck at zero (e.g. action[5] vs action[3] mapping issue).

  2. Translation sync — scatter Δstate.xyz against dt·action.xyz. Should fall
     on y = x if the recorder pairs state[k] with action[k] correctly. A
     scatter that's smeared, biased, or has wrong slope means there's a
     desync bug between read_observation and get_spacemouse_action.

  3. Rotation sync — compare the *magnitude* of the actual frame-to-frame
     rotation (computed correctly via rotation matrices, not by subtracting
     axis-angle vectors) to dt · |action[3:6]|. Component-by-component
     comparisons of axis-angle would be misleading because that
     representation is nonlinear.

Output is either interactive matplotlib windows (default) or PNG files in a
directory if --save is passed.

Usage:
    python3 check_data.py path/to/recording_dir
    python3 check_data.py path/to/recording_dir --episode 5 --fps 25
    python3 check_data.py path/to/episode_000000.parquet
    python3 check_data.py path/to/recording_dir --save plots/
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Channels that are essentially constant produce harmless polyfit/correlation
# warnings. We surface those degenerate channels in the printed text already,
# so silencing the warnings keeps the output readable.
_RankWarning = getattr(np, "RankWarning", None) or getattr(
    getattr(np, "exceptions", None), "RankWarning", UserWarning
)
warnings.filterwarnings("ignore", category=_RankWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message="invalid value encountered in divide")

# Matplotlib is optional. The text statistics are the primary diagnostic;
# plots only run when --plot or --save is passed and matplotlib is importable.
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception as _e:
    plt = None
    _HAS_MPL = False
    _MPL_ERR = str(_e)


# ---------------------------------------------------------------------------
# Pure-numpy rotation utilities (no scipy dependency)
# ---------------------------------------------------------------------------

def _skew(v):
    """v: (..., 3) -> (..., 3, 3) skew-symmetric matrices."""
    out = np.zeros(v.shape[:-1] + (3, 3), dtype=v.dtype)
    out[..., 0, 1] = -v[..., 2]
    out[..., 0, 2] =  v[..., 1]
    out[..., 1, 0] =  v[..., 2]
    out[..., 1, 2] = -v[..., 0]
    out[..., 2, 0] = -v[..., 1]
    out[..., 2, 1] =  v[..., 0]
    return out


def axis_angle_to_rotmat(aa):
    """aa: (..., 3) -> (..., 3, 3). Rodrigues' formula."""
    aa = np.asarray(aa, dtype=np.float64)
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    theta_safe = np.where(theta > 1e-8, theta, 1.0)
    axis = aa / theta_safe
    K = _skew(axis)
    I = np.eye(3, dtype=aa.dtype)
    sin_t = np.sin(theta)[..., None]
    cos_t = np.cos(theta)[..., None]
    return I + sin_t * K + (1.0 - cos_t) * (K @ K)


def rotation_angle_between(aa_a, aa_b):
    """Magnitude of the rotation that takes aa_a to aa_b. Returns (..., ) angles."""
    Ra = axis_angle_to_rotmat(aa_a)
    Rb = axis_angle_to_rotmat(aa_b)
    Rdiff = Rb @ np.swapaxes(Ra, -1, -2)
    trace = np.trace(Rdiff, axis1=-2, axis2=-1)
    cos_t = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cos_t)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_episode(parquet_path: Path):
    df = pd.read_parquet(parquet_path)

    # Auto-detect schema:
    #  - LIBERO style (tcp_record.py):    "state" + "actions"
    #  - LeRobot style (lerobot_ur3...):  "observation.state" + "action"
    if "observation.state" in df.columns:
        state_col, action_col = "observation.state", "action"
    elif "state" in df.columns:
        state_col, action_col = "state", "actions"
    else:
        raise ValueError(
            f"Unknown schema in {parquet_path}: columns = {df.columns.tolist()}"
        )

    state = np.stack(df[state_col].to_numpy())   # (T, state_dim)
    action = np.stack(df[action_col].to_numpy()) # (T, action_dim)
    return state, action, df, (state_col, action_col)


def find_parquet(path: Path, episode: int) -> Path:
    if path.is_file():
        return path
    p = path / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
    if not p.exists():
        raise FileNotFoundError(p)
    return p


# ---------------------------------------------------------------------------
# Check 1 — action axis usage
# ---------------------------------------------------------------------------

def check_action_axis_usage(action: np.ndarray) -> None:
    print("=" * 64)
    print("1) Action axis usage")
    print("=" * 64)

    n = action.shape[1]
    names = ["Δx", "Δy", "Δz", "Δrx", "Δry", "Δrz"]
    if n >= 7:
        names.append("gripper")

    print(f"{'axis':<10} {'mean':>11} {'std':>11} {'q01':>11} {'q99':>11}")
    print("-" * 64)
    for i in range(min(n, len(names))):
        c = action[:, i]
        print(f"{names[i]:<10} {c.mean():>11.5f} {c.std():>11.5f} "
              f"{np.quantile(c, 0.01):>11.5f} {np.quantile(c, 0.99):>11.5f}")

    if n >= 6:
        rx_std = action[:, 3].std()
        rz_std = action[:, 5].std()
        print()
        print(f"  rotation channels — Δrx std = {rx_std:.5f},  Δrz std = {rz_std:.5f}")
        if rx_std > 1e-3 and rz_std < rx_std / 10:
            print("  >>> Δrz is dead while Δrx is active.")
            print("      Likely: SpaceMouse 'twist' maps to robot Δrx, not Δrz.")
            print("      Fix:   change _TX_ZUP_SPNAV in the recorder OR teach")
            print("             operators to lean the puck for yaw.")
        elif rx_std < 1e-3 and rz_std < 1e-3:
            print("  >>> Both rotation channels are dead. Operators didn't")
            print("      rotate the puck during teleop.")
        else:
            print("  >>> Rotation axis usage looks fine.")
    print()


# ---------------------------------------------------------------------------
# Check 2 — translation sync
# ---------------------------------------------------------------------------

def check_translation_sync(state: np.ndarray, action: np.ndarray, fps: int,
                           title: str, save_to: Path | None, do_plot: bool):
    dt = 1.0 / fps
    dpos = np.diff(state[:, :3], axis=0)
    cmd  = dt * action[:-1, :3]

    print("=" * 64)
    print("2) Translation sync (Δstate.xyz vs dt · action.xyz)")
    print("=" * 64)

    # Always print stats (no matplotlib needed)
    for i, name in enumerate("xyz"):
        x, y = cmd[:, i], dpos[:, i]
        if x.std() > 1e-9:
            corr = float(np.corrcoef(x, y)[0, 1])
            slope = float(np.polyfit(x, y, 1)[0])
            verdict = "OK" if (corr > 0.9 and 0.5 < slope < 1.5) else "SUSPECT"
            print(f"  {name}: corr={corr:.4f}  slope={slope:.4f}  → {verdict}")
        else:
            print(f"  {name}: action variance ≈ 0")

    if not do_plot:
        print()
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for i, name in enumerate("xyz"):
        x, y = cmd[:, i], dpos[:, i]
        ax = axes[i]
        ax.scatter(x, y, s=2, alpha=0.3)
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        if lo < hi:
            ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="y = x")
        if x.std() > 1e-9:
            corr = float(np.corrcoef(x, y)[0, 1])
            slope = float(np.polyfit(x, y, 1)[0])
            ax.set_title(f"{name}: corr={corr:.3f}, slope={slope:.3f}")
        else:
            ax.set_title(f"{name}: action const")
        ax.set_xlabel("dt · action.vel")
        ax.set_ylabel("Δstate.pos")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(f"{title} — Translation sync (fps={fps})")
    fig.tight_layout()

    if save_to is not None:
        fig.savefig(save_to, dpi=110)
        print(f"  [saved] {save_to}")
    print()
    return fig


# ---------------------------------------------------------------------------
# Check 3 — rotation sync (uses rotation matrices to avoid axis-angle traps)
# ---------------------------------------------------------------------------

def check_rotation_sync(state: np.ndarray, action: np.ndarray, fps: int,
                        title: str, save_to: Path | None, do_plot: bool):
    dt = 1.0 / fps

    aa = state[:, 3:6]
    drot_angle = rotation_angle_between(aa[:-1], aa[1:])     # (T-1,) actual
    cmd_angmag = np.linalg.norm(action[:-1, 3:6], axis=1) * dt  # expected

    print("=" * 64)
    print("3) Rotation sync (|Δrot| actual vs dt · |ω_cmd|)")
    print("=" * 64)

    # Stats always
    if cmd_angmag.std() > 1e-9:
        corr = float(np.corrcoef(cmd_angmag, drot_angle)[0, 1])
        slope = float(np.polyfit(cmd_angmag, drot_angle, 1)[0])
        verdict = "OK" if (corr > 0.85 and 0.4 < slope < 1.6) else "SUSPECT"
        print(f"  |·|: corr={corr:.4f}  slope={slope:.4f}  → {verdict}")
    else:
        print("  |·|: angular command variance ≈ 0")

    if not do_plot:
        print()
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # (a) magnitude scatter
    ax = axes[0]
    ax.scatter(cmd_angmag, drot_angle, s=2, alpha=0.3)
    lo = 0.0
    hi = float(max(cmd_angmag.max(), drot_angle.max()))
    if hi > 0:
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="y = x")
    if cmd_angmag.std() > 1e-9:
        corr = float(np.corrcoef(cmd_angmag, drot_angle)[0, 1])
        slope = float(np.polyfit(cmd_angmag, drot_angle, 1)[0])
        ax.set_title(f"|Δrot| vs dt·|ω_cmd|: corr={corr:.3f}, slope={slope:.3f}")
    else:
        ax.set_title("|Δrot| vs dt·|ω_cmd|: cmd const")
    ax.set_xlabel("dt · |action[3:6]|")
    ax.set_ylabel("|actual rotation Δ| (rad)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (b) time-series
    ax = axes[1]
    t = np.arange(len(drot_angle)) * dt
    ax.plot(t, drot_angle, label="|Δrot| actual", alpha=0.7)
    ax.plot(t, cmd_angmag, label="dt · |ω_cmd|", alpha=0.7)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("rotation magnitude (rad)")
    ax.set_title("Rotation magnitude over time")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(f"{title} — Rotation sync (fps={fps})")
    fig.tight_layout()

    if save_to is not None:
        fig.savefig(save_to, dpi=110)
        print(f"  [saved] {save_to}")
    print()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("path", type=Path,
                    help="Episode .parquet file OR a recording directory.")
    ap.add_argument("--episode", type=int, default=0,
                    help="Episode index when path is a directory (default 0).")
    ap.add_argument("--fps", type=int, default=25,
                    help="Recording fps for converting velocities to displacements.")
    ap.add_argument("--save", type=Path, default=None,
                    help="If set, save plots to this directory instead of opening windows.")
    ap.add_argument("--plot", action="store_true",
                    help="Open interactive plots (requires matplotlib).")
    args = ap.parse_args()

    do_plot = bool(args.plot or args.save is not None)
    if do_plot and not _HAS_MPL:
        print(f"[warn] --plot/--save requested but matplotlib is unavailable: {_MPL_ERR}",
              file=sys.stderr)
        print("       Falling back to text-only output. "
              "Install matplotlib to enable plots.",
              file=sys.stderr)
        do_plot = False

    try:
        parquet_path = find_parquet(args.path, args.episode)
    except FileNotFoundError as e:
        print(f"[error] missing parquet: {e}", file=sys.stderr)
        return 1

    print(f"[loading] {parquet_path}")
    state, action, df, (sc, ac) = load_episode(parquet_path)
    print(f"[loaded ] {len(state)} frames | "
          f"state[{sc}] dim={state.shape[1]} | action[{ac}] dim={action.shape[1]}")
    if state.shape[1] != 8:
        print(f"[warn   ] expected 8-dim state (3 pos + 3 axis-angle + 2 gripper). "
              f"Got {state.shape[1]}; translation/rotation checks may be off.")
    print()

    save_dir = args.save
    if save_dir is not None and do_plot:
        save_dir.mkdir(parents=True, exist_ok=True)

    title = parquet_path.stem
    check_action_axis_usage(action)
    check_translation_sync(
        state, action, args.fps, title,
        save_to=(save_dir / f"trans_{title}.png") if (do_plot and save_dir) else None,
        do_plot=do_plot,
    )
    check_rotation_sync(
        state, action, args.fps, title,
        save_to=(save_dir / f"rot_{title}.png") if (do_plot and save_dir) else None,
        do_plot=do_plot,
    )

    if do_plot and save_dir is None:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
