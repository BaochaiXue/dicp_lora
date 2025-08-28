#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# gridworld/plot_best_streams.py
"""
Compare mean return of ALL streams vs. TOP-p% best streams (by a ranking criterion).
Saves a figure for each selected env instance.

Usage (example, from gridworld/):
  python gridworld/plot_best_streams.py \
    -ec cfg/env/darkroom.yaml \
    -ac cfg/alg/ppo_dr.yaml \
    -t datasets_dr_noise50 \
    -o plots \
    --env-index train \
    --max-envs 1 \
    --n-stream 100 \
    --source-timesteps -1 \
    --top-percent 50 \
    --rank-by mean \
    --seed 0
"""

import os, argparse, random
import numpy as np
import h5py
import matplotlib.pyplot as plt

from utils import get_config, get_traj_file_name  # provided in repo


# ---------- helpers: align T to horizon, returns computation ----------


def compute_use_T_like_dataset(T, horizon, source_timesteps):
    """
    Dataset-style behavior:
      - If source_timesteps < 0: fallback to the largest multiple of horizon not exceeding T.
      - Else: use min(source_timesteps, T); must be divisible by horizon.
    """
    if source_timesteps is None:
        raise ValueError("source_timesteps 不能为空；请显式设置。")
    if source_timesteps < 0:
        use_T = (T // horizon) * horizon
        if use_T <= 0:
            raise ValueError(
                f"T={T} 与 horizon={horizon} 不匹配，无法形成一个完整 episode。"
            )
        print(f"[INFO] source_timesteps<0: 回退 use_T={use_T}（与 horizon 对齐）")
        return use_T
    use_T = min(int(source_timesteps), T)
    if use_T % horizon != 0:
        raise ValueError(
            f"dataset 风格要求 source_timesteps 可被 horizon 整除；"
            f"当前 source_timesteps={source_timesteps}, horizon={horizon}."
        )
    return use_T


def returns_first_n_streams_dataset_style(
    rewards_TS, horizon, source_timesteps, n_stream
):
    """
    Parameters
    ----------
    rewards_TS : np.ndarray, shape (T, S)
    horizon : int
    source_timesteps : int
    n_stream : int

    Returns
    -------
    returns_SE : np.ndarray, shape (S0, E)
      per-stream per-episode returns
    E : int, number of episodes that fit
    use_T : int, number of timesteps used
    S0 : int, number of streams used
    """
    T, S = rewards_TS.shape
    S0 = min(int(n_stream), S)
    use_T = compute_use_T_like_dataset(T, horizon, source_timesteps)
    E = use_T // horizon
    # reshape to (E, horizon, S0) then sum over horizon → (E, S0) → transpose to (S0, E)
    ret_SE = rewards_TS[:use_T, :S0].reshape(E, horizon, S0).sum(axis=1).T
    return ret_SE, E, use_T, S0


# ---------- env 选择：与数据集一致（先打乱再切分） ----------


def resolve_env_indices_dataset_style(cfg, which):
    if which in ("train", "test", "all"):
        if cfg["env"] == "darkroom":
            n_total = cfg["grid_size"] ** 2
        elif cfg["env"] == "darkroompermuted":
            n_total = 120
        elif cfg["env"] == "darkkeytodoor":
            # 按你仓库 dataset 绘图脚本的用法（避免超大），可适当缩减；这里保持与之前一致
            n_total = round((cfg["grid_size"] ** 4) / 4)
        else:
            raise ValueError("Unsupported env")

        idx = list(range(n_total))
        rng = random.Random(int(cfg.get("env_split_seed", 0)))
        rng.shuffle(idx)

        n_train = round(n_total * cfg["train_env_ratio"])
        if which == "train":
            return idx[:n_train]
        elif which == "test":
            return idx[n_train:]
        else:
            return idx
    # Comma list of indices
    return [int(x) for x in which.split(",") if x != ""]


# ---------- 工作目录自适应：为 include 相对路径兜底 ----------


def _maybe_chdir_for_includes(env_cfg_abs, alg_cfg_abs):
    candidates = []
    for p in (env_cfg_abs, alg_cfg_abs):
        if p:
            d = os.path.abspath(os.path.join(os.path.dirname(p), "..", ".."))
            candidates.append(d)
    for root in candidates:
        if os.path.isdir(os.path.join(root, "cfg")) and os.path.isfile(
            os.path.join(root, "utils.py")
        ):
            try:
                os.chdir(root)
                print(f"[INFO] 工作目录已切换到 {root} 以适配 include 相对路径。")
            except Exception as e:
                print(f"[WARN] 切换工作目录到 {root} 失败：{e}")
            break


# ---------- 根据数据集目录名生成子目录标签 ----------


def compute_dataset_tag(traj_dir_abs, cfg):
    tag = os.path.basename(os.path.normpath(traj_dir_abs))
    if tag.lower() in {"", ".", "datasets", "dataset", "data", "trajectories", "traj"}:
        tag = get_traj_file_name(cfg)
    return tag


def sanitize_tag(tag: str) -> str:
    safe_chars = []
    for c in tag:
        if c.isalnum() or c in "._-":
            safe_chars.append(c)
        else:
            safe_chars.append("_")
    return "".join(safe_chars)


# ---------- 排名与绘图 ----------


def rank_streams(returns_SE: np.ndarray, criterion: str):
    """
    returns_SE: (S, E)
    criterion: 'mean' | 'last' | 'max' | 'auc'
    returns: ndarray of shape (S,), higher is better
    """
    if criterion == "mean":
        score = returns_SE.mean(axis=1)
    elif criterion == "last":
        score = returns_SE[:, -1]
    elif criterion == "max":
        score = returns_SE.max(axis=1)
    elif criterion == "auc":
        # sum across episodes (equiv. to mean*E)
        score = returns_SE.sum(axis=1)
    else:
        raise ValueError(f"Unknown rank criterion: {criterion}")
    return score


def plot_best_vs_all(returns_SE, top_idx, out_path, title):
    """
    Plot mean curves of ALL vs TOP
    """
    all_mean = returns_SE.mean(axis=0)
    top_mean = (
        returns_SE[top_idx].mean(axis=0)
        if len(top_idx) > 0
        else np.zeros_like(all_mean)
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(all_mean, label=f"ALL (n={returns_SE.shape[0]})", linewidth=2)
    ax.plot(top_mean, label=f"TOP (n={len(top_idx)})", linewidth=2)

    ax.set_xlabel("episode")
    ax.set_ylabel("return")
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


# ---------- 主流程 ----------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--env-config",
        "-ec",
        required=True,
        help="eg. cfg/env/darkroom.yaml 或 gridworld/cfg/env/darkroom.yaml",
    )
    ap.add_argument(
        "--alg-config",
        "-ac",
        required=True,
        help="eg. cfg/alg/ppo_dr.yaml 或 gridworld/cfg/alg/ppo_dr.yaml",
    )
    ap.add_argument("--traj-dir", "-t", default="./datasets", help="目录，内含 *.hdf5")
    ap.add_argument("--out", "-o", default="./plots", help="输出图的根目录")
    ap.add_argument(
        "--env-index",
        "-g",
        default="train",
        help='绘制的 env 组索引："0" 或 "0,1,2" 或 "train"/"test"/"all"（与数据集一致）',
    )
    ap.add_argument(
        "--max-envs",
        type=int,
        default=1,
        help="最多绘制多少个 env（从 --env-index 里取前K；0 表示不限制）",
    )
    ap.add_argument(
        "--n-stream",
        type=int,
        default=100,
        help="每个 env 参与统计的 stream 数（只看前 n_stream 条）",
    )
    ap.add_argument(
        "--source-timesteps",
        type=int,
        default=-1,
        help="用于统计的时间步数；<0 将回退到不超过 T 的最大整倍数（与 horizon 对齐）",
    )
    ap.add_argument(
        "--top-percent", type=float, default=50.0, help="选择前 p%% 的最佳流，默认 50%%"
    )
    ap.add_argument(
        "--rank-by",
        choices=["mean", "last", "max", "auc"],
        default="mean",
        help="最优流的排序指标，默认 mean",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="用于 env 索引随机性的种子；默认使用 config['env_split_seed']",
    )
    args = ap.parse_args()

    # 绝对路径 & 工作目录自适应
    env_cfg_abs = os.path.abspath(args.env_config)
    alg_cfg_abs = os.path.abspath(args.alg_config)
    traj_dir_abs = os.path.abspath(args.traj_dir)
    out_root_abs = os.path.abspath(args.out)

    _maybe_chdir_for_includes(env_cfg_abs, alg_cfg_abs)

    # 读取配置
    cfg = get_config(env_cfg_abs)
    cfg.update(get_config(alg_cfg_abs))

    # 随机种子（仅用于 env 切分的一致性）
    seed_to_use = (
        args.seed if args.seed is not None else int(cfg.get("env_split_seed", 0))
    )
    random.seed(seed_to_use)
    np.random.seed(seed_to_use % (2**32 - 1))

    # 路径与输出目录
    h5_path = os.path.join(traj_dir_abs, get_traj_file_name(cfg) + ".hdf5")
    dataset_tag = compute_dataset_tag(traj_dir_abs, cfg)
    dataset_tag_safe = sanitize_tag(dataset_tag)
    out_dir_by_dataset = os.path.join(out_root_abs, dataset_tag)
    os.makedirs(out_dir_by_dataset, exist_ok=True)
    print(
        f"[INFO] 输出目录：{out_dir_by_dataset}  （dataset_tag='{dataset_tag}', filename_tag='{dataset_tag_safe}'）"
    )

    # env indices
    env_indices = resolve_env_indices_dataset_style(cfg, args.env_index)
    if args.max_envs and args.max_envs > 0:
        env_indices = env_indices[: args.max_envs]
    print(f"[INFO] 将绘制的 env 实例索引：{env_indices}")

    # 安全检查
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"未找到 HDF5 文件：{h5_path}")

    top_percent = float(args.top_percent)
    if not (0 < top_percent <= 100):
        raise ValueError(f"--top-percent 必须在 (0, 100]，收到 {top_percent}")

    with h5py.File(h5_path, "r") as f:
        for i in env_indices:
            # 读取 rewards(T,S) 并计算 returns(S,E)
            rewards_TS = f[str(i)]["rewards"][()]  # (T, S)
            T, S = rewards_TS.shape
            horizon = cfg["horizon"]

            returns_SE, E, use_T, S0 = returns_first_n_streams_dataset_style(
                rewards_TS, horizon, args.source_timesteps, args.n_stream
            )
            # 排名与选择 top-k
            score = rank_streams(returns_SE, args.rank_by)  # shape (S0,)
            order_desc = np.argsort(-score)  # 高分在前
            k = max(1, int(np.ceil(S0 * top_percent / 100.0)))
            top_idx = order_desc[:k]

            print(
                f"[env {i}] S0={S0}, E={E}, use_T={use_T}, "
                f"rank_by={args.rank_by}, top_percent={top_percent:.1f}%, top_k={k}"
            )
            print(f"Top-{k} indices (by {args.rank_by}): {top_idx.tolist()}")

            # 画图
            title = (
                f"Env {i} — {cfg['env']} — Mean return: "
                f"ALL (n={S0}) vs TOP-{k} ({top_percent:.1f}%, by {args.rank_by})"
            )
            out_fig = os.path.join(
                out_dir_by_dataset,
                f"best_vs_all_{dataset_tag_safe}_env{i}_top{int(round(top_percent))}_{args.rank_by}.png",
            )
            plot_best_vs_all(returns_SE, top_idx, out_fig, title)
            print(f"[OK] 保存：{out_fig}")


if __name__ == "__main__":
    main()
