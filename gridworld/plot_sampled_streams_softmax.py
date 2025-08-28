#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Softmax 版本的流采样可视化（与数据集实现完全一致）

- metric: stability + improvement
  stability = 1 + mean(negative diffs) / (max_reward - min(return_stream))
  improvement = (mean(return_stream) + (max-min))/2/max_reward
- 概率: 对按 metric 升序排列后的 metric 做 softmax(metric/T)，再线性缩放到 [0,1]
- 采样: 阈值扫描（从高到低），直到凑满 n_stream，允许重复
- 随机源: 使用 random.uniform，与数据集一致（并用 env_split_seed 初始化）

输出4张图：All/Selected/Not-selected 的网格，以及三组均值曲线对比
"""
import os, argparse, random
import numpy as np
import h5py
import matplotlib.pyplot as plt

from utils import get_config, get_traj_file_name  # 同目录 utils.py


# ---------- 与数据集风格一致的 returns/采样 ----------

def compute_use_T_like_dataset(T, horizon, source_timesteps):
    """
    数据集里直接传入 source_timesteps 并 reshape(E, horizon)，因此要求可被 horizon 整除。
    这里保持一致：如果 <0 则回退到 <=T 的最大整倍数（给可视化兜底）；否则必须整除。
    """
    if source_timesteps is None:
        raise ValueError("source_timesteps 不能为空；请显式设置。")
    if source_timesteps < 0:
        use_T = (T // horizon) * horizon
        print(
            f"[WARN] dataset 风格通常要求 source_timesteps>0 且可被 horizon 整除；"
            f"收到 {source_timesteps}，这里退回到 use_T={use_T}（与 horizon 对齐）。"
        )
        return use_T
    use_T = min(int(source_timesteps), T)
    if use_T % horizon != 0:
        raise ValueError(
            f"dataset 风格要求 source_timesteps 可被 horizon 整除；"
            f"当前 source_timesteps={source_timesteps}, horizon={horizon}."
        )
    return use_T


def returns_first_n_streams_dataset_style(rewards_TS, horizon, source_timesteps, n_stream):
    """
    按数据集逻辑：只看前 n_stream 条流，取前 use_T 个时间步，按 (E, horizon) reshape 后逐 episode 求和。
    返回：
      returns_SE: 形状 (S0, E)，S0=min(n_stream, S)
      E: episode 数
      use_T: 实际使用的时间步
      S0: 实际使用的流数
    """
    T, S = rewards_TS.shape
    S0 = min(int(n_stream), S)
    use_T = compute_use_T_like_dataset(T, horizon, source_timesteps)
    E = use_T // horizon
    ret_SE = rewards_TS[:use_T, :S0].reshape(E, horizon, S0).sum(axis=1).T  # (S0, E)
    return ret_SE, E, use_T, S0


def select_streams_softmax_dataset_style(
    returns_SE,
    max_reward,
    n_select,
    stability_coeff=1.0,
    temperature=0.125,   # 与数据集默认一致
):
    """
    完全对齐你的数据集实现：
    1) 升序排序 metric
    2) softmax(metric/T) -> 归一化 -> 线性缩放到 [0,1]
    3) 从高到低做阈值扫描（允许重复），直到抽满 n_select
    返回：sel_idx(含重复)、metric、probs_sorted(升序)、sorted_index_lst(升序索引)
    """
    S, E = returns_SE.shape
    metric_lst = []

    for j in range(S):
        r = returns_SE[j]                   # (E,)
        diff = np.diff(r)
        neg = diff[diff < 0]               # 严格小于0（与你的实现一致）

        denom = (max_reward - float(np.min(r)))  # 与数据集一致的分母
        # 保持原实现语义，不做额外数值防护
        stability = 1.0 + (np.mean(neg) if neg.size > 0 else 0.0) / denom
        improvement = (float(np.mean(r)) + (float(np.max(r)) - float(np.min(r)))) / 2.0 / max_reward

        metric = stability_coeff * stability + 1.0 * improvement
        metric_lst.append(float(metric))

    # 升序排序
    sorted_index_lst = sorted(range(S), key=lambda i: metric_lst[i])
    metric_array_sorted = np.array([metric_lst[i] for i in sorted_index_lst], dtype=float)

    # softmax(metric/T) 后线性缩放到 [0,1]
    softmax_vals = np.exp(metric_array_sorted / float(temperature))
    softmax_vals /= np.sum(softmax_vals)
    new_softmax_vals = softmax_vals - np.min(softmax_vals)
    if np.max(new_softmax_vals) > 0:
        probabilities_lst = new_softmax_vals / np.max(new_softmax_vals)
    else:
        probabilities_lst = np.zeros_like(new_softmax_vals)

    # 打印与数据集一致
    print(f"Scaled softmax probability (sorted from small to large) = {probabilities_lst}")

    # 阈值扫描（从高到低），允许重复，直到凑满 n_select
    temp_slice = []
    S_scan = min(S, n_select)  # 与原始结构保持一致
    while len(temp_slice) < n_select:
        for potato in reversed(range(S_scan)):  # 从高到低扫描
            selected_stream = sorted_index_lst[potato]
            selected_stream_prob = probabilities_lst[potato]
            if random.uniform(0, 1) <= selected_stream_prob:
                temp_slice.append(selected_stream)
            if len(temp_slice) >= n_select:
                break

    sel_idx = np.array(temp_slice[:n_select], dtype=np.int32)
    probs_sorted = probabilities_lst  # 升序概率
    return sel_idx, np.array(metric_lst, dtype=float), probs_sorted, sorted_index_lst


# ---------- 绘图 ----------

def plot_grid(returns_SE, selected_idx, out_path, title, cols=10):
    if len(selected_idx) == 0:
        print("[WARN] 没有可绘制的 stream，输出占位图：", title)
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.axis("off")
        ax.text(0.5, 0.5, "No streams in this group", ha="center", va="center")
        fig.suptitle(title, fontsize=12)
        fig.tight_layout()
        fig.savefig(out_path, dpi=240)
        plt.close(fig)
        return

    selR = returns_SE[selected_idx]
    rows = int(np.ceil(len(selected_idx) / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 2.1, rows * 1.7), sharex=True, sharey=True
    )
    axes = np.ravel(axes)
    for k, ax in enumerate(axes):
        if k < len(selected_idx):
            ax.plot(selR[k])
            ax.set_title(f"s{selected_idx[k]}", fontsize=8)
        else:
            ax.axis("off")
        ax.tick_params(labelsize=6)
    fig.suptitle(title, fontsize=12)
    fig.text(0.5, 0.02, "episode", ha="center")
    fig.text(0.02, 0.5, "return", va="center", rotation="vertical")
    fig.tight_layout(rect=[0.03, 0.04, 1, 0.96])
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_mean_three_groups(returns_SE, selected_idx, out_path, title):
    all_mean = returns_SE.mean(axis=0)
    S0 = returns_SE.shape[0]
    sel_unique = np.unique(selected_idx) if len(selected_idx) > 0 else np.array([], dtype=int)
    not_sel = np.setdiff1d(np.arange(S0), sel_unique, assume_unique=True)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(all_mean, label=f"all candidates (n={S0})", linewidth=2)

    if sel_unique.size > 0:
        ax.plot(
            returns_SE[sel_unique].mean(axis=0),
            label=f"selected (n={sel_unique.size})",
            linewidth=2,
        )
    else:
        ax.plot(np.zeros_like(all_mean), label="selected (n=0)", linewidth=2)

    if not_sel.size > 0:
        ax.plot(
            returns_SE[not_sel].mean(axis=0),
            label=f"not selected (n={not_sel.size})",
            linewidth=2,
            linestyle="--",
        )
    else:
        ax.plot(
            np.zeros_like(all_mean),
            label="not selected (n=0)",
            linewidth=2,
            linestyle="--",
        )

    ax.set_xlabel("episode")
    ax.set_ylabel("return")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


# ---------- env 选择：与数据集一致（先打乱再切分） ----------

def resolve_env_indices_dataset_style(cfg, which):
    if which in ("train", "test", "all"):
        if cfg["env"] == "darkroom":
            n_total = cfg["grid_size"] ** 2
        elif cfg["env"] == "darkroompermuted":
            n_total = 120
        elif cfg["env"] == "darkkeytodoor":
            n_total = round((cfg["grid_size"] ** 4) / 4)  # 若有自定义，可按需调整；darkroom 用不到
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


# ---------- 主流程 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-config", "-ec", required=True,
                    help="eg. cfg/env/darkroom.yaml 或 gridworld/cfg/env/darkroom.yaml")
    ap.add_argument("--alg-config", "-ac", required=True,
                    help="eg. cfg/alg/ppo_dr.yaml 或 gridworld/cfg/alg/ppo_dr.yaml")
    ap.add_argument("--traj-dir", "-t", default="./datasets")
    ap.add_argument("--out", "-o", default="./plots")
    ap.add_argument("--env-index", "-g", default="train",
                    help='绘制的 env 组索引："0" 或 "0,1,2" 或 "train"/"test"/"all"（与数据集一致）')
    ap.add_argument("--max-envs", type=int, default=1,
                    help="最多绘制多少个 env（在 --env-index 选出的列表里取前K个；0 表示不限制）。默认 1。")
    ap.add_argument("--n-stream", type=int, default=100,
                    help="每个 env 参与候选与采样的 stream 数（只看前 n_stream 条）")
    ap.add_argument("--source-timesteps", type=int, default=-1,
                    help="用于统计的时间步数；数据集风格要求 >0 且能被 horizon 整除；<0 将回退到不超过 T 的最大整倍数")
    ap.add_argument("--stability-coeff", type=float, default=1.0,
                    help="与数据集一致，默认 1.0")
    ap.add_argument("--temperature", type=float, default=0.125,
                    help="Softmax 温度（与数据集默认一致为 0.125）")
    ap.add_argument("--seed", type=int, default=None,
                    help="随机种子；默认使用 config['env_split_seed']，以与数据集实现复现一致。")
    args = ap.parse_args()

    # 绝对路径
    env_cfg_abs = os.path.abspath(args.env_config)
    alg_cfg_abs = os.path.abspath(args.alg_config)
    traj_dir_abs = os.path.abspath(args.traj_dir)
    out_root_abs = os.path.abspath(args.out)

    # 适配 include 相对路径
    _maybe_chdir_for_includes(env_cfg_abs, alg_cfg_abs)

    # 读取配置
    cfg = get_config(env_cfg_abs)
    cfg.update(get_config(alg_cfg_abs))

    # 随机性（与数据集一致：使用 random 作为主要随机源）
    seed_to_use = args.seed if args.seed is not None else int(cfg.get("env_split_seed", 0))
    random.seed(seed_to_use)
    np.random.seed(seed_to_use % (2**32 - 1))

    # 数据 & 输出路径
    h5_path = os.path.join(traj_dir_abs, get_traj_file_name(cfg) + ".hdf5")
    dataset_tag = compute_dataset_tag(traj_dir_abs, cfg)
    dataset_tag_safe = sanitize_tag(dataset_tag)
    out_dir_by_dataset = os.path.join(out_root_abs, dataset_tag)
    os.makedirs(out_dir_by_dataset, exist_ok=True)
    print(f"[INFO] 输出目录：{out_dir_by_dataset}  （dataset_tag='{dataset_tag}', filename_tag='{dataset_tag_safe}'）")

    # env 选择（与数据集一致的打乱&切分方式）
    env_indices = resolve_env_indices_dataset_style(cfg, args.env_index)
    if args.max_envs and args.max_envs > 0:
        env_indices = env_indices[: args.max_envs]
    print(f"[INFO] 将绘制的 env 实例索引：{env_indices}")

    with h5py.File(h5_path, "r") as f:
        for i in env_indices:
            rewards_TS = f[str(i)]["rewards"][()]  # (T, S)
            T, S = rewards_TS.shape
            horizon = cfg["horizon"]

            returns_SE, E, use_T, S0 = returns_first_n_streams_dataset_style(
                rewards_TS, horizon, args.source_timesteps, args.n_stream
            )

            sel_idx, metric, probs_sorted, sorted_index_lst = select_streams_softmax_dataset_style(
                returns_SE=returns_SE,
                max_reward=cfg["max_reward"],
                n_select=min(args.n_stream, S0),
                stability_coeff=args.stability_coeff,
                temperature=args.temperature,
            )

            # 与数据集一致的打印
            print(f'Env {i}: there are {len(sel_idx)} streams randomly selected!')
            print(f'The selected streams are {list(sel_idx)}')
            print('----------------------------------------------')

            sel_unique = np.unique(sel_idx)
            all_idx = np.arange(S0, dtype=int)
            not_sel_idx = np.setdiff1d(all_idx, sel_unique, assume_unique=True)

            print(
                f"Env {i}: selected={len(sel_idx)} (unique={sel_unique.size}), "
                f"not_selected={not_sel_idx.size}, candidates={S0}"
            )

            # 1) All candidates
            title_all = f"Env {i} — {cfg['env']} — All candidates (n={S0}, T={use_T}, episodes={E})"
            out_grid_all = os.path.join(
                out_dir_by_dataset,
                f"returns_all_candidates_{dataset_tag_safe}_env{i}.png",
            )
            plot_grid(returns_SE, all_idx, out_grid_all, title_all)

            # 2) Selected (with duplicates)
            title_sel = (
                f"Env {i} — {cfg['env']} — Selected "
                f"(selections={len(sel_idx)}, unique={sel_unique.size}, from {S0} candidates)"
            )
            out_grid_sel = os.path.join(
                out_dir_by_dataset, f"returns_selected_{dataset_tag_safe}_env{i}.png"
            )
            plot_grid(returns_SE, sel_idx, out_grid_sel, title_sel)

            # 3) Not selected
            title_not = f"Env {i} — {cfg['env']} — Not selected (n={not_sel_idx.size}, from {S0} candidates)"
            out_grid_not = os.path.join(
                out_dir_by_dataset,
                f"returns_not_selected_{dataset_tag_safe}_env{i}.png",
            )
            plot_grid(returns_SE, not_sel_idx, out_grid_not, title_not)

            # 4) Mean curves
            title_mean = f"Env {i} — {cfg['env']} — Mean return: all vs selected vs not-selected"
            out_mean3 = os.path.join(
                out_dir_by_dataset,
                f"mean_all_selected_not_{dataset_tag_safe}_env{i}.png",
            )
            plot_mean_three_groups(returns_SE, sel_idx, out_mean3, title_mean)

            print(
                f"[OK] env {i}: 保存：\n"
                f"  - {out_grid_all}\n"
                f"  - {out_grid_sel}\n"
                f"  - {out_grid_not}\n"
                f"  - {out_mean3}"
            )


if __name__ == "__main__":
    main()
