import argparse, h5py, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
import math

def smooth_same(y, win=1):
    if win <= 1 or len(y) < 2: return y
    k = np.ones(win, dtype=float) / win
    return np.convolve(y, k, mode="same")

def accumulate_returns_one_group(rewards, dones, cap=None):
    """
    rewards/dones: 形状 (T, N_stream)
    返回 list[stream] -> list[episode return]
    """
    T, N = rewards.shape
    acc = np.zeros(N, dtype=float)
    returns = [[] for _ in range(N)]
    for t in range(T):
        acc += rewards[t]
        done_idx = np.nonzero(dones[t])[0]
        for s in done_idx:
            returns[s].append(acc[s])
            acc[s] = 0.0
    if cap is not None:
        returns = [r[:cap] for r in returns]
    return returns

def grid_dims(n):
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    return rows, cols

def plot_grid(returns, env_id, out_path, smooth=1, title_prefix=""):
    n_stream = len(returns)
    rows, cols = grid_dims(n_stream)
    fig_w, fig_h = cols * 2.2, rows * 1.9
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), sharex=True, sharey=True)
    axes = np.atleast_2d(axes).reshape(rows, cols)

    for s in range(n_stream):
        ax = axes[s // cols, s % cols]
        y = np.asarray(returns[s], dtype=float)
        x = np.arange(1, len(y) + 1)
        ax.plot(x, smooth_same(y, smooth), linewidth=1.0)
        ax.set_title(f"S{s}", fontsize=8)
        ax.grid(True, linestyle=":", linewidth=0.4)

    # 去掉多余子图
    for k in range(n_stream, rows * cols):
        axes[k // cols, k % cols].axis("off")

    fig.supxlabel("episode")
    fig.supylabel("return")
    fig.suptitle(f"{title_prefix}Returns per Stream (env {env_id})", fontsize=12, y=0.995)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.97])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"✅ 网格总览已保存: {out_path}")

def save_each_stream(returns, out_dir, smooth=1):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for s, r in enumerate(returns):
        y = np.asarray(r, dtype=float)
        x = np.arange(1, len(y) + 1)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(x, smooth_same(y, smooth), linewidth=1.2)
        ax.set_xlabel("episode"); ax.set_ylabel("return"); ax.set_title(f"stream {s}")
        ax.grid(True, linestyle=":", linewidth=0.6)
        fig.tight_layout()
        fp = out_dir / f"stream_{s:03d}.png"
        fig.savefig(fp, dpi=200)
        plt.close(fig)
    print(f"✅ 单流图已保存到目录: {out_dir}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", required=True, help="HDF5 数据文件")
    p.add_argument("--env-id", type=int, default=0, help="只画该分组（group）的流")
    p.add_argument("--num", type=int, default=None, help="每条 stream 取前多少个 episode（默认取完）")
    p.add_argument("--smooth", type=int, default=1, help="滑动平均窗口，>1 可平滑曲线")
    p.add_argument("--outgrid", default=None, help="网格总览图输出路径（缺省自动生成）")
    p.add_argument("--outdir", default="", help="如提供，则逐流输出到该目录（每个 stream 一张图）")
    args = p.parse_args()

    h5_path = Path(args.h5)
    assert h5_path.exists(), f"找不到文件：{h5_path}"

    with h5py.File(h5_path, "r") as f:
        assert str(args.env_id) in f.keys(), f"env-id {args.env_id} 不在 HDF5 分组中"
        g = f[str(args.env_id)]
        rewards = g["rewards"][()]  # (T, N_stream)
        dones = g["dones"][()]      # (T, N_stream) bool
        returns = accumulate_returns_one_group(rewards, dones, cap=args.num)
        per_stream_eps = dones.sum(axis=0)
        print(f"[env {args.env_id}] n_stream={rewards.shape[1]}, episodes/stream 平均={per_stream_eps.mean():.0f} (min={per_stream_eps.min()}, max={per_stream_eps.max()})")

    outgrid = args.outgrid or (h5_path.parent / f"returns_env{args.env_id}_grid.png")
    plot_grid(returns, args.env_id, outgrid, smooth=args.smooth)

    if args.outdir:
        save_each_stream(returns, args.outdir, smooth=args.smooth)

if __name__ == "__main__":
    main()
