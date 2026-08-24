"""Parse the 40 MB training log + TensorBoard events into a convergence figure.

Train metrics only exist inside tqdm progress-bar lines (Lightning logs them to the
bar, not to TB), so they are recovered by regex over carriage-return-separated
records. Note these are Lightning's *running epoch means*, not per-step values --
they reset each epoch, which is why the train curve has a sawtooth at epoch
boundaries early on. Val metrics come from the TB event file, one point per epoch.
"""

import argparse
import glob
import os
import re

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402


def setup_cjk():
    for p in ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",):
        if os.path.exists(p):
            font_manager.fontManager.addfont(p)
            plt.rcParams["font.sans-serif"] = [
                font_manager.FontProperties(fname=p).get_name(),
                "DejaVu Sans",
            ]
            # WenQuanYi has no U+2212 MINUS SIGN, which matplotlib uses by default
            # on log axes -- fall back to ASCII hyphen instead of tofu.
            plt.rcParams["axes.unicode_minus"] = False
            plt.rcParams["mathtext.default"] = "regular"
            return True
    return False


BAR = re.compile(
    r"Epoch (\d+):\s+\d+%\|[^|]*\|\s*(\d+)/(\d+).*?"
    r"train_loss=([\d.e+-]+).*?train/recon_loss=([\d.e+-]+).*?"
    r"train/cos_sim_metric=([\d.e+-]+).*?train/state_loss=([\d.e+-]+).*?"
    r"train/lr=([\d.e+-]+)"
)


def parse_log(path):
    rows = []
    with open(path, "rb") as f:
        blob = f.read().decode("utf-8", errors="replace")
    for rec in blob.replace("\r", "\n").split("\n"):
        m = BAR.search(rec)
        if not m:
            continue
        ep, it, tot = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            rows.append(
                (
                    ep + it / max(tot, 1),
                    float(m.group(4)),
                    float(m.group(5)),
                    float(m.group(6)),
                    float(m.group(7)),
                    float(m.group(8)),
                )
            )
        except ValueError:
            continue
    a = np.array(rows)
    # the bar repeats each record twice (pre/post step); dedupe on the x axis
    _, keep = np.unique(a[:, 0], return_index=True)
    return a[np.sort(keep)]


def parse_tb(vdir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    f = sorted(glob.glob(os.path.join(vdir, "events.out.tfevents.*")))[0]
    ea = EventAccumulator(f, size_guidance={"scalars": 0})
    ea.Reload()
    out = {}
    for t in ea.Tags()["scalars"]:
        s = ea.Scalars(t)
        out[t] = (np.array([x.step for x in s]), np.array([x.value for x in s]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="/home/ma-user/work/lam_runs/train.log")
    ap.add_argument("--vdir", default="/home/ma-user/work/lam_runs/vggt_vae_libero/version_0")
    ap.add_argument("--out", default="/home/ma-user/work/lam_runs/viz/training_curves.png")
    ap.add_argument("--csv", default="/home/ma-user/work/lam_runs/viz/curves.csv")
    args = ap.parse_args()

    setup_cjk()
    tr = parse_log(args.log)
    tb = parse_tb(args.vdir)
    print(f"[curves] train points={len(tr)} epochs={tr[-1,0]:.2f}")

    # val is logged per-epoch; TB 'step' is the global step, so rebuild the epoch axis
    n_val = len(tb["val/recon_loss"][1])
    vx = np.arange(n_val) + 1.0
    vrec = tb["val/recon_loss"][1]
    vcos = tb["val/cos_sim_metric"][1]
    vstate = tb["val/state_loss"][1]

    fig, ax = plt.subplots(2, 2, figsize=(14, 9))

    # (a) recon loss, log scale -- the headline curve
    a = ax[0, 0]
    a.plot(tr[:, 0], tr[:, 2], color="C0", lw=0.8, alpha=0.45, label="train/recon_loss (epoch running mean)")
    a.plot(vx, vrec, "o-", color="C3", ms=4, lw=1.8, label="val/recon_loss")
    best = int(np.argmin(vrec))
    a.plot(vx[best], vrec[best], "*", color="k", ms=16, zorder=5)
    a.annotate(
        f"最低 ep{best} = {vrec[best]:.4f}",
        (vx[best], vrec[best]),
        textcoords="offset points",
        xytext=(12, 18),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", lw=0.8),
    )
    a.axhline(vrec[-1], color="gray", ls=":", lw=1)
    a.set_yscale("log")
    a.set_xlabel("epoch")
    a.set_ylabel("smooth-L1 (标准化特征空间)")
    a.set_title(f"重建损失：0.130 → {vrec[-1]:.4f}  (对数轴)", fontsize=12)
    a.legend(fontsize=8.5)
    a.grid(alpha=0.3, which="both")

    # (b) val only, linear -- shows the plateau honestly
    a = ax[0, 1]
    a.plot(vx, vrec, "o-", color="C3", ms=4, lw=1.8)
    tail = vrec[24:]
    a.axhspan(tail.min(), tail.max(), color="C1", alpha=0.15)
    a.annotate(
        f"ep24 后在 {tail.min():.4f}~{tail.max():.4f} 震荡\n"
        f"(val 只有 109 样本，±0.005 是噪声)",
        (30, tail.max()),
        textcoords="offset points",
        xytext=(-140, 42),
        fontsize=8.5,
        arrowprops=dict(arrowstyle="->", lw=0.8),
    )
    a.set_xlabel("epoch")
    a.set_ylabel("val/recon_loss")
    a.set_title("验证损失（线性轴）：ep24 之后已进入平台期", fontsize=12)
    a.grid(alpha=0.3)

    # (c) train/val gap -- the overfitting check
    a = ax[1, 0]
    tr_ep = np.array([tr[(tr[:, 0] > e) & (tr[:, 0] <= e + 1), 2][-1] for e in range(n_val)])
    a.plot(vx, tr_ep, "s-", color="C0", ms=3.5, lw=1.4, label="train (epoch 末)")
    a.plot(vx, vrec, "o-", color="C3", ms=3.5, lw=1.4, label="val")
    a2 = a.twinx()
    gap = vrec / np.maximum(tr_ep, 1e-9)
    a2.plot(vx, gap, "--", color="C2", lw=1.4, label="val/train 比值")
    a2.axhline(1.0, color="gray", lw=0.8, ls=":")
    a2.set_ylabel("val / train", color="C2")
    a2.set_ylim(0, max(3.0, gap.max() * 1.15))
    a2.tick_params(axis="y", colors="C2")
    a.set_xlabel("epoch")
    a.set_ylabel("recon_loss")
    a.set_title(f"过拟合检查：gap 稳定在 {gap[5:].mean():.2f}×，无发散", fontsize=12)
    a.legend(fontsize=8.5, loc="upper right")
    a.grid(alpha=0.3)

    # (d) the auxiliary signals
    a = ax[1, 1]
    a.plot(vx, vcos, "o-", color="C4", ms=3.5, lw=1.5, label="val/cos_sim_metric")
    a.set_xlabel("epoch")
    a.set_ylabel("cosine", color="C4")
    a.tick_params(axis="y", colors="C4")
    a.set_ylim(0.96, 1.0)
    a3 = a.twinx()
    a3.plot(vx, vstate, "^-", color="C5", ms=3.5, lw=1.5, label="val/state_loss")
    a3.set_yscale("log")
    a3.set_ylabel("state_loss (log)", color="C5")
    a3.tick_params(axis="y", colors="C5")
    a.set_title(
        f"辅助指标：cos {vcos[0]:.3f}→{vcos[-1]:.3f}；state_loss 降 {vstate[0]/vstate[-1]:.0f}×",
        fontsize=12,
    )
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = a3.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=8.5, loc="center right")
    a.grid(alpha=0.3)

    fig.suptitle(
        "LaWAM Stage 1 (VGGT-1B 冻结编码器) 训练收敛曲线 — LIBERO, 40 epochs, 8×A100-80G, ~56 h",
        fontsize=14,
        y=0.995,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[curves] wrote {args.out}")

    with open(args.csv, "w") as f:
        f.write("epoch,train_recon,val_recon,val_cos_sim,val_state_loss\n")
        for i in range(n_val):
            f.write(f"{i},{tr_ep[i]:.6f},{vrec[i]:.6f},{vcos[i]:.6f},{vstate[i]:.8f}\n")
    print(f"[curves] wrote {args.csv}")
    print(f"[curves] final train={tr_ep[-1]:.4f} val={vrec[-1]:.4f} gap={gap[-1]:.2f}x")


if __name__ == "__main__":
    main()
