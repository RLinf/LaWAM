"""Contact sheets of candidate samples, so a human can pick which ones to plot.

The automatic "spread out the motion centroids" selection in viz_action_heatmap.py
kept choosing frames where the arm sits at the top edge and the gripper is out of
view -- fine by the centroid metric, useless to look at. Faster to dump thumbnails
and let a person pick.

Loads no model: dataset + the eval-time aug only, so it runs in well under a minute.
Each cell is o_t with a red contour marking where the pixels actually changed over
the 1.6 s horizon, i.e. where the arm and objects moved.
"""

import argparse
import importlib.machinery
import os
import sys
import types

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402


def _setup_cjk_font():
    for path in ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",):
        if os.path.exists(path):
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.sans-serif"] = [
                font_manager.FontProperties(fname=path).get_name(), "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return


REPO = "/home/ma-user/work/dataset/xxd-dataset/dataset_yhw/WAM/LaWAM_official"
sys.path.insert(0, REPO)


def _stub(name, **attrs):
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    mod.__path__ = []
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


class _Callback:
    pass


_stub("lightning", LightningModule=torch.nn.Module)
_stub("lightning.pytorch", Callback=_Callback)
_stub("lightning.pytorch.callbacks", Callback=_Callback)
sys.modules["lightning"].pytorch = sys.modules["lightning.pytorch"]
_stub("wandb", Image=lambda *a, **k: None, log=lambda *a, **k: None)

from latent_action_model.data_loader.lerobot_dataset import LeRobotLAMDataset  # noqa: E402
from latent_action_model.data_loader.collate import lam_collate  # noqa: E402
from latent_action_model.data_loader.video_aug import (  # noqa: E402
    gpu_two_view_video_aug,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


def denorm_to_uint8(img):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    x = (img.cpu() * std + mean).clamp(0, 1)
    return (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/ma-user/work/lam_runs/viz_pick")
    ap.add_argument("--pool", type=int, default=96)
    ap.add_argument("--per-sheet", type=int, default=24)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    _setup_cjk_font()
    torch.manual_seed(args.seed)
    dev = "cuda"

    ds = LeRobotLAMDataset(
        data_root_dir="/home/ma-user/work/lam_datasets",
        data_mix="libero",
        num_frames=2,
        mode="val",
        val_tail_ratio=0.05,
        video_backend="pyav",
        image_hw=(256, 256),
        frame_dt_sec=1.6,
        debug_repeat_batch=False,
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.pool, num_workers=8, shuffle=True,
        collate_fn=lambda b: lam_collate(b, max_state_dim=32), drop_last=True,
    )
    batch = next(iter(loader))
    v1, _ = gpu_two_view_video_aug(
        batch["videos"].to(dev), output_size=(256, 256), training=False, dual_view_aug=False)

    B = v1.shape[0]
    frames_t = [denorm_to_uint8(v1[i, 0]) for i in range(B)]
    frames_T = [denorm_to_uint8(v1[i, 1]) for i in range(B)]

    # pixel-level motion, only to draw the contour
    diffs, cents = [], []
    for i in range(B):
        d = np.abs(frames_T[i].astype(np.float32) - frames_t[i].astype(np.float32)).mean(-1)
        # cheap blur so the contour is one blob instead of speckle
        t = torch.tensor(d)[None, None]
        d = torch.nn.functional.avg_pool2d(t, 9, stride=1, padding=4)[0, 0].numpy()
        diffs.append(d)
        w = np.clip(d - np.percentile(d, 50), 0, None)
        w = w / max(w.sum(), 1e-8)
        yy, xx = np.mgrid[0:d.shape[0], 0:d.shape[1]]
        cents.append(((w * yy).sum(), (w * xx).sum()))
    cents = np.array(cents)
    mags = np.array([d.mean() for d in diffs])

    print(f"[pick] pool={B}  运动量 min={mags.min():.2f} max={mags.max():.2f}")
    print(f"[pick] 运动重心 y 范围 {cents[:,0].min():.0f}-{cents[:,0].max():.0f}px  "
          f"x 范围 {cents[:,1].min():.0f}-{cents[:,1].max():.0f}px")

    nsheets = int(np.ceil(B / args.per_sheet))
    for s in range(nsheets):
        ids = list(range(s * args.per_sheet, min((s + 1) * args.per_sheet, B)))
        cols = args.cols
        rowsn = int(np.ceil(len(ids) / cols))
        fig, axes = plt.subplots(rowsn, cols, figsize=(2.1 * cols, 2.3 * rowsn))
        axes = np.atleast_2d(axes)
        for k, i in enumerate(ids):
            ax = axes[k // cols, k % cols]
            ax.imshow(frames_t[i])
            d = diffs[i]
            ax.contour(d, levels=[np.percentile(d, 96)], colors="r", linewidths=1.2)
            ax.set_title(f"#{i}  运动{mags[i]:.1f}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
        for k in range(len(ids), rowsn * cols):
            axes[k // cols, k % cols].axis("off")
        fig.suptitle(f"候选样本 第{s+1}/{nsheets}组  (红线=1.6s内实际动过的区域)", fontsize=11)
        plt.tight_layout()
        p = f"{args.out}/sheet{s+1}.png"
        plt.savefig(p, dpi=80, bbox_inches="tight")
        plt.close(fig)
        print(f"[pick] wrote {p}  ({os.path.getsize(p)//1024} KB)  样本 {ids[0]}-{ids[-1]}")


if __name__ == "__main__":
    main()
