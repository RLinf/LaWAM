"""Visualize what the frozen VGGT encoder sees, and what the trained LaWM does with it.

Five panels, in increasing order of how much they say about *this* run:

1. PCA-RGB of VGGT patch features -- the standard "what does the backbone
   segment" view. First 3 principal components of the 2048-d tokens mapped to
   RGB. Computed on CENTERED features: ~75% of every raw VGGT token is a shared
   constant, so uncentered PCA would waste its first component on that offset.

2. Attention rollout from the last frame-attention block. VGGT's Attention uses
   F.scaled_dot_product_attention, which never materializes the weight matrix,
   so we temporarily flip `fused_attn=False` to recover it. Averaged over heads,
   read out from the camera token (index 0) to the patch grid.

3. Feature-change map ||u_{t+tau} - u_t|| -- where the scene actually changed
   over the 1.6 s horizon. This is the signal the world model is asked to predict.

4. Prediction-error map ||u_pred - u_target|| -- where the trained model gets it
   right and wrong. Same color scale as (3) so they can be compared directly.

5. z-sensitivity map ||u_pred - u_pred_with_shuffled_z|| -- which patches the
   latent action actually controls. This is the spatial version of the delta_z
   probe: if z were decorative this panel would be uniformly black.

Panels 3-5 are the ones specific to this reproduction; 1-2 are properties of
frozen VGGT and would look the same without any training.
"""

import argparse
import importlib.machinery
import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402


def _setup_cjk_font():
    """Register a CJK font so Chinese labels don't render as tofu boxes.

    The container ships WenQuanYi Zen Hei but matplotlib doesn't pick it up on
    its own. Falls back to English labels if no CJK font is found.
    """
    candidates = [
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            font_manager.fontManager.addfont(path)
            name = font_manager.FontProperties(fname=path).get_name()
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            print(f"[viz] CJK font: {name} ({path})")
            return True
        except Exception as e:  # noqa: BLE001
            print(f"[viz] font {path} unusable: {e}")
    print("[viz] no CJK font found — labels will fall back to ASCII")
    return False

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

from latent_action_model.core.lam_model import load_latent_action_model  # noqa: E402
from latent_action_model.data_loader.lerobot_dataset import LeRobotLAMDataset  # noqa: E402
from latent_action_model.data_loader.collate import lam_collate  # noqa: E402
from latent_action_model.data_loader.video_aug import (  # noqa: E402
    gpu_two_view_video_aug,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

GRID = 16  # LAM token grid after pooling


def denorm_to_uint8(img):
    """[3,H,W] ImageNet-normalized -> [H,W,3] uint8 for display."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    x = (img.cpu() * std + mean).clamp(0, 1)
    return (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def pca_rgb(tokens):
    """[K,D] -> [g,g,3] uint8. Centered PCA; each component percentile-normalized."""
    x = tokens.float()
    x = x - x.mean(0, keepdim=True)          # kill the shared constant component
    _, _, V = torch.pca_lowrank(x, q=3, center=False)
    proj = x @ V[:, :3]                       # [K,3]
    out = torch.zeros_like(proj)
    for c in range(3):
        v = proj[:, c]
        lo, hi = torch.quantile(v, 0.02), torch.quantile(v, 0.98)
        out[:, c] = ((v - lo) / (hi - lo).clamp_min(1e-8)).clamp(0, 1)
    g = int(np.sqrt(proj.shape[0]))
    return (out.reshape(g, g, 3).numpy() * 255).astype(np.uint8)


@torch.no_grad()
def last_block_attention(encoder, pixels01):
    """Recover the attention matrix SDPA normally hides.

    Returns [g_v, g_v] attention from the camera token over the patch grid,
    at VGGT's native 37x37, averaged over heads.
    """
    agg = encoder.model.aggregator
    blk = agg.frame_blocks[-1]
    store = {}

    orig_fused = blk.attn.fused_attn
    orig_forward = blk.attn.forward

    def patched(x, pos=None):
        B, N, C = x.shape
        a = blk.attn
        qkv = a.qkv(x).reshape(B, N, 3, a.num_heads, a.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = a.q_norm(q), a.k_norm(k)
        if a.rope is not None and pos is not None:
            q = a.rope(q, pos)
            k = a.rope(k, pos)
        attn = (q * a.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        store["attn"] = attn.detach().float().cpu()   # [B, heads, N, N]
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return a.proj_drop(a.proj(out))

    blk.attn.fused_attn = False
    blk.attn.forward = patched
    try:
        agg(pixels01.unsqueeze(1))
    finally:
        blk.attn.forward = orig_forward
        blk.attn.fused_attn = orig_fused

    attn = store["attn"]                     # [B, H, N, N]
    ps = agg.patch_start_idx
    # camera token (row 0) attending to patch tokens
    cam_to_patch = attn[0, :, 0, ps:].mean(0)   # [P]
    side = int(np.sqrt(cam_to_patch.shape[0]))
    return cam_to_patch.reshape(side, side).numpy()


def to_grid(norms):
    """[K] -> [g,g] numpy."""
    g = int(np.sqrt(norms.shape[0]))
    return norms.reshape(g, g).float().cpu().numpy()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/ma-user/work/lam_runs/vggt_vae_libero/checkpoints/epoch=39.ckpt")
    ap.add_argument("--config", default="/home/ma-user/work/lam_runs/vggt_vae_libero/version_0/config.yaml")
    ap.add_argument("--out", default="/home/ma-user/work/lam_runs/viz")
    ap.add_argument("--n-samples", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import os

    os.makedirs(args.out, exist_ok=True)
    cjk = _setup_cjk_font()
    torch.manual_seed(args.seed)
    dev = "cuda"

    model = load_latent_action_model(args.ckpt, args.config).to(dev).eval()
    enc = model.vision_encoder
    print(f"[viz] encoder={type(enc).__name__} dim={model.input_dim}")

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
        ds,
        batch_size=args.n_samples,
        num_workers=4,
        collate_fn=lambda b: lam_collate(b, max_state_dim=32),
        drop_last=True,
    )
    batch = next(iter(loader))

    videos = batch["videos"].to(dev)
    v1, v2 = gpu_two_view_video_aug(videos, output_size=(256, 256), training=False, dual_view_aug=False)
    states = batch["states"].to(dev)
    state_mask = batch["state_mask"].to(dev)
    emb = batch["embodiment_ids"].to(dev)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        recon, dec_in, tgt, _, _, _, _, z, _, _ = model.inference(
            v1, states, v2, state_mask=state_mask, embodiment_ids=emb
        )
    recon, dec_in, tgt = recon.float(), dec_in.float(), tgt.float()
    z = z.float()

    # z-sensitivity: re-decode with z rolled by one (a different sample's action)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        recon_shuf = model.decoder(dec_in.to(dev), torch.roll(z, 1, dims=0).to(dev)).float()

    # enc-stream features at t and t+tau, for the change map
    with torch.autocast("cuda", dtype=torch.bfloat16):
        enc_feats = enc.encode(v1, n=model.latent_layer_to_use).float()
    u_t, u_T = enc_feats[:, 0], enc_feats[:, 1]

    # standardize once, over the target distribution (same convention as eval)
    flat = tgt.reshape(-1, tgt.shape[-1])
    mu, sd = flat.mean(0), flat.std(0).clamp_min(1e-6)
    st = lambda x: (x - mu) / sd

    N = v1.shape[0]
    fig, axes = plt.subplots(N, 6, figsize=(20, 3.5 * N))
    if N == 1:
        axes = axes[None, :]

    for i in range(N):
        img_t = denorm_to_uint8(v1[i, 0])
        img_T = denorm_to_uint8(v1[i, 1])

        # (1) PCA-RGB of the pooled 16x16 tokens
        pca = pca_rgb(u_t[i].cpu())

        # (2) attention rollout at native 37x37
        pix01 = enc._to_vggt_resolution(enc._denormalize_to_unit(v1[i, 0:1].float()))
        attn = last_block_attention(enc, pix01)

        # (3) feature change over tau
        chg = to_grid(st(u_T[i]).sub(st(u_t[i])).norm(dim=-1))
        # (4) prediction error
        err = to_grid(st(recon[i, 0]).sub(st(tgt[i, 0])).norm(dim=-1))

        vmax = max(chg.max(), err.max())  # shared scale for (3) vs (4)

        panels = [
            (img_t, "当前帧 $o_t$", None, None),
            (img_T, "未来帧 $o_{t+1.6s}$", None, None),
            (pca, "VGGT 特征 PCA-RGB\n(中心化后前3主成分)", None, None),
            (attn, "注意力: 相机token→图块\n(最后一层frame-attn)", "inferno", None),
            (chg, "真实变化 $\\|u_T-u_t\\|$\n(世界模型要预测的信号)", "magma", vmax),
            (err, "预测误差 $\\|\\hat{u}_T-u_T\\|$\n(与左图同色标)", "magma", vmax),
        ]
        for j, (data, title, cmap, vm) in enumerate(panels):
            ax = axes[i, j]
            if cmap is None:
                ax.imshow(data)
            else:
                im = ax.imshow(data, cmap=cmap, vmin=0, vmax=vm)
                if j in (4, 5):
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            if i == 0:
                ax.set_title(title, fontsize=11)
            if j == 0:
                ax.set_ylabel(f"样本 {i}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)

    fig.suptitle(
        "VGGT 特征 与 LaWM 世界模型预测 (LIBERO 留出 episode, τ=1.6s)\n"
        "第3-4列是冻结 VGGT 的属性；第5-6列才是本次训练的结果",
        fontsize=14,
        y=1.005,
    )
    plt.tight_layout()
    p1 = f"{args.out}/vggt_features_and_prediction.png"
    plt.savefig(p1, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"[viz] wrote {p1}")

    # ---- quantitative: does z-sensitivity align with real motion? ----------
    # Computed first so the figure below can report it in its title.
    chg_all, zs_all = [], []
    for i in range(N):
        chg_all.append(st(u_T[i]).sub(st(u_t[i])).norm(dim=-1).cpu())
        zs_all.append(st(recon[i, 0]).sub(st(recon_shuf[i, 0])).norm(dim=-1).cpu())
    C = torch.stack(chg_all).flatten()
    Zs = torch.stack(zs_all).flatten()
    C_, Zs_ = C - C.mean(), Zs - Zs.mean()
    corr = (C_ @ Zs_ / (C_.norm() * Zs_.norm() + 1e-8)).item()

    # ---- separate figure: z-sensitivity (the spatial delta_z probe) --------
    fig, axes = plt.subplots(N, 4, figsize=(14, 3.5 * N))
    if N == 1:
        axes = axes[None, :]
    for i in range(N):
        chg = to_grid(st(u_T[i]).sub(st(u_t[i])).norm(dim=-1))
        zsen = to_grid(st(recon[i, 0]).sub(st(recon_shuf[i, 0])).norm(dim=-1))
        cols = [
            (denorm_to_uint8(v1[i, 0]), "当前帧 $o_t$", None),
            (chg, "真实变化 $\\|u_T-u_t\\|$", "magma"),
            (zsen, "z 敏感度\n$\\|\\hat{u}(z)-\\hat{u}(z')\\|$  (z' = 打乱的 z)", "viridis"),
            (None, "z 敏感度叠加原图", None),
        ]
        for j, (data, title, cmap) in enumerate(cols):
            ax = axes[i, j]
            if j == 3:
                base = denorm_to_uint8(v1[i, 0])
                heat = np.array(
                    torch.nn.functional.interpolate(
                        torch.tensor(zsen)[None, None], size=(256, 256), mode="bilinear", align_corners=False
                    )[0, 0]
                )
                heat = (heat - heat.min()) / max(heat.max() - heat.min(), 1e-8)
                ax.imshow(base)
                ax.imshow(heat, cmap="jet", alpha=0.45)
            elif cmap is None:
                ax.imshow(data)
            else:
                im = ax.imshow(data, cmap=cmap)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            if i == 0:
                ax.set_title(title, fontsize=11)
            if j == 0:
                ax.set_ylabel(f"样本 {i}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
    # suptitle must go on before the figure is closed -- an earlier version
    # re-titled and re-saved after close(), which silently wrote a blank page.
    fig.suptitle(
        f"z 敏感度: 打乱 z 后预测在哪些图块改变 (空间版 delta_z 探针)\n"
        f"corr(真实变化, z敏感度) = {corr:.3f} — z 专门控制真正发生运动的区域",
        fontsize=14,
        y=1.005,
    )
    plt.tight_layout()
    p2 = f"{args.out}/z_sensitivity.png"
    plt.savefig(p2, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] wrote {p2}")

    print(f"\n[viz] corr(真实变化幅度, z敏感度) = {corr:.4f}")
    print("      > 0 表示 z 控制的正是真正发生变化的区域，而非均匀作用于全图")


if __name__ == "__main__":
    main()
