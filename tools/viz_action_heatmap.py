"""Action heatmap: which patches does the latent action z actually control?

This is a sharper version of the z-sensitivity panel in viz_vggt_lam.py. That one
swapped in a single other sample's z, so its map conflated "z governs this patch"
with "this particular replacement z happened to differ a lot". Here we hold the
observation fixed, decode it under K different z's, and take the per-patch standard
deviation of the prediction. A patch the action cannot influence has std ~= 0
regardless of which z shows up.

The hypothesis being tested is the obvious physical one: an action should move the
arm and the objects it contacts, and leave the table, floor and cabinet alone. So
the script also prints a separation ratio -- mean sensitivity over patches that
really changed vs. patches that did not -- which is what makes this falsifiable
rather than a pretty picture.
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
    for path in (
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ):
        if not os.path.exists(path):
            continue
        try:
            font_manager.fontManager.addfont(path)
            name = font_manager.FontProperties(fname=path).get_name()
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            print(f"[viz] CJK font: {name}")
            return True
        except Exception as e:  # noqa: BLE001
            print(f"[viz] font {path} unusable: {e}")
    print("[viz] no CJK font found")
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


def denorm_to_uint8(img):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    x = (img.cpu() * std + mean).clamp(0, 1)
    return (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def to_grid(v):
    g = int(np.sqrt(v.shape[0]))
    return v.reshape(g, g).float().cpu().numpy()


def upsample(grid, size=256, mode="bicubic"):
    t = torch.tensor(grid)[None, None].float()
    out = torch.nn.functional.interpolate(t, size=(size, size), mode=mode, align_corners=False)
    return out[0, 0].numpy()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/ma-user/work/lam_runs/vggt_vae_libero/checkpoints/epoch=39.ckpt")
    ap.add_argument("--config", default="/home/ma-user/work/lam_runs/vggt_vae_libero/version_0/config.yaml")
    ap.add_argument("--out", default="/home/ma-user/work/lam_runs/viz")
    ap.add_argument("--batch", type=int, default=16, help="batch size; also the number of distinct z's")
    ap.add_argument("--n-show", type=int, default=6, help="how many samples to draw")
    ap.add_argument("--pool", type=int, default=0,
                    help="if >0, draw this many samples and display the n-show whose MOTION "
                         "is most spread out in image space. Consecutive val slices all put "
                         "the arm in the same place, which hides whether the heatmap follows it.")
    ap.add_argument("--indices", default="",
                    help="comma-separated sample indices to display, overriding --pool's "
                         "automatic choice. Use --contact-sheet first to pick them by eye: "
                         "max-spread selection favours centroids near the border, which in "
                         "practice means the gripper is cropped out of frame.")
    ap.add_argument("--contact-sheet", action="store_true",
                    help="dump the whole pool as a labelled grid and exit, for picking indices")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true",
                    help="draw from across episodes; without it the val loader returns "
                         "consecutive slices of one scene, which confounds 'shared "
                         "positional bias' with 'same scene'")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    _setup_cjk_font()
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
        batch_size=max(args.batch, args.pool),
        num_workers=4,
        shuffle=args.shuffle,
        collate_fn=lambda b: lam_collate(b, max_state_dim=32),
        drop_last=True,
    )
    batch = next(iter(loader))
    print(f"[viz] shuffle={args.shuffle} pool={args.pool}")

    videos = batch["videos"].to(dev)
    v1, v2 = gpu_two_view_video_aug(videos, output_size=(256, 256), training=False, dual_view_aug=False)
    states = batch["states"].to(dev)
    state_mask = batch["state_mask"].to(dev)
    emb = batch["embodiment_ids"].to(dev)

    # --- pick a visually diverse subset ----------------------------------------
    # Consecutive val slices put the arm in nearly the same place every time, which
    # makes it impossible to see whether the heatmap tracks the arm or sits still.
    # Score each candidate by where its motion is centred, then greedily take the
    # ones whose motion centroids are farthest apart.
    if args.contact_sheet:
        P0 = v1.shape[0]
        ncol = 8
        nrow = (P0 + ncol - 1) // ncol
        figc, axc = plt.subplots(nrow, ncol, figsize=(2.0 * ncol, 2.1 * nrow))
        axc = np.atleast_2d(axc)
        for k in range(nrow * ncol):
            a = axc[k // ncol, k % ncol]
            if k < P0:
                a.imshow(denorm_to_uint8(v1[k, 0]))
                a.set_title(str(k), fontsize=9, pad=2)
            a.set_xticks([]); a.set_yticks([])
            for s_ in a.spines.values():
                s_.set_visible(False)
        figc.suptitle("候选帧（用 --indices 指定要画哪几个）", fontsize=13)
        plt.tight_layout()
        pc = f"{args.out}/contact_sheet.png"
        plt.savefig(pc, dpi=100, bbox_inches="tight")
        plt.close(figc)
        print(f"[viz] wrote {pc}  ({P0} 个候选)")
        return

    if args.indices:
        sel = torch.tensor([int(t) for t in args.indices.split(",")], device=dev)
        print(f"[viz] 手动指定样本: {sel.tolist()}")
    elif args.pool > 0:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pf = enc.encode(v1, n=model.latent_layer_to_use).float()
        pm = pf[:, 1].sub(pf[:, 0]).norm(dim=-1)              # [P0, 256] crude motion map
        g = int(np.sqrt(pm.shape[1]))
        ys, xs = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij")
        ys, xs = ys.flatten().float().to(dev), xs.flatten().float().to(dev)
        w = (pm - pm.min(dim=1, keepdim=True).values).clamp_min(0)
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-8)
        cy, cx = (w * ys).sum(1), (w * xs).sum(1)             # motion centroid per sample
        cen = torch.stack([cy, cx], dim=1)

        keep = [int(pm.sum(1).argmax())]                       # start from the largest motion
        while len(keep) < args.n_show:
            d = torch.cdist(cen, cen[keep]).min(dim=1).values  # distance to nearest kept
            d[torch.tensor(keep, device=dev)] = -1
            keep.append(int(d.argmax()))
        sel = torch.tensor(keep, device=dev)
        spread = torch.cdist(cen[sel], cen[sel])
        print(f"[viz] pool={args.pool} -> picked {keep}")
        print(f"[viz] 选中样本的运动重心两两距离 min={spread[spread>0].min():.1f} "
              f"max={spread.max():.1f} (图块单位, 16x16 网格)")
    else:
        sel = None

    with torch.autocast("cuda", dtype=torch.bfloat16):
        recon, dec_in, tgt, _, _, _, _, z, _, _ = model.inference(
            v1, states, v2, state_mask=state_mask, embodiment_ids=emb
        )
    recon, dec_in, tgt, z = recon.float(), dec_in.float(), tgt.float(), z.float()
    print(f"[viz] dec_in={tuple(dec_in.shape)} z={tuple(z.shape)} recon={tuple(recon.shape)}")

    with torch.autocast("cuda", dtype=torch.bfloat16):
        enc_feats = enc.encode(v1, n=model.latent_layer_to_use).float()
    u_t, u_T = enc_feats[:, 0], enc_feats[:, 1]

    flat = tgt.reshape(-1, tgt.shape[-1])
    mu, sd = flat.mean(0), flat.std(0).clamp_min(1e-6)
    st = lambda x: (x - mu) / sd

    B = dec_in.shape[0]
    K = B  # every sample's z gets applied to every observation

    # --- the probe: hold the observation fixed, sweep z over all K actions ------
    sens, chgs = [], []
    for i in range(B):
        rep = dec_in[i : i + 1].repeat(K, *([1] * (dec_in.dim() - 1)))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            preds = model.decoder(rep, z).float()          # [K, 1, P, D]
        p = st(preds[:, 0])                                # [K, P, D]
        # per-patch spread of the prediction across actions
        sens.append(p.std(dim=0).norm(dim=-1).cpu())       # [P]
        chgs.append(st(u_T[i]).sub(st(u_t[i])).norm(dim=-1).cpu())

    S = torch.stack(sens)        # [B, P]
    C = torch.stack(chgs)        # [B, P]

    # Every sample's map has a red blob in the same upper-left corner, so the raw
    # ratio could just be a fixed positional bias lining up with the fact that the
    # arm always lives in that half of the frame. Subtract the map shared by all
    # samples and re-score: what survives is the part that tracks THIS sample.
    bias = S.mean(0)                                       # [P]
    resid = S - bias

    def score(maps):
        out = []
        for i in range(B):
            m, c = maps[i], C[i]
            k = max(1, c.numel() // 4)
            moving = torch.zeros_like(c, dtype=torch.bool)
            moving[torch.topk(c, k).indices] = True
            static = c <= torch.quantile(c, 0.5)
            out.append((m[moving].mean() / m[static].mean().clamp_min(1e-8)).item())
        return np.array(out)

    ratios = score(S)
    # for the residual, shift to positive before ratioing (it is zero-mean by construction)
    resid_pos = resid - resid.min(dim=1, keepdim=True).values
    ratios_r = score(resid_pos)

    # how much of each map is just the shared bias?
    bias_frac = (bias.norm() / S.norm(dim=1).mean()).item()

    # The norm ratio above is a weak statistic: these maps are all non-negative, so
    # they share a large DC component by construction and the ratio is inflated.
    # The honest version -- remove each map's OWN spatial mean, then correlate maps
    # from DIFFERENT samples. If the spatial pattern really is sample-independent
    # this stays high; if each map tracked its own frame it would fall to ~0.
    Sc = S - S.mean(dim=1, keepdim=True)
    Sn = Sc / Sc.norm(dim=1, keepdim=True).clamp_min(1e-8)
    G = Sn @ Sn.T                                          # [B, B] cosines
    off = G[~torch.eye(B, dtype=torch.bool)]
    # control: same statistic on the real-change maps, which DO track each frame
    Cc = C - C.mean(dim=1, keepdim=True)
    Cn = Cc / Cc.norm(dim=1, keepdim=True).clamp_min(1e-8)
    Gc = Cn @ Cn.T
    off_c = Gc[~torch.eye(B, dtype=torch.bool)]

    cc = []
    for i in range(B):
        a, b = C[i] - C[i].mean(), S[i] - S[i].mean()
        cc.append((a @ b / (a.norm() * b.norm() + 1e-8)).item())
    cc = np.array(cc)

    print("\n=== 动作热力图 vs 真实运动区域 ===")
    print(f"  样本数 {B} | 每个观测扫过 K={K} 个不同的 z")
    print(f"  [原始]   运动区/静止区 敏感度比 = {ratios.mean():.2f} ± {ratios.std():.2f}")
    print(f"           逐样本: {np.array2string(ratios, precision=2, floatmode='fixed')}")
    print(f"  [扣偏置] 同上，但先减掉所有样本共享的平均图 = {ratios_r.mean():.2f} ± {ratios_r.std():.2f}")
    print(f"           逐样本: {np.array2string(ratios_r, precision=2, floatmode='fixed')}")
    print(f"\n  --- 热力图到底有多'共享' ---")
    print(f"  范数比 ‖mean‖/‖单样本‖ = {bias_frac:.3f}   (偏乐观：图全非负，天然有共同分量)")
    print(f"  跨样本相关(各自去空间均值后) = {off.mean():.3f} ± {off.std():.3f}   <- 严谨版")
    print(f"  同一统计量用在'真实变化'图上   = {off_c.mean():.3f} ± {off_c.std():.3f}   <- 对照组")
    print(f"  corr(真实变化, 敏感度) 逐样本 = {cc.mean():.3f} ± {cc.std():.3f}")
    print("  > 若敏感度图真的跟着每一帧走，跨样本相关应接近对照组；接近 1 则是固定图案")

    # --- centroids: the most direct form of the question ------------------------
    # "Does the heatmap follow the arm?" == "when the motion centroid moves, does
    # the sensitivity centroid move with it?" Two numbers settle it: how far each
    # centroid wanders across samples, and whether the two wander together.
    gg = int(np.sqrt(S.shape[1]))
    yy, xx = torch.meshgrid(torch.arange(gg).float(), torch.arange(gg).float(), indexing="ij")
    yy, xx = yy.flatten(), xx.flatten()

    def centroid(M):
        w = (M - M.min(dim=1, keepdim=True).values).clamp_min(0)
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return torch.stack([(w * yy).sum(1), (w * xx).sum(1)], dim=1)

    cenS, cenC = centroid(S), centroid(C)
    spreadS = cenS.std(0).norm().item()
    spreadC = cenC.std(0).norm().item()
    dS = (cenS - cenS.mean(0))
    dC = (cenC - cenC.mean(0))
    follow = (dS * dC).sum() / (dS.norm() * dC.norm() + 1e-8)

    print(f"\n  --- 重心会不会跟着动 ---")
    print(f"  真实运动重心的游走幅度   = {spreadC:.2f} 图块  <- 手臂确实到处跑")
    print(f"  z 敏感度重心的游走幅度   = {spreadS:.2f} 图块  <- 热力图几乎钉死")
    print(f"  两者游走方向的相关       = {follow:.3f}")
    print(f"  比值 (敏感度/真实)       = {spreadS / max(spreadC, 1e-8):.2f}"
          f"   <- 1.0 才叫'跟着走'")

    # --- figure -----------------------------------------------------------------
    rows = list(range(min(args.n_show, B))) if sel is None else [int(i) for i in sel]
    N = len(rows)
    fig, axes = plt.subplots(N, 5, figsize=(18, 3.6 * N))
    if N == 1:
        axes = axes[None, :]

    for r, i in enumerate(rows):
        frame = denorm_to_uint8(v1[i, 0])
        heat = to_grid(S[i])
        res = to_grid(resid[i])
        chg = to_grid(C[i])
        ptS, ptC = cenS[i], cenC[i]

        hn = (res - res.min()) / max(res.max() - res.min(), 1e-8)
        hi = upsample(hn)
        thr = float(np.quantile(hi, 0.75))

        axes[r, 0].imshow(frame)
        # raw map -- dominated by the shared positional bias
        im1 = axes[r, 1].imshow(heat, cmap="turbo", interpolation="nearest")
        plt.colorbar(im1, ax=axes[r, 1], fraction=0.046, pad=0.02)
        # bias-removed map -- the part that actually tracks this frame
        im2 = axes[r, 2].imshow(res, cmap="turbo", interpolation="nearest")
        plt.colorbar(im2, ax=axes[r, 2], fraction=0.046, pad=0.02)
        # overlay of the bias-removed map
        axes[r, 3].imshow(frame)
        axes[r, 3].imshow(hi, cmap="turbo", alpha=0.5)
        axes[r, 3].contour(hi, levels=[thr], colors="w", linewidths=1.6)
        # reference -- where the scene really changed
        im4 = axes[r, 4].imshow(chg, cmap="magma", interpolation="nearest")
        plt.colorbar(im4, ax=axes[r, 4], fraction=0.046, pad=0.02)

        # the punchline: put both centroids on the raw map AND the real-change map.
        # The white circle barely moves down the rows; the red cross moves a lot.
        for c in (1, 4):
            axes[r, c].plot(ptS[1], ptS[0], "o", mfc="none", mec="w", mew=2.5, ms=14)
            axes[r, c].plot(ptC[1], ptC[0], "x", mec="r", mew=3, ms=13)

        if r == 0:
            for j, t in enumerate([
                "当前帧 $o_t$",
                "① 动作热力图 (原始)\n〇=热力重心  ✕=真实运动重心",
                "② 扣掉共享偏置后",
                "③ ②叠加原图 + 白线=前25%",
                "④ 对照: 真实变化\n〇 ✕ 同上",
            ]):
                axes[r, j].set_title(t, fontsize=11)
        axes[r, 0].set_ylabel(f"样本 {i}\n原始 {ratios[i]:.2f} / 扣偏置 {ratios_r[i]:.2f}", fontsize=9)
        for j in range(5):
            axes[r, j].set_xticks([])
            axes[r, j].set_yticks([])
            for s_ in axes[r, j].spines.values():
                s_.set_visible(False)

    fig.suptitle(
        f"动作热力图: 固定观测，扫过 {K} 个不同的 latent action，逐图块取标准差\n"
        f"✕(真实运动重心) 逐行大幅移动，〇(热力重心) 几乎不动 — "
        f"游走幅度 {spreadC:.2f} vs {spreadS:.2f} 图块 (比值 {spreadS/max(spreadC,1e-8):.2f})\n"
        f"运动区/静止区比值 原始 {ratios.mean():.2f} → 扣掉共享偏置后 {ratios_r.mean():.2f}",
        fontsize=13,
        y=1.004,
    )
    plt.tight_layout()
    p = f"{args.out}/action_heatmap.png"
    plt.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[viz] wrote {p}")

    # --- decomposition figure: 原始 = 公共底图 + 本帧独有 ------------------------
    # The point of this one is to make "shared bias" concrete: column 2 is literally
    # the same image in every row, and column 3 is what is left over.
    bias_g = to_grid(bias)
    rmax = float(resid[rows].abs().max())
    fig2, ax2 = plt.subplots(N, 4, figsize=(15, 3.6 * N))
    if N == 1:
        ax2 = ax2[None, :]
    vmax_s = float(S[rows].max())
    for r, i in enumerate(rows):
        ax2[r, 0].imshow(denorm_to_uint8(v1[i, 0]))
        a = ax2[r, 1].imshow(to_grid(S[i]), cmap="turbo", vmin=0, vmax=vmax_s, interpolation="nearest")
        plt.colorbar(a, ax=ax2[r, 1], fraction=0.046, pad=0.02)
        b = ax2[r, 2].imshow(bias_g, cmap="turbo", vmin=0, vmax=vmax_s, interpolation="nearest")
        plt.colorbar(b, ax=ax2[r, 2], fraction=0.046, pad=0.02)
        c = ax2[r, 3].imshow(to_grid(resid[i]), cmap="coolwarm", vmin=-rmax, vmax=rmax,
                             interpolation="nearest")
        plt.colorbar(c, ax=ax2[r, 3], fraction=0.046, pad=0.02)
        if r == 0:
            for j, t in enumerate([
                "当前帧 $o_t$",
                "① 原始热力图",
                "② 公共底图 (所有样本的平均)\n每一行都是同一张",
                "③ 剩下的 = ① − ②\n红=比平均热, 蓝=比平均冷",
            ]):
                ax2[r, j].set_title(t, fontsize=11)
        ax2[r, 0].set_ylabel(f"样本 {i}", fontsize=10)
        for j in range(4):
            ax2[r, j].set_xticks([])
            ax2[r, j].set_yticks([])
            for s_ in ax2[r, j].spines.values():
                s_.set_visible(False)
    fig2.suptitle(
        "共享偏置是什么: 原始热力图 ① 拆成「所有样本共有的 ②」+「这一帧独有的 ③」\n"
        f"② 三列同一张图；③ 的量级只有 ① 的一小部分 — 跨样本相关 {off.mean():.2f} "
        f"(对照: 真实变化图只有 {off_c.mean():.2f})",
        fontsize=14,
        y=1.004,
    )
    plt.tight_layout()
    p2 = f"{args.out}/action_heatmap_decomposition.png"
    plt.savefig(p2, dpi=110, bbox_inches="tight")
    plt.close(fig2)
    print(f"[viz] wrote {p2}")


if __name__ == "__main__":
    main()
