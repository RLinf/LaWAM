"""LaWAM Stage-1 (VGGT) evaluation: is the latent action `z` actually alive?

A low `val/recon_loss` and a 0.99 cosine only prove the world model can predict
future VGGT features. They do NOT prove `z` carries action information -- the
decoder could be ignoring `z` entirely and just learning "copy the input plus an
average motion prior". VGGT features make this especially easy to fake: ~75% of
every token is a constant shared across all tokens, so even the identity map
scores a high raw cosine.

Two probes decide whether Stage 2 is viable at all:

1. delta_z  -- re-run the decoder with `z` shuffled across the batch. If the loss
   barely moves, `z` is decorative and no Stage-2 policy can recover actions.
   Reported against the trivial baselines (predict-the-mean, predict-identity)
   so the numbers are interpretable rather than just "small".

2. z -> action linear probe -- ridge regression from `z` to the ground-truth
   action chunk, scored by R^2 on held-out episodes. Stage 2 trains a policy to
   emit `z`; if even a *linear* map cannot recover actions from `z`, a learned
   policy has nothing to aim at.

All feature-space metrics are computed on standardized features. Raw VGGT
cosines are rank-inverted (cross-episode pairs score *higher* than temporal
pairs) because of that shared constant component, so uncentered numbers would be
actively misleading.
"""

import argparse
import importlib.machinery
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

REPO = "/home/ma-user/work/dataset/xxd-dataset/dataset_yhw/WAM/LaWAM_official"
sys.path.insert(0, REPO)


def _stub(name, **attrs):
    """Stub a module that `latent_action_model.core` imports but we don't need."""
    mod = types.ModuleType(name)
    # accelerate probes importlib.util.find_spec("wandb"), which raises if
    # __spec__ is None on an already-imported module.
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
from latent_action_model.data_loader.video_aug import gpu_two_view_video_aug  # noqa: E402


def standardize(x, mu, sd):
    return (x - mu) / sd


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/ma-user/work/lam_runs/vggt_vae_libero/checkpoints/epoch=39.ckpt")
    ap.add_argument("--config", default="/home/ma-user/work/lam_runs/vggt_vae_libero/version_0/config.yaml")
    ap.add_argument("--n-batches", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--val-tail-ratio", type=float, default=0.05, help="held-out episode fraction")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda"

    model = load_latent_action_model(args.ckpt, args.config).to(dev).eval()
    print(f"[eval] ckpt={args.ckpt}")
    print(f"[eval] encoder={type(model.vision_encoder).__name__} input_dim={model.input_dim}")

    # Held-out episodes. The training run used val_tail_ratio=0.001 (1 episode);
    # 5% (~85 episodes) gives probe estimates that aren't dominated by noise.
    ds = LeRobotLAMDataset(
        data_root_dir="/home/ma-user/work/lam_datasets",
        data_mix="libero",
        num_frames=2,
        mode="val",
        val_tail_ratio=args.val_tail_ratio,
        video_backend="pyav",
        image_hw=(256, 256),
        frame_dt_sec=1.6,
        debug_repeat_batch=False,
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=8,
        collate_fn=lambda b: lam_collate(b, max_state_dim=32),
        drop_last=True,
    )

    # ---- pass 1: collect features, latents, predictions -------------------
    all_dec_in, all_tgt, all_recon, all_z = [], [], [], []
    all_traj, all_base = [], []

    it = iter(loader)
    for i in range(args.n_batches):
        try:
            batch = next(it)
        except StopIteration:
            print(f"[eval] loader exhausted after {i} batches")
            break

        videos = batch["videos"].to(dev)
        # eval mode: no augmentation, video1 == video2 (matches validation_step)
        v1, v2 = gpu_two_view_video_aug(
            videos, output_size=(256, 256), training=False, dual_view_aug=False
        )
        states = batch["states"].to(dev)
        state_mask = batch["state_mask"].to(dev)
        emb = batch["embodiment_ids"].to(dev)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            recon, dec_in, tgt, _, _, _, _, z, _, _ = model.inference(
                v1, states, v2, state_mask=state_mask, embodiment_ids=emb
            )

        all_dec_in.append(dec_in.float().cpu())
        all_tgt.append(tgt.float().cpu())
        all_recon.append(recon.float().cpu())
        all_z.append(z.float().reshape(z.shape[0], -1).cpu())
        all_traj.extend(batch["trajectory_ids"])
        all_base.append(batch["base_indices"])
        if (i + 1) % 10 == 0:
            print(f"[eval] pass1 {i+1}/{args.n_batches}")

    dec_in = torch.cat(all_dec_in)   # [N,1,K,D] features at time t
    tgt = torch.cat(all_tgt)         # [N,1,K,D] features at t+tau (target)
    recon = torch.cat(all_recon)     # [N,1,K,D] prediction
    Z = torch.cat(all_z)             # [N, code_dim]
    base_idx = torch.cat(all_base).numpy()
    N = dec_in.shape[0]
    print(f"\n[eval] N={N} samples | z dim={Z.shape[1]}")

    # ---- standardize in feature space -------------------------------------
    flat = tgt.reshape(-1, tgt.shape[-1])
    mu, sd = flat.mean(0), flat.std(0).clamp_min(1e-6)
    T_ = standardize(tgt, mu, sd)
    R_ = standardize(recon, mu, sd)
    I_ = standardize(dec_in, mu, sd)

    mse = lambda a, b: (a - b).pow(2).mean().item()
    mse_pred = mse(R_, T_)
    mse_mean = mse(torch.zeros_like(T_), T_)      # predict dataset mean == 1.0
    mse_identity = mse(I_, T_)                    # predict "no change"

    print("\n=== 特征空间基线 (标准化后) ===")
    print(f"  mse_pred     = {mse_pred:.4f}   <- 模型预测")
    print(f"  mse_mean     = {mse_mean:.4f}   <- 预测数据集均值 (定义上=1.00)")
    print(f"  mse_identity = {mse_identity:.4f}   <- 预测 u_t (抄袭输入)")
    beats = mse_pred < min(mse_mean, mse_identity)
    print(f"  => 击败所有平凡基线: {'YES' if beats else 'NO'}")

    # centered cosines (Fig.10-style curves)
    def cos_c(a, b):
        a2 = (a - a.mean(dim=-1, keepdim=True)).reshape(-1, a.shape[-1])
        b2 = (b - b.mean(dim=-1, keepdim=True)).reshape(-1, b.shape[-1])
        return F.cosine_similarity(a2, b2, dim=-1).mean().item()

    print(f"  cos_pred_gt   = {cos_c(R_, T_):.4f}  (预测 vs 真实未来)")
    print(f"  cos_init_gt   = {cos_c(I_, T_):.4f}  (当前 vs 真实未来)")
    print(f"  cos_pred_init = {cos_c(R_, I_):.4f}  (预测 vs 当前)")

    # ---- probe 1: delta_z (shuffle z, re-decode) ---------------------------
    # Re-run only the decoder with permuted latents; everything else identical.
    print("\n=== Probe 1: delta_z (打乱 z 重新解码) ===")
    perm = torch.randperm(N)
    shuf_mse = []
    bs = args.batch_size
    for s in range(0, N, bs):
        e = min(s + bs, N)
        feats = dec_in[s:e].to(dev)
        z_shuf = Z[perm[s:e]].to(dev).unsqueeze(1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            r = model.decoder(feats, z_shuf).float().cpu()
        shuf_mse.append(standardize(r, mu, sd))
    R_shuf = torch.cat(shuf_mse)
    mse_shuf = mse(R_shuf, T_)
    delta_z = mse_shuf - mse_pred
    print(f"  mse_pred     = {mse_pred:.4f}")
    print(f"  mse_shuffled = {mse_shuf:.4f}")
    print(f"  delta_z      = {delta_z:.4f}   (相对提升 {100*delta_z/max(mse_pred,1e-9):.1f}%)")
    print(f"  => z 是否在驱动预测: {'YES' if delta_z > 0.02 else 'NO — z 基本是死的'}")

    # ---- probe 2: z -> action linear probe --------------------------------
    print("\n=== Probe 2: z -> 真实动作 线性探针 (ridge, R^2) ===")
    import pyarrow.parquet as pq

    tbl = pq.read_table(
        "/home/ma-user/work/lam_datasets/libero_merged_no_noops_20hz/data/chunk-000/file-000.parquet",
        columns=["episode_index", "frame_index", "action"],
    )
    ep_arr = np.asarray(tbl.column("episode_index"))
    fr_arr = np.asarray(tbl.column("frame_index"))
    act_arr = np.stack(tbl.column("action").to_numpy(zero_copy_only=False))
    # (episode, frame) -> row
    key2row = {(int(e), int(f)): i for i, (e, f) in enumerate(zip(ep_arr, fr_arr))}

    # z spans [t, t+32] (frame_dt_sec=1.6 @ 20fps). Use the whole action chunk.
    STRIDE = 32
    X_rows, Y_rows = [], []
    for i in range(N):
        ep = int(all_traj[i])
        b = int(base_idx[i])
        chunk = []
        ok = True
        for k in range(0, STRIDE, 4):   # subsample the 32-step chunk -> 8 x 7 = 56 dims
            r = key2row.get((ep, b + k))
            if r is None:
                ok = False
                break
            chunk.append(act_arr[r])
        if ok:
            X_rows.append(Z[i].numpy())
            Y_rows.append(np.concatenate(chunk))

    if len(X_rows) < 50:
        print(f"  跳过: 只对齐上 {len(X_rows)} 个样本")
        return

    X = np.asarray(X_rows, dtype=np.float64)
    Y = np.asarray(Y_rows, dtype=np.float64)
    print(f"  对齐样本 {X.shape[0]} | z {X.shape[1]}维 -> action {Y.shape[1]}维")

    # split by episode so train/test never share an episode
    eps = np.array([int(all_traj[i]) for i in range(N)][: len(X_rows)])
    uniq = np.unique(eps)
    rng = np.random.RandomState(0)
    rng.shuffle(uniq)
    n_tr = max(1, int(len(uniq) * 0.7))
    tr_eps = set(uniq[:n_tr].tolist())
    tr = np.array([e in tr_eps for e in eps])
    te = ~tr
    if te.sum() < 10 or tr.sum() < 10:
        print(f"  跳过: episode 切分后训练/测试太小 ({tr.sum()}/{te.sum()})")
        return

    Xm, Xs = X[tr].mean(0), X[tr].std(0) + 1e-8
    Ym = Y[tr].mean(0)
    Xtr, Xte = (X[tr] - Xm) / Xs, (X[te] - Xm) / Xs
    Ytr, Yte = Y[tr] - Ym, Y[te] - Ym

    best = (-1e9, None)
    for lam in [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]:
        A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
        W = np.linalg.solve(A, Xtr.T @ Ytr)
        P = Xte @ W
        ss_res = ((Yte - P) ** 2).sum()
        ss_tot = (Yte ** 2).sum()
        r2 = 1 - ss_res / max(ss_tot, 1e-12)
        if r2 > best[0]:
            best = (r2, lam)
    r2, lam = best
    print(f"  episode 级切分: train {tr.sum()} / test {te.sum()} 样本, {len(uniq)} episodes")
    print(f"  best ridge lambda={lam}")
    print(f"  z2action_r2  = {r2:.4f}")
    print(f"  => {'YES' if r2 > 0.3 else 'NO — 线性解不出动作'}")

    print("\n=== 汇总 ===")
    print(f"  mse_pred {mse_pred:.4f} vs identity {mse_identity:.4f} vs mean {mse_mean:.4f}")
    print(f"  delta_z      = {delta_z:.4f}  (目标 > 0.02, 理想 > 0.1)")
    print(f"  z2action_r2  = {r2:.4f}  (目标 > 0.3)")


if __name__ == "__main__":
    main()
