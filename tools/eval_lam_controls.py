"""Controls for the LaWAM-VGGT Stage-1 probes -- the numbers `eval_lam_probes.py`
reports are not interpretable on their own.

Two gaps in the original evaluation, both of which could make a dead `z` look alive:

A. `z2action_r2 = 0.55` has no baseline. LIBERO actions are strongly predictable
   from the current frame alone (the arm's position implies where it is going).
   If a 32-d PCA of `u_t` scores the same R^2, `z` contributes nothing and the
   "z encodes action" claim collapses. We match dimensionality exactly (32 vs 32)
   so the comparison is about *content*, not capacity, and fit the PCA on train
   episodes only.

B. `delta_z` shuffles `z` across the whole batch, which mixes two effects: wrong
   *motion* and wrong *scene/task*. A `z` that only encoded "which LIBERO suite
   is this" would still produce a large delta_z. Shuffling *within* an episode
   holds scene and task fixed, so only the motion component can explain the gap.

Also characterizes what the decoder does with a wrong `z` (collapse to identity,
or confidently wrong motion?) and adds z=0 / random-z references.
"""

import argparse
import importlib.machinery
import sys
import types
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

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
from latent_action_model.data_loader.video_aug import gpu_two_view_video_aug  # noqa: E402


def standardize(x, mu, sd):
    return (x - mu) / sd


def ridge_r2(X, Y, tr, te, tag):
    """Episode-split ridge with a lambda sweep. Returns (r2, lambda)."""
    Xm, Xs = X[tr].mean(0), X[tr].std(0) + 1e-8
    Ym = Y[tr].mean(0)
    Xtr, Xte = (X[tr] - Xm) / Xs, (X[te] - Xm) / Xs
    Ytr, Yte = Y[tr] - Ym, Y[te] - Ym
    best = (-1e9, None)
    for lam in [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3]:
        A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
        W = np.linalg.solve(A, Xtr.T @ Ytr)
        P = Xte @ W
        r2 = 1 - ((Yte - P) ** 2).sum() / max((Yte ** 2).sum(), 1e-12)
        if r2 > best[0]:
            best = (r2, lam)
    print(f"  {tag:36s} R^2 = {best[0]:.4f}   (lambda={best[1]})")
    return best


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/ma-user/work/lam_runs/vggt_vae_libero/checkpoints/epoch=39.ckpt")
    ap.add_argument("--config", default="/home/ma-user/work/lam_runs/vggt_vae_libero/version_0/config.yaml")
    ap.add_argument("--n-batches", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--val-tail-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda"

    model = load_latent_action_model(args.ckpt, args.config).to(dev).eval()
    print(f"[ctrl] encoder={type(model.vision_encoder).__name__} input_dim={model.input_dim}")

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
        ds, batch_size=args.batch_size, num_workers=8,
        collate_fn=lambda b: lam_collate(b, max_state_dim=32), drop_last=True,
    )

    all_dec_in, all_tgt, all_recon, all_z, all_traj, all_base = [], [], [], [], [], []
    it = iter(loader)
    for i in range(args.n_batches):
        try:
            batch = next(it)
        except StopIteration:
            print(f"[ctrl] loader exhausted after {i} batches")
            break
        videos = batch["videos"].to(dev)
        v1, v2 = gpu_two_view_video_aug(videos, output_size=(256, 256), training=False, dual_view_aug=False)
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
        if (i + 1) % 20 == 0:
            print(f"[ctrl] pass1 {i+1}/{args.n_batches}")

    dec_in = torch.cat(all_dec_in)
    tgt = torch.cat(all_tgt)
    recon = torch.cat(all_recon)
    Z = torch.cat(all_z)
    base_idx = torch.cat(all_base).numpy()
    eps_all = np.array([int(t) for t in all_traj])
    N = dec_in.shape[0]
    print(f"\n[ctrl] N={N} samples | z dim={Z.shape[1]} | {len(np.unique(eps_all))} episodes")

    flat = tgt.reshape(-1, tgt.shape[-1])
    mu, sd = flat.mean(0), flat.std(0).clamp_min(1e-6)
    T_ = standardize(tgt, mu, sd)
    R_ = standardize(recon, mu, sd)
    I_ = standardize(dec_in, mu, sd)
    mse = lambda a, b: (a - b).pow(2).mean().item()
    mse_pred, mse_identity = mse(R_, T_), mse(I_, T_)
    print(f"[ctrl] mse_pred={mse_pred:.4f}  mse_identity={mse_identity:.4f}  mse_mean={mse(torch.zeros_like(T_), T_):.4f}")

    def decode_with(zs):
        """Re-run only the decoder with the given latents. Returns standardized pred."""
        outs = []
        for s in range(0, N, args.batch_size):
            e = min(s + args.batch_size, N)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                r = model.decoder(dec_in[s:e].to(dev), zs[s:e].to(dev).unsqueeze(1)).float().cpu()
            outs.append(standardize(r, mu, sd))
        return torch.cat(outs)

    # ---- B. delta_z variants ----------------------------------------------
    print("\n=== 对照 B: delta_z 的四种 z 替换 ===")
    g = torch.Generator().manual_seed(args.seed)

    # (b1) shuffle across the whole set -- the original probe
    R_glob = decode_with(Z[torch.randperm(N, generator=g)])

    # (b2) shuffle WITHIN episode -- holds scene/task fixed, only motion differs
    by_ep = defaultdict(list)
    for i, e in enumerate(eps_all):
        by_ep[int(e)].append(i)
    perm_w = np.arange(N)
    n_swappable = 0
    for e, idxs in by_ep.items():
        if len(idxs) < 2:
            continue
        a = np.array(idxs)
        b = a.copy()
        # derangement-ish: roll by 1 within the episode
        perm_w[a] = np.roll(b, 1)
        n_swappable += len(idxs)
    R_within = decode_with(Z[torch.from_numpy(perm_w)])

    # (b3) z = 0  and  (b4) z ~ N(0, I)
    R_zero = decode_with(torch.zeros_like(Z))
    R_rand = decode_with(torch.randn(Z.shape, generator=g) * Z.std())

    rows = [
        ("真实 z",              mse_pred,            R_),
        ("跨全集打乱 z",         mse(R_glob, T_),     R_glob),
        ("同 episode 内打乱 z",  mse(R_within, T_),   R_within),
        ("z = 0",              mse(R_zero, T_),     R_zero),
        ("z ~ 高斯噪声",         mse(R_rand, T_),     R_rand),
        ("抄袭 u_t (identity)",  mse_identity,        I_),
    ]
    print(f"  {'替换方式':22s} {'mse':>8s} {'delta_z':>9s} {'与u_t距离':>10s}")
    for name, m, arr in rows:
        d = m - mse_pred
        di = mse(arr, I_)
        print(f"  {name:22s} {m:8.4f} {d:9.4f} {di:10.4f}")
    print(f"  可同-episode 交换的样本: {n_swappable}/{N}")
    print("  注: '与u_t距离'=0 表示解码器退化成原样抄袭输入")

    # ---- A. z vs u_t as action predictors ---------------------------------
    print("\n=== 对照 A: z 与 u_t 谁能解出动作 (同为32维, 同 episode 切分) ===")
    import pyarrow.parquet as pq

    tbl = pq.read_table(
        "/home/ma-user/work/lam_datasets/libero_merged_no_noops_20hz/data/chunk-000/file-000.parquet",
        columns=["episode_index", "frame_index", "action"],
    )
    ep_arr = np.asarray(tbl.column("episode_index"))
    fr_arr = np.asarray(tbl.column("frame_index"))
    act_arr = np.stack(tbl.column("action").to_numpy(zero_copy_only=False))
    key2row = {(int(e), int(f)): i for i, (e, f) in enumerate(zip(ep_arr, fr_arr))}

    STRIDE = 32
    keep, Y_rows = [], []
    for i in range(N):
        e, b = int(eps_all[i]), int(base_idx[i])
        chunk, ok = [], True
        for k in range(0, STRIDE, 4):
            r = key2row.get((e, b + k))
            if r is None:
                ok = False
                break
            chunk.append(act_arr[r])
        if ok:
            keep.append(i)
            Y_rows.append(np.concatenate(chunk))
    keep = np.array(keep)
    Y = np.asarray(Y_rows, dtype=np.float64)
    eps_k = eps_all[keep]
    print(f"  对齐样本 {len(keep)} | action {Y.shape[1]}维")

    uniq = np.unique(eps_k)
    rng = np.random.RandomState(0)
    rng.shuffle(uniq)
    tr_eps = set(uniq[: max(1, int(len(uniq) * 0.7))].tolist())
    tr = np.array([e in tr_eps for e in eps_k])
    te = ~tr
    print(f"  episode 级切分: train {tr.sum()} / test {te.sum()}, {len(uniq)} episodes")

    Xz = Z[keep].numpy().astype(np.float64)

    # u_t -> 32 dims via PCA fit on TRAIN episodes only (no leakage)
    U = dec_in[keep, 0].mean(dim=1).numpy().astype(np.float64)   # [n, 2048] token-mean
    Um = U[tr].mean(0)
    Uc = U - Um
    _, _, Vt = np.linalg.svd(Uc[tr], full_matrices=False)
    Xu = Uc @ Vt[:32].T

    # also the delta feature u_t itself at full token resolution, PCA-32
    Uf = dec_in[keep, 0].reshape(len(keep), -1).numpy().astype(np.float64)
    Ufm = Uf[tr].mean(0)
    Ufc = Uf - Ufm
    _, _, Vt2 = np.linalg.svd(Ufc[tr], full_matrices=False)
    Xuf = Ufc @ Vt2[:32].T

    r2_z, _ = ridge_r2(Xz, Y, tr, te, "z (32维, 本模型的latent action)")
    r2_u, _ = ridge_r2(Xu, Y, tr, te, "PCA32(u_t token均值)  [对照]")
    r2_uf, _ = ridge_r2(Xuf, Y, tr, te, "PCA32(u_t 全token)   [对照]")
    r2_cat, _ = ridge_r2(np.concatenate([Xz, Xuf], 1), Y, tr, te, "z + PCA32(u_t) 拼接")

    print("\n=== 汇总 ===")
    print(f"  z 单独            R^2 = {r2_z:.4f}")
    print(f"  u_t 单独 (最好)    R^2 = {max(r2_u, r2_uf):.4f}")
    print(f"  增量 (z - u_t)          = {r2_z - max(r2_u, r2_uf):+.4f}")
    print(f"  拼接              R^2 = {r2_cat:.4f}")
    print(f"  delta_z 跨全集          = {mse(R_glob, T_) - mse_pred:.4f}")
    print(f"  delta_z 同episode内      = {mse(R_within, T_) - mse_pred:.4f}  <- 排除场景/任务混淆")


if __name__ == "__main__":
    main()
