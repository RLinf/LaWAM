"""Export a slim, inference-only artifact from a 7.7 GB Lightning training checkpoint.

The training checkpoint is 7.7 GB, of which 5.2 GB is AdamW optimizer state and
2.58 GB is fp32 weights. Nobody downloading this needs the optimizer state, and
Stage 2 needs only the decoder -- the IDM encoder exists to produce z during
Stage 1 training and is discarded afterwards.

So this writes three safetensors files:

  lawm_decoder.safetensors    the world model. This is what Stage 2 imports.
  lam_idm_encoder.safetensors the inverse dynamics model, for anyone who wants to
                              re-derive z from an (o_t, o_T) pair or fine-tune.
  state_head.safetensors      the auxiliary state-delta head.

Weights stay fp32: these are small enough that halving them buys little, and the
world model is consumed as a feature predictor where fp32 costs nothing at
inference. The frozen VGGT-1B backbone is NOT included -- it is an unmodified
third-party checkpoint that users should pull from its own source.
"""

import argparse
import hashlib
import json
import os
import shutil
import tempfile

import torch
from safetensors.torch import save_file

GROUPS = {
    "lawm_decoder": "decoder.",
    "lam_idm_encoder": "encoder.",
    "state_head": "state_decoder.",
}


def save_via_shm(tensors, dest, metadata):
    """safetensors serializes with mmap, which the root-squash NFS home rejects
    outright (EPERM). Write to tmpfs first, then copy the finished bytes over.
    Same class of problem as [[dataset-yhw-nfs-no-rename]].
    """
    tmpdir = tempfile.mkdtemp(dir="/dev/shm", prefix="lam_export_")
    try:
        tmp = os.path.join(tmpdir, os.path.basename(dest))
        save_file(tensors, tmp, metadata=metadata)
        shutil.copyfile(tmp, dest)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/ma-user/work/lam_runs/vggt_vae_libero/checkpoints/epoch=39.ckpt")
    ap.add_argument("--out", default="/home/ma-user/work/lam_release/weights")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"[export] loading {args.ckpt}")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    print(f"[export] epoch={ck.get('epoch')} global_step={ck.get('global_step')}")

    # strip the Lightning 'lam.' prefix; drop the frozen backbone if present
    clean = {}
    for k, v in sd.items():
        kk = k[4:] if k.startswith("lam.") else k
        if kk.startswith("vision_encoder."):
            continue
        clean[kk] = v.contiguous()

    manifest = {
        "source_checkpoint": os.path.basename(args.ckpt),
        "epoch": int(ck.get("epoch", -1)),
        "global_step": int(ck.get("global_step", -1)),
        "dtype": "float32",
        "vision_encoder": "VGGT-1B (frozen, NOT included -- fetch from facebook/VGGT-1B)",
        "files": {},
    }

    used = set()
    for name, prefix in GROUPS.items():
        part = {k: v for k, v in clean.items() if k.startswith(prefix)}
        if not part:
            print(f"[export] WARNING: no tensors matched prefix '{prefix}'")
            continue
        used |= set(part)
        path = os.path.join(args.out, f"{name}.safetensors")
        save_via_shm(part, path, {"format": "pt", "component": name})
        n = sum(v.numel() for v in part.values())
        sz = os.path.getsize(path)
        manifest["files"][f"{name}.safetensors"] = {
            "prefix": prefix,
            "tensors": len(part),
            "params": n,
            "params_readable": f"{n/1e6:.1f}M",
            "bytes": sz,
            "size_readable": f"{sz/1e6:.1f} MB",
            "sha256": sha256(path),
        }
        print(f"[export] {name:20s} {len(part):4d} tensors  {n/1e6:7.1f} M  {sz/1e6:8.1f} MB")

    leftover = {k: v for k, v in clean.items() if k not in used}
    if leftover:
        # vq/* buffers and anything else that doesn't fit a group -- keep them so
        # the export is lossless rather than silently dropping state.
        path = os.path.join(args.out, "misc.safetensors")
        save_via_shm(leftover, path, {"format": "pt", "component": "misc"})
        manifest["files"]["misc.safetensors"] = {
            "keys": sorted(leftover),
            "bytes": os.path.getsize(path),
            "sha256": sha256(path),
        }
        print(f"[export] misc: {sorted(leftover)}")

    total_in = os.path.getsize(args.ckpt)
    total_out = sum(os.path.getsize(os.path.join(args.out, f)) for f in manifest["files"])
    manifest["total_bytes"] = total_out
    manifest["total_readable"] = f"{total_out/1e9:.2f} GB"
    manifest["original_checkpoint_bytes"] = total_in
    manifest["shrink_factor"] = round(total_in / max(total_out, 1), 2)

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[export] 训练 ckpt {total_in/1e9:.2f} GB -> 发布权重 {total_out/1e9:.2f} GB "
          f"({manifest['shrink_factor']}× 缩小)")
    print(f"[export] Stage 2 只需 lawm_decoder.safetensors = "
          f"{manifest['files']['lawm_decoder.safetensors']['size_readable']}")
    print(f"[export] wrote manifest to {args.out}/manifest.json")


if __name__ == "__main__":
    main()
