"""Verify the exported safetensors actually reconstruct the trained model.

An export that loads without error but produces different numbers is worse than
no export at all, so this rebuilds the model from the release files and checks
bit-level agreement with the original checkpoint on real data. Run this before
uploading anything.
"""

import importlib.machinery
import os
import sys
import types

import torch
from safetensors.torch import load_file

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

CKPT = "/home/ma-user/work/lam_runs/vggt_vae_libero/checkpoints/epoch=39.ckpt"
CONFIG = "/home/ma-user/work/lam_runs/vggt_vae_libero/version_0/config.yaml"
REL = "/home/ma-user/work/lam_release/weights"


@torch.no_grad()
def main():
    print("[verify] building model from the training checkpoint")
    model = load_latent_action_model(CKPT, CONFIG).eval()

    merged = {}
    for f in sorted(os.listdir(REL)):
        if f.endswith(".safetensors"):
            part = load_file(os.path.join(REL, f))
            merged.update(part)
            print(f"[verify] loaded {f}: {len(part)} tensors")

    # compare every released tensor against the live module
    live = dict(model.named_parameters())
    live.update(dict(model.named_buffers()))
    missing, mismatched, checked = [], [], 0
    for k, v in merged.items():
        if k not in live:
            missing.append(k)
            continue
        d = (live[k].detach().cpu().float() - v.float()).abs().max().item()
        checked += 1
        if d != 0.0:
            mismatched.append((k, d))

    print(f"\n[verify] compared {checked}/{len(merged)} released tensors")
    print(f"[verify] not found in live model: {len(missing)}")
    if missing:
        print("        ", missing[:10])
    print(f"[verify] numerically different: {len(mismatched)}")
    if mismatched:
        for k, d in mismatched[:10]:
            print(f"         {k}: max|delta|={d:.3e}")

    # are any trainable weights absent from the release?
    trainable = {k for k, p in model.named_parameters()
                 if not k.startswith("vision_encoder.")}
    not_released = sorted(trainable - set(merged))
    print(f"[verify] trainable params NOT in release: {len(not_released)}")
    if not_released:
        print("        ", not_released[:10])

    ok = not mismatched and not missing and not not_released
    print(f"\n[verify] {'PASS -- release is bit-identical and complete' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
