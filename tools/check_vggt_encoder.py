"""Standalone checks for VGGTEncoder (no Lightning needed)."""
import importlib.util, sys, torch

# Load the module by path: latent_action_model/core/__init__.py pulls in Lightning,
# which these checks do not need.
_spec = importlib.util.spec_from_file_location(
    "vjepa_encoder",
    "/home/ma-user/work/dataset/xxd-dataset/dataset_yhw/WAM/LaWAM_official/latent_action_model/core/vjepa_encoder.py",
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["vjepa_encoder"] = _mod
_spec.loader.exec_module(_mod)
build_vision_encoder = _mod.build_vision_encoder

WEIGHTS = "/home/ma-user/work/dataset/xxd-dataset/dataset_yhw/StarVLA/playground/Pretrained_models/VGGT-1B"

enc, dim = build_vision_encoder(WEIGHTS, num_latent_layers=1, norm_layer_type="ln", enable_norm=True)
print("feature_dim:", dim, "| image_size:", enc.image_size, "| patch_size:", enc.patch_size)

dev = "cuda"
enc = enc.to(dev)
enc.train()  # must stay eval
assert not enc.model.training, "train() did not force eval"
assert all(not p.requires_grad for p in enc.model.parameters()), "params not frozen"
print("[ok] frozen + eval-locked")

torch.manual_seed(0)
# video_aug output convention: ImageNet-normalized float, [B, T, C, 256, 256]
B, T = 3, 4
imgs = (torch.rand(B, T, 3, 256, 256, device=dev) - 0.449) / 0.226

with torch.autocast("cuda", dtype=torch.bfloat16):
    out = enc.encode(imgs, n=-2)
print("shape:", tuple(out.shape), out.dtype)
assert out.shape == (B, T, 256, 2048), out.shape
print("[ok] shape [B,T,256,2048]")

# (b) batch-dim stacking must be bit-identical to encoding the sub-batch alone
with torch.autocast("cuda", dtype=torch.bfloat16):
    out_sub = enc.encode(imgs[:1], n=-2)
delta = (out[:1].float() - out_sub.float()).abs().max().item()
print("max|delta| batch-stack vs alone:", delta)
assert delta == 0.0, f"batch stacking is not independent: {delta}"
print("[ok] frames are mutually independent (no temporal leak)")

# (c) latent layer selection actually differs
with torch.autocast("cuda", dtype=torch.bfloat16):
    out_last = enc.encode(imgs, n=-1)
print("mean|f(-1) - f(-2)|:", (out_last.float() - out.float()).abs().mean().item())
assert not torch.allclose(out_last, out), "-1 and -2 returned the same layer"
print("[ok] -1 (block 23) != -2 (block 17)")

# (d) 4D input path
with torch.autocast("cuda", dtype=torch.bfloat16):
    out4 = enc.encode(imgs[0], n=-2)
assert out4.shape == (T, 1, 256, 2048), out4.shape
print("[ok] 4D input ->", tuple(out4.shape))

# (e) de-normalization round-trip: feeding normalized pixels must equal feeding
#     the raw [0,1] image straight to the aggregator at 518.
import torch.nn.functional as F
raw01 = torch.rand(1, 3, 256, 256, device=dev)
norm = (raw01 - enc._imagenet_mean.to(dev)) / enc._imagenet_std.to(dev)
with torch.autocast("cuda", dtype=torch.bfloat16):
    via_enc = enc.encode(norm.unsqueeze(1), n=-1)
    up = F.interpolate(raw01, size=(518, 518), mode="bilinear", align_corners=False)
    tl, ps = enc.model.aggregator(up.unsqueeze(1))
    ref = tl[-1][:, 0, ps:, :]
    ref = F.adaptive_avg_pool2d(ref.reshape(1, 37, 37, 2048).permute(0, 3, 1, 2), (16, 16))
    ref = ref.flatten(2).transpose(1, 2)
    ref = enc.latent_norms[0](ref)
print("max|delta| denorm round-trip:", (via_enc[0].float() - ref.float()).abs().max().item())
print("\nALL CHECKS PASSED")
