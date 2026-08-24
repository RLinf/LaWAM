"""End-to-end forward through the real LatentLAMModel with the VGGT encoder.

Lightning/wandb are not installed in this env yet, and `latent_action_model.core.__init__`
imports them transitively. They are irrelevant to `LatentLAMModel`, so stub them out.
"""
import importlib.machinery, sys, types, torch

REPO = "/home/ma-user/work/dataset/xxd-dataset/dataset_yhw/WAM/LaWAM_official"
sys.path.insert(0, REPO)


def _stub(name, **attrs):
    mod = types.ModuleType(name)
    # accelerate probes `importlib.util.find_spec("wandb")`, which raises if
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
_stub("wandb", Image=lambda *a, **k: None, log=lambda *a, **k: None, Video=lambda *a, **k: None)

from latent_action_model.core.lam_model import LatentLAMModel  # noqa: E402

WEIGHTS = "/home/ma-user/work/dataset/xxd-dataset/dataset_yhw/StarVLA/playground/Pretrained_models/VGGT-1B"

model = LatentLAMModel(
    dim=1024,
    num_heads=16,
    ffn_expansion_factor=4,
    enc_layers=24,
    dec_layers=12,
    codebook_size=32,
    code_dim=32,
    max_state_dim=32,
    num_frames=2,
    num_queries=1,
    vq_type="vae",
    vq_kwargs={"layer_norm": True},
    norm_latents=True,
    norm_latents_type="ln",
    vision_model_id=WEIGHTS,
    enc_add_state=False,
    enc_modal_mask=True,
    latent_layer_to_use=-2,
    multi_input=False,
    num_embodiments=32,
    image_hw=(256, 256),
    patch_size=16,
    dropout=0.0,
).cuda()
model.train()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"input_dim (VGGT) : {model.input_dim}")
print(f"grid             : {model.grid_height}x{model.grid_width}")
print(f"trainable params : {trainable/1e6:.1f} M")
print(f"frozen params    : {frozen/1e6:.1f} M")
assert model.input_dim == 2048
assert not model.vision_encoder.model.training, "vision encoder left training mode"

B, T = 2, 2
videos = torch.randn(B, T, 3, 256, 256, device="cuda")      # enc stream (aug A)
dec_videos = torch.randn(B, T, 3, 256, 256, device="cuda")  # dec stream (aug B)
states = torch.randn(B, T, 32, device="cuda")
state_mask = torch.ones(B, T, 32, device="cuda")
emb_ids = torch.ones(B, dtype=torch.long, device="cuda")

with torch.autocast("cuda", dtype=torch.bfloat16):
    out = model(videos, states, dec_videos, state_mask=state_mask, embodiment_ids=emb_ids)

if isinstance(out, dict):
    items = {k: (tuple(v.shape) if torch.is_tensor(v) else v) for k, v in out.items()}
else:
    items = [tuple(v.shape) if torch.is_tensor(v) else v for v in out]
print("forward output:", items)

# Unpack in the same order lam_lightinng.py:605 does.
recon, dec_in, tgt, perplexity, indices, delta_s_pred, features, _, entropy_loss, vq_loss = out
assert recon.shape == tgt.shape == (B, 1, 256, 2048), (recon.shape, tgt.shape)
assert delta_s_pred.shape == (B, 32), delta_s_pred.shape
print(f"recon/tgt shape  : {tuple(recon.shape)}  (VGGT feature space)")
print(f"delta_s_pred     : {tuple(delta_s_pred.shape)}  (aux state-delta head)\nlatent z         : {tuple(_.shape)}")

# Same loss the Lightning module builds: smooth_l1 in feature space + KL.
loss = torch.nn.functional.smooth_l1_loss(recon.float(), tgt.float()) + vq_loss.float()
assert loss is not None, "could not locate a scalar loss in the forward output"
assert torch.isfinite(loss), f"loss is not finite: {loss}"
loss.float().backward()

grads = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
gnorm = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in grads))
print(f"loss={loss.item():.6f} | params with grad={len(grads)} | grad_norm={gnorm.item():.4f}")
assert torch.isfinite(gnorm) and gnorm > 0, "bad gradient"
assert all(p.grad is None for p in model.vision_encoder.parameters()), "VGGT received gradients"
print("\nEND-TO-END FORWARD/BACKWARD OK")
