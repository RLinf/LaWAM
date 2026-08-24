---
license: mit
tags:
  - robotics
  - world-model
  - latent-action-model
  - vggt
  - libero
  - lawam
library_name: pytorch
---

# LaWAM Stage 1 — VGGT-1B 特征空间上的 Latent World Model

复现 [LaWAM](https://github.com/RLinf/LaWAM) 的 **Stage 1**，其它一切不变，
**只把冻结的视觉编码器从 DINOv3 换成 VGGT-1B**。在 LIBERO 上训练 40 epoch。

世界模型预测的是 **1.6 秒后的 VGGT 特征**（不是像素），并以一个 32 维的
latent action `z` 为条件。

## 文件

| 文件 | 大小 | 参数 | 用途 |
|---|---|---|---|
| `lawm_decoder.safetensors` | 923.7 MB | 230.9 M | **世界模型本体**。Stage 2 只需要这个 |
| `lam_idm_encoder.safetensors` | 1243.0 MB | 310.7 M | IDM，从 (o_t, o_T) 反推 z。Stage 1 训完即弃 |
| `state_head.safetensors` | 411.6 MB | 102.9 M | 辅助 state-delta 头 |
| `misc.safetensors` | 9.2 KB | — | VAE 的 mu / logvar / pre_norm |

fp32。每个文件的 sha256 见 `manifest.json`。

**不含冻结的 VGGT-1B 主干**（909 M）—— 那是未修改的第三方权重，
请从 [`facebook/VGGT-1B`](https://huggingface.co/facebook/VGGT-1B) 自行获取。

## 评估结果

在 **84 个留出 episode / 640 样本**上，全部在标准化特征空间计算：

| 指标 | 结果 | 判据 |
|---|---|---|
| `mse_pred` | **0.1586** | vs identity 0.5496 / mean 1.0000 → 好 3.5× |
| `delta_z`（打乱 z 后变差多少） | **0.3822** | 目标 > 0.1 ✅ |
| `z2action_r2`（z → 真实动作线性探针） | **0.5523** | 目标 > 0.30 ✅ |

**最干净的一个结果**：把 z 打乱后 MSE 掉到 0.5408，几乎正好等于恒等基线 0.5496 ——
即模型相对"抄袭输入"的**全部增益都来自 z**。

## 已知局限（请一并阅读）

1. **数据量比论文少约 700×**（273 K 帧 vs 3000 h + 1500 h），
   只有 LIBERO 单一本体（Franka），**跨本体泛化未验证**。
2. **z 的空间作用范围有很强的固定先验。** 把观测固定、扫过 16 个不同的 z 取逐图块标准差，
   得到的热力图在**跨 episode** 的不同场景之间相关性高达 **0.922**，
   而真实运动图之间只有 0.522。扣掉这个共享分量后，
   "运动区/静止区"敏感度比从 2.65 掉到 **1.22**。
   → z 主要控制"画面里机械臂常驻的那片区域"，而非逐帧精确跟随物体。
   32 维 z 要驱动 256 图块 × 2048 维输出，带宽本就不够。
3. τ = 1.6 s（取自官方 config；论文正文写 1.2 s）；β = 5e-5（类默认值，论文写 1e-5）。
4. 单视角（官方 dataloader 行为），无人类第一人称视频。

## 用法

```python
from safetensors.torch import load_file
sd = load_file("lawm_decoder.safetensors")   # 键名带 "decoder." 前缀
```

完整加载路径、`VGGTEncoder` 实现、训练配置和复现脚本见 GitHub 仓库。

## 训练

8×A100-80G，约 56 h，40 epoch，batch 32。
`val/recon_loss` 0.130 → 0.0561，`cos_sim` 0.970 → 0.992，train/val gap 1.43×，无过拟合。

⚠️ VGGT 的 global attention **跨帧混合**，把多帧拼进 S 维会泄漏未来
（实测 `max|Δ| = 20.3`）。本实现内部强制 **S=1**，所有帧压到 batch 维（实测逐位相同）。
这是承重的实现细节。
