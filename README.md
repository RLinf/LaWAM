# LaWAM Stage 1 · 把 DINOv3 换成冻结的 VGGT-1B

这是 [RLinf/LaWAM](https://github.com/RLinf/LaWAM) 的 fork。**只改了 Stage 1**：把原本冻结的
DINOv3 视觉编码器换成**冻结的 VGGT-1B**（3D 几何基础模型，2048 维），在 LIBERO 上重训 40 epoch。
Stage 2 / 策略训练部分未触碰。

上游原始 README 完整保留在 [`README_UPSTREAM.md`](README_UPSTREAM.md)。

| | |
|---|---|
| 视觉编码器 | `facebook/VGGT-1B`，**冻结**，patch-14 @ 518 → 37×37 池化到 16×16 = 256 token |
| 训练数据 | `jialei02/libero_merged_no_noops_20hz`（LeRobot v3.0） |
| 训练规模 | 8×A100-80G，约 56 h，40 epoch，global_step 42720 |
| 权重 | **[🤗 YuanhaoXD/Lam_VGGT](https://huggingface.co/YuanhaoXD/Lam_VGGT)**（2.58 GB；Stage 2 只需其中 924 MB 的 decoder） |
| 结果 | val recon 0.130 → 0.0561，cos_sim 0.992，`z2action_r2` = 0.5523 |

## 文档

| 文件 | 内容 |
|---|---|
| **[`LAWAM_VGGT_STAGE1_REPORT.md`](LAWAM_VGGT_STAGE1_REPORT.md)** | 主报告。配置、训练曲线、评估、§11 可视化逐图解读 |
| [`LAWAM_VGGT_PROBE_ANALYSIS.md`](LAWAM_VGGT_PROBE_ANALYSIS.md) | 三个探针（recon / z2action / z-sensitivity）的分析 |
| [`experiments/`](experiments/) | 训练曲线、原始日志、TB event、配置快照、样本图 |

**报告里最值得看的是 §11.3** —— 一个反转了早期乐观结论的负面结果：
用"扣掉共享偏置"的动作热力图证明 latent action `z` 控制的是一块**固定的**工作区区域，
**并不跟着机械臂走**（原始比值 2.38 → 扣偏置后仅 1.17；跨样本相关 0.94 对照组 0.53）。
早期那个看起来很漂亮的 2.43 是被共享偏置伪装出来的。

## 改了什么

对上游的全部改动（基线 `4ea6fda`）：

```
 latent_action_model/core/vjepa_encoder.py     +248   ← VGGTEncoder 主体
 latent_action_model/core/lam_lightinng.py      +28   ← ckpt 瘦身钩子（剥掉冻结主干）
 latent_action_model/core/lam_model.py          +10
 latent_action_model/config/vggt_vae.yaml       新增   ← 训练配置，改动处标了 # [VGGT]
 tools/*.py                                     新增 10 个（评估 + 可视化 + 权重导出）
 latent_action_model/train{,_ddp}.sh            见下方安全说明
```

### 三个实现上的坑

1. **VGGT 的交替注意力会泄漏未来。** global attention 会把所有 S 帧混在一起，
   所以 `VGGTEncoder` 强制 **S=1**、走 batch 维堆叠。实测 batch 维堆叠逐位一致
   （`max|Δ|=0.0`），而 S 维堆叠 `max|Δ|=20.3, cos=0.9796` —— 未来帧信息漏进了当前帧。
2. **特征必须标准化后再比。** 每个 VGGT token 约 75% 是共享常量
   （‖mean token‖=76.4 vs 残差 43.5），直接算余弦相似度会得到**排序反转**的结论。
   所有 MSE / cosine 都在标准化后计算，PCA 必须中心化。
3. **`vq_type: "vae"` 下 `perplexity=0.000` 是正常的**，连续隐变量没有码本。

## 复现

```bash
# 训练
REPO_DIR=$PWD PY=/path/to/python NPROC=8 ./launch_vggt_lam.sh

# 评估三探针
python tools/eval_lam_probes.py --n-batches 40

# 出图
python tools/viz_vggt_lam.py --n-samples 6 --seed 0
python tools/viz_action_heatmap.py --batch 16 --n-show 6 --shuffle
python tools/plot_training_curves.py
```

数据用 HF 上的 **LeRobot v3.0** 版 `jialei02/libero_merged_no_noops_20hz`（1.97 GB）。
v2.1 的 LIBERO 上游 fork 不认。国内可走 `HF_ENDPOINT=https://hf-mirror.com`。

> ⚠️ **`tools/*.py` 和 `config/vggt_vae.yaml` 里留着作者机器上的绝对路径。**
> 这些脚本是原样提交的（就是跑出上面那些结果的那一份，没有为了发布而改写，
> 以免代码与实验记录脱节）。换机器跑之前至少要改：
> `vggt_vae.yaml` 的 `vision_model_id` / `data_root_dir` / `dirpath` / `save_dir`，
> 以及各 `tools/*.py` 顶部的 `REPO` / `WEIGHTS` 常量（多数也可用 argparse 参数覆盖）。

## ⚠️ 安全：上游泄漏的 W&B key 已在本 fork 中替换

上游 `latent_action_model/train.sh:40` 和 `train_ddp.sh:37` 各有一个**明文真实
`WANDB_API_KEY`**（两处是同一个 key）。这是上游 RLinf/LaWAM 自己泄的，非本 fork 引入，
本次训练全程 `WANDB_MODE=disabled` 从未使用它。

本 fork 已把两处都换成 `${WANDB_API_KEY:?...}` 占位符。但**该 key 仍存在于本仓库的
git 历史中**（继承自上游），也仍在上游仓库里公开着 —— 建议上游持有者轮换它。

## 许可

上游代码沿用 StarVLA 的 MIT。但注意 **VGGT-1B 是 `cc-by-nc-4.0`（禁止商用）**，
因此在 HF 上发布的那批权重按更严格的一方标注为 `cc-by-nc-4.0`。

## 引用

本工作基于 [LaWAM](https://arxiv.org/abs/2606.15768) 与
[VGGT](https://arxiv.org/abs/2503.11651)（`arXiv:2503.11651`），请一并引用。
