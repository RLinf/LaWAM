# 实验产物

本次 Stage 1 训练（VGGT-1B 版本）的原始记录与分析产物。

| 路径 | 内容 |
|---|---|
| `figures/training_curves.png` | 四面板收敛曲线；`curves.csv` 是逐 epoch 数值表 |
| `logs/train.log.gz` | 完整训练日志（解压后 40 MB） |
| `logs/events.out.tfevents.*` | TensorBoard event（val 指标，逐 epoch） |
| `configs/` | 训练时实际生效的配置快照（`config.yaml` / `hparams.yaml` / `vggt_vae.yaml`） |
| `samples/sheet1-4.png` | 抽样检查用的样本组，每张含多组机械臂位置差异较大的样本 |
| `manifest.json` | HF 上那批权重的 sha256 / 参数量 / 文件大小 |
| `WEIGHTS.md` | 权重分组说明（哪个文件对应哪个子模块） |

## 训练曲线

`figures/training_curves.png` 的数据来自两处：TensorBoard event 提供 val（逐 epoch），
而 train 曲线是从 40 MB 日志的 tqdm 进度条里正则抽出的 **42721 个点** —— Lightning
没把 train 指标写进 TB。

| 面板 | 内容 |
|---|---|
| 左上 | recon loss 对数轴，train + val 叠加，标注最低点 ep24 = 0.0526 |
| 右上 | val 线性轴，高亮 ep24 之后的震荡带 |
| 左下 | train/val 并列 + 比值曲线（过拟合检查） |
| 右下 | cos_sim 0.970→0.992、state_loss 降 56× |

关键数字：**val 0.130 → 0.0561，train/val gap 1.43×，全程无发散。**

## 两个读数陷阱

1. **训练期 val 集只有 109 个样本**（`val_tail_ratio: 0.001`），±0.005 纯属噪声，
   不要拿曲线上的小抖动讲故事。真正可信的评估用 5% 留出（84 episodes / 640 样本），
   见主报告 §4。
2. 日志里 `PermissionError: Operation not permitted` 出现了 **416 次**，全部无害 ——
   是 root-squash NFS 拒绝 `os.replace`，不影响训练。

## 权重

不在本仓库（GitHub 不适合放 GB 级二进制），在
**[🤗 YuanhaoXD/Lam_VGGT](https://huggingface.co/YuanhaoXD/Lam_VGGT)**。
`manifest.json` 里的 sha256 可用于校验下载完整性。

冻结的 VGGT-1B（909 M）两边都没有 —— 未经修改的第三方权重，
从 `facebook/VGGT-1B` 自取即可。
