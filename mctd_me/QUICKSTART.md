# MCTD-ME 快速上手指南

**硬件前提：** 48 GB 显存 GPU（如魔改 RTX 4090）
**存储路径：** `/root/rivermind-data/`
**网络环境：** 中国大陆，使用 HuggingFace 镜像

---

## 目录

1. [目录结构约定](#1-目录结构约定)
2. [创建 Conda 环境](#2-创建-conda-环境)
3. [配置 HuggingFace 镜像](#3-配置-huggingface-镜像)
4. [下载模型权重](#4-下载模型权重)
5. [下载基准数据集](#5-下载基准数据集)
6. [快速验证（冒烟测试）](#6-快速验证冒烟测试)
7. [完整任务运行](#7-完整任务运行)
8. [常用参数速查](#8-常用参数速查)

---

## 1. 目录结构约定

```
/root/rivermind-data/
├── models/
│   ├── dplm_150m/        # DPLM-2 150M (~600 MB)
│   ├── dplm_650m/        # DPLM-2 650M (~2.6 GB)
│   ├── dplm_3b/          # DPLM-2 3B   (~12 GB)
│   ├── esmfold_v1/       # ESMFold     (~2.5 GB)
│   └── ProteinMPNN/      # ProteinMPNN (~50 MB, git clone)
└── data/
    ├── cameo2022/        # CAMEO 2022 benchmark
    └── pdb_date/         # PDB date-split subset
```

---

## 2. 创建 Conda 环境

```bash
cd /home/dds/Protein/mctd_me

conda env create -f environment.yml
conda activate mctd_me

pip install -e .
```

---

## 3. 配置 HuggingFace 镜像

**临时生效（当前终端）：**

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/rivermind-data/models
```

**永久写入（推荐）：**

```bash
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
echo 'export HF_HOME=/root/rivermind-data/models'  >> ~/.bashrc
source ~/.bashrc
```

---

## 4. 下载模型权重

确保已激活环境并设置好镜像变量后执行：

```bash
mkdir -p /root/rivermind-data/models

# DPLM-2 150M（约 600 MB，快速验证用）
HF_ENDPOINT=https://hf-mirror.com \
huggingface-cli download airkingbd/dplm_150m \
    --local-dir /root/rivermind-data/models/dplm_150m \
    --local-dir-use-symlinks False

# DPLM-2 650M（约 2.6 GB）
HF_ENDPOINT=https://hf-mirror.com \
huggingface-cli download airkingbd/dplm_650m \
    --local-dir /root/rivermind-data/models/dplm_650m \
    --local-dir-use-symlinks False

# DPLM-2 3B（约 12 GB，完整实验用）
HF_ENDPOINT=https://hf-mirror.com \
huggingface-cli download airkingbd/dplm_3b \
    --local-dir /root/rivermind-data/models/dplm_3b \
    --local-dir-use-symlinks False

# ESMFold（约 2.5 GB，用作结构预测 critic）
HF_ENDPOINT=https://hf-mirror.com \
huggingface-cli download facebook/esmfold_v1 \
    --local-dir /root/rivermind-data/models/esmfold_v1 \
    --local-dir-use-symlinks False

# ProteinMPNN（逆折叠专家，git clone）
cd /root/rivermind-data/models
git clone https://ghfast.top/https://github.com/dauparas/ProteinMPNN.git
# 若上面镜像失败，改用：
# git clone https://github.com/dauparas/ProteinMPNN.git
```

> 下载中断可直接重新执行相同命令，支持断点续传。

---

## 5. 下载基准数据集

```bash
conda activate mctd_me
cd /home/dds/Protein/mctd_me

python scripts/download_data.py \
    --all \
    --output /root/rivermind-data/data/
```

---

## 6. 快速验证（冒烟测试）

**目的：** 验证整条链路（模型加载 → masked diffusion → ESMFold critic → MCTS → 输出）全部正常，约 3–5 分钟完成。

```bash
conda activate mctd_me
cd /home/dds/Protein/mctd_me
export HF_ENDPOINT=https://hf-mirror.com

python scripts/run_folding.py \
    --sequence "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGD" \
    --output ./outputs/smoke_test/ \
    --cache_dir /root/rivermind-data/models \
    --experts /root/rivermind-data/models/dplm_150m \
    --num_iters 3 \
    --num_rollouts 1 \
    --top_k_children 2 \
    --max_depth 2 \
    --diffusion_steps 20 \
    --return_top_k 2 \
    --half_precision \
    --device cuda \
    --log_level DEBUG
```

**参数对比（快速验证 vs 论文默认）：**

| 参数 | 快速验证 | 论文默认 |
|------|----------|----------|
| experts | 仅 150M | 150M + 650M + 3B |
| num_iters | 3 | 100 |
| num_rollouts | 1 | 3 |
| top_k_children | 2 | 3 |
| max_depth | 2 | 5 |
| diffusion_steps | 20 | 150 |
| half_precision | 是 | 否 |

**预期输出：**

```
INFO  Loading DPLM-2 model: .../dplm_150m
INFO  Loading ESMFold ...
INFO  MCTS iter 1/3 ...
INFO  MCTS iter 2/3 ...
INFO  MCTS iter 3/3 ...
=== Best Design ===
  Reward : 0.xxxx
  Seq    : MKTAY...
```

输出文件位于 `./outputs/smoke_test/`：
- `designs.fasta` — 设计的序列
- `summary.json` — 每条序列的 reward / AAR / scTM / pLDDT
- `run.log` — 完整运行日志

---

## 7. 完整任务运行

### 7.1 逆折叠（Inverse Folding）

给定蛋白质骨架 PDB，设计对应序列：

```bash
python scripts/run_inverse_folding.py \
    --pdb /root/rivermind-data/data/cameo2022/7dz2_C.pdb \
    --chain A \
    --output ./outputs/inverse_folding/ \
    --cache_dir /root/rivermind-data/models \
    --experts \
        /root/rivermind-data/models/dplm_150m \
        /root/rivermind-data/models/dplm_650m \
    --use_proteinmpnn \
    --proteinmpnn_path /root/rivermind-data/models/ProteinMPNN \
    --num_iters 100 \
    --num_rollouts 3 \
    --top_k_children 3 \
    --max_depth 5 \
    --diffusion_steps 150 \
    --return_top_k 10 \
    --half_precision \
    --device cuda
```

### 7.2 折叠优化（Folding Lead Optimization）

给定初始序列，优化其折叠结构质量：

```bash
python scripts/run_folding.py \
    --sequence "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGD" \
    --output ./outputs/folding/ \
    --cache_dir /root/rivermind-data/models \
    --experts \
        /root/rivermind-data/models/dplm_150m \
        /root/rivermind-data/models/dplm_650m \
        /root/rivermind-data/models/dplm_3b \
    --num_iters 100 \
    --num_rollouts 3 \
    --top_k_children 3 \
    --max_depth 5 \
    --diffusion_steps 150 \
    --half_precision \
    --device cuda
```

### 7.3 Motif 支架设计（Motif Scaffolding）

固定功能性 motif，设计周围支架：

```bash
python scripts/run_motif_scaffolding.py \
    --pdb /root/rivermind-data/data/motifs/1YCR.pdb \
    --chain A \
    --motif_residues 10-25 \
    --scaffold_length 100 \
    --output ./outputs/motif/ \
    --cache_dir /root/rivermind-data/models \
    --experts \
        /root/rivermind-data/models/dplm_150m \
        /root/rivermind-data/models/dplm_650m \
    --num_iters 100 \
    --half_precision \
    --device cuda
```

### 7.4 评估已有结果

```bash
python scripts/evaluate.py \
    --fasta ./outputs/inverse_folding/designs.fasta \
    --reference /root/rivermind-data/data/cameo2022/7dz2_C.pdb \
    --chain A \
    --output ./eval_results/ \
    --cache_dir /root/rivermind-data/models \
    --device cuda
```

---

## 8. 常用参数速查

### 专家模型

| 想用的专家 | `--experts` 参数值 |
|------------|-------------------|
| 仅 150M（最快） | `/root/rivermind-data/models/dplm_150m` |
| 150M + 650M | `...dplm_150m ...dplm_650m` |
| 全量三专家 | `...dplm_150m ...dplm_650m ...dplm_3b` |
| 加 ProteinMPNN | 追加 `--use_proteinmpnn --proteinmpnn_path ...` |

### 速度 vs 质量

| 场景 | num_iters | diffusion_steps | 预计耗时 |
|------|-----------|-----------------|----------|
| 冒烟测试 | 3 | 20 | 3–5 min |
| 快速实验 | 20 | 50 | 10–20 min |
| 论文默认 | 100 | 150 | 5–15 min/target |

> 论文 Table 11：单目标 multi-expert 平均耗时约 **800 秒（~13 分钟）**，单 A100 GPU。

### 显存节省

```bash
--half_precision          # fp16，显存减约 40%，强烈推荐
```
