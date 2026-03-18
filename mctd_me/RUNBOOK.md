# MCTD-ME 运行手册

**运行环境**
- 项目路径：`~/rivermind-data/Protein/mctd_me`
- 数据/模型路径：`~/rivermind-data/`
- GPU：双卡 RTX 4090 24GB（cuda:0 跑专家模型，cuda:1 跑 ESMFold）
- 网络：中国大陆，使用 HuggingFace 镜像

---

## 目录

1. [路径约定](#1-路径约定)
2. [创建 Conda 环境](#2-创建-conda-环境)
3. [配置 HuggingFace 镜像](#3-配置-huggingface-镜像)
4. [下载模型](#4-下载模型)
5. [下载基准数据](#5-下载基准数据)
6. [验证安装](#6-验证安装)
7. [冒烟测试（快速验证）](#7-冒烟测试快速验证)
8. [完整任务运行](#8-完整任务运行)
9. [已知问题与修复](#9-已知问题与修复)

---

## 1. 路径约定

```
~/rivermind-data/
├── Protein/
│   └── mctd_me/              # 本项目代码
├── models/
│   ├── dplm_150m/            # DPLM-2 150M  (~600 MB)
│   ├── dplm_650m/            # DPLM-2 650M  (~2.6 GB)
│   ├── dplm_3b/              # DPLM-2 3B    (~12 GB)
│   ├── esmfold_v1/           # ESMFold      (~2.5 GB)
│   └── ProteinMPNN/          # ProteinMPNN  (git clone, 已完成)
└── data/
    ├── motifs/               # EvoDiff motif set (23个)
    └── pdb_date/             # PDB date-split subset
```

**双卡显存分配**

| 卡 | 负责 | 峰值显存 |
|----|------|----------|
| cuda:0 | DPLM-2 专家（150M+650M+3B，fp16）| ~8 GB |
| cuda:1 | ESMFold critic | ~14 GB |

---

## 2. 创建 Conda 环境

```bash
cd ~/rivermind-data/Protein/mctd_me

conda env create -f environment.yml
conda activate mctd_me

pip install -e .
```

> **注意**：`environment.yml` 已固定 `numpy<2`，避免与 PyTorch 编译版本冲突。
> 若已有旧环境出现 `_ARRAY_API not found` 警告，重建环境即可。

---

## 3. 配置 HuggingFace 镜像

```bash
# 写入 ~/.bashrc（永久生效）
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
echo 'export HF_HOME=~/rivermind-data/models'   >> ~/.bashrc
source ~/.bashrc
```

---

## 4. 下载模型

确保已执行 `source ~/.bashrc` 后运行：

```bash
mkdir -p ~/rivermind-data/models

# DPLM-2 150M（~600 MB）
HF_ENDPOINT=https://hf-mirror.com \
huggingface-cli download airkingbd/dplm_150m \
    --local-dir ~/rivermind-data/models/dplm_150m \
    --local-dir-use-symlinks False

# DPLM-2 650M（~2.6 GB）
HF_ENDPOINT=https://hf-mirror.com \
huggingface-cli download airkingbd/dplm_650m \
    --local-dir ~/rivermind-data/models/dplm_650m \
    --local-dir-use-symlinks False

# DPLM-2 3B（~12 GB）
HF_ENDPOINT=https://hf-mirror.com \
huggingface-cli download airkingbd/dplm_3b \
    --local-dir ~/rivermind-data/models/dplm_3b \
    --local-dir-use-symlinks False

# ESMFold（~2.5 GB）
HF_ENDPOINT=https://hf-mirror.com \
huggingface-cli download facebook/esmfold_v1 \
    --local-dir ~/rivermind-data/models/esmfold_v1 \
    --local-dir-use-symlinks False
```

> ProteinMPNN 已 clone 完成，跳过。
> 所有下载支持断点续传，中断后重新执行即可。

---

## 5. 下载基准数据

```bash
conda activate mctd_me
cd ~/rivermind-data/Protein/mctd_me

python scripts/download_data.py \
    --all \
    --output ~/rivermind-data/data/
```

---

## 6. 验证安装

### 6.1 单元测试（无需 GPU）

```bash
conda activate mctd_me
cd ~/rivermind-data/Protein/mctd_me

pytest tests/ -v
```

### 6.2 模型加载测试

```bash
python - <<'EOF'
import os, torch
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from mctd_me.experts import DPLM2Expert

expert = DPLM2Expert(
    model_name=os.path.expanduser("~/rivermind-data/models/dplm_150m"),
    device="cuda:0",
    half_precision=True,
)
seq = "ACDEFGHIKLMNPQRSTVWY" * 2
logp = expert.get_logprobs(seq, [5, 10, 15])
print(f"log_probs shape : {logp.shape}")   # 期望 [3, 20]
print(f"cuda:0 已用显存 : {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
print("✓ Expert 加载成功")
EOF
```

---

## 7. 冒烟测试（快速验证）

验证完整链路：专家生成 → ESMFold 打分 → MCTS 搜索 → 输出。
预计耗时 **3–5 分钟**。

```bash
conda activate mctd_me
cd ~/rivermind-data/Protein/mctd_me

python scripts/run_folding.py \
    --sequence "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGD" \
    --output ~/rivermind-data/outputs/smoke_test/ \
    --cache_dir ~/rivermind-data/models \
    --experts \
        ~/rivermind-data/models/dplm_150m \
    --device cuda:0 \
    --critic_device cuda:1 \
    --esmfold_chunk_size 64 \
    --half_precision \
    --num_iters 3 \
    --num_rollouts 1 \
    --top_k_children 2 \
    --max_depth 2 \
    --diffusion_steps 20 \
    --return_top_k 2 \
    --log_level DEBUG
```

**参数说明（冒烟 vs 论文默认）**

| 参数 | 冒烟测试 | 论文默认 | 说明 |
|------|----------|----------|------|
| experts | 仅 150M | 150M+650M+3B | 最快加载 |
| num_iters | 3 | 100 | MCTS 迭代数 |
| num_rollouts | 1 | 3 | 每次展开的 rollout 数 |
| top_k_children | 2 | 3 | 保留子节点数 |
| max_depth | 2 | 5 | 树深度 |
| diffusion_steps | 20 | 150 | 扩散步数 |
| critic_device | cuda:1 | — | ESMFold 独占第二卡 |

**预期输出**

```
INFO  Loading DPLM-2 model: .../dplm_150m   [cuda:0]
INFO  Loading ESMFold ...                    [cuda:1]
INFO  MCTS iter 1/3 ...
INFO  MCTS iter 2/3 ...
INFO  MCTS iter 3/3 ...

=== Best Design ===
  Reward : 0.xxxx
  Seq    : MKTAY...
```

输出文件：`~/rivermind-data/outputs/smoke_test/`
- `designs.fasta` — 设计序列
- `summary.json` — reward / AAR / scTM / pLDDT
- `run.log` — 完整日志

---

## 8. 完整任务运行

所有任务均使用 `--device cuda:0 --critic_device cuda:1`。

### 8.1 折叠优化（Folding）

```bash
python scripts/run_folding.py \
    --sequence "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGD" \
    --output ~/rivermind-data/outputs/folding/ \
    --cache_dir ~/rivermind-data/models \
    --experts \
        ~/rivermind-data/models/dplm_150m \
        ~/rivermind-data/models/dplm_650m \
        ~/rivermind-data/models/dplm_3b \
    --device cuda:0 \
    --critic_device cuda:1 \
    --half_precision \
    --num_iters 100 \
    --num_rollouts 3 \
    --top_k_children 3 \
    --max_depth 5 \
    --diffusion_steps 150
```

### 8.2 逆折叠（Inverse Folding）

```bash
python scripts/run_inverse_folding.py \
    --pdb ~/rivermind-data/data/motifs/1YCR.pdb \
    --chain A \
    --output ~/rivermind-data/outputs/inverse_folding/ \
    --cache_dir ~/rivermind-data/models \
    --experts \
        ~/rivermind-data/models/dplm_150m \
        ~/rivermind-data/models/dplm_650m \
        ~/rivermind-data/models/dplm_3b \
    --use_proteinmpnn \
    --proteinmpnn_path ~/rivermind-data/models/ProteinMPNN \
    --device cuda:0 \
    --critic_device cuda:1 \
    --half_precision \
    --num_iters 100
```

### 8.3 Motif 支架设计（Motif Scaffolding）

```bash
python scripts/run_motif_scaffolding.py \
    --pdb ~/rivermind-data/data/motifs/1YCR.pdb \
    --chain A \
    --motif_residues 10-25 \
    --scaffold_length 100 \
    --output ~/rivermind-data/outputs/motif/ \
    --cache_dir ~/rivermind-data/models \
    --experts \
        ~/rivermind-data/models/dplm_150m \
        ~/rivermind-data/models/dplm_650m \
        ~/rivermind-data/models/dplm_3b \
    --device cuda:0 \
    --critic_device cuda:1 \
    --half_precision \
    --num_iters 100
```

### 8.4 评估结果

```bash
python scripts/evaluate.py \
    --fasta ~/rivermind-data/outputs/inverse_folding/designs.fasta \
    --reference ~/rivermind-data/data/motifs/1YCR.pdb \
    --chain A \
    --output ~/rivermind-data/outputs/eval/ \
    --cache_dir ~/rivermind-data/models \
    --device cuda:1
```

---

## 9. 已知问题与修复

### NumPy 2.x 兼容性警告

**现象**
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
UserWarning: Failed to initialize NumPy: _ARRAY_API not found
```

**原因**：conda 环境安装了 numpy 2.x，与 PyTorch 编译版本不兼容。

**修复**：`environment.yml` 已更新为 `numpy>=1.24,<2`，重建环境即可：

```bash
conda deactivate
conda env remove -n mctd_me
conda env create -f ~/rivermind-data/Protein/mctd_me/environment.yml
conda activate mctd_me
pip install -e ~/rivermind-data/Protein/mctd_me/
```

### ESMFold 显存不足（OOM）

将 `--esmfold_chunk_size` 调低：

```bash
--esmfold_chunk_size 64   # 默认 128，越小越省显存
--esmfold_chunk_size 32   # 更省，但推理变慢
```

### 模型路径找不到

检查路径是否包含 `~`（部分 shell 不展开）：

```bash
# 用绝对路径替代 ~
--cache_dir $HOME/rivermind-data/models
--experts $HOME/rivermind-data/models/dplm_150m
```
