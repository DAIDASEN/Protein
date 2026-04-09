# MCTD-ME 安装与运行指南

本指南适用于以下环境：

- GPU：48 GB 显存（如魔改 RTX 4090）
- 模型存储：`/root/rivermind-data/`
- 网络：中国大陆，使用 HuggingFace 镜像站

---

## 显存预估（48 GB 是否足够？）

| 配置 | 峰值显存 | 是否可行 |
|------|----------|----------|
| 单专家 DPLM-2 650M + ESMFold (fp16) | ~16 GB | ✅ |
| 三专家 150M+650M+ProteinMPNN + ESMFold (fp16) | ~18 GB | ✅ |
| 三专家 150M+650M+3B + ESMFold (fp16) | ~24 GB | ✅ |
| 全量（含 3B，fp32） | ~42 GB | ✅ 刚好 |

**结论：48 GB 可以跑完整论文配置（fp16 推荐，显存宽裕）。**

---

## 目录结构约定

```
/root/rivermind-data/
├── models/
│   ├── dplm_150m/          # DPLM-2 150M 模型
│   ├── dplm_650m/          # DPLM-2 650M 模型
│   ├── dplm_3b/            # DPLM-2 3B 模型
│   ├── esmfold_v1/         # ESMFold
│   └── ProteinMPNN/        # ProteinMPNN（git clone）
└── data/
    ├── cameo2022/           # CAMEO 2022 benchmark
    └── pdb_date/            # PDB date-split benchmark
```

---

## 第一步：创建 Conda 环境

```bash
cd /home/dds/Protein/mctd_me

conda env create -f environment.yml
conda activate mctd_me

# 安装本项目
pip install -e .
```

---

## 第二步：配置 HuggingFace 镜像（中国大陆）

每次使用前执行，或写入 `~/.bashrc` / `~/.zshrc`：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/rivermind-data/models
```

永久写入：

```bash
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
echo 'export HF_HOME=/root/rivermind-data/models' >> ~/.bashrc
source ~/.bashrc
```

验证镜像可用：

```bash
curl -I https://hf-mirror.com
# 应返回 HTTP/2 200 或 302
```

---

## 第三步：下载 DPLM-2 模型

```bash
conda activate mctd_me

# 创建目录
mkdir -p /root/rivermind-data/models

# 下载三个 DPLM-2 模型（通过镜像）
HF_ENDPOINT=https://hf-mirror.com \
huggingface-cli download airkingbd/dplm_150m \
    --local-dir /root/rivermind-data/models/dplm_150m \
    --local-dir-use-symlinks False

HF_ENDPOINT=https://hf-mirror.com \
huggingface-cli download airkingbd/dplm_650m \
    --local-dir /root/rivermind-data/models/dplm_650m \
    --local-dir-use-symlinks False

HF_ENDPOINT=https://hf-mirror.com \
huggingface-cli download airkingbd/dplm_3b \
    --local-dir /root/rivermind-data/models/dplm_3b \
    --local-dir-use-symlinks False
```

---

## 第四步：下载 ESMFold

ESMFold 用作结构预测 critic（计算 pLDDT、scTM、RMSD）：

```bash
HF_ENDPOINT=https://hf-mirror.com \
huggingface-cli download facebook/esmfold_v1 \
    --local-dir /root/rivermind-data/models/esmfold_v1 \
    --local-dir-use-symlinks False
```

> **注意**：ESMFold 权重约 2.5 GB，但推理时含 ESM-2 骨干网络，总显存占用约 14 GB。

---

## 第五步：下载 ProteinMPNN（逆折叠专家）

```bash
cd /root/rivermind-data/models
git clone https://github.com/dauparas/ProteinMPNN.git
```

如果 GitHub 访问慢，可用 Gitee 镜像或 ghproxy：

```bash
git clone https://ghfast.top/https://github.com/dauparas/ProteinMPNN.git
```

---

## 第六步：下载基准数据集

```bash
conda activate mctd_me
cd /home/dds/Protein/mctd_me

python scripts/download_data.py \
    --all \
    --output /root/rivermind-data/data/
```

> 该脚本会下载 CAMEO 2022（183 个目标）和 PDB date-split 子集（449 条链）。
> 数据来源为公开 PDB，无需镜像。

---

## 第七步：验证安装

```bash
conda activate mctd_me
cd /home/dds/Protein/mctd_me

# 运行单元测试（不加载真实模型）
pytest tests/ -v

# 快速冒烟测试（加载 150M 模型验证流程）
python - <<'EOF'
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from mctd_me.experts import DPLM2Expert
import torch

expert = DPLM2Expert(
    model_name="/root/rivermind-data/models/dplm_150m",
    device="cuda",
    half_precision=True,
)
seq = "ACDEFGHIKLMNPQRSTVWY" * 2  # 40残基测试序列
mask = [5, 10, 15]
logp = expert.get_logprobs(seq, mask)
print(f"log_probs shape: {logp.shape}")  # 期望: torch.Size([3, 20])
print(f"GPU 显存已用: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print("✓ 安装验证通过")
EOF
```

---

## 第八步：运行任务

所有脚本均通过 `--cache_dir` 指定模型路径，无需修改代码。

### 8.1 逆折叠（Inverse Folding）

给定蛋白质骨架结构，设计对应序列：

```bash
python scripts/run_inverse_folding.py \
    --pdb /root/rivermind-data/data/cameo2022/7dz2_C.pdb \
    --chain A \
    --output ./outputs/inverse_folding/ \
    --cache_dir /root/rivermind-data/models \
    --experts airkingbd/dplm_150m airkingbd/dplm_650m \
    --use_proteinmpnn \
    --proteinmpnn_path /root/rivermind-data/models/ProteinMPNN \
    --half_precision \
    --num_iters 100 \
    --device cuda
```

### 8.2 折叠优化（Lead Optimization - Folding）

给定初始序列，优化其折叠质量：

```bash
python scripts/run_folding.py \
    --sequence "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGD" \
    --output ./outputs/folding/ \
    --cache_dir /root/rivermind-data/models \
    --experts airkingbd/dplm_150m airkingbd/dplm_650m airkingbd/dplm_3b \
    --half_precision \
    --num_iters 100 \
    --device cuda
```

### 8.3 Motif 支架设计（Motif Scaffolding）

固定功能性 motif，设计周围支架序列：

```bash
python scripts/run_motif_scaffolding.py \
    --pdb /root/rivermind-data/data/motifs/1YCR.pdb \
    --chain A \
    --motif_residues 10-25 \
    --scaffold_length 100 \
    --output ./outputs/motif/ \
    --cache_dir /root/rivermind-data/models \
    --half_precision \
    --device cuda
```

### 8.4 批量评估

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

## 常用配置调整

### 仅用 150M + 650M（节省显存）

```bash
--experts airkingbd/dplm_150m airkingbd/dplm_650m
```

### 开启 fp16（推荐，节省约 40% 显存）

```bash
--half_precision
```

### 调整 MCTS 迭代次数（越多越好但更慢）

```bash
--num_iters 50    # 快速测试
--num_iters 100   # 论文默认
--num_iters 200   # 更充分搜索
```

### 修改模型缓存目录（Python 代码中）

```python
from mctd_me.config import MCTDMEConfig

cfg = MCTDMEConfig(
    experts=[
        "/root/rivermind-data/models/dplm_150m",
        "/root/rivermind-data/models/dplm_650m",
        "/root/rivermind-data/models/dplm_3b",
    ],
    use_proteinmpnn=True,
    proteinmpnn_path="/root/rivermind-data/models/ProteinMPNN",
    esmfold_model="/root/rivermind-data/models/esmfold_v1",
    cache_dir="/root/rivermind-data/models",
    output_dir="./outputs",
    device="cuda",
    task="inverse_folding",
    # fp16 节省显存
)
```

> 本地路径与 HuggingFace model ID 均可传给 `experts` 列表，代码会自动识别。

---

## 磁盘空间汇总

| 内容 | 大小 |
|------|------|
| conda 环境 | ~9 GB |
| DPLM-2 150M | ~600 MB |
| DPLM-2 650M | ~2.6 GB |
| DPLM-2 3B | ~12 GB |
| ESMFold | ~2.5 GB |
| ProteinMPNN | ~50 MB |
| CAMEO + PDB 数据 | ~500 MB |
| **合计** | **~28 GB** |

---

## 常见问题

### `huggingface-cli` 下载中断怎么办？

重新运行同一命令即可，支持断点续传。

### ESMFold OOM（显存不足）？

```bash
# 将 ESMFold 的 chunk_size 调小（在 critics.py 中已有此参数）
--esmfold_chunk_size 64   # 默认 128，越小越省显存但越慢
```

### 模型加载很慢？

第一次加载需从磁盘读取，后续运行会快很多。可用 `--half_precision` 加速加载并减少显存。

### 如何只跑单专家对比实验？

```bash
python scripts/run_inverse_folding.py \
    --experts airkingbd/dplm_650m \   # 只用一个专家
    --num_iters 100 \
    ...
```

---

## 参考文献

```
Liu et al., "Monte Carlo Tree Diffusion with Multiple Experts for Protein Design"
arXiv:2509.15796, 2025
```
