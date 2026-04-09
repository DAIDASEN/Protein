# MCTD-Maze 运行手册

**运行环境**
- 项目路径：`/root/rivermind-data/planning/mctd_maze`
- 数据/模型路径：`/root/rivermind-data/`
- GPU：单卡 RTX 4090 24GB（`cuda:0`），或双卡 RTX 4090（`cuda:0` + `cuda:1`）
- 网络：中国大陆，使用镜像加速

---

## 目录

1. [路径约定](#1-路径约定)
2. [克隆官方代码](#2-克隆官方代码)
3. [创建 Conda 环境](#3-创建-conda-环境)
4. [配置 WandB](#4-配置-wandb)
5. [准备 OGBench 数据集](#5-准备-ogbench-数据集)
6. [验证安装](#6-验证安装)
7. [冒烟测试（快速验证）](#7-冒烟测试快速验证)
8. [训练扩散规划器](#8-训练扩散规划器)
9. [MCTD 评估](#9-mctd-评估)
10. [完整 Benchmark 复现](#10-完整-benchmark-复现)
11. [已知问题与修复](#11-已知问题与修复)

---

## 1. 路径约定

```
/root/rivermind-data/
├── planning/
│   ├── mctd_maze/            # 本项目代码（git clone + pip install -e）
│   └── mctd_official/        # 官方原始代码（参考用，git clone ahn-ml/mctd）
├── data/
│   └── ogbench/              # OGBench 数据集（首次运行自动下载）
├── checkpoints/
│   └── mctd_maze/            # 训练保存的模型权重
│       ├── pointmaze_medium/
│       ├── pointmaze_large/
│       ├── antmaze_medium/
│       └── ...
└── outputs/
    └── mctd_maze/            # 评估结果
```

**GPU 显存分配**

| 卡 | 负责 | 峰值显存 |
|----|------|----------|
| cuda:0 | 扩散规划器训练 / MCTD 搜索 | ~10–16 GB |
| cuda:1（双卡时）| 并行跑另一个任务 | ~10–16 GB |

---

## 2. 克隆官方代码

```bash
mkdir -p /root/rivermind-data/planning

# 克隆本项目（已含封装脚本）
git clone <本仓库地址> /root/rivermind-data/planning/mctd_maze
# 或直接 rsync 同步本地目录（若已从本地机器同步）

# 克隆官方原始代码（参考实现，用于对照）
git clone https://github.com/ahn-ml/mctd.git \
    /root/rivermind-data/planning/mctd_official
# 中国大陆加速：
# git clone https://ghfast.top/https://github.com/ahn-ml/mctd.git \
#     /root/rivermind-data/planning/mctd_official
```

---

## 3. 创建 Conda 环境

```bash
cd /root/rivermind-data/planning/mctd_maze

conda env create -f environment.yml
conda activate mctd_maze

pip install -e .
```

> **注意**：`environment.yml` 已固定 `numpy<2` 避免 PyTorch 编译冲突，
> 并使用 `pytorch-cuda=12.1` 适配 RTX 4090。

---

## 4. 配置 WandB

训练和评估均通过 WandB 记录指标。

```bash
# 注册账号：https://wandb.ai/，然后：
wandb login   # 输入 API Key

# 写入 ~/.bashrc（永久）
echo 'export WANDB_ENTITY=<你的用户名>'      >> ~/.bashrc
echo 'export WANDB_PROJECT=mctd_maze'        >> ~/.bashrc
echo 'export OGBENCH_DATA=/root/rivermind-data/data/ogbench' >> ~/.bashrc
source ~/.bashrc
```

> WandB 可选：训练时加 `--no_wandb` 跳过（见脚本参数）。

---

## 5. 准备 OGBench 数据集

数据集在首次调用环境时自动下载到 `$OGBENCH_DATA`。

```bash
conda activate mctd_maze
export OGBENCH_DATA=/root/rivermind-data/data/ogbench
mkdir -p $OGBENCH_DATA

# 预下载所有任务的数据集（可选，避免训练时临时下载）
python - <<'EOF'
import os
os.environ["OGBENCH_DATASETS_PATH"] = os.environ["OGBENCH_DATA"]
import ogbench

envs = [
    "pointmaze-medium-navigate-v0",
    "pointmaze-large-navigate-v0",
    "pointmaze-giant-navigate-v0",
    "antmaze-medium-navigate-v0",
    "antmaze-large-navigate-v0",
    "antmaze-giant-navigate-v0",
    "cube-single-play-v0",
    "cube-double-play-v0",
]
for name in envs:
    print(f"Downloading {name} ...")
    env, dataset = ogbench.make_env_and_datasets(name)
    print(f"  obs shape: {dataset['observations'].shape}")
    env.close()
print("All done.")
EOF
```

---

## 6. 验证安装

### 6.1 单元测试（无需 GPU）

```bash
conda activate mctd_maze
cd /root/rivermind-data/planning/mctd_maze

pytest tests/ -v
```

### 6.2 环境加载测试

```bash
python - <<'EOF'
import os, torch
os.environ["OGBENCH_DATASETS_PATH"] = "/root/rivermind-data/data/ogbench"
import ogbench

env, dataset = ogbench.make_env_and_datasets("pointmaze-medium-navigate-v0")
print(f"obs dim   : {env.observation_space.shape}")
print(f"action dim: {env.action_space.shape}")
print(f"dataset   : {dataset['observations'].shape[0]} transitions")
print(f"cuda avail: {torch.cuda.is_available()}")
print(f"GPU       : {torch.cuda.get_device_name(0)}")
print("✓ 环境加载成功")
EOF
```

---

## 7. 冒烟测试（快速验证）

验证完整链路：数据加载 → 模型初始化 → 1 轮 MCTD 搜索 → 输出。
预计耗时 **3–5 分钟**。

```bash
conda activate mctd_maze
cd /root/rivermind-data/planning/mctd_maze
export OGBENCH_DATASETS_PATH=/root/rivermind-data/data/ogbench

python scripts/train.py \
    --env pointmaze-medium-navigate-v0 \
    --save_dir /root/rivermind-data/checkpoints/mctd_maze/smoke_test \
    --device cuda:0 \
    --train_steps 100 \
    --batch_size 64 \
    --no_wandb \
    --log_level DEBUG
```

**参数说明（冒烟 vs 论文默认）**

| 参数 | 冒烟测试 | 论文默认 | 说明 |
|------|----------|----------|------|
| train_steps | 100 | 200,005 | 训练步数 |
| batch_size | 64 | 1024 | 批次大小 |
| no_wandb | 是 | 否 | 跳过 WandB |

---

## 8. 训练扩散规划器

### 8.1 单卡训练（cuda:0）

```bash
conda activate mctd_maze
cd /root/rivermind-data/planning/mctd_maze
export OGBENCH_DATASETS_PATH=/root/rivermind-data/data/ogbench

# Pointmaze Medium（基准任务，约 4–6 小时）
python scripts/train.py \
    --env pointmaze-medium-navigate-v0 \
    --save_dir /root/rivermind-data/checkpoints/mctd_maze/pointmaze_medium \
    --device cuda:0 \
    --train_steps 200005 \
    --batch_size 1024 \
    --lr 5e-4 \
    --precision 16-mixed

# Antmaze Medium
python scripts/train.py \
    --env antmaze-medium-navigate-v0 \
    --save_dir /root/rivermind-data/checkpoints/mctd_maze/antmaze_medium \
    --device cuda:0 \
    --train_steps 200005 \
    --batch_size 1024 \
    --lr 5e-4 \
    --precision 16-mixed
```

### 8.2 双卡并行（各跑一个任务）

```bash
# 终端 1
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --env pointmaze-large-navigate-v0 \
    --save_dir /root/rivermind-data/checkpoints/mctd_maze/pointmaze_large \
    --train_steps 200005 --batch_size 1024 --lr 5e-4

# 终端 2
CUDA_VISIBLE_DEVICES=1 python scripts/train.py \
    --env antmaze-large-navigate-v0 \
    --save_dir /root/rivermind-data/checkpoints/mctd_maze/antmaze_large \
    --train_steps 200005 --batch_size 1024 --lr 5e-4
```

**训练时间估计（单卡 RTX 4090）**

| 环境 | 估计时间 |
|------|---------|
| Pointmaze Medium | 4–6 h |
| Pointmaze Large | 6–8 h |
| Pointmaze Giant | 8–12 h |
| Antmaze Medium | 6–8 h |
| Antmaze Large | 8–12 h |
| Antmaze Giant | 12–16 h |

---

## 9. MCTD 评估

```bash
conda activate mctd_maze
cd /root/rivermind-data/planning/mctd_maze
export OGBENCH_DATASETS_PATH=/root/rivermind-data/data/ogbench

python scripts/evaluate.py \
    --env pointmaze-medium-navigate-v0 \
    --checkpoint /root/rivermind-data/checkpoints/mctd_maze/pointmaze_medium/best.pt \
    --output /root/rivermind-data/outputs/mctd_maze/pointmaze_medium/ \
    --device cuda:0 \
    --num_eval_episodes 100 \
    --max_search 500 \
    --num_subplans 5 \
    --partial_denoise_steps 20 \
    --jumpy_interval 10 \
    --ddim_eta 0.0
```

**MCTD 核心参数（论文 Table 10）**

| 参数 | 值 | 说明 |
|------|----|------|
| `--num_subplans` S | 5 | 子计划数（N=500 总步长） |
| `--max_search` | 500 | MCTS 最大迭代次数 |
| `--partial_denoise_steps` | 20 | 每节点去噪步数 |
| `--jumpy_interval` C | 10 | DDIM 快速仿真跳跃间隔 |
| `--ddim_eta` | 0.0 | 确定性采样（无随机噪声） |

---

## 10. 完整 Benchmark 复现

所有任务按顺序跑（单卡）或两个并行（双卡）：

```bash
for ENV in \
    pointmaze-medium-navigate-v0 \
    pointmaze-large-navigate-v0 \
    pointmaze-giant-navigate-v0 \
    antmaze-medium-navigate-v0 \
    antmaze-large-navigate-v0 \
    antmaze-giant-navigate-v0; do

    NAME=$(echo $ENV | sed 's/-navigate-v0//' | tr '-' '_')

    python scripts/train.py \
        --env $ENV \
        --save_dir /root/rivermind-data/checkpoints/mctd_maze/$NAME \
        --train_steps 200005 --batch_size 1024 --lr 5e-4

    python scripts/evaluate.py \
        --env $ENV \
        --checkpoint /root/rivermind-data/checkpoints/mctd_maze/$NAME/best.pt \
        --output /root/rivermind-data/outputs/mctd_maze/$NAME/ \
        --num_eval_episodes 100
done
```

---

## 11. 已知问题与修复

### MuJoCo 渲染报错（Antmaze）

```bash
# 设置无显示器渲染后端
export MUJOCO_GL=egl    # 首选
# 或
export MUJOCO_GL=osmesa # 备选（需安装 libosmesa6）

apt-get install -y libosmesa6-dev libglew-dev
```

### NumPy 2.x 兼容警告

```
_ARRAY_API not found
```

`environment.yml` 已固定 `numpy<2`。若旧环境报错，重建：

```bash
conda deactivate
conda env remove -n mctd_maze
conda env create -f /root/rivermind-data/planning/mctd_maze/environment.yml
conda activate mctd_maze
pip install -e /root/rivermind-data/planning/mctd_maze/
```

### CUDA OOM（显存不足）

```bash
# 降低 batch size
python scripts/train.py ... --batch_size 512

# 降低 MCTD 搜索预算
python scripts/evaluate.py ... --max_search 100
```

### OGBench 数据路径找不到

```bash
export OGBENCH_DATASETS_PATH=/root/rivermind-data/data/ogbench
# 或用绝对路径参数替代 ~
--data_dir /root/rivermind-data/data/ogbench
```

### WandB 离线模式（无网络时）

```bash
export WANDB_MODE=offline
# 之后上传：
wandb sync /root/rivermind-data/outputs/mctd_maze/wandb/offline-run-*
```
