# MCTD-Maze 快速上手

**硬件前提：** RTX 4090 24GB（单卡或双卡）
**存储路径：** `/root/rivermind-data/`
**网络：** 中国大陆，使用镜像加速

---

## 1. 目录结构

```
/root/rivermind-data/
├── planning/mctd_maze/         # 本项目代码
├── data/ogbench/               # OGBench 数据集
├── checkpoints/mctd_maze/      # 训练权重
└── outputs/mctd_maze/          # 评估结果
```

---

## 2. 创建环境

```bash
cd /root/rivermind-data/planning/mctd_maze

conda env create -f environment.yml
conda activate mctd_maze
pip install -e .
```

---

## 3. 设置环境变量

```bash
echo 'export OGBENCH_DATASETS_PATH=/root/rivermind-data/data/ogbench' >> ~/.bashrc
echo 'export WANDB_PROJECT=mctd_maze' >> ~/.bashrc
source ~/.bashrc
```

---

## 4. 快速验证（冒烟测试）

```bash
conda activate mctd_maze
cd /root/rivermind-data/planning/mctd_maze

python scripts/train.py \
    --env pointmaze-medium-navigate-v0 \
    --save_dir /root/rivermind-data/checkpoints/mctd_maze/smoke_test \
    --device cuda:0 \
    --train_steps 100 \
    --batch_size 64 \
    --no_wandb \
    --log_level DEBUG
```

**预期输出：**

```
INFO  Loading dataset: pointmaze-medium-navigate-v0
INFO  obs_dim=4  action_dim=2  horizon=200
INFO  Model params: 3.2M
INFO  Step   10 | loss=0.xxxx
INFO  Step   50 | loss=0.xxxx
INFO  Step  100 | loss=0.xxxx
✓ 冒烟测试通过
```

---

## 5. 完整训练

```bash
python scripts/train.py \
    --env pointmaze-medium-navigate-v0 \
    --save_dir /root/rivermind-data/checkpoints/mctd_maze/pointmaze_medium \
    --device cuda:0 \
    --train_steps 200005 \
    --batch_size 1024 \
    --lr 5e-4 \
    --precision 16-mixed
```

---

## 6. MCTD 评估

```bash
python scripts/evaluate.py \
    --env pointmaze-medium-navigate-v0 \
    --checkpoint /root/rivermind-data/checkpoints/mctd_maze/pointmaze_medium/best.pt \
    --output /root/rivermind-data/outputs/mctd_maze/pointmaze_medium/ \
    --device cuda:0 \
    --num_eval_episodes 100
```

---

## 参数速查

### 速度 vs 质量

| 场景 | train_steps | batch_size | 评估 max_search | 预计训练时间 |
|------|-------------|------------|-----------------|-------------|
| 冒烟测试 | 100 | 64 | 50 | 3–5 min |
| 快速实验 | 20,000 | 512 | 100 | 30–60 min |
| 论文默认 | 200,005 | 1024 | 500 | 4–16 h |

### 显存节省

```bash
--batch_size 512      # 默认 1024，减半显存
--precision 16-mixed  # 训练时 fp16，强烈推荐（论文配置）
```
