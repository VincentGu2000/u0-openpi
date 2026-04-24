# U0Bot 训练与评估指南

## 1. 环境准备

```bash
mamba create -n pi05 python==3.11
conda activate pi05
pip install uv
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
conda install -c conda-forge ffmpeg -y
pip install --force-reinstall nvidia-cudnn-cu12==9.12.0.46
```

## 2. 计算归一化统计量

在训练之前，需要先为数据集计算归一化统计量（norm stats）：

```bash
python scripts/compute_norm_stats.py \
    --config_name pi05_u0bot \
    --repo_id /data/gujunwen/project/fish-vla/dataset/lerobot_full
```

## 3. JAX 训练

```bash
tmux new -s my_training "source ~/miniconda3/bin/activate pi05 && \
HF_HUB_OFFLINE=1 WANDB_MODE=offline XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi05_u0bot \
    --exp-name=u0bot_finetune_v1 \
    --overwrite"
```

## 4. 评估动作预测 MSE

使用 `eval_action_mse.py` 在测试数据集上逐轨迹评估模型预测动作与真实动作之间的均方误差（MSE）。

### 4.1 基本用法

指定训练数据集路径加载 norm_stats，并指定仅评估前 5 条轨迹。评估时还可选择将预测动作与真实动作进行可视化对比。每个动作维度会生成一张子图，展示 gt action、pred action 和 state 的时序对比。

```bash
python scripts/eval_action_mse.py \
    --config_name pi05_u0bot \
    --checkpoint_dir checkpoints/pi05_u0bot/u0bot_finetune_v1/10999 \
    --test_repo_id /data/gujunwen/project/fish-vla/dataset/lerobot_test \
    --train_repo_id /data/gujunwen/project/fish-vla/dataset/lerobot_full \
    --max_trajs 5 \
    --save_csv_path results/eval_u0bot_test.csv \
    --save_plot_path results/plots \
    --eval_horizon 16
```

### 4.2 在整个测试集进行评估

```bash
python scripts/eval_action_mse.py \
    --config_name pi05_u0bot \
    --checkpoint_dir checkpoints/pi05_u0bot/u0bot_finetune_bs32/21999 \
    --test_repo_id /data/gujunwen/project/fish-vla/dataset/lerobot_test \
    --save_csv_path results/eval_u0bot_test_sample.csv

# For base model, use the path below:
    --checkpoint_dir /data/gujunwen/model/pi05_base
```

## 5. 启动策略服务

```bash
python scripts/serve_policy.py \
    --config pi05_u0bot \
    --checkpoint_dir checkpoints/pi05_u0bot/u0bot_finetune_v1/10999
```
