# U0Bot 训练与评估指南（USIM / π0.5）

> 使用前先配置路径环境变量：`cp .env.example .env` 并按机器实际情况编辑，随后在执行下方命令前先 `source .env`。
> 下文用 `$MODEL_BASE_DIR` / `$DATA_BASE_DIR` 分别表示权重与数据根目录。英文版说明见 [README.md](README.md)。

## 1. 环境准备

```bash
mamba create -n pi05 python==3.11
conda activate pi05
pip install uv
GIT_LFS_SKIP_SMUDGE=1 uv sync --all-groups
conda install -c conda-forge ffmpeg -y
pip install --force-reinstall nvidia-cudnn-cu12==9.12.0.46
```

## 2. 数据准备

```bash
# 下载 USIM 数据集（LeRobot 格式）
hf download Vincent2025hello/usim --local-dir $DATA_BASE_DIR/usim
```

## 3. 计算归一化统计量

在训练之前，需要先为数据集计算归一化统计量（norm stats）：

```bash
python scripts/compute_norm_stats.py \
    --config_name pi05_u0bot \
    --repo_id $DATA_BASE_DIR/usim/train
```

## 4. JAX 训练

```bash
tmux new -s my_training "source .env && source ~/miniconda3/bin/activate pi05 && \
HF_HUB_OFFLINE=1 WANDB_MODE=offline XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi05_u0bot \
    --exp-name=u0bot_finetune_bs32 \
    --overwrite"
```

## 5. 评估动作预测 MSE

使用 `eval_action_mse.py` 在测试数据集上逐轨迹评估模型预测动作与真实动作之间的均方误差（MSE）。

### 5.1 基本用法

指定训练数据集路径加载 norm_stats，并指定仅评估前 5 条轨迹。评估时还可选择将预测动作与真实动作进行可视化对比。每个动作维度会生成一张子图，展示 gt action、pred action 和 state 的时序对比。

```bash
python scripts/eval_action_mse.py \
    --config_name pi05_u0bot \
    --checkpoint_dir checkpoints/pi05_u0bot/u0bot_finetune_v1/10999 \
    --test_repo_id $DATA_BASE_DIR/usim/test \
    --train_repo_id $DATA_BASE_DIR/usim/train \
    --max_trajs 5 \
    --save_csv_path results/eval_u0bot_test.csv \
    --save_plot_path results/plots \
    --eval_horizon 16
```

### 5.2 在整个测试集进行评估

```bash
python scripts/eval_action_mse.py \
    --config_name pi05_u0bot \
    --checkpoint_dir checkpoints/pi05_u0bot/u0bot_finetune_bs32/21999 \
    --test_repo_id $DATA_BASE_DIR/usim/test \
    --save_csv_path results/eval_u0bot_test_pi05_2ep_abs.csv

# For base model, use the path below:
    --checkpoint_dir $MODEL_BASE_DIR/pi05_base/params
```

## 6. 启动策略服务

> 注意：启动服务会从数据集目录（`$DATA_BASE_DIR/usim/train/`）读取 `norm_stats.json`。
> 若未进行训练，请先执行第 3 节的归一化统计计算步骤。

### 6.1 WebSocket 策略服务（原生）

```bash
python scripts/serve_policy.py \
    --config pi05_u0bot \
    --checkpoint_dir checkpoints/pi05_u0bot/u0bot_finetune_bs32/21999
```

### 6.2 HTTP 推理服务（仿真闭环测评）

为 [u0env](https://github.com/VincentGu2000/u0env) 仿真闭环测评提供与 GR00T 兼容的 HTTP 推理接口。
`ros_gr00t_bridge.py` 无需任何修改，只需指向本服务即可。

单模型部署：
```bash
pip install uvicorn fastapi json-numpy requests
python scripts/inference_service_openpi.py \
    --config pi05_u0bot \
    --checkpoint_dir checkpoints/pi05_u0bot/u0bot_finetune_bs32/21999 \
    --host 0.0.0.0 \
    --port 8000 \
    --debug-dir logs/gr00t
```

多模型部署：
```bash
python scripts/launch_multi_gpu.py \
    --num-instances 4 \
    --base-port 8000 \
    --gpus 0,1 \
    --config pi05_u0bot \
    --checkpoint-dir checkpoints/pi05_u0bot/u0bot_finetune_bs32/21999
```
