# U0Bot 训练指南

## 环境准备

```bash
mamba create -n pi05 python==3.11
conda activate pi05
pip install uv
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
conda install -c conda-forge ffmpeg -y
pip install --force-reinstall nvidia-cudnn-cu12==9.12.0.46
```

## JAX 训练

```bash
tmux new -s my_training "source ~/miniconda3/bin/activate pi05 && \
WANDB_MODE=offline XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi05_u0bot \
    --exp-name=u0bot_finetune_v1 \
    --overwrite"
```

## PyTorch 训练（暂时放弃）

### 安装 PyTorch

```bash
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 torchcodec \
    --index-url https://download.pytorch.org/whl/cu130 \
    --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 替换 Transformers 模块

```bash
uv pip show transformers
cp -r ./src/openpi/models_pytorch/transformers_replace/* \
    /data/gujunwen/miniconda3/envs/pi05/lib/python3.11/site-packages/transformers/
```

### 转换模型权重

```bash
python examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /data/gujunwen/model/pi05_base \
    --config_name pi05_u0bot \
    --output_path /data/gujunwen/model/pi05_base_pytorch
```

### 启动训练

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/train_pytorch.py pi05_u0bot \
    --exp-name u0bot_finetune_v1 \
    --overwrite \
    --no-wandb-enabled \
    --pytorch-weight-path /data/gujunwen/model/pi05_base_pytorch
```
