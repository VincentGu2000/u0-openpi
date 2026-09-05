# π0.5 Baseline on USIM

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
![Python 3.11](https://img.shields.io/badge/Python-3.11-green.svg)

π0.5 fine-tuned on the **USIM** underwater Vision-Language-Action dataset — the π0.5
baseline of the paper *USIM and U0: A Vision-Language-Action Dataset and Model for General
Underwater Robots* ([arXiv:2510.07869](https://arxiv.org/abs/2510.07869)).
This repository provides the fine-tuning config, open-loop action-MSE evaluation, and a
GR00T-compatible HTTP inference service for closed-loop evaluation in the
[u0env](https://github.com/VincentGu2000/u0env) simulator.

> **Note**: This repository is forked from
> [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)
> (base commit `650c5b0`) and customized for the u0bot underwater robot and the USIM
> dataset. See [Changes vs. Upstream](#changes-vs-upstream). The upstream README is
> preserved in [README_UPSTREAM.md](README_UPSTREAM.md); a Chinese quick guide is available
> in [README_zh.md](README_zh.md).

Companion resources:

| Resource | Link |
|---|---|
| Main model (U0, based on GR00T N1.5) | [u0model](https://github.com/VincentGu2000/u0model) |
| OpenVLA baseline | [u0-openvla](https://github.com/VincentGu2000/u0-openvla) |
| Simulation environment | [u0env](https://github.com/VincentGu2000/u0env) |
| USIM dataset (LeRobot format) | [`Vincent2025hello/usim`](https://huggingface.co/datasets/Vincent2025hello/usim) |
| Fine-tuned π0.5 weights | [`Vincent2025hello/pi05-u0bot`](https://huggingface.co/Vincent2025hello/pi05-u0bot) |

## Features

- __Fine-Tuning__: `pi05_u0bot` train config (frozen PaliGemma LLM, trainable action expert
  and vision tower, absolute actions, batch 64) on the USIM LeRobot dataset
- __Evaluation__: per-trajectory open-loop action MSE over the 525-trajectory USIM test split
- __Inference__: GR00T-compatible HTTP server (`POST /act`, `GET /health`), the native
  WebSocket policy server, and a multi-instance, multi-GPU launcher
- __Data__: consumes the USIM dataset in LeRobot format directly (no conversion needed)

The u0bot uses 13-dim actions (5 joint positions + 8 thruster PWMs) and a 29-dim
proprioceptive state; the transforms live in `src/openpi/policies/u0bot_policy.py`.

---

## 1. Installation

```bash
conda create -n pi05 python=3.11 -y
conda activate pi05
git clone https://github.com/VincentGu2000/u0-openpi.git
cd u0-openpi
pip install uv
GIT_LFS_SKIP_SMUDGE=1 uv sync --all-groups
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
conda install -c conda-forge ffmpeg -y
pip install --force-reinstall nvidia-cudnn-cu12==9.12.0.46

# HTTP inference service dependencies (scripts/inference_service_openpi.py)
pip install uvicorn fastapi json-numpy requests
```

## 2. Quick Setup: Configure Local Paths

All file paths (model weights, datasets) are configured through environment variables so the
same commands work across machines.

```bash
cp .env.example .env    # then edit .env with your actual paths
source .env             # before running any command below
```

## 3. Download Model Weights

### Option A: Use Pre-Fine-Tuned Weights (No Training Required)

Download our fine-tuned π0.5 checkpoint directly from Hugging Face:

```bash
hf download Vincent2025hello/pi05-u0bot --local-dir $MODEL_BASE_DIR/pi05-u0bot
```

Point `--checkpoint_dir` at `$MODEL_BASE_DIR/pi05-u0bot` in the evaluation and inference
steps below.

### Option B: Use the Official π0.5 Base Weights (For Self Fine-Tuning)

The `pi05_u0bot` config loads the official base weights from
`gs://openpi-assets/checkpoints/pi05_base/params`; openpi downloads and caches them
automatically on first use (no manual step required).

## 4. Download Dataset

The USIM dataset is consumed directly in LeRobot format (no RLDS conversion needed):

```bash
hf download Vincent2025hello/usim --local-dir $DATA_BASE_DIR/usim
```

## 5. Fine-Tuning

Before training, compute the dataset normalization statistics once:

```bash
source .env
python scripts/compute_norm_stats.py \
    --config_name pi05_u0bot \
    --repo_id $DATA_BASE_DIR/usim/train
```

The recipe used in the paper (batch 64, 22000 steps = 2 epochs, frozen PaliGemma LLM,
absolute actions; the final checkpoint is saved at step 21999):

```bash
tmux new -s my_training "source .env && conda activate pi05 && \
HF_HUB_OFFLINE=1 WANDB_MODE=offline XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi05_u0bot \
    --exp-name=u0bot_finetune_bs32 \
    --overwrite"
```

## 6. Evaluation

Open-loop evaluation computes the per-trajectory action MSE on the USIM test split
(525 trajectories):

```bash
source .env
python scripts/eval_action_mse.py \
    --config_name pi05_u0bot \
    --checkpoint_dir $MODEL_BASE_DIR/pi05-u0bot \
    --test_repo_id $DATA_BASE_DIR/usim/test \
    --save_csv_path results/eval_u0bot_test_pi05_2ep_abs.csv
```

Useful options: `--max_trajs 5` (first 5 trajectories), `--train_repo_id` (norm stats
source, defaults to the config's `repo_id`), `--save_plot_path results/plots`
(per-dimension action plots), `--eval_horizon 16`. Run
`python scripts/eval_action_mse.py --help` for the full list.

### Results

The `results/` directory contains the per-trajectory CSVs produced by our runs
(each CSV ends with `simple_avg` / `weighted_avg` / `median` summary rows):

| Checkpoint | CSV | Mean action MSE (simple avg) |
|---|---|---|
| π0.5 fine-tuned on USIM (2 epochs, released) | `results/eval_u0bot_test_pi05_2ep_abs.csv` | **0.0809** |
| π0.5 base (not fine-tuned) | `results/eval_u0bot_test_pi05_pretrained.csv` | 0.1405 |

An additional delta-action variant run is released as-is: `results/eval_u0bot_test_pi05_2ep.csv`.

## 7. Inference

### 7.1 WebSocket Policy Server (Native)

```bash
python scripts/serve_policy.py \
    --config pi05_u0bot \
    --checkpoint_dir $MODEL_BASE_DIR/pi05-u0bot
```

### 7.2 HTTP Inference Service (GR00T-Compatible)

Starts a GR00T-compatible HTTP server for closed-loop evaluation in u0env:

```bash
source .env
python scripts/inference_service_openpi.py \
    --config pi05_u0bot \
    --checkpoint_dir $MODEL_BASE_DIR/pi05-u0bot \
    --host 0.0.0.0 \
    --port 8000
```

Test the endpoints:

```bash
curl http://localhost:8000/health
# POST /act with {"observation": {gr00t_obs_dict}} returns
# {"action.pwm": (H, 8), "action.joint_pos": (H, 5)}
```

### 7.3 Multi-Instance Inference on Multiple GPUs

```bash
source .env
python scripts/launch_multi_gpu.py \
    --num-instances 4 \
    --base-port 8000 \
    --gpus 0,1 \
    --config pi05_u0bot \
    --checkpoint-dir $MODEL_BASE_DIR/pi05-u0bot
```

Press `Ctrl+C` to stop all instances. Logs are written to `./logs/`.

### 7.4 Closed-Loop Evaluation with u0env

Refer to the [u0env](https://github.com/VincentGu2000/u0env) project for the simulation
environment and the automated evaluation pipeline. Its `ros_gr00t_bridge.py` talks to this
service without any modification — simply point it at the host/port above.

## Changes vs. Upstream

Based on [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)
@ `650c5b0`:

**Added**

- `src/openpi/policies/u0bot_policy.py` — u0bot input/output transforms (29-dim state
  assembly, 13-dim action handling)
- `pi05_u0bot` train config + `LeRobotU0BotDataConfig` in `src/openpi/training/config.py`
- `scripts/eval_action_mse.py` — open-loop action MSE evaluation on the USIM test set
- `scripts/inference_service_openpi.py` — GR00T-compatible HTTP inference service
- `scripts/launch_multi_gpu.py` — multi-instance, multi-GPU service launcher
- `results/*.csv`, `README_zh.md`, `.env.example`

**Modified**

- `scripts/train.py`, `scripts/compute_norm_stats.py` — compatibility fix for legacy LeRobot
  dataset metadata; wandb image logging gated behind `config.wandb_enabled`
- `pyproject.toml` / `uv.lock` — `jax[cuda13]==0.8.0`, `torch==2.10.0`, `torchcodec==0.10`,
  `orbax-checkpoint==0.11.17`, numpy ≥ 2, lerobot pinned to rev `0cf8648`
- `packages/openpi-client/pyproject.toml` — numpy version cap removed
- `.gitignore`

**Unchanged**: `examples/`, `docs/`, `src/openpi/` otherwise, `scripts/serve_policy.py`.

## Citation

If you use this work, please cite:

```bibtex
@misc{gu2025usimu0visionlanguageactiondataset,
      title={USIM and U0: A Vision-Language-Action Dataset and Model for General Underwater Robots}, 
      author={Junwen Gu and Zhiheng Wu and Pengxuan Si and Shuang Qiu and Yukai Feng and Luoyang Sun and Laien Luo and Lianyi Yu and Jian Wang and Zhengxing Wu},
      year={2025},
      eprint={2510.07869},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2510.07869}, 
}
```

## Acknowledgments

This project is built on top of [openpi](https://github.com/Physical-Intelligence/openpi).
We thank the Physical Intelligence team for open-sourcing the π0 / π0.5 models and framework.

## License

This project is licensed under the Apache License 2.0 (inherited from the upstream
repository). The π0.5 weights additionally fall under the Gemma terms — see
[LICENSE_GEMMA.txt](LICENSE_GEMMA.txt).
