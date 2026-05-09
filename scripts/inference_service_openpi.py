#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenPI Inference Service (HTTP)
为 fish-vla 仿真闭环测评提供与 GR00T HTTP Server 兼容的推理接口。

暴露与 GR00T 完全相同的 API：
    POST /act
    Request:  {"observation": {gr00t_obs_dict}}
    Response: {"action.pwm": (H, 8), "action.joint_pos": (H, 5)}

这样 ros_gr00t_bridge.py 无需任何修改，只需指向本服务即可。

注意：不使用 json_numpy.patch()，因为它会全局替换 json.loads()，
导致 orbax（JAX checkpoint 加载）和 scipy 等库内部使用 json.loads 时崩溃。
改为在 HTTP 请求/响应处理中显式使用 json_numpy.dumps()/loads()。

Dependencies:
    => Server: pip install uvicorn fastapi json_numpy
    => Client: pip install requests json_numpy

Usage:
    # 启动服务
    python scripts/inference_service_openpi.py \
        --config pi05_u0bot \
        --checkpoint_dir checkpoints/pi05_u0bot/u0bot_finetune_v1/10999 \
        --host 0.0.0.0 \
        --port 8000

    # Bridge 对接（无需修改 ros_gr00t_bridge.py）
    rosrun bluerov2_control ros_gr00t_bridge.py _host:=0.0.0.0 _port:=8000 _mode:=pwm
"""

import os
# JAX 显存分配策略：必须在 JAX 导入之前设置
# 设置为 false 表示按需分配，而不是预分配固定比例的 GPU 显存
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import logging
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

# 导入 openpi 模块（必须在 json_numpy 之前，因为 orbax 不兼容 json_numpy patch）
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

# 仅导入 json_numpy，不调用 patch()
# 在 HTTP 请求/响应处理中显式使用 json_numpy.dumps()/loads()
import json_numpy

logger = logging.getLogger(__name__)


# ============================================================================
# 观测格式转换：GR00T → OpenPI
# ============================================================================
def convert_obs_gr00t_to_openpi(gr00t_obs: dict) -> dict:
    """将 GR00T Bridge 发送的观测格式转换为 OpenPI policy 期望的格式。

    GR00T obs keys:
        video.ego:          (1, 240, 320, 3) uint8
        video.wrist:        (1, 240, 320, 3) uint8
        state.joint_pos:    (1, 5) float32
        state.pwm:          (1, 8) float32
        state.joint_v:      (1, 5) float32
        state.dvl_v:        (1, 3) float32
        state.imu_av:       (1, 3) float32
        state.imu_la:       (1, 3) float32
        state.pressure:     (1, 1) float32
        state.dvl_h:        (1, 1) float32
        annotation.human.action.task_description: [str]

    OpenPI expected keys:
        observation/ego_image:   (H, W, 3) uint8
        observation/wrist_image: (H, W, 3) uint8
        observation/state:       (29,) float32
        prompt:                  str
    """
    # 提取图像：去掉 batch 维度 (1, H, W, 3) -> (H, W, 3)
    ego_image = gr00t_obs["video.ego"]
    if ego_image.ndim == 4:
        ego_image = ego_image[0]

    wrist_image = gr00t_obs["video.wrist"]
    if wrist_image.ndim == 4:
        wrist_image = wrist_image[0]

    # 拼接 state：29 维
    # joint_pos(5) + pwm(8) + joint_v(5) + dvl_v(3) + imu_av(3) + imu_la(3) + pressure(1) + dvl_h(1) = 29
    state_parts = []
    for key in [
        "state.joint_pos",
        "state.pwm",
        "state.joint_v",
        "state.dvl_v",
        "state.imu_av",
        "state.imu_la",
        "state.pressure",
        "state.dvl_h",
    ]:
        val = gr00t_obs.get(key)
        if val is not None:
            val = np.asarray(val).flatten()
            state_parts.append(val)
        else:
            # 如果某个 state 缺失，用零填充（根据维度推断）
            dim_map = {
                "state.joint_pos": 5,
                "state.pwm": 8,
                "state.joint_v": 5,
                "state.dvl_v": 3,
                "state.imu_av": 3,
                "state.imu_la": 3,
                "state.pressure": 1,
                "state.dvl_h": 1,
            }
            state_parts.append(np.zeros(dim_map[key], dtype=np.float32))

    state = np.concatenate(state_parts).astype(np.float32)

    # 提取任务描述
    task_desc = gr00t_obs.get("annotation.human.action.task_description", "")
    if isinstance(task_desc, list):
        task_desc = task_desc[0] if len(task_desc) > 0 else ""
    prompt = str(task_desc)

    return {
        "observation/ego_image": ego_image,
        "observation/wrist_image": wrist_image,
        "observation/state": state,
        "prompt": prompt,
    }


# ============================================================================
# 动作格式转换：OpenPI → GR00T
# ============================================================================
def convert_action_openpi_to_gr00t(openpi_output: dict) -> dict:
    """将 OpenPI policy 输出转换为 GR00T Bridge 期望的格式。

    OpenPI output (after U0BotOutputs transform):
        {"actions": (action_horizon, 13)}
        前 5 维 = joint_pos, 后 8 维 = pwm

    GR00T expected:
        {"action.pwm": (action_horizon, 8), "action.joint_pos": (action_horizon, 5)}
    """
    actions = np.asarray(openpi_output["actions"])

    # actions: (action_horizon, 13)
    # 前 5 维 -> joint_pos, 后 8 维 -> pwm
    action_joint_pos = actions[:, :5].astype(np.float32)
    action_pwm = actions[:, 5:].astype(np.float32)

    return {
        "action.pwm": action_pwm,
        "action.joint_pos": action_joint_pos,
    }


# ============================================================================
# HTTP 推理服务
# ============================================================================
class OpenPIInferenceServer:
    """HTTP 推理服务，暴露与 GR00T 兼容的 /act 端点。"""

    def __init__(
        self,
        config_name: str,
        checkpoint_dir: str,
        host: str = "0.0.0.0",
        port: int = 8000,
        default_prompt: Optional[str] = None,
        pytorch_device: Optional[str] = None,
    ):
        self.host = host
        self.port = port

        # 加载 OpenPI policy
        logger.info(f"Loading OpenPI policy: config={config_name}, checkpoint={checkpoint_dir}")
        train_config = _config.get_config(config_name)
        self.policy = _policy_config.create_trained_policy(
            train_config,
            checkpoint_dir,
            default_prompt=default_prompt,
            pytorch_device=pytorch_device,
        )
        logger.info("Policy loaded successfully.")

        # 创建 FastAPI app
        self.app = FastAPI(title="OpenPI Inference Server", version="1.0.0")
        self.app.post("/act")(self.predict_action)
        self.app.get("/health")(self.health_check)

    async def predict_action(self, request: Request) -> Response:
        """推理端点，兼容 GR00T HTTP Server 格式。

        使用 Request/Response 直接处理原始字节，
        通过 json_numpy 显式序列化/反序列化 numpy 数组。

        接收: {"observation": {gr00t_obs_dict}}
        返回: {"action.pwm": ..., "action.joint_pos": ...}
        """
        try:
            # 读取原始请求体，使用 json_numpy 反序列化（支持 numpy 数组）
            body = await request.body()
            payload = json_numpy.loads(body)

            # 提取 observation
            if "observation" not in payload:
                raise HTTPException(
                    status_code=400,
                    detail="Missing 'observation' field in payload. Expected: {'observation': {...}}",
                )

            gr00t_obs = payload["observation"]

            # 转换观测格式: GR00T → OpenPI
            openpi_obs = convert_obs_gr00t_to_openpi(gr00t_obs)

            # 推理
            infer_start = time.time()
            openpi_output = self.policy.infer(openpi_obs)
            infer_time = time.time() - infer_start
            logger.info(f"Inference time: {infer_time:.3f}s")

            # 转换输出格式: OpenPI → GR00T
            gr00t_action = convert_action_openpi_to_gr00t(openpi_output)

            # 使用 json_numpy 序列化响应（支持 numpy 数组）
            response_body = json_numpy.dumps(gr00t_action)
            return Response(content=response_body, media_type="application/json")

        except HTTPException:
            raise
        except Exception as e:
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: {str(e)}",
            )

    def health_check(self) -> Dict[str, str]:
        """健康检查端点。"""
        return {"status": "healthy", "model": "OpenPI"}

    def run(self) -> None:
        """启动 HTTP 服务。"""
        logger.info(f"Starting OpenPI HTTP server on {self.host}:{self.port}")
        logger.info("Available endpoints:")
        logger.info("  POST /act    - Get action prediction from observation")
        logger.info("  GET  /health - Health check")
        uvicorn.run(self.app, host=self.host, port=self.port)


# ============================================================================
# 命令行入口
# ============================================================================
@dataclass
class ServerConfig:
    """命令行参数配置。"""

    # 模型配置名称（对应 openpi/src/openpi/training/config.py 中的 _CONFIGS）
    config: str = "pi05_u0bot"
    """OpenPI 训练配置名称。"""

    # 模型 checkpoint 目录
    checkpoint_dir: str = "checkpoints/pi05_u0bot/u0bot_finetune_v1/10999"
    """Path to the model checkpoint directory."""

    # 服务器配置
    host: str = "0.0.0.0"
    """Server host address."""
    port: int = 8000
    """Server port."""

    # 可选配置
    default_prompt: Optional[str] = None
    """Default prompt if not provided in observation."""
    pytorch_device: Optional[str] = None
    """PyTorch device (e.g., 'cuda:0', 'cpu'). Auto-detected if not specified."""


def main():
    import tyro
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    cfg = tyro.cli(ServerConfig)
    server = OpenPIInferenceServer(
        config_name=cfg.config,
        checkpoint_dir=cfg.checkpoint_dir,
        host=cfg.host,
        port=cfg.port,
        default_prompt=cfg.default_prompt,
        pytorch_device=cfg.pytorch_device,
    )
    server.run()


if __name__ == "__main__":
    main()
