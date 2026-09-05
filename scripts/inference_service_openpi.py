#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenPI Inference Service (HTTP)
Provides a GR00T HTTP Server-compatible inference endpoint for closed-loop
evaluation in the u0env simulator (https://github.com/VincentGu2000/u0env).

Exposes exactly the same API as the GR00T HTTP Server:
    POST /act
    Request:  {"observation": {gr00t_obs_dict}}
    Response: {"action.pwm": (H, 8), "action.joint_pos": (H, 5)}

This way ros_gr00t_bridge.py (from the u0env ROS workspace) needs no
modification -- simply point it to this server.

Note: json_numpy.patch() is NOT used, because it globally replaces json.loads()
and breaks orbax (JAX checkpoint loading) and scipy, which call json.loads()
internally. json_numpy.dumps()/loads() are used explicitly in the HTTP
request/response handling instead.

Dependencies:
    => Server: pip install uvicorn fastapi json_numpy
    => Client: pip install requests json_numpy

Usage:
    # Start the server
    python scripts/inference_service_openpi.py \
        --config pi05_u0bot \
        --checkpoint_dir $MODEL_BASE_DIR/u0_pi05 \
        --host 0.0.0.0 \
        --port 8000

    # Connect the bridge (no modification needed in ros_gr00t_bridge.py)
    rosrun bluerov2_control ros_gr00t_bridge.py _host:=0.0.0.0 _port:=8000 _mode:=pwm
"""

import os
# JAX 显存分配策略：必须在 JAX 导入之前设置
# 设置为 false 表示按需分配，而不是预分配固定比例的 GPU 显存
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import datetime
import logging
import os
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
# Debug 记录器
# ============================================================================
class DebugRecorder:
    """记录推理服务的输入观测和输出动作，用于调试和验证数据正确性。

    输出为人类可读的 .log 文本文件，可直接用文本编辑器打开查看。
    """

    def __init__(self, record_dir: str):
        self._record_dir = os.path.abspath(record_dir)
        os.makedirs(self._record_dir, exist_ok=True)
        self._record_step = 0
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_path = os.path.join(self._record_dir, f"debug_{timestamp}.log")
        self._log_file = open(self._log_path, "w", encoding="utf-8")
        self._log_file.write(f"{'='*80}\n")
        self._log_file.write(f"DebugRecorder - 推理调试日志\n")
        self._log_file.write(f"开始时间: {datetime.datetime.now().isoformat()}\n")
        self._log_file.write(f"{'='*80}\n\n")
        logger.info(f"[DebugRecorder] 日志文件: {self._log_path}")

    @staticmethod
    def _format_value(val, indent=2):
        """将值格式化为可读字符串。"""
        prefix = " " * indent
        if isinstance(val, np.ndarray):
            lines = [f"{prefix}ndarray: shape={val.shape}, dtype={val.dtype}"]
            flat = val.flatten()
            max_show = 20
            if len(flat) <= max_show:
                lines.append(f"{prefix}  values: {np.array2string(flat, precision=6, separator=', ')}")
            else:
                lines.append(f"{prefix}  前{max_show}个值: {np.array2string(flat[:max_show], precision=6, separator=', ')}")
                lines.append(f"{prefix}  ... 共 {len(flat)} 个元素")
            return "\n".join(lines)
        elif isinstance(val, dict):
            if not val:
                return f"{prefix}{{}}"
            lines = []
            for k, v in val.items():
                formatted = DebugRecorder._format_value(v, indent + 2)
                lines.append(f"{prefix}{k}:")
                lines.append(formatted)
            return "\n".join(lines)
        elif isinstance(val, (list, tuple)):
            if not val:
                return f"{prefix}[]"
            lines = [f"{prefix}[len={len(val)}]"]
            for i, v in enumerate(val[:5]):
                formatted = DebugRecorder._format_value(v, indent + 2)
                lines.append(f"{prefix}  [{i}]:")
                lines.append(formatted)
            if len(val) > 5:
                lines.append(f"{prefix}  ... 共 {len(val)} 个元素")
            return "\n".join(lines)
        elif isinstance(val, (int, float)):
            return f"{prefix}{val}"
        elif isinstance(val, str):
            return f'{prefix}"{val}"'
        else:
            return f"{prefix}{str(val)}"

    def record(self, obs: dict, action: dict, extra: dict = None) -> None:
        """记录一次推理的输入和输出。"""
        f = self._log_file
        f.write(f"{'─'*80}\n")
        f.write(f"Step {self._record_step} | {datetime.datetime.now().isoformat()}\n")
        f.write(f"{'─'*80}\n")

        f.write("[输入观测 (observation)]:\n")
        f.write(self._format_value(obs) + "\n")

        f.write("[输出动作 (action)]:\n")
        f.write(self._format_value(action) + "\n")

        if extra is not None:
            f.write("[附加信息 (extra)]:\n")
            f.write(self._format_value(extra) + "\n")

        f.write("\n")
        f.flush()
        logger.info(f"[DebugRecorder] 已记录 step {self._record_step}")
        self._record_step += 1

    def close(self):
        """关闭日志文件。"""
        if self._log_file and not self._log_file.closed:
            self._log_file.write(f"{'='*80}\n")
            self._log_file.write(f"DebugRecorder 结束 | 总步数: {self._record_step}\n")
            self._log_file.write(f"{'='*80}\n")
            self._log_file.close()
            logger.info(f"[DebugRecorder] 日志已关闭: {self._log_path}")


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
        debug_dir: Optional[str] = None,
    ):
        self.host = host
        self.port = port

        # Debug 记录器
        self.debug_recorder = DebugRecorder(debug_dir) if debug_dir else None
        if self.debug_recorder:
            logger.info(f"[Debug] 已启用 debug 模式，记录到: {debug_dir}")

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

            # Debug 记录
            if self.debug_recorder:
                self.debug_recorder.record(
                    obs=gr00t_obs,
                    action=gr00t_action,
                    extra={"infer_time": infer_time},
                )

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
    checkpoint_dir: str = os.path.join(
        os.path.expanduser(os.environ.get("MODEL_BASE_DIR", "~/models")), "u0_pi05"
    )
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

    # Debug 模式
    debug_dir: Optional[str] = None
    """Debug 记录目录。如果指定，将记录每次推理的输入观测和输出动作到该目录。"""


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
        debug_dir=cfg.debug_dir,
    )
    try:
        server.run()
    finally:
        if server.debug_recorder:
            server.debug_recorder.close()


if __name__ == "__main__":
    main()
