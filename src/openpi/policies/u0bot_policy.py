import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_u0bot_example() -> dict:
    """Creates a random input example for the u0bot policy."""
    return {
        "observation/ego_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.rand(29),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class U0BotInputs(transforms.DataTransformFn):
    """Converts inputs from the u0bot format to the model's expected format.

    Used for both training and inference.

    The u0bot dataset provides:
    - observation.images.ego: third-person ego camera (240x320x3)
    - observation.images.wrist: wrist camera (240x320x3)
    - observation.state: 29-dim state (joint_pos, pwm, joint_v, dvl_v, imu_av, imu_la, pressure, dvl_h)
    - action: 13-dim actions (joint_pos, pwm)
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Parse images to uint8 (H, W, C) format.
        # LeRobot stores images as float32 (C, H, W), so we need to convert.
        ego_image = _parse_image(data["observation/ego_image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # pi0/pi0.5 models support three image inputs: base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb.
        # We map ego -> base_0_rgb, wrist -> left_wrist_0_rgb, and pad right_wrist with zeros.
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": ego_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(ego_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (language instruction) to the model.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class U0BotOutputs(transforms.DataTransformFn):
    """Converts model outputs back to the u0bot format.

    Used for inference only. Trims the padded actions back to 13 dimensions.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first 13 actions (the rest is padding from action_dim=32).
        return {"actions": np.asarray(data["actions"][:, :13])}
