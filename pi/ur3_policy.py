"""UR3 input/output transforms for openpi.

Adapted from the UR5 example (examples/ur5/README.md) for datasets recorded
with teleop_vision_record.py, which stores data in LeRobot v2.1 format with
the following field names:

    observation.state             float32 (7,)   joints(6) + gripper(1)
    action                        float32 (7,)
    observation.images.cam_high   video   224×224 RGB
    observation.images.cam_wrist  video   224×224 RGB
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    """Convert image to uint8 HWC.

    LeRobot stores video frames as float32 CHW in [0, 1]. During policy
    inference images arrive as uint8 HWC, so this handles both cases.
    """
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


# RepackTransform structure: {output_key: lerobot_dataset_key}
REPACK_STRUCTURE = {
    "observation/image":       "observation.images.cam_high",
    "observation/wrist_image": "observation.images.cam_wrist",
    "observation/state":       "observation.state",
    "actions":                 "action",
    "prompt":                  "prompt",
}


@dataclasses.dataclass(frozen=True)
class UR3Inputs(transforms.DataTransformFn):
    """Maps UR3 LeRobot dataset fields to openpi model inputs.

    Expects the dict produced by RepackTransform(REPACK_STRUCTURE):
        observation/image       uint8 or float32 (H,W,3) RGB
        observation/wrist_image uint8 or float32 (H,W,3) RGB
        observation/state       float32 (7,)
        actions                 float32 (7,)   [training only]
        prompt                  str            [optional]
    """

    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # observation.state is already (7,): joints(6) + gripper(1)
        state = np.asarray(data["observation/state"], dtype=np.float32)

        base_image  = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb":        base_image,
                "left_wrist_0_rgb":  wrist_image,
                # UR3 has no right wrist camera — pad with zeros
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb":        np.True_,
                "left_wrist_0_rgb":  np.True_,
                "right_wrist_0_rgb": (
                    np.True_
                    if self.model_type == _model.ModelType.PI0_FAST
                    else np.False_
                ),
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UR3Outputs(transforms.DataTransformFn):
    """Extracts UR3 actions (7 dims) from model output."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}
