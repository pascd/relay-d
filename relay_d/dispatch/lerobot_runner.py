"""LeRobot inference backend: runs a policy trained with LeRobot (ACT, Diffusion, ...) on live observations.

Same interface as ``RobomimicPolicyRunner``: ``get_action(obs_dict) -> np.ndarray (A,)``, ``reset()`` and ``obs_keys``.
The checkpoint is a LeRobot ``pretrained_model`` folder (``config.json`` + ``model.safetensors`` + the pre/post-processor
files), e.g. ``<run>/output/checkpoints/last/pretrained_model`` from the Train tab. LeRobot itself normalizes the inputs and
un-normalizes the action with the statistics stored in that folder; this module only has to give it the same inputs the
dataset had:

- ``observation.state``: the state obs side by side, in the dataset's order (read from the dataset's ``meta/info.json``);
- ``observation.images.<name>``: the camera obs, converted BGR -> RGB (the dataset video is RGB) and to CHW float [0, 1],
  cropped to the size the policy was trained with (the writer crops odd sizes to even).

Importing this module needs only numpy; torch and lerobot are imported when a runner is created.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from relay_d.utils.coloring_logger import logger

STATE = "observation.state"
IMAGE_PREFIX = "observation.images."


def resolve_checkpoint(path) -> Optional[Path]:
    """The ``pretrained_model`` folder for ``path`` (that folder, or a folder holding one, e.g. ``checkpoints/last``)."""
    path = Path(path)
    for candidate in (path, path / "pretrained_model"):
        if (candidate / "config.json").is_file():
            return candidate
    return None


def state_layout_from_names(names: List[str]) -> List[Tuple[str, int]]:
    """``['a.0', 'a.1', 'b']`` (the names the converter writes) -> ``[('a', 2), ('b', 1)]``: obs key and width, in order."""
    layout: List[Tuple[str, int]] = []
    for name in names:
        m = re.fullmatch(r"(.+)\.(\d+)", name)
        key = m.group(1) if m else name
        if m and layout and layout[-1][0] == key:
            layout[-1] = (key, layout[-1][1] + 1)
        else:
            layout.append((key, 1))
    return layout


def dataset_info(checkpoint: Path) -> Optional[dict]:
    """``meta/info.json`` of the dataset this checkpoint was trained on (via ``train_config.json``), if it is still there."""
    try:
        root = json.loads((checkpoint / "train_config.json").read_text())["dataset"]["root"]
        return json.loads((Path(root) / "meta" / "info.json").read_text())
    except (OSError, ValueError, KeyError, TypeError):
        return None


def build_batch(obs: Dict[str, np.ndarray], state_keys: List[str], images: Dict[str, Tuple[str, tuple]]) -> Dict[str, np.ndarray]:
    """The LeRobot input for one tick, as numpy arrays without a batch dimension.

    ``images`` maps an obs name to ``(feature name, (C, H, W))`` of the policy's image input.
    """
    missing = [k for k in [*state_keys, *images] if k not in obs]
    if missing:
        raise KeyError(f"[LeRobotPolicyRunner] observations missing: {missing}. Available: {sorted(obs)}")
    batch = {STATE: np.concatenate([np.asarray(obs[k], dtype=np.float32).reshape(-1) for k in state_keys])}
    for name, (feature, (_c, h, w)) in images.items():
        frame = np.asarray(obs[name])
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.shape[0] < h or frame.shape[1] < w or frame.shape[0] - h > 1 or frame.shape[1] - w > 1:
            raise ValueError(
                f"[LeRobotPolicyRunner] camera '{name}' is {frame.shape}, the policy was trained on ({h}, {w}, 3) "
                "(only an odd row / column of difference is cropped, like the dataset writer does)."
            )
        rgb = frame[:h, :w, ::-1]  # live frames are BGR, the dataset video (and so the policy) is RGB
        batch[feature] = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
    return batch


class LeRobotPolicyRunner:
    """Runs a LeRobot policy checkpoint on ObsBuilder's observations."""

    def __init__(
        self,
        checkpoint: str,
        device: str = "cpu",
        state_keys: Optional[List[str]] = None,
        fallback_state_keys: Optional[List[str]] = None,
    ):
        """``state_keys``: obs names that form ``observation.state``, in order. If None they are read from the dataset
        the policy was trained on; if that is gone, ``fallback_state_keys`` (the obs_config order) is used with a warning."""
        folder = resolve_checkpoint(checkpoint)
        if folder is None:
            raise FileNotFoundError(
                f"'{checkpoint}' is not a LeRobot checkpoint: expected a folder with config.json and model.safetensors, "
                "e.g. <run>/output/checkpoints/last/pretrained_model"
            )
        try:
            import torch
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        except ImportError as e:
            raise ImportError(
                "LeRobot is required for --backend lerobot but is not installed in this environment: "
                "pip install 'lerobot[training]' (LeRobot 0.6 needs torch < 2.12). See README.md."
            ) from e
        self._torch = torch

        cfg = PreTrainedConfig.from_pretrained(str(folder))
        cfg.device = device
        self._policy = get_policy_class(cfg.type).from_pretrained(str(folder), config=cfg)
        self._policy.eval()
        self._pre, self._post = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=str(folder), preprocessor_overrides={"device_processor": {"device": device}}
        )

        features = cfg.input_features or {}
        unsupported = [k for k in features if k != STATE and not k.startswith(IMAGE_PREFIX)]
        if unsupported:
            raise ValueError(f"[LeRobotPolicyRunner] unsupported policy inputs {unsupported}: only {STATE} and {IMAGE_PREFIX}* are fed")
        self._state_dim = int(features[STATE].shape[0]) if STATE in features else 0
        self._images = {k[len(IMAGE_PREFIX):]: (k, tuple(f.shape)) for k, f in features.items() if k.startswith(IMAGE_PREFIX)}

        self.state_keys = self._resolve_state_keys(folder, state_keys, fallback_state_keys)
        self.obs_keys: List[str] = [*self.state_keys, *self._images]
        logger.info(
            f"[LeRobotPolicyRunner] policy={cfg.type} device={device} state={self.state_keys} "
            f"cameras={list(self._images)} checkpoint={folder}"
        )
        self.reset()

    def _resolve_state_keys(self, folder: Path, explicit, fallback) -> List[str]:
        if explicit:
            return list(explicit)
        info = dataset_info(folder)
        names = ((info or {}).get("features", {}).get(STATE) or {}).get("names")
        if isinstance(names, list):
            layout = state_layout_from_names(names)
            if self._state_dim and sum(w for _, w in layout) != self._state_dim:
                raise ValueError(
                    f"[LeRobotPolicyRunner] the dataset's {STATE} has {sum(w for _, w in layout)} values, "
                    f"the policy expects {self._state_dim}: the checkpoint and its dataset do not match."
                )
            return [k for k, _ in layout]
        if fallback:
            logger.warning(
                "[LeRobotPolicyRunner] the training dataset (train_config.json -> dataset.root) was not found, so the "
                f"state order is taken from obs_config.yaml: {list(fallback)}. Check it, or pass --state-keys."
            )
            return list(fallback)
        raise ValueError("[LeRobotPolicyRunner] cannot tell which observations form observation.state: pass --state-keys")

    def reset(self) -> None:
        self._policy.reset()
        self._pre.reset()
        self._post.reset()

    def get_action(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        batch = build_batch(obs_dict, self.state_keys, self._images)
        if self._state_dim and batch[STATE].shape[0] != self._state_dim:
            widths = {k: int(np.asarray(obs_dict[k]).size) for k in self.state_keys}
            raise ValueError(
                f"[LeRobotPolicyRunner] {STATE} has {batch[STATE].shape[0]} values {widths}, the policy expects {self._state_dim}. "
                "Check the observations (or --state-keys) against the dataset the policy was trained on."
            )
        tensors = {k: self._torch.from_numpy(v) for k, v in batch.items()}
        with self._torch.no_grad():
            action = self._post(self._policy.select_action(self._pre(tensors)))
        return action.detach().cpu().numpy().astype(np.float32).reshape(-1)
