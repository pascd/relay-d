"""
inference_node.py
-----------------
CLI entry point. Thin wrapper around api.InferenceSession, which does the
actual work: grab latest obs -> run policy forward pass -> dispatch action.

Two backends, chosen from the checkpoint unless --backend says otherwise:
  robomimic  a robomimic checkpoint file (.pth), run with robomimic's own RolloutPolicy
  lerobot    a LeRobot pretrained_model folder (ACT, Diffusion, ...), run with LeRobot

Usage (after `pip install -e .`):
    lfd-inference-node \
        --checkpoint path/to/model.pth \
        --obs-config path/to/obs_config.yaml \
        --action-config path/to/action_config.yaml \
        --device cpu

    lfd-inference-node \
        --checkpoint path/to/run/output/checkpoints/last/pretrained_model \
        --obs-config path/to/obs_config.yaml \
        --action-config path/to/action_config.yaml
"""

import argparse

from relay_d.utils import user_settings

from .api import InferenceSession


def main():
    parser = argparse.ArgumentParser(description="LfD Inference Node")
    parser.add_argument("--checkpoint",    required=True,
                        help="robomimic checkpoint (.pth, a pickle that runs code on load: only use checkpoints you trust) "
                             "or a LeRobot pretrained_model folder (e.g. .../checkpoints/last/pretrained_model)")
    parser.add_argument("--obs-config",    required=True,  help="Path to obs_config.yaml")
    parser.add_argument("--action-config", required=True,  help="Path to action_config.yaml")
    parser.add_argument("--device",        default=user_settings.get("infer_device"),  help="Torch device: cpu | cuda (default: the infer_device setting)")
    parser.add_argument("--hz",            type=float, default=user_settings.get("infer_hz"),
                        help="Max inference rate in Hz (0 = as fast as possible; default: the infer_hz setting)")
    parser.add_argument("--max-obs-age",   type=float, default=user_settings.get("infer_max_obs_age_s"),
                        help="Skip dispatching while any input is older than this many seconds (default: the infer_max_obs_age_s setting)")
    parser.add_argument("--obs-wait",      type=float, default=user_settings.get("infer_obs_wait_s"),
                        help="Seconds to wait for all observations before starting inference (default: the infer_obs_wait_s setting)")
    parser.add_argument("--backend",       choices=["auto", "robomimic", "lerobot"], default=user_settings.get("infer_backend"),
                        help="Inference backend. auto: robomimic for a .pth/.pt file, lerobot for a pretrained_model "
                             "folder (default: the infer_backend setting)")
    parser.add_argument("--state-keys",    default=None,
                        help="lerobot only: comma-separated observations that form observation.state, in order. "
                             "Default: read from the dataset the policy was trained on")
    args = parser.parse_args()

    state_keys = [k.strip() for k in args.state_keys.split(",")] if args.state_keys else None

    session = InferenceSession(
        checkpoint=args.checkpoint,
        obs_config=args.obs_config,
        action_config=args.action_config,
        device=args.device,
        backend=args.backend,
        state_keys=state_keys,
    )
    try:
        session.run(hz=args.hz, validate_timeout=args.obs_wait, max_obs_age=args.max_obs_age)
    finally:
        session.close()


if __name__ == "__main__":
    main()
