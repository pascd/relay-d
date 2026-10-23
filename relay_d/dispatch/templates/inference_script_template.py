import argparse
import os
import sys

from relay_d.dispatch.api import InferenceSession, detect_backend
from relay_d.dispatch.reexec import reexec_if_needed

# Default configuration file paths
ACTION_CONFIG="./action_config.yaml"
OBS_CONFIG="./obs_config.yaml"

def main():
  # Create argument parser for command-line interface
  parser = argparse.ArgumentParser(description="LfD Inference Node")

  # Required checkpoint path argument
  parser.add_argument("--checkpoint",
                      required=True,
                      help="robomimic checkpoint (.pth) or LeRobot pretrained_model folder")

  # Observation configuration file path
  parser.add_argument("--obs-config",
                      help="Path to obs_config.yaml",
                      default=OBS_CONFIG)

  # Action configuration file path
  parser.add_argument("--action-config",
                      help="Path to action_config.yaml",
                      default=ACTION_CONFIG)

  # Torch device selection (CPU or CUDA)
  parser.add_argument("--device",
                      default="cuda",
                      help="Torch device: cpu | cuda")

  # Maximum inference rate in Hz (0 = unlimited)
  parser.add_argument("--hz",
                      type=float,
                      default=0.0,
                      help="Max inference rate in Hz (0 = as fast as possible)")

  # Wait time for all observations before inference starts
  parser.add_argument("--obs-wait",
                      type=float,
                      default=5.0,
                      help="Seconds to wait for all observations before starting inference (default: 5.0)")

  # Inference backend: robomimic .pth file or LeRobot pretrained_model folder
  parser.add_argument("--backend",
                      choices=["auto", "robomimic", "lerobot"],
                      default="auto",
                      help="auto: robomimic for a .pth/.pt file, lerobot for a pretrained_model folder")

  # LeRobot only: which observations form observation.state
  parser.add_argument("--state-keys",
                      default=None,
                      help="lerobot only: comma-separated observations that form observation.state, in order "
                            "(default: read from the dataset the policy was trained on)")

  # Parse command-line arguments
  args = parser.parse_args()

  # The package of the backend (lerobot / robomimic) may live in another environment, e.g. the one built by the Train tab:
  # then this script starts again with that Python (same arguments).
  try:
    reexec_if_needed(args.backend if args.backend != "auto" else detect_backend(args.checkpoint),
                     [os.path.abspath(sys.argv[0]), *sys.argv[1:]])
  except ValueError:
    pass  # an unknown checkpoint: InferenceSession reports it

  # Initialize inference session with parsed arguments
  session = InferenceSession(
          checkpoint=args.checkpoint,
          obs_config=args.obs_config,
          action_config=args.action_config,
          device=args.device,
          backend=args.backend,
          state_keys=[k.strip() for k in args.state_keys.split(",")] if args.state_keys else None,
      )

  # Start inference session
  session.run(hz=args.hz, validate_timeout=args.obs_wait)

if __name__ == "__main__":
  main()
