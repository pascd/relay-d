# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **"Generate Defaults..." buttons** on the Config and Post-process pages: copy
  `config_template.yaml` and `postprocess_config_template.yaml` into a folder you pick. That
  folder is saved as the `config_folder` setting and is where the Insert / Load Config /
  Load Options dialogs open; when unset they use the HDF5 load folder from Settings.
- **Headless converter** `relay-d-convert` (also `scripts/convert_dataset.py`): convert recordings to
  robomimic or LeRobot without the GUI or ROS (`relay_d/acquisition/convert_cli.py`).
- A `test` job in CI running the ROS-free pytest suite; CI now runs on `main` only.
- **`postprocess_config.yaml`**: conversion options kept apart from the acquisition config
  (`postprocess.robomimic`: `metadata` env args, `training_model`, `splits`;
  `postprocess.lerobot`: `task`, `robot_type`, `state_streams`). Picked up automatically
  from the folder of the loaded config, or via "Load Options..." on the Post-process page,
  and copied into the dataset folder. The same keys inside the acquisition config still
  work but log a deprecation warning. See `relay_d/acquisition/qt_app/ui/config/templates/postprocess_config_template.yaml`;
  `AppAPI` accepts `postprocess_config_path`.
- **"Load Config..." on the Post-process page**: load an acquisition config for
  conversion only, without any live ROS topics/TF (no availability check, no
  Record-page containers; the Record page is never affected). "Use Config page"
  switches back to the Config page's config.
- **LeRobotDataset v3.0 output format** in the Post-process page and `AppAPI.convert_recording_to_dataset(output_format="lerobot")`.
  Reuses the existing acquisition config: `actions_to_extract` -> `action`,
  `output.streams` -> `observation.state`, image bridge-map entries ->
  `observation.images.<key>` mp4 videos, `recording_frequency` -> fps. Optional
  `config.lerobot.task` / `config.lerobot.state_streams`. Adds `pyarrow` and `av`
  dependencies and the first pytest tests (`tests/test_lerobot_writer.py`).
- Open-source project scaffolding: `LICENSE` (GNU GPLv3), `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md`, `SECURITY.md`, GitHub issue/PR templates, and a CI
  workflow.
- `license`, `authors`, and classifier metadata in `pyproject.toml`.
- A `## Dependencies` section in the README enumerating all pip and
  ROS 2/system dependencies in one place.

### Changed
- The config templates moved from the top-level `config/` folder into the package
  (`relay_d/acquisition/qt_app/ui/config/templates/`) so installed copies include them.
- `DataRecorder.stop_recording()` returns False (and keeps the samples in memory) when the
  HDF5 save fails; it used to report success.
- Record page: the start/pause/resume/stop logic moved from `RecordPage` into
  `RecordingController`, which also owns the Start/Pause/Stop buttons. The Pause button is
  connected once and toggles Pause/Resume (no more disconnecting/reconnecting slots).
  `RecordPage` keeps its public methods and attributes as delegates.
- `ContainerManager.create_containers` no longer freezes the window with `time.sleep`: it
  waits while processing Qt events and ignores re-entrant requests. The unused
  `wait_for_all_data` was removed.
- `pyarrow` and `av` are now an optional extra (`pip install ".[lerobot]"`) instead of base
  dependencies; they are only needed for LeRobot conversion. Anyone who relied on them
  being installed with the package must install the extra.

### Removed
- Dead code: `DataNormalizer` (`record/data_normalization.py`, never configured), the unused
  `gripper_states` recording, `DataRecorder.get_recording_status`/`cleanup`,
  `YAMLDrivenConverter.convert_dataset`, the `ACT` scaffold and `"act"` model type in
  `model_runner.py`, the unused `UI.auto_arrange_containers`/`set_forced_layout`, and
  `data_acquisition_node.py` / `interface_menu.py`.

### Fixed
- The exported `inference_script` crashed at startup (`--model-type` was never defined).
- "Remove" on the Config page left the Post-process page converting with the removed config.
- A demo that failed to write no longer corrupts the rest of a robomimic conversion, and a
  conversion with no demos now reports failure instead of success.
- A failed copy of `acquisition_config.yaml` after conversion is now reported to the user.
- A config without an `input` section no longer raises `AttributeError` in `get_obs_config()`.
- Stopping a recording no longer logs a spurious warning per TF input.
- `scripts/reprocess_recorded_data.py` handles mixed absolute/relative paths and requires `--input_dir`.
- Post-process page: selecting an output format now unchecks the other format button
  (both RoboMimic and LeRobot used to stay highlighted).
- LeRobot conversion of large camera frames uses about half the memory (no unused `next_obs`
  copy, strided image statistics).
- `get_obs_config()` no longer raises `KeyError` when the config has no
  `sync_tolerance_sec`; the `sync` section is simply omitted (check disabled).
- LeRobot conversion no longer demands PyAV (`av`) for recordings that contain no camera
  images, even if the config declares an image bridge-map entry.
- Post-processing no longer silently skips the training config and `action_dispatcher/`
  files when a config has no `training_model`: it now defaults to `bc_rnn_gmm`, and any
  remaining error in that step is shown to the user.
- Removed leftover local absolute file paths (`/home/<user>/...`) from
  `pyuic5`-generated header comments in the compiled Qt UI modules.

## [0.1.0] — 2026-07-28

### Added
- Upload of RelayD two subsystems:
  - `relay_d.acquisition` — the `RelayDApp` GUI for recording robot sensor
    data to HDF5 and converting it to Robomimic format.
  - `relay_d.dispatch` — the `lfd-inference-node` CLI for running a trained
    LfD policy live on the robot.
- Shared `relay_d.utils` module (colored console logger, config loader).
- Packaging via `pyproject.toml` (setuptools backend), with `RelayDApp` and
  `lfd-inference-node` console-script entry points.

## [0.1.1] — 2026-10-8

### Fixed
- Startup crash under NumPy 2: removed the compiled `cv_bridge` dependency (replaced by pure-numpy `imgmsg_to_cv2`/`cv2_to_imgmsg` in `relay_d.utils.ros_media_codec`) and dropped the `numpy<2` pin.
- Qt "xcb" plugin abort when the full `opencv-python` wheel is installed: `sanitize_qt_env()` undoes cv2's Qt plugin-path override.
