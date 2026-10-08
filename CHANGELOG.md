# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Open-source project scaffolding: `LICENSE` (GNU GPLv3), `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md`, `SECURITY.md`, GitHub issue/PR templates, and a CI
  workflow.
- `license`, `authors`, and classifier metadata in `pyproject.toml`.
- A `## Dependencies` section in the README enumerating all pip and
  ROS 2/system dependencies in one place.

<!-- ### Changed
- Nothing was changed. -->

### Fixed
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
