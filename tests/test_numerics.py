"""Maths shared by training-data conversion and live inference, and the converter rules built on it."""

import h5py
import numpy as np
import pytest

from relay_d.acquisition.qt_app.ui.postprocess.postprocess_config_loader import PostprocessConfigLoader
from relay_d.acquisition.qt_app.ui.postprocess.yaml_driven_converter import (
    DataTransformEngine,
    YAMLDrivenConverter,
    _apply_normalization,
)
from relay_d.utils.numerics import normalize, quat_to_euler


def test_normalize_maps_range_and_leaves_degenerate_range_alone():
    assert normalize(5.0, 0.0, 10.0) == 0.0
    assert normalize(10.0, 0.0, 10.0, 0.0, 1.0) == 1.0
    assert normalize(7.0, 3.0, 3.0) == 7.0  # upper == lower
    assert normalize(7.0, 3.0, 1.0) == 7.0  # upper < lower


def test_training_normalization_matches_shared_rule_for_degenerate_limits():
    data = np.array([[7.0, 5.0], [7.0, 10.0]], dtype=np.float32)
    limits = [{"lower_lim": 3.0, "upper_lim": 3.0}, {"lower_lim": 0.0, "upper_lim": 10.0}]
    out = _apply_normalization(data, limits, [-1.0, 1.0])
    assert out[:, 0].tolist() == [7.0, 7.0]  # same as inference: the value is passed through
    assert out[:, 1].tolist() == [0.0, 1.0]


def test_quat_to_euler_known_rotations():
    s = np.sqrt(0.5)
    q = np.array([[0, 0, 0, 1], [0, 0, s, s], [s, 0, 0, s]], dtype=np.float64)  # id, yaw 90, roll 90
    rpy = quat_to_euler(q)
    assert np.allclose(rpy, [[0, 0, 0], [0, 0, np.pi / 2], [np.pi / 2, 0, 0]])


@pytest.fixture
def converter():
    return YAMLDrivenConverter(PostprocessConfigLoader())


def _mask(converter, n, train, val, test):
    with h5py.File("m.h5", "w", driver="core", backing_store=False) as f:
        g = f.create_group("mask")
        ok = converter._create_data_mask(g, [f"demo_{i}" for i in range(n)], train, val, test)
        return ok, {k: len(g[k]) for k in g}


def test_split_mask_normal_and_few_demos(converter):
    assert _mask(converter, 10, 0.8, 0.1, 0.1) == (True, {"train": 8, "valid": 1, "eval": 1})
    ok, sizes = _mask(converter, 2, 0.8, 0.1, 0.1)
    assert ok and sizes["train"] >= 1  # int(2 * 0.8) would be 1 here, int(1 * 0.8) would be 0


def test_split_mask_rejects_bad_fractions(converter):
    assert _mask(converter, 10, 0.9, 0.2, 0.0)[0] is False
    assert "more than 1" in converter.last_error
    assert _mask(converter, 10, -0.1, 0.1, 0.1)[0] is False
    assert _mask(converter, 10, "0.8", 0.1, 0.1)[0] is False


def test_generate_actions_fails_on_missing_key():
    cfg = {"actions_to_extract": ["robot_joint_pos"], "output": {"robot_joint_pos": "robot/joint_positions"}}
    engine = DataTransformEngine(cfg)
    with h5py.File("a.h5", "w", driver="core", backing_store=False) as f:
        with pytest.raises(ValueError, match="not found in data"):
            engine.generate_actions(f, {}, 5)
        f.create_dataset("robot/joint_positions", data=np.ones((5, 3), dtype=np.float32))
        assert engine.generate_actions(f, {}, 5).shape == (5, 3)
    with pytest.raises(ValueError, match="not found in bridge_map"):
        DataTransformEngine({"actions_to_extract": ["nope"]}).generate_actions(None, {}, 5)


def test_zero_quaternion_in_an_rpy_stream_fails_instead_of_writing_nan():
    from relay_d.acquisition.qt_app.ui.postprocess.yaml_driven_converter import _quat_to_rpy

    good = np.array([[0, 0, 0, 1.0], [0, 0, np.sqrt(0.5), np.sqrt(0.5)]], dtype=np.float32)
    assert np.allclose(_quat_to_rpy(good, "eef"), [[0, 0, 0], [0, 0, np.pi / 2]], atol=1e-6)
    with pytest.raises(ValueError, match=r"stream 'eef' has 1 row"):
        _quat_to_rpy(np.array([[0, 0, 0, 1.0], [0, 0, 0, 0.0]], dtype=np.float32), "eef")
