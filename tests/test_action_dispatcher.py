"""Pure action math of ActionDispatcher. Needs a sourced ROS 2 (rclpy) and torch, so it is skipped in CI."""

import numpy as np
import pytest

pytest.importorskip("rclpy")
pytest.importorskip("torch")  # relay_d.dispatch imports the model runner
from relay_d.dispatch.action_dispatcher import ActionDispatcher  # noqa: E402


@pytest.fixture
def d():
    return ActionDispatcher.__new__(ActionDispatcher)  # skip ROS setup: only pure methods are used


def test_limits_broadcast_clip_and_none(d):
    lims = [{"lower_lim": 0.0, "upper_lim": 10.0}]
    out = d._apply_limits(np.array([-1.0, 0.0, 1.0, 5.0]), lims, [-1.0, 1.0])
    assert out.tolist() == [0.0, 5.0, 10.0, 10.0]  # out-of-range input is clipped
    out = d._apply_limits(np.array([1.0, 0.3]), [None, {"lower_lim": 0, "upper_lim": 2}], [-1.0, 1.0])
    assert out[0] == 1.0  # None entry: unchanged
    assert d._apply_limits(np.array([0.5]), lims, [1.0, 1.0]).tolist() == [0.5]  # zero span: unchanged


def test_render_command(d):
    cmd = d._render_command("run {I[0]} {I[1]} {I[10]}", np.array([3.0, 0.1234567]))
    assert cmd == "run 3 0.123457 {I[10]}"  # {I[1]} must not eat {I[10]}


def test_dispatch_drops_non_finite_action():
    d = ActionDispatcher.__new__(ActionDispatcher)
    d._action_configs = {"x": {"indices": [0]}}
    d._last_sent = {}
    d.dispatch(np.array([np.nan]))  # would raise (no publishers set up) if it got past the guard
    d.dispatch(np.array([np.inf]))
