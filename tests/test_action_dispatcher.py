"""Pure action math of ActionDispatcher. Needs a sourced ROS 2 (rclpy), so it is skipped in CI."""

import numpy as np
import pytest

pytest.importorskip("rclpy")
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


def test_dispatch_drops_non_finite_action(d):
    d._action_configs = {"x": {"indices": [0]}}
    d._last_sent = {}
    d.dispatch(np.array([np.nan]))  # would raise (no publishers set up) if it got past the guard
    d.dispatch(np.array([np.inf]))


def test_set_nested_casts_to_field_type():
    from types import SimpleNamespace

    from relay_d.dispatch.action_dispatcher import _set_nested

    msg = SimpleNamespace(count=0, flag=False, speed=0.0)
    _set_nested(msg, "count", 2.6)
    _set_nested(msg, "flag", 1.0)
    _set_nested(msg, "speed", 3)
    assert (msg.count, msg.flag, msg.speed) == (3, True, 3.0)  # rclpy rejects a float for an int/bool field


def test_fill_msg_reports_failed_mapping():
    from types import SimpleNamespace

    from relay_d.dispatch.action_dispatcher import _fill_msg

    msg = SimpleNamespace(x=0.0)
    values = np.array([1.0, 2.0])
    assert _fill_msg(msg, {"mapping": {"x": 1}}, values, None) is True
    assert msg.x == 2.0
    assert _fill_msg(msg, {"mapping": {"x": 0, "missing": 1}}, values, None) is False  # caller must not publish


def test_failed_send_is_not_remembered_so_it_is_retried(d):
    d._action_configs = {"grip": {"indices": [0], "action": True}}
    d._last_sent, d._procs = {}, {}
    sent = []
    results = iter([False, True, True])
    d._dispatch_action = lambda name, cfg, values: sent.append(name) or next(results)
    for _ in range(3):
        d.dispatch(np.array([1.0]))
    assert sent == ["grip", "grip"]  # 1st failed -> retried; 2nd succeeded -> same command is not re-sent


def test_dispatch_sends_nothing_if_one_entry_cannot_be_computed(d):
    d._action_configs = {
        "arm": {"indices": [0], "topic": "/arm"},
        "tool": {"indices": [0], "transform": {"type": "nope"}},
    }
    d._last_sent, d._procs = {}, {}
    d._dispatch_topic = lambda *a: pytest.fail("arm must not move while the tool entry is broken")
    with pytest.raises(ValueError):
        d.dispatch(np.array([1.0]))


def test_check_mapping_rejects_a_field_the_message_does_not_have(d):
    from types import SimpleNamespace

    msg_class = lambda: SimpleNamespace(x=0.0)  # noqa: E731
    d._node = None  # the message has no header, so the clock is never asked
    d._check_mapping("arm", {"indices": [0, 1], "mapping": {"x": 0}}, msg_class)
    with pytest.raises(ValueError):
        d._check_mapping("arm", {"indices": [0, 1], "mapping": {"x": 0, "typo": 1}}, msg_class)  # would move the arm, skip the rest


def test_dispatch_sends_nothing_if_a_transform_gives_nan(d):
    d._action_configs = {
        "arm": {"indices": [0], "topic": "/arm"},
        "tool": {"indices": [0], "transform": {"type": "scale", "in_min": 1, "in_max": 1, "out_min": 0, "out_max": 1}},
    }
    d._last_sent, d._procs = {}, {}
    d._dispatch_topic = lambda *a: pytest.fail("nothing may be sent when one output is NaN")
    d.dispatch(np.array([1.0]))
