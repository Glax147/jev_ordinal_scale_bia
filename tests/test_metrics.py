import pytest

from jev_bias.metrics import summarize


def test_full_balanced_utilization():
    result = summarize([0, 1, 2, 3], [0, 1, 2, 3], None, 4, True)
    assert result["accuracy"] == 1
    assert result["u_arg"] == pytest.approx(1)
    assert result["gold_relative_utilization"] == pytest.approx(1)
    assert result["support_deviation"] == pytest.approx(0)
    assert result["tvd"] == pytest.approx(0)


def test_support_deviation_from_one_anchor():
    result = summarize([0, 1, 2, 3], [0, 0, 0, 0], None, 4, True)
    assert result["u_arg"] == pytest.approx(0.25)
    assert result["gold_relative_utilization"] == pytest.approx(0.25)
    assert result["support_deviation"] == pytest.approx(0.75)
    assert result["tvd"] == pytest.approx(0.75)


def test_qwk_reference_cases():
    perfect = summarize([0, 1, 2], [0, 1, 2], None, 3, True)
    reversed_order = summarize([0, 1, 2], [2, 1, 0], None, 3, True)
    assert perfect["qwk"] == pytest.approx(1.0)
    assert reversed_order["qwk"] == pytest.approx(-1.0)
