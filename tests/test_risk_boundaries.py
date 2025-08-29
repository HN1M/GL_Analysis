import math
import pytest
from analysis.anomaly import _risk_from


@pytest.mark.parametrize("z_abs", [0.0, 1.0, 3.0, 10.0])
@pytest.mark.parametrize("pm_value", [0.0, 1.0, 1e6, 1e12])
def test_risk_from_boundary_is_finite(z_abs, pm_value):
    a, f, k, score = _risk_from(z_abs=z_abs, amount=1_000_000, pm=pm_value)
    assert 0.0 <= score <= 1.0
    assert math.isfinite(score)


def test_risk_from_monotonic_in_z_when_pm_fixed():
    pm = 1e6
    s1 = _risk_from(0.5, amount=1_000_000, pm=pm)[-1]
    s2 = _risk_from(3.0, amount=1_000_000, pm=pm)[-1]
    s3 = _risk_from(10.0, amount=1_000_000, pm=pm)[-1]
    assert s1 <= s2 <= s3


@pytest.mark.parametrize("pm_lo,pm_hi", [(0.0, 1e9)])
def test_risk_from_non_decreasing_in_pm(pm_lo, pm_hi):
    z = 3.0
    slo = _risk_from(z, amount=1_000_000, pm=pm_lo)[-1]
    shi = _risk_from(z, amount=1_000_000, pm=pm_hi)[-1]
    assert slo <= shi

