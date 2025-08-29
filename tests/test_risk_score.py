import math
from analysis.anomaly import _risk_from
from config import Z_SIGMOID_SCALE


def test_risk_from_basic():
    a, f, k, score = _risk_from(z_abs=3.0, amount=1_000_000_000, pm=500_000_000)
    exp_a = 1.0 / (1.0 + math.exp(-(3.0/float(Z_SIGMOID_SCALE or 1.0))))
    assert abs(a - exp_a) < 1e-9
    assert f == 1.0                 # PM 대비 캡 1
    assert k == 1.0                 # KIT
    # 가중합: 0.5*a + 0.4*1 + 0.1*1
    expected = 0.5*a + 0.4 + 0.1
    assert abs(score - expected) < 1e-9


def test_risk_from_zero_pm_guard():
    a, f, k, score = _risk_from(z_abs=0.0, amount=0, pm=0)
    assert f == 0.0 and k == 0.0    # 분모 가드 동작
    assert 0.0 <= a <= 0.5          # z=0이면 a는 0.5 근처

