from analysis.anomaly import _risk_from
from analysis.anomaly import _assertions_for_row


def test_risk_from_none_and_negative_pm():
    for pm in (None, -1, -1000):
        a, f, k, score = _risk_from(z_abs=2.0, amount=1_000_000, pm=pm)
        assert f == 0.0 and k == 0.0
        assert 0.0 <= a <= 1.0
        assert 0.0 <= score <= 1.0

def test_assertions_mapping_rules():
    # 항상 A 포함
    assert "A" in _assertions_for_row(0.0)
    # 큰 양의 이탈 → E 포함
    assert set(_assertions_for_row(+2.5)) >= {"A","E"}
    # 큰 음의 이탈 → C 포함
    assert set(_assertions_for_row(-2.5)) >= {"A","C"}

