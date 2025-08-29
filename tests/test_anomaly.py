# Z-Score 및 위험 점수 단조성 테스트
import pandas as pd
from analysis.anomaly import calculate_grouped_stats_and_zscore, _risk_from

def test_zscore_and_risk_monotonic():
    df = pd.DataFrame({
        "계정코드": ["100"]*5 + ["200"]*5,
        "차변":     [0,0,0,0,0] + [0,0,0,0,0],
        "대변":     [10,20,30,40,50] + [5,5,5,5,5],
    })
    out = calculate_grouped_stats_and_zscore(df, target_accounts=["100","200"])
    assert "Z-Score" in out.columns
    a1 = _risk_from(1.0, amount=1_000, pm=100_000)[-1]
    a2 = _risk_from(3.0, amount=1_000, pm=100_000)[-1]
    b1 = _risk_from(1.0, amount= 50_000, pm=100_000)[-1]
    b2 = _risk_from(1.0, amount=150_000, pm=100_000)[-1]
    assert a2 > a1
    assert b2 > b1


