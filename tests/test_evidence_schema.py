from dataclasses import asdict
from analysis.anomaly import run_anomaly_module
from analysis.contracts import LedgerFrame
import pandas as pd


def test_anomaly_emits_evidence_minimal():
    # 간단 가짜 DF
    df = pd.DataFrame({
        "row_id":["a","b","c"],
        "회계일자": pd.to_datetime(["2024-01-01","2024-01-02","2024-01-03"]),
        "계정코드": ["400","400","400"],
        "계정명":   ["매출","매출","매출"],
        "차변": [0, 0, 0],
        "대변": [10_000_000, 100, 50],
    })
    lf = LedgerFrame(df=df, meta={})
    mod = run_anomaly_module(lf, target_accounts=["400"], topn=2, pm_value=500_000_000)
    assert isinstance(mod.evidences, list)
    assert len(mod.evidences) >= 1
    d = asdict(mod.evidences[0])
    for key in ["row_id","reason","anomaly_score","financial_impact","risk_score","is_key_item","impacted_assertions","links"]:
        assert key in d

