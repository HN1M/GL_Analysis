from analysis.contracts import EvidenceDetail, ModuleResult
from analysis.assertion_risk import build_matrix
import pandas as pd


def _ev(row_id, acct, assertions, score):
    return EvidenceDetail(
        row_id=row_id, reason="t", anomaly_score=0.0, financial_impact=0.0,
        risk_score=score, is_key_item=False, impacted_assertions=assertions,
        links={"account_name": acct}
    )


def test_build_matrix_max_aggregation():
    mod = ModuleResult(
        name="anomaly",
        summary={}, tables={}, figures={},
        evidences=[
            _ev("r1", "매출", ["A","E"], 0.40),
            _ev("r2", "매출", ["E"],     0.65),
            _ev("r3", "매입", ["C"],     0.30),
        ],
        warnings=[]
    )
    mat, emap = build_matrix([mod])
    assert float(mat.loc["매출","E"]) == 0.65   # 동일 셀은 최대값
    assert float(mat.loc["매출","A"]) == 0.40
    assert float(mat.loc["매입","C"]) == 0.30
    assert ("매출","E") in emap and set(emap[("매출","E")]) == {"r1","r2"}

