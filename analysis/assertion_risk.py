from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from analysis.contracts import ModuleResult, EvidenceDetail, ASSERTIONS


HEATMAP_BS_RISK = "max"  # or "balance_only" / "weighted"


def _agg_bs_risk(rows: pd.DataFrame) -> float:
    """BS 셀 위험도 집계 규칙(기본 max).
    - EvidenceDetail.measure가 제공되는 경우에만 적용 가능.
    - 'weighted'는 balance 0.6, flow 0.4 가중.
    """
    if rows is None or rows.empty:
        return 0.0
    if HEATMAP_BS_RISK == "balance_only":
        r = rows.loc[rows.get("measure", pd.Series()).eq("balance"), "risk_score"]
        return float(r.max() if not r.empty else rows["risk_score"].max())
    if HEATMAP_BS_RISK == "weighted":
        w = rows.get("measure", pd.Series(index=rows.index)).map({"balance": 0.6, "flow": 0.4}).fillna(0.5)
        try:
            return float(np.average(rows["risk_score"].astype(float), weights=w))
        except Exception:
            return float(rows["risk_score"].max())
    return float(rows["risk_score"].max())


def build_matrix(modules: List[ModuleResult]):
    """
    모듈 EvidenceDetail → (계정 × 주장) 최대 risk_score 매트릭스 + 드릴다운 맵
    반환: (matrix_df[account_name x ASSERTIONS], evidence_map[(acct, asrt)] -> [row_id...])
    """
    bucket_rows: Dict[Tuple[str,str], List[Dict]] = {}
    emap: Dict[Tuple[str,str], List[str]] = {}
    accts: set[str] = set()

    for mod in modules:
        for ev in (mod.evidences or []):
            acct = ev.links.get("account_name") or ev.links.get("account_code") or "UNMAPPED"
            accts.add(acct)
            for a in (ev.impacted_assertions or []):
                key = (acct, a)
                bucket_rows.setdefault(key, []).append({
                    "risk_score": float(ev.risk_score),
                    "measure": getattr(ev, "measure", None)
                })
                emap.setdefault(key, []).append(str(ev.row_id))

    idx = sorted(accts)
    mat = pd.DataFrame(index=idx, columns=ASSERTIONS, data=0.0)
    for (acct, asrt), rows in bucket_rows.items():
        df = pd.DataFrame(rows)
        mat.loc[acct, asrt] = _agg_bs_risk(df) if not df.empty else 0.0
    return mat.fillna(0.0), emap


