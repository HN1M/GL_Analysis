from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Optional
from utils.helpers import find_column_by_keyword


def compute_amount_columns(df: pd.DataFrame) -> pd.DataFrame:
    """발생액(절대 규모) / 순액(차-대) 계산."""
    dcol = find_column_by_keyword(df.columns, '차변')
    ccol = find_column_by_keyword(df.columns, '대변')
    df = df.copy()
    if not dcol or not ccol:
        df['발생액'] = 0.0; df['순액'] = 0.0; df['거래금액'] = 0.0
        return df
    d = pd.to_numeric(df[dcol], errors='coerce').fillna(0.0)
    c = pd.to_numeric(df[ccol], errors='coerce').fillna(0.0)
    row_amt = np.where((d > 0) & (c == 0), d,
              np.where((c > 0) & (d == 0), c,
              np.where((d == 0) & (c == 0), 0.0, np.abs(d - c))))
    df['발생액'] = row_amt
    df['순액']  = d - c
    df['거래금액'] = df['순액']
    return df


def calculate_grouped_stats_and_zscore(df: pd.DataFrame, target_accounts: List[str], data_type: str = "당기") -> pd.DataFrame:
    """선택 계정 그룹의 발생액 분포 기준 Z-Score 산출."""
    acct_col = find_column_by_keyword(df.columns, '계정코드')
    df = compute_amount_columns(df.copy())
    if not acct_col:
        df['Z-Score'] = 0.0
        return df
    is_target = df[acct_col].astype(str).isin([str(x) for x in target_accounts])
    tgt = df.loc[is_target, '발생액'].astype(float)
    df['Z-Score'] = 0.0
    if tgt.empty:
        return df
    mu = float(tgt.mean()); std = float(tgt.std(ddof=1))
    if std and std > 0:
        df.loc[is_target, 'Z-Score'] = (df.loc[is_target, '발생액'] - mu) / std
    else:
        med = float(tgt.median()); mad = float((np.abs(tgt - med)).median())
        df.loc[is_target, 'Z-Score'] = 0.0 if mad == 0 else 0.6745 * (df.loc[is_target, '발생액'] - med) / mad
    return df

# --- NEW: ensure_zscore ---
def ensure_zscore(df: pd.DataFrame, account_codes: List[str]):
    """
    Recompute Z-Score for the given account subset and return (df, ok).
    ok=True only if Z-Score column exists and has at least one non-null value.
    """
    df2 = calculate_grouped_stats_and_zscore(df.copy(), target_accounts=[str(x) for x in account_codes] if account_codes else [])
    z = df2.get('Z-Score')
    ok = (z is not None) and (z.notna().any())
    return df2, bool(ok)




# === (ADD) v0.18: ModuleResult 러너 ===
from analysis.contracts import ModuleResult, EvidenceDetail
from config import PM_DEFAULT, RISK_WEIGHT_A, RISK_WEIGHT_F, RISK_WEIGHT_K, Z_SIGMOID_SCALE, Z_SIGMOID_DIVISOR
import plotly.express as px
import numpy as np
import pandas as pd


def _z_bins_025_sigma(series: pd.Series):
    """0.25σ 간격 bin (±3σ 테일 포함)."""
    # 경계에 +3.0 포함(+inf 테일) → 총 bin 수 = 24(코어) + 2(테일) = 26
    edges = [-np.inf] + [round(x, 2) for x in np.arange(-3.0, 3.0 + 0.25, 0.25)] + [np.inf]
    core_lefts = [x for x in np.arange(-3.0, 3.0, 0.25)]  # 24개
    labels_mid = [f"{a:.2f}~{a+0.25:.2f}σ" for a in core_lefts]
    labels = ["≤-3σ"] + labels_mid + ["≥3σ"]               # 26개
    cats = pd.cut(
        series.astype(float),
        bins=edges,
        labels=labels,
        right=False,
        include_lowest=True,
    )
    # 빈 구간도 0으로 채워 순서 유지
    counts = cats.value_counts(sort=False).reindex(labels, fill_value=0)
    out = pd.DataFrame({"구간": labels, "건수": counts.values})
    order = labels
    return out, order


def _sigmoid(x: float) -> float:
    import math
    return 1.0 / (1.0 + math.exp(-x))


def _risk_from(z_abs: float, amount: float, pm: float):
    """리스크 점수 구성요소 계산.
    - a: 시그모이드 정규화된 이탈 강도(|Z|/scale). scale은 설정값.
    - f: PM 대비 금액비율(0~1로 캡). PM이 0/음수면 0으로 강제.
    - k: Key Item 플래그(PM 초과시 1). PM이 0/음수면 0으로 강제.
    """
    # 우선순위: Z_SIGMOID_DIVISOR(신규 노브) > Z_SIGMOID_SCALE(구명). 기본 1.0
    div = None
    try:
        div = float(Z_SIGMOID_DIVISOR)
    except Exception:
        div = None
    if not div or div <= 0:
        try:
            div = float(Z_SIGMOID_SCALE)
        except Exception:
            div = 1.0
    if not div or div <= 0:
        div = 1.0
    a = _sigmoid(float(abs(z_abs)) / float(div))      # anomaly_score
    # PM 가드: pm<=0이면 f=0, k=0
    if pm is None or float(pm) <= 0:
        f = 0.0
        k = 0.0
    else:
        f = min(1.0, abs(float(amount)) / float(pm))  # PM ratio (capped at 1)
        k = 1.0 if abs(float(amount)) >= float(pm) else 0.0
    score = RISK_WEIGHT_A * a + RISK_WEIGHT_F * f + RISK_WEIGHT_K * k
    return a, f, k, score


def _assertions_for_row(z_val: float) -> List[str]:
    # 기본 규칙: A는 항상 포함. 음의 큰 이탈(C), 양의 큰 이탈(E)을 보강.
    out = {"A"}
    try:
        if float(z_val) <= -2.0:
            out.add("C")
        if float(z_val) >=  2.0:
            out.add("E")
    except Exception:
        pass
    return sorted(out)


def run_anomaly_module(lf, target_accounts=None, topn=20, pm_value: Optional[float] = None):
    df = lf.df.copy()
    acct_col = find_column_by_keyword(df.columns, '계정코드')
    if not acct_col:
        return ModuleResult("anomaly", {}, {}, {}, [], ["계정코드 컬럼을 찾지 못했습니다."])

    # 대상 계정 서브셋
    if target_accounts:
        codes = [str(x) for x in target_accounts]
        df = df[df[acct_col].astype(str).isin(codes)].copy()

    # Z-Score 계산
    df = calculate_grouped_stats_and_zscore(df, target_accounts=df[acct_col].astype(str).unique().tolist())
    if '회계일자' in df.columns:
        df['연월'] = df['회계일자'].dt.to_period('M').astype(str)

    # 이상치 플래그 (±3σ)
    df['is_outlier'] = df['Z-Score'].abs() >= 3

    # 이상치 후보 테이블 (절댓값 기준 상위)
    out_cols = [c for c in ['row_id','회계일자','연월','계정코드','계정명','거래처','적요','발생액','Z-Score'] if c in df.columns]
    cand = (df.assign(absz=df['Z-Score'].abs())
              .sort_values('absz', ascending=False)
              .drop(columns=['absz'])
              .head(int(topn)))
    table = cand[out_cols + (['is_outlier'] if 'is_outlier' in cand.columns and 'is_outlier' not in out_cols else [])] if out_cols else cand

    # === EvidenceDetail 생성 (KIT + |Z| 기준) ===
    pm = float(pm_value) if pm_value is not None else float(PM_DEFAULT)
    ev_rows: List[EvidenceDetail] = []
    # 증거 채집 대상: (1) PM 초과 or (2) |Z|>=2.5 or (3) 상위 topn
    mask_key = df['발생액'].abs() >= pm if '발생액' in df.columns else pd.Series(False, index=df.index)
    mask_z   = df['Z-Score'].abs() >= 2.5 if 'Z-Score' in df.columns else pd.Series(False, index=df.index)
    idx_sel  = set(df.index[mask_key | mask_z].tolist()) | set(table.index.tolist())
    sub = df.loc[sorted(idx_sel)].copy() if len(idx_sel)>0 else df.head(0).copy()
    for _, r in sub.iterrows():
        z  = float(r.get('Z-Score', 0.0)) if pd.notna(r.get('Z-Score', np.nan)) else 0.0
        za = abs(z)
        amt = float(r.get('발생액', 0.0))
        a, f, k, score = _risk_from(za, amt, pm)
        ev_rows.append(EvidenceDetail(
            row_id=str(r.get('row_id','')),
            reason=f"amt_z={z:+.2f}",
            anomaly_score=float(a),
            financial_impact=abs(amt),
            risk_score=float(score),
            is_key_item=bool(abs(amt) >= pm),
            impacted_assertions=_assertions_for_row(z),
            links={
                "account_code": str(r.get('계정코드','')),
                "account_name": str(r.get('계정명','')),
                "vendor":      str(r.get('거래처','')),
                "narration":   str(r.get('적요','')),
                "cluster_name": str(r.get('cluster_name','')) if 'cluster_name' in r.index else "",
                "cluster_group": str(r.get('cluster_group','')) if 'cluster_group' in r.index else "",
                "month":       str(r.get('연월','')) if '연월' in r.index else "",
                "period_tag": str(r.get('period_tag','')),
            }
        ))

    # step-σ bin 분포 막대
    figures = {}
    try:
        dist_df, order = _z_bins_025_sigma(df['Z-Score'])
        total_n = int(len(df))
        outlier_rate = float((df['Z-Score'].abs() >= 3).mean() * 100) if total_n else 0.0
        title = f"Z-Score 분포 (0.25σ bin, ±3σ 집계) — N={total_n:,}, outlier≈{outlier_rate:.1f}%"
        fig = px.bar(dist_df, x='구간', y='건수', title=title)
        fig.update_yaxes(separatethousands=True)
        fig.update_layout(bargap=0.10)
        figures = {"zscore_hist": fig}
    except Exception:
        pass

    summary = {
        "n_rows": int(len(df)),
        "n_candidates": int(len(table)),
        "accounts": sorted(df[acct_col].astype(str).unique().tolist()),
        "period_tag_coverage": dict(df.get('period_tag', pd.Series(dtype=str)).value_counts()) if 'period_tag' in df.columns else {}
    }
    # Evidence 미리보기 테이블(선택)
    try:
        import pandas as _pd
        ev_tbl = _pd.DataFrame([{
            "row_id": e.row_id,
            "계정코드": e.links.get("account_code",""),
            "계정명":   e.links.get("account_name",""),
            "risk_score": e.risk_score,
            "is_key_item": e.is_key_item,
            "impacted": ",".join(e.impacted_assertions),
            "reason": e.reason,
        } for e in ev_rows]).sort_values("risk_score", ascending=False).head(100)
    except Exception:
        ev_tbl = None

    return ModuleResult(
        name="anomaly",
        summary=summary,
        tables={"anomaly_top": table, **({"evidence_preview": ev_tbl} if ev_tbl is not None else {})},
        figures=figures,
        evidences=ev_rows,
        warnings=[]
    )