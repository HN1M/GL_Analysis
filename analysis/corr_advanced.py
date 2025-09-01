from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict
from analysis.contracts import ModuleResult, EvidenceDetail, LedgerFrame
from config import CORR_THRESHOLD_DEFAULT, CORR_MAX_LAG_DEFAULT, CORR_ROLLWIN_DEFAULT
from utils.viz import apply_corr_heatmap_theme
import plotly.express as px


def _pivot_monthly_flow(lf: LedgerFrame, accounts: List[str]) -> pd.DataFrame:
    """
    입력 accounts 가 '계정코드' 또는 '계정명' 어느 쪽이든 동작하도록 방어.
    (UI에서 계정명을 전달했을 때 빈 피벗이 되던 문제 수정)
    """
    df = lf.df.copy()
    accs = {str(a) for a in (accounts or [])}
    if not accs:
        return pd.DataFrame()
    code_mask = df["계정코드"].astype(str).isin(accs)
    name_mask = df["계정명"].astype(str).isin(accs) if "계정명" in df.columns else False
    df = df[code_mask | name_mask].copy()
    if df.empty:
        return pd.DataFrame()
    df["월"] = pd.to_datetime(df["회계일자"], errors="coerce").dt.to_period("M").astype(str)
    # 월 기준 발생액(절대값) 합계 피벗
    pivot = df.pivot_table(index="월", columns="계정명", values="거래금액_절대값", aggfunc="sum").fillna(0.0)
    return pivot.sort_index()


def _corr_with_lag(a: pd.Series, b: pd.Series, lag: int, *, min_overlap: int = 6) -> float:
    """시차 상관: B를 lag만큼 shift해 A(t) vs B(t-lag).
    - lag 음수면 반대 방향 이동
    - 둘 다 0인 월 제거
    - 유효 표본 길이(min_overlap) 미만이면 NaN
    """
    b_shift = b.shift(lag)
    df = pd.concat([a, b_shift], axis=1, keys=["A", "B"]).copy()
    df = df[(df["A"].fillna(0) != 0) | (df["B"].fillna(0) != 0)]
    df = df.dropna()
    if len(df) < int(min_overlap):
        return np.nan
    return float(df["A"].corr(df["B"]))


def _best_lag_pair(pivot: pd.DataFrame, max_lag: int, *, min_overlap: int = 6) -> List[Dict[str, object]]:
    cols = list(pivot.columns)
    out: List[Dict[str, object]] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            s1, s2 = pivot[cols[i]], pivot[cols[j]]
            best_lag, best_val = 0, np.nan
            for lag in range(-max_lag, max_lag + 1):
                v = _corr_with_lag(s1, s2, lag, min_overlap=min_overlap)
                if not np.isnan(v) and (np.isnan(best_val) or abs(v) > abs(best_val)):
                    best_lag, best_val = lag, v
            if not np.isnan(best_val):
                out.append({"계정A": cols[i], "계정B": cols[j], "최적시차": best_lag, "상관계수": best_val})
    out.sort(key=lambda x: abs(x["상관계수"]), reverse=True)
    return out


def _rolling_stability(pivot: pd.DataFrame, window: int = 6) -> List[Dict[str, object]]:
    cols = list(pivot.columns)
    out: List[Dict[str, object]] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = pivot[cols[i]].rolling(window).corr(pivot[cols[j]])
            if len(r.dropna()) == 0:
                continue
            vol = float(r.std(skipna=True))
            mean = float(r.mean(skipna=True))
            out.append({"계정A": cols[i], "계정B": cols[j], "롤링평균": mean, "롤링변동성": vol})
    out.sort(key=lambda x: x["롤링변동성"])  # 낮은 변동성 우선
    return out


def run_corr_advanced(
    lf: LedgerFrame,
    accounts: List[str],
    *,
    method: str = "pearson",
    corr_threshold: float = CORR_THRESHOLD_DEFAULT,
    max_lag: int = CORR_MAX_LAG_DEFAULT,
    rolling_window: int = CORR_ROLLWIN_DEFAULT,
    cycles_map=None,
    within_same_cycle: bool=False,
) -> ModuleResult:
    name = "corr_advanced"
    if lf is None or getattr(lf, "df", None) is None:
        return ModuleResult(name=name, summary={}, tables={}, figures={}, evidences=[], warnings=["LedgerFrame 없음"])
    if not accounts:
        return ModuleResult(name=name, summary={"n_accounts": 0}, tables={}, figures={}, evidences=[], warnings=["선택 계정 없음"])

    pivot = _pivot_monthly_flow(lf, accounts)
    if pivot.empty or len(pivot.columns) < 2:
        return ModuleResult(name=name, summary={"n_accounts": len(accounts)}, tables={}, figures={}, evidences=[], warnings=["데이터 부족"])

    corr = pivot.corr(method=method).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 히트맵 (계정명으로)
    fig_heat = px.imshow(
        corr,
        text_auto=False,
        color_continuous_scale="Blues",
        labels=dict(color="상관계수"),
        x=corr.columns,
        y=corr.index,
        title="계정 간 월별 상관 히트맵",
    )
    fig_heat = apply_corr_heatmap_theme(fig_heat)

    # 임계치 이상 쌍
    strong: List[Dict[str, object]] = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = float(corr.iloc[i, j])
            if abs(v) >= float(corr_threshold):
                strong.append({"계정A": cols[i], "계정B": cols[j], "상관계수": v})
    # 동일 사이클 필터(선택)
    if within_same_cycle and cycles_map and "계정명" in lf.df.columns:
        # 이름->코드 역매핑
        nm2cd = (lf.df.drop_duplicates("계정명")
                    .assign(계정코드=lambda d: d["계정코드"].astype(str))
                    .set_index("계정명")["계정코드"].astype(str).to_dict())
        def _same(a,b):
            ca, cb = nm2cd.get(a), nm2cd.get(b)
            return (cycles_map.get(str(ca)) == cycles_map.get(str(cb))) if (ca and cb and isinstance(cycles_map, dict)) else True
        strong = [r for r in strong if _same(r["계정A"], r["계정B"])]
    strong_df = pd.DataFrame(strong)

    # 최적 시차 상관
    lag_pairs = pd.DataFrame(_best_lag_pair(pivot, int(max_lag)))

    # 롤링 안정성(낮은 변동성 우선)
    roll = pd.DataFrame(_rolling_stability(pivot, int(rolling_window)))

    # Evidence 샘플(상위 10개): |r|을 anomaly_score로 활용
    evid: List[EvidenceDetail] = []
    # 각 계정의 월별 절대발생액 합(규모) → 증거의 재무영향 추정치로 사용
    size_by_acct = pivot.abs().sum(axis=0).astype(float).to_dict()
    for row in strong[: min(10, len(strong))]:
        a, b = str(row["계정A"]), str(row["계정B"])
        r = float(row["상관계수"])
        evid.append(EvidenceDetail(
            row_id=f"{a}|{b}",
            reason=f"corr={r:+.2f} (|r|≥{corr_threshold})",
            anomaly_score=abs(r),
            financial_impact=float(min(size_by_acct.get(a, 0.0), size_by_acct.get(b, 0.0))),
            risk_score=abs(r),
            is_key_item=False,
            impacted_assertions=[],
            links={"account_a": a, "account_b": b, "type": "corr_strong"},
        ))

    summary = {
        "n_accounts": int(len(accounts)),
        "n_pairs_over_threshold": int(len(strong)),
        "corr_threshold": float(corr_threshold),
        "max_lag": int(max_lag),
        "rolling_window": int(rolling_window),
    }

    tables = {
        "corr_matrix": corr,
        "strong_pairs": strong_df,
        "lagged_pairs": lag_pairs,
        "rolling_stability": roll,
    }
    figures = {"heatmap": fig_heat}

    return ModuleResult(name=name, summary=summary, tables=tables, figures=figures, evidences=evid, warnings=[])


