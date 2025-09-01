from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict
from analysis.contracts import ModuleResult, EvidenceDetail, LedgerFrame
import plotly.express as px


def _pivot_monthly_flow(lf: LedgerFrame, accounts: List[str]) -> pd.DataFrame:
    df = lf.df.copy()
    df = df[df["계정코드"].astype(str).isin([str(a) for a in accounts])]
    df["월"] = pd.to_datetime(df["회계일자"], errors="coerce").dt.to_period("M").astype(str)
    # 월 기준 발생액(절대값) 합계 피벗
    pivot = df.pivot_table(index="월", columns="계정명", values="거래금액_절대값", aggfunc="sum").fillna(0.0)
    return pivot.sort_index()


def _corr_with_lag(a: pd.Series, b: pd.Series, lag: int) -> float:
    if lag > 0:
        return a.iloc[lag:].corr(b.iloc[:-lag])
    elif lag < 0:
        return a.iloc[:lag].corr(b.iloc[-lag:])
    else:
        return a.corr(b)


def _best_lag_pair(pivot: pd.DataFrame, max_lag: int) -> List[Dict[str, object]]:
    cols = list(pivot.columns)
    out: List[Dict[str, object]] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            s1, s2 = pivot[cols[i]], pivot[cols[j]]
            best_lag, best_val = 0, np.nan
            for lag in range(-max_lag, max_lag + 1):
                v = _corr_with_lag(s1, s2, lag)
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
    corr_threshold: float = 0.7,
    max_lag: int = 6,
    rolling_window: int = 6,
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

    # 임계치 이상 쌍
    strong: List[Dict[str, object]] = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = float(corr.iloc[i, j])
            if abs(v) >= float(corr_threshold):
                strong.append({"계정A": cols[i], "계정B": cols[j], "상관계수": v})
    strong_df = pd.DataFrame(strong)

    # 최적 시차 상관
    lag_pairs = pd.DataFrame(_best_lag_pair(pivot, int(max_lag)))

    # 롤링 안정성(낮은 변동성 우선)
    roll = pd.DataFrame(_rolling_stability(pivot, int(rolling_window)))

    # Evidence 샘플
    evid: List[EvidenceDetail] = []
    for row in strong[: min(10, len(strong))]:
        evid.append(EvidenceDetail(
            row_id=f"{row['계정A']}|{row['계정B']}",
            reason=f"corr={row['상관계수']:+.2f} (|r|≥{corr_threshold})",
            risk_score=min(1.0, abs(float(row["상관계수"]))),
            financial_impact=0.0,
            is_key_item=False,
            impacted_assertions=[],
            links={"account_a": row["계정A"], "account_b": row["계정B"], "type": "corr_strong"},
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


