from typing import List, Dict, Any
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from itertools import product
from analysis.contracts import LedgerFrame, ModuleResult
from utils.viz import add_materiality_threshold, add_pm_badge


# 공통: 금액 컬럼 후보에서 하나를 고르고, 필요 시 절대값으로 변환
_AMOUNT_CANDIDATES = ["거래금액_절대값", "거래금액", "발생액", "amount", "금액"]

def _pick_amount_column(df: pd.DataFrame) -> str | None:
    for c in _AMOUNT_CANDIDATES:
        if c in df.columns:
            return c
    return None


def create_pareto_figure(ledger_df: pd.DataFrame, min_amount: float = 0, include_others: bool = True, pm_value: float | None = None):
    """거래처별 거래금액 파레토 차트.
    - min_amount 이상인 거래처만 개별 표기
    - 나머지는 '기타'로 합산(옵션)
    - 1차 Y축에 PM 점선/라벨 표시
    """
    if ledger_df is None or ledger_df.empty or "연도" not in ledger_df.columns:
        return None

    cy_df = ledger_df[ledger_df["연도"] == ledger_df["연도"].max()]
    if "거래처" not in cy_df.columns or cy_df["거래처"].nunique() < 1:
        return None

    amt_col = _pick_amount_column(cy_df)
    if amt_col is None:
        return None

    # 절대값 기준 합계 (표시의 일관성을 위해)
    series_raw = cy_df.groupby("거래처")[amt_col].sum()
    vendor_sum = series_raw.abs()

    if vendor_sum.empty:
        return None

    # 임계치 필터
    above = vendor_sum[vendor_sum >= float(min_amount)].sort_values(ascending=False)
    etc_sum = float(vendor_sum.sum() - above.sum())

    series = above
    if include_others and etc_sum > 0:
        series = pd.concat([above, pd.Series({"기타": etc_sum})])

    cum_ratio = series.cumsum() / series.sum() * 100.0

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=series.index, y=series.values, name="거래 금액"), secondary_y=False)
    fig.add_trace(go.Scatter(x=series.index, y=cum_ratio.values, name="누적 비율(%)", mode="lines+markers"), secondary_y=True)
    fig.update_layout(title="거래처 집중도 분석 (Pareto)", yaxis_title="금액", yaxis2_title="누적 비율(%)")

    # 🔢 축/툴팁 포맷
    fig.update_yaxes(separatethousands=True, tickformat=",.0f", showexponent="none", exponentformat="none", secondary_y=False)
    fig.update_yaxes(tickformat=".1f", ticksuffix="%", secondary_y=True)
    fig.update_traces(hovertemplate="%{x}<br>%{y:,.0f} 원<extra></extra>", selector=dict(type="bar"))
    fig.update_traces(hovertemplate="%{x}<br>누적 비율=%{y:.1f}%<extra></extra>", selector=dict(type="scatter"))

    # PM 점선/라벨 (1차 y축)
    if pm_value and float(pm_value) > 0:
        add_materiality_threshold(fig, pm_value=float(pm_value))

    return fig


def create_vendor_heatmap(ledger_df: pd.DataFrame, min_amount: float = 0, include_others: bool = True, pm_value: float | None = None):
    """거래처별 월별 활동 히트맵.
    - min_amount 이상인 거래처만 개별 표기
    - 나머지는 '기타'로 월별 합산(옵션)
    - 우측 상단에 PM 배지 표시
    """
    if ledger_df is None or ledger_df.empty or "거래처" not in ledger_df.columns:
        return None

    df = ledger_df.copy()
    if "회계일자" not in df.columns:
        return None

    amt_col = _pick_amount_column(df)
    if amt_col is None:
        return None

    df["연월"] = pd.to_datetime(df["회계일자"], errors="coerce").dt.to_period("M").astype(str)
    pivot = df.pivot_table(index="거래처", columns="연월", values=amt_col, aggfunc="sum").fillna(0).abs()
    if pivot.empty:
        return None

    totals = pivot.sum(axis=1)
    above_idx = totals >= float(min_amount)
    pivot_above = pivot.loc[above_idx].copy()
    pivot_above["_tot_"] = pivot_above.sum(axis=1)
    pivot_above = pivot_above.sort_values("_tot_", ascending=False).drop(columns=["_tot_"])

    pivot_final = pivot_above
    if include_others:
        below = pivot.loc[~above_idx]
        if not below.empty:
            etc_row = pd.DataFrame([below.sum(axis=0)], index=["기타"])
            pivot_final = pd.concat([pivot_above, etc_row], axis=0)

    fig = px.imshow(pivot_final, title="거래처 월별 활동 히트맵", labels=dict(x="연월", y="거래처", color="거래금액"))

    # 🔢 컬러바/툴팁 포맷
    fig.update_coloraxes(colorbar=dict(tickformat=",.0f"))
    fig.update_traces(hovertemplate="연월=%{x}<br>거래처=%{y}<br>거래금액=%{z:,.0f} 원<extra></extra>")

    # PM 배지
    if pm_value and float(pm_value) > 0:
        add_pm_badge(fig, pm_value=float(pm_value))

    return fig


def create_vendor_detail_figure(ledger_df: pd.DataFrame, vendor_name: str, all_months: List[str]):
    """특정 거래처의 월별 거래액을 계정별 누적 막대그래프로 생성합니다. (전체 기간 X축 보장)"""
    vendor_df = ledger_df[ledger_df["거래처"] == vendor_name].copy()
    amt_col = _pick_amount_column(vendor_df)
    if amt_col is None:
        return None

    vendor_df["연월"] = pd.to_datetime(vendor_df["회계일자"], errors="coerce").dt.to_period("M").astype(str)
    summary = vendor_df.groupby(["연월", "계정명"], as_index=False)[amt_col].sum()
    summary[amt_col] = summary[amt_col].abs()

    unique_accounts = vendor_df["계정명"].unique()
    axis_labels = {"연월": "거래월", amt_col: "거래금액"}

    if len(unique_accounts) == 0:
        empty_df = pd.DataFrame({"연월": all_months, "계정명": [None] * len(all_months), amt_col: [0] * len(all_months)})
        fig = px.bar(empty_df, x="연월", y=amt_col, labels=axis_labels)
    else:
        template_df = pd.DataFrame(list(product(all_months, unique_accounts)), columns=["연월", "계정명"])
        merged_summary = pd.merge(template_df, summary, on=["연월", "계정명"], how="left").fillna(0)
        fig = px.bar(
            merged_summary,
            x="연월", y=amt_col, color="계정명",
            category_orders={"연월": all_months},
            labels=axis_labels,
        )

    fig.update_layout(
        barmode="stack",
        title=f"'{vendor_name}' 거래처 월별/계정별 상세 내역"
    )
    # 🔢 축/툴팁 포맷: 천단위 쉼표, SI 제거
    fig.update_yaxes(separatethousands=True, tickformat=",.0f", showexponent="none", exponentformat="none")
    fig.update_traces(hovertemplate="연월=%{x}<br>금액=%{y:,.0f} 원<br>계정명=%{fullData.name}<extra></extra>")
    return fig


def run_vendor_module(lf: LedgerFrame, account_codes: List[str] | None = None,
                      min_amount: float = 0, include_others: bool = True) -> ModuleResult:
    """거래처 모듈: 선택 계정 필터 + 최소 거래금액 필터('기타' 합산) + PM 라벨/배지."""
    df = lf.df
    use_df = df.copy()
    if account_codes:
        acs = [str(a) for a in account_codes]
        use_df = use_df[use_df["계정코드"].astype(str).isin(acs)]

    figures: Dict[str, Any] = {}
    warnings: List[str] = []
    pm_value = (lf.meta or {}).get("pm_value")

    pareto = create_pareto_figure(use_df, min_amount=min_amount, include_others=include_others, pm_value=pm_value)
    heatmap = create_vendor_heatmap(use_df, min_amount=min_amount, include_others=include_others, pm_value=pm_value)

    if pareto: figures["pareto"] = pareto
    else: warnings.append("Pareto 그래프 생성 불가(데이터 부족).")
    if heatmap: figures["heatmap"] = heatmap
    else: warnings.append("히트맵 생성 불가(데이터 부족).")

    # 요약 정보
    if "연도" in use_df.columns:
        cy = use_df[use_df["연도"] == use_df["연도"].max()]
    else:
        cy = use_df.iloc[:0]

    amt_col = _pick_amount_column(cy) if not cy.empty else None
    vendor_sum = cy.groupby("거래처")[amt_col].sum().abs() if (amt_col and "거래처" in cy.columns and not cy.empty) else pd.Series(dtype=float)
    n_above = int((vendor_sum >= float(min_amount)).sum()) if not vendor_sum.empty else 0
    n_below = int((vendor_sum < float(min_amount)).sum()) if not vendor_sum.empty else 0

    summary = {
        "filtered_accounts": [str(a) for a in account_codes] if account_codes else [],
        "min_amount": float(min_amount),
        "include_others": bool(include_others),
        "n_above_threshold": n_above,
        "n_below_threshold": n_below,
        "n_figures": len(figures),
        "period_tag_coverage": dict(use_df["period_tag"].value_counts()) if "period_tag" in use_df.columns else {},
    }
    return ModuleResult(
        name="vendor",
        summary=summary,
        tables={},
        figures=figures,
        evidences=[],
        warnings=warnings
    )
