from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.express as px

from typing import Optional
from analysis.timeseries import build_trend_validation_data
from utils.viz import add_period_guides, add_materiality_threshold


def render_trend_validation_section(
    monthly: pd.DataFrame,
    *,
    section_id: str,
    is_bs: bool,
    pm_value: Optional[float] = None,
    title: str = "데이터 검증: 월별 추세 분석(막대그래프로 직접 대조)",
):
    """
    - monthly: ['date','flow'] (+ 'balance' 선택)
    - section_id: 토글 고유키에 섞을 식별자(계정코드 등)
    - is_bs: BS 여부
    - pm_value: PM 표시(점선 보조선)
    """
    df_val = build_trend_validation_data(monthly, is_bs=is_bs)
    if df_val.empty:
        st.info("검증용 데이터가 없습니다.")
        return

    st.subheader(title)

    # 고유 key로 토글 충돌 방지
    show_cfg = st.toggle("이 차트의 통계 설정 보기", key=f"trend_cfg_{section_id}")

    # 막대: 실측, 선: 예측
    fig = px.bar(
        df_val,
        x="date",
        y="actual",
        color="measure",
        barmode="group",
        title="actual vs predicted (검증)",
    )
    # 동일 x축 위에 예측선 오버레이
    for m in df_val["measure"].unique():
        sub = df_val[df_val["measure"] == m]
        fig.add_scatter(
            x=sub["date"],
            y=sub["predicted"],
            mode="lines+markers",
            name=f"predicted ({m})",
        )

    add_period_guides(fig, df_val["date"])
    if pm_value:
        add_materiality_threshold(fig, pm_value)

    # 숫자 포맷
    fig.update_yaxes(separatethousands=True, tickformat=",.0f", showexponent="none", exponentformat="none")
    fig.update_traces(hovertemplate="month=%{x}<br>value=%{y:,.0f}<extra></extra>")

    st.plotly_chart(fig, use_container_width=True)

    if show_cfg:
        by_m = df_val.groupby("measure").agg(
            n_points=("date", "count"),
            last_actual=("actual", "last"),
            last_pred=("predicted", "last"),
        )
        st.dataframe(by_m, use_container_width=True)


