# utils/viz.py
# 목적: Materiality(Performance Materiality, PM) 보조선/배지 추가 유틸
# - Plotly Figure에 빨간 점선(가로선) + "PM=xxx원" 라벨을 안전하게 추가
# - 현재 y축 범위에 PM이 없으면 축을 자동 확장해서 선이 보이게 함
# - Pareto(보조축 있음)에서도 1차 y축에 정확히 그려줌

from __future__ import annotations
from typing import Optional
import pandas as pd
import math


def _is_plotly_fig(fig) -> bool:
    try:
        # 지연 임포트 (환경에 plotly 미설치일 때도 함수 자체는 import 가능하도록)
        import plotly.graph_objects as go  # noqa: F401
        from plotly.graph_objs import Figure
        return isinstance(fig, Figure)
    except Exception:
        return False


def _get_primary_y_data_bounds(fig):
    """
    1차 y축 데이터의 (min, max) 추정.
    - secondary_y=True로 올라간 trace는 제외
    - trace.y가 수치 배열일 때만 집계
    """
    ymin, ymax = math.inf, -math.inf
    for tr in getattr(fig, "data", []):
        # 보조축 여부: trace.yaxis 가 'y2'/'y3'... 이면 보조축
        yaxis = getattr(tr, "yaxis", "y")
        if yaxis and str(yaxis).lower() != "y":  # 'y2' 등은 제외
            continue
        y = getattr(tr, "y", None)
        if y is None:
            continue
        try:
            for v in y:
                if v is None:
                    continue
                fv = float(v)
                if math.isfinite(fv):
                    ymin = min(ymin, fv)
                    ymax = max(ymax, fv)
        except Exception:
            # 숫자 배열이 아니면 스킵
            continue
    if ymin is math.inf:  # 데이터가 비어있는 경우
        return (0.0, 0.0)
    return (ymin, ymax)


def _ensure_y_contains(fig, y_value: float, pad_ratio: float = 0.05):
    """
    y_value가 y축 범위에 포함되도록 레이아웃을 조정.
    - 기존 auto-range라도 PM이 축 밖이면 강제로 range 부여
    - pad_ratio만큼 여유를 둬서 라벨이 잘리지 않게 함
    """
    if not math.isfinite(y_value):
        return
    # 현재 1차 y축 데이터 범위 추정
    ymin_data, ymax_data = _get_primary_y_data_bounds(fig)
    # 데이터가 전부 음수이거나 전부 양수일 수 있음 → PM이 더 큰 쪽에 있으면 확장
    base_min = min(0.0, ymin_data) if math.isfinite(ymin_data) else 0.0
    base_max = max(0.0, ymax_data) if math.isfinite(ymax_data) else 0.0
    tgt_min = min(base_min, y_value)
    tgt_max = max(base_max, y_value)
    if tgt_min == tgt_max:
        # 완전 평평하면 살짝 폭 추가
        span = abs(y_value) if y_value != 0 else 1.0
        tgt_min -= span * 0.5
        tgt_max += span * 0.5
    # 여유 패딩
    span = (tgt_max - tgt_min) or 1.0
    pad = span * float(pad_ratio)
    final_min = tgt_min - pad
    final_max = tgt_max + pad
    # yaxis는 레이아웃 키 'yaxis' (서브플롯 아닌 기본 도면 기준)
    if "yaxis" not in fig.layout:
        fig.update_layout(yaxis=dict(range=[final_min, final_max]))
    else:
        fig.layout.yaxis.update(range=[final_min, final_max])


def add_materiality_threshold(fig, pm_value: Optional[float], *, label: bool = True):
    """
    Plotly Figure에 PM 가로 점선 + 라벨 추가.
    - pm_value가 None/0/음수면 아무 것도 하지 않음
    - Pareto(보조축)도 1차 y축에 라인을 그림 (yref='y')
    - 축 범위를 자동 확장해서 항상 보이게 함
    반환: 동일 Figure (in-place 수정 후)
    """
    if not _is_plotly_fig(fig):
        return fig
    try:
        pm = float(pm_value) if pm_value is not None else 0.0
    except Exception:
        pm = 0.0
    if pm <= 0:
        return fig

    # y축 범위에 PM이 포함되도록 먼저 보장
    _ensure_y_contains(fig, pm, pad_ratio=0.08)

    # 점선 라인 추가
    # xref='paper'로 0~1 전폭에 걸쳐 수평선, yref='y'로 1차 y축 기준 고정
    line_shape = dict(
        type="line",
        xref="paper", x0=0, x1=1,
        yref="y",     y0=pm, y1=pm,
        line=dict(color="red", width=2, dash="dot"),
        layer="above"
    )
    shapes = list(fig.layout.shapes) if getattr(fig.layout, "shapes", None) else []
    shapes.append(line_shape)
    fig.update_layout(shapes=shapes)

    # 라벨(오른쪽 끝)
    if label:
        annotations = list(fig.layout.annotations) if getattr(fig.layout, "annotations", None) else []
        annotations.append(dict(
            x=1.0, xref="paper",
            y=pm, yref="y",
            xanchor="left", yanchor="bottom",
            text=f"PM {pm:,.0f}원",
            showarrow=False,
            font=dict(color="red", size=11),
            bgcolor="rgba(255,255,255,0.6)",
            bordercolor="red",
            borderwidth=0.5,
            align="left"
        ))
        fig.update_layout(annotations=annotations)

    return fig


def add_pm_badge(fig, pm_value: Optional[float], *, text: str | None = None):
    """
    Heatmap처럼 선을 긋기 애매한 그래프에 우측 상단 배지 추가.
    반환: 동일 Figure (in-place)
    """
    if not _is_plotly_fig(fig):
        return fig
    try:
        pm = float(pm_value) if pm_value is not None else 0.0
    except Exception:
        pm = 0.0
    if pm <= 0:
        return fig

    label = text or f"PM {pm:,.0f}원"
    annotations = list(fig.layout.annotations) if getattr(fig.layout, "annotations", None) else []
    annotations.append(dict(
        x=0.995, xref="paper",
        y=0.995, yref="paper",
        xanchor="right", yanchor="top",
        text=label,
        showarrow=False,
        font=dict(color="red", size=11),
        bgcolor="rgba(255,255,255,0.6)",
        bordercolor="red",
        borderwidth=0.5,
        align="right"
    ))
    fig.update_layout(annotations=annotations)
    return fig



def add_time_dividers(fig, xdates, show_quarter: bool = True, show_year_label: bool = True):
    """
    xdates: datetime Series/List (차트의 x 값)
    - 연도 경계(1/1): 굵은 실선
    - 분기 경계(4/1, 7/1, 10/1): 얇은 점선
    """
    if xdates is None:
        return fig
    try:
        ts = pd.to_datetime(pd.Series(xdates)).dropna().sort_values()
    except Exception:
        return fig
    if ts.empty:
        return fig

    # 연도 경계 (첫 해는 제외, 다음 해 1/1 지점)
    years = ts.dt.year.unique()
    for y in years[1:]:
        x = pd.Timestamp(year=y, month=1, day=1)
        if x < ts.iloc[0] or x > ts.iloc[-1]:
            continue
        try:
            fig.add_vline(
                x=x, line_width=2, line_dash="solid",
                line_color="rgba(0,0,0,0.35)",
                annotation_text=(str(y) if show_year_label else None),
                annotation_position="top", annotation_font_color="rgba(0,0,0,0.55)"
            )
        except Exception:
            continue

    # 분기 경계: 4/1, 7/1, 10/1
    if show_quarter:
        start = pd.Timestamp(ts.iloc[0].year, ts.iloc[0].month, 1)
        end   = pd.Timestamp(ts.iloc[-1].year, ts.iloc[-1].month, 1)
        for x in pd.date_range(start, end, freq="MS"):
            if x.month in (4, 7, 10):
                try:
                    fig.add_vline(
                        x=x, line_width=1, line_dash="dot",
                        line_color="rgba(0,0,0,0.18)"
                    )
                except Exception:
                    continue
    return fig


def add_period_guides(fig, x_series):
    """
    월 단위 시계열에 연/분기 경계선 추가.
    - 연말(12월): 굵은 점선(검정)
    - 분기말(3/6/9/12월): 얇은 점선(회색)
    """
    import pandas as _pd
    try:
        xs = _pd.to_datetime(x_series)
    except Exception:
        try:
            xs = _pd.to_datetime(_pd.Index(x_series))
        except Exception:
            return fig
    if xs is None or len(xs) == 0:
        return fig
    x_min, x_max = xs.min(), xs.max()
    if _pd.isna(x_min) or _pd.isna(x_max):
        return fig
    months = _pd.date_range(x_min, x_max, freq="M")

    # 연말: 12월(굵은 선)
    year_ends = [m for m in months if m.month == 12]
    for x in year_ends:
        try:
            fig.add_vline(x=x, line=dict(width=2, dash="dash"), line_color="black")
        except Exception:
            continue

    # 분기말: 3/6/9/12월(얇은 점선)
    quarter_ends = [m for m in months if m.month in (3, 6, 9, 12)]
    for x in quarter_ends:
        try:
            fig.add_vline(x=x, line=dict(width=1, dash="dot"), line_color="gray")
        except Exception:
            continue
    return fig


# === NEW: Correlation heatmap theming & explanation utilities ===
def apply_corr_heatmap_theme(fig, *, title: str | None = None):
    """
    상관 히트맵 공통 테마 적용:
    - 색상/스케일: Blues, [-1, 1]
    - 축 타입: category
    - 호버 템플릿: 계정명 × 계정명 + r 표시
    """
    if title:
        fig.update_layout(title=title)
    try:
        fig.update_coloraxes(cmin=-1, cmax=1, colorscale="Blues", colorbar=dict(tickformat=".2f"))
    except Exception:
        pass
    try:
        fig.update_xaxes(type="category")
        fig.update_yaxes(type="category")
    except Exception:
        pass
    try:
        fig.update_traces(hovertemplate="계정: %{y} × %{x}<br>상관계수: %{z:.3f}<extra></extra>")
    except Exception:
        pass
    return fig


def narrate_corr_meaning(r_signed: float, r_abs: float, r_lag: float | None, lag: int | None) -> str:
    """
    기본(부호 포함) vs 고급(절대값/규모) vs 최적시차 결과를 자연어로 설명.
    UI에서 특정 계정쌍 선택 시 보조 텍스트로 활용.
    """
    def _sgn(x: float) -> str:
        return "양( + )" if float(x) >= 0 else "음( - )"

    lines: list[str] = []
    lines.append(f"- 기본 상관(부호포함): r={float(r_signed):+ .2f} → {'같이 오르고 내리는' if float(r_signed)>=0 else '한쪽↑ 다른쪽↓'} 경향.")
    lines.append(f"- 고급 상관(규모/절대값): r={float(r_abs):+ .2f} → 금액 규모가 {'같이 커지거나 작아지는' if float(r_abs)>=0 else '규모 관점에서는 엇갈리는'} 경향.")
    if lag is not None and r_lag is not None:
        when = "A가 B보다" if int(lag) > 0 else "B가 A보다"
        lines.append(f"- 최적시차: {int(lag):+d}개월 ⇒ {when} {abs(int(lag))}개월 선행/후행 시 상관 r={float(r_lag):+ .2f}.")
    lines.append("※ 기본과 고급이 다르면: '방향 동행'(부호)과 '규모 동행'(절대값)이 다른 신호를 준다는 뜻입니다.")
    return "\n".join(lines)
