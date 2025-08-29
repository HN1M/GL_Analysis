# utils/viz.py
# 목적: Materiality(Performance Materiality, PM) 보조선/배지 추가 유틸
# - Plotly Figure에 빨간 점선(가로선) + "PM=xxx원" 라벨을 안전하게 추가
# - 현재 y축 범위에 PM이 없으면 축을 자동 확장해서 선이 보이게 함
# - Pareto(보조축 있음)에서도 1차 y축에 정확히 그려줌

from __future__ import annotations
from typing import Optional
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


