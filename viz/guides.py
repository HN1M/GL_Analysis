# Materiality 가이드라인(붉은 점선) 유틸
# - matplotlib/plotly 모두 지원. 사용하는 라이브러리에 맞춰 호출만 붙이면 됨.

def add_materiality_lines_matplotlib(ax, *, y_threshold=None, x_threshold=None, label_prefix="Materiality"):
    # y축 기준 수평선
    if y_threshold is not None and ax is not None:
        ax.axhline(y_threshold, linestyle="--", color="red", linewidth=1.25, alpha=0.9)  # 붉은 점선
        ax.text(ax.get_xlim()[0], y_threshold, f"{label_prefix}: {y_threshold:,.0f}",
                va="bottom", ha="left", fontsize=9, color="red", alpha=0.9)
    # x축 기준 수직선
    if x_threshold is not None and ax is not None:
        ax.axvline(x_threshold, linestyle="--", color="red", linewidth=1.25, alpha=0.9)
        ax.text(x_threshold, ax.get_ylim()[1], f"{label_prefix}: {x_threshold:,.0f}",
                va="top", ha="right", fontsize=9, color="red", alpha=0.9)


def add_materiality_lines_plotly(fig, *, y_threshold=None, x_threshold=None, label_prefix="Materiality"):
    # Plotly Figure에 가이드라인 추가
    if fig is None:
        return fig
    if y_threshold is not None:
        try:
            fig.add_hline(y=y_threshold, line_dash="dash", line_color="red", opacity=0.9)
            fig.add_annotation(xref="paper", x=0.0, y=y_threshold, yref="y",
                               text=f"{label_prefix}: {y_threshold:,.0f}",
                               showarrow=False, align="left", yanchor="bottom", font=dict(color="red", size=10))
        except Exception:
            pass
    if x_threshold is not None:
        try:
            fig.add_vline(x=x_threshold, line_dash="dash", line_color="red", opacity=0.9)
            fig.add_annotation(yref="paper", y=1.0, x=x_threshold, xref="x",
                               text=f"{label_prefix}: {x_threshold:,.0f}",
                               showarrow=False, align="right", xanchor="right", font=dict(color="red", size=10))
        except Exception:
            pass
    return fig


