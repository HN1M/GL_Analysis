import pandas as pd
import plotly.express as px
from typing import List, Dict, Any, Optional
from analysis.contracts import LedgerFrame, ModuleResult
from utils.viz import add_materiality_threshold
from utils.helpers import is_credit_account


def create_monthly_trend_figure(ledger_df: pd.DataFrame, master_df: pd.DataFrame, account_code: str, account_name: str):
    """BS/PL, 차/대변 성격을 반영하여 월별 추이 그래프를 생성합니다."""
    mrow = master_df[master_df['계정코드'] == account_code]
    if mrow.empty:
        return None  # 안전 가드
    master_row = mrow.iloc[0]
    bspl = str(master_row.get('BS/PL', 'PL') or 'PL').upper()
    dc = master_row.get('차변/대변', None)
    # 대변 성격이면 그래프 부호를 뒤집어 시각화
    sign = -1.0 if is_credit_account(bspl, dc) else 1.0

    if '연도' not in ledger_df.columns or ledger_df['연도'].isna().all():
        return None
    current_year = int(ledger_df['연도'].max())
    df_filtered = ledger_df[(ledger_df['계정코드'] == account_code) & (ledger_df['연도'].isin([current_year, current_year - 1]))]
    months = list(range(1, 13))
    plot_df_list = []

    if bspl == 'BS':
        def _f(x):
            try:
                v = float(x)
                return 0.0 if pd.isna(v) else v
            except Exception:
                return 0.0
        bop_cy = _f(master_row.get('전기말잔액', 0))
        bop_py = _f(master_row.get('전전기말잔액', 0))
        for year, bop, year_label in [(current_year, bop_cy, 'CY'), (current_year - 1, bop_py, 'PY')]:
            monthly_flow = df_filtered[df_filtered['연도'] == year].groupby('월')['거래금액'].sum() if '거래금액' in df_filtered.columns else pd.Series(dtype=float)
            monthly_series = pd.Series(index=months, data=0.0)
            monthly_series.update(monthly_flow)
            monthly_balance = bop + monthly_series.cumsum()
            plot_df_list.append(pd.DataFrame({'월': months, '금액': monthly_balance.values * sign, '구분': year_label}))
        title_suffix = "월별 잔액 추이 (BS · 월말)"
    else:
        # PL: 금액 컬럼 유연 인식
        cand = ['거래금액', '발생액', '거래금액_절대값', 'amount', '금액']
        amt_col = next((c for c in cand if c in df_filtered.columns), None)
        if amt_col is None:
            return None
        monthly_sum = df_filtered.groupby(['연도', '월'])[amt_col].sum().reset_index()
        for year, year_label in [(current_year, 'CY'), (current_year - 1, 'PY')]:
            year_data = monthly_sum[monthly_sum['연도'] == year]
            monthly_series = pd.Series(index=months, data=0.0)
            monthly_series.update(year_data.set_index('월')[amt_col])
            plot_df_list.append(pd.DataFrame({'월': months, '금액': monthly_series.values * sign, '구분': year_label}))
        title_suffix = "월별 발생액 추이 (PL · 월합계)"

    if not plot_df_list:
        return None

    plot_df = pd.concat(plot_df_list)
    fig = px.bar(
        plot_df,
        x='월', y='금액', color='구분', barmode='group',
        title=f"'{account_name}' ({account_code}) {title_suffix}",
        labels={'월': '월', '금액': '금액', '구분': '연도'},
        color_discrete_map={'PY': '#a9a9a9', 'CY': '#1f77b4'}
    )
    fig.update_xaxes(dtick=1)
    # 🔢 축/툴팁 포맷: 천단위 쉼표, SI 단위 제거
    fig.update_yaxes(separatethousands=True, tickformat=',.0f', showexponent='none', exponentformat='none')
    _note = "시점=월말" if bspl == 'BS' else "집계=월합계"
    fig.update_traces(hovertemplate=f'월=%{{x}}<br>금액=%{{y:,.0f}} 원<br>{_note}<br>구분=%{{fullData.name}}<extra></extra>')
    fig.update_layout(xaxis_title="월", yaxis_title="금액(원)")
    return fig


# (제거됨) 자동 추천 로직: 사용자가 명시적으로 선택한 계정만 사용


def run_trend_module(lf: LedgerFrame, accounts: Optional[List[str]] = None) -> ModuleResult:
    """월별 추이 모듈: 선택 계정 필터 + 월별 발생액 tables 제공."""
    df = lf.df.copy()
    master_df = lf.meta.get("master_df")
    if master_df is None:
        return ModuleResult(
            name="trend",
            summary={},
            tables={},
            figures={},
            evidences=[],
            warnings=["Master DF가 없습니다."]
        )

    # 계정 필터 강제
    acc_codes = [str(a) for a in (accounts or [])]
    if acc_codes:
        df = df[df['계정코드'].astype(str).isin(acc_codes)]

    # 월별 발생액 집계 테이블
    try:
        if '발생액' not in df.columns:
            # 유연 인식: 거래금액/금액 등에서 대체 가능하면 사용
            cand = ['거래금액', '거래금액_절대값', 'amount', '금액']
            alt = next((c for c in cand if c in df.columns), None)
            if alt is not None:
                _df = df.rename(columns={alt: '발생액'})
            else:
                _df = df.copy()
        else:
            _df = df.copy()
        _df['월'] = _df['회계일자'].dt.to_period('M').astype(str)
        flow_tbl = (
            _df.groupby(['계정코드', '월'])['발생액'].sum().reset_index().rename(columns={'발생액': '월별발생액'})
        )
    except Exception:
        flow_tbl = df.head(0).copy()

    figures: Dict[str, Any] = {}
    warns: List[str] = []
    pm_value = (lf.meta or {}).get("pm_value")
    for code in sorted(flow_tbl['계정코드'].astype(str).unique().tolist() if not flow_tbl.empty else acc_codes):
        m = master_df[master_df['계정코드'].astype(str) == code]
        if m.empty:
            warns.append(f"계정코드 {code}가 Master에 없습니다.")
            continue
        name = m.iloc[0].get('계정명', str(code))
        fig = create_monthly_trend_figure(df, master_df, code, name)
        if fig:
            if pm_value:
                add_materiality_threshold(fig, pm_value=pm_value)
            figures[f"{code}:{name}"] = fig
        else:
            warns.append(f"{name}({code}) 그림 생성 불가(데이터 부족).")

    summary = {
        "picked_accounts": acc_codes or sorted(flow_tbl['계정코드'].astype(str).unique().tolist() if not flow_tbl.empty else []),
        "n_figures": len(figures),
        "period_tag_coverage": dict(df['period_tag'].value_counts()) if 'period_tag' in df.columns else {},
    }

    return ModuleResult(
        name="trend",
        summary=summary,
        tables={"monthly_flow": flow_tbl},
        figures=figures,
        evidences=[],
        warnings=warns
    )

