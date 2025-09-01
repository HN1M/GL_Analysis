from __future__ import annotations
import pandas as pd
import numpy as np
import plotly.express as px
from typing import List, Dict, Any, Tuple, Optional, Mapping, Sequence
import re
from analysis.contracts import LedgerFrame, ModuleResult
from utils.helpers import find_column_by_keyword


def _monthly_pivot(df: pd.DataFrame, acct_col: str) -> pd.DataFrame:
    """계정코드×연월 피벗(거래금액 합계). PL/BS 모두 월 흐름 기준."""
    if '회계일자' not in df.columns:
        raise ValueError("회계일자 필요")
    g = (df.assign(연월=df['회계일자'].dt.to_period('M').astype(str))
           .groupby([acct_col, '연월'])['거래금액'].sum()
           .unstack('연월', fill_value=0.0)
           .sort_index())
    return g
 
def _filter_accounts_for_corr(piv: pd.DataFrame, min_active_months: int = 6) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    - Drop accounts with zero variance across months (std == 0) OR
      with insufficient active months (abs(value)>0 in fewer than min_active_months months).
    - Return filtered pivot and an exclusions dataframe with reasons.
    """
    if piv.empty:
        return piv, pd.DataFrame(columns=['계정코드','사유','활동월수','표준편차'])
    std = piv.std(axis=1)
    active = (piv.abs() > 0).sum(axis=1)
    reason = []
    idx = piv.index.astype(str)
    keep = (std > 0) & (active >= int(min_active_months))
    for code, s, a, k in zip(idx, std, active, keep):
        if k:
            continue
        r = []
        if s == 0:
            r.append("변동없음(표준편차 0)")
        if a < int(min_active_months):
            r.append(f"활동 월 부족(<{int(min_active_months)})")
        reason.append((code, " & ".join(r) if r else "제외", int(a), float(s)))
    excluded = pd.DataFrame(reason, columns=['계정코드','사유','활동월수','표준편차'])
    return piv.loc[keep], excluded


def _infer_cycle(account_name: str, cycles_map: Mapping[str, Sequence[str]]) -> Optional[str]:
    """
    STANDARD_ACCOUNTING_CYCLES 기반의 간단한 키워드 매핑.
    가장 먼저 매칭되는 사이클을 반환(우선순위: dict 정의 순서).
    """
    name = str(account_name or "").lower()
    for cycle, keywords in cycles_map.items():
        for kw in keywords:
            if kw and re.search(re.escape(str(kw).lower()), name):
                return cycle
    return None


def map_accounts_to_cycles(accounts: List[str], *, cycles_map: Mapping[str, Sequence[str]]) -> Dict[str, Optional[str]]:
    """배치 매핑: 계정명 리스트 → {계정명: 사이클(or None)}.
    cycles_map은 상위 레이어(app/services)에서 주입합니다.
    """
    return {acc: _infer_cycle(acc, cycles_map) for acc in accounts}


def run_correlation_module(
    lf: LedgerFrame,
    accounts: List[str] | None = None,
    *,
    corr_threshold: float = 0.7,
    min_active_months: int = 6,
    cycles_map: Mapping[str, Sequence[str]] | None = None,
) -> ModuleResult:
    df = lf.df.copy()
    acct_col = find_column_by_keyword(df.columns, '계정코드')
    if not acct_col:
        return ModuleResult("correlation", {}, {}, {}, [], ["계정코드 컬럼을 찾지 못했습니다."])

    # 대상 계정 필터
    if accounts:
        codes = [str(a) for a in accounts]
        df = df[df[acct_col].astype(str).isin(codes)].copy()

    if df.empty:
        return ModuleResult("correlation", {}, {}, {}, [], ["선택된 데이터가 없습니다."])

    piv = _monthly_pivot(df, acct_col)
    piv_f, excluded = _filter_accounts_for_corr(piv, min_active_months=min_active_months)
    if piv_f.shape[0] < 2:
        warn = "상관을 계산할 계정이 2개 미만입니다."
        if not excluded.empty:
            warn += f" (제외된 계정 {len(excluded)}개: 변동없음/활동월 부족)"
        return ModuleResult("correlation", {}, {"excluded_accounts": excluded}, {}, [], [warn])

    corr = piv_f.T.corr(method='pearson')  # 계정×계정
    # 계정코드 → 계정명 매핑
    try:
        name_map = (
            df.drop_duplicates('계정코드')
              .assign(계정코드=lambda d: d['계정코드'].astype(str))
              .set_index('계정코드')['계정명']
              .astype(str).to_dict()
        )
    except Exception:
        name_map = {}
    xn = [name_map.get(str(c), str(c)) for c in corr.columns]
    yn = [name_map.get(str(r), str(r)) for r in corr.index]
    fig = px.imshow(
        corr,
        text_auto=False,
        title="계정 간 월별 상관 히트맵",
        labels=dict(x="계정", y="계정", color="상관계수"),
        aspect='auto',
        x=xn,
        y=yn,
    )
    fig.update_traces(hovertemplate="계정: %{y} × %{x}<br>상관계수: %{z:.3f}<extra></extra>")
    fig.update_coloraxes(cmin=-1, cmax=1)
    fig.update_xaxes(type='category')
    fig.update_yaxes(type='category')

    # 임계 상관쌍 테이블 (idempotent-safe)
    def build_strong_pairs(corr_matrix: pd.DataFrame, code_to_name: dict, threshold: float = 0.7) -> pd.DataFrame:
        import numpy as _np
        import pandas as _pd
        mask = _np.triu(_np.ones_like(corr_matrix, dtype=bool), k=1)
        cm = corr_matrix.copy().mask(mask)
        rows = []
        abs_vals = cm.abs().values
        idx_i, idx_j = _np.where(abs_vals >= threshold)
        for i, j in zip(idx_i, idx_j):
            rows.append({
                "계정코드_A": corr_matrix.index[i],
                "계정코드_B": corr_matrix.columns[j],
                "상관계수": float(cm.values[i, j]),
            })
        pairs_df = _pd.DataFrame(rows)
        if pairs_df.empty:
            return pairs_df
        pairs_df = pairs_df.assign(
            계정명_A=pairs_df["계정코드_A"].map(code_to_name),
            계정명_B=pairs_df["계정코드_B"].map(code_to_name),
        )
        base_cols = ["계정명_A", "계정코드_A", "계정명_B", "계정코드_B", "상관계수"]
        pairs_df = pairs_df[base_cols]
        pairs_df = pairs_df.reindex(
            pairs_df["상관계수"].abs().sort_values(ascending=False).index
        )
        return pairs_df

    pairs_df = build_strong_pairs(corr, name_map, threshold=float(corr_threshold))

    # 사이클 매핑 요약(계정명 필요하므로 별도 표에서는 계정명 매핑 필요 시 upstream에서 처리)
    summary = {
        "n_accounts": int(corr.shape[0]),
        "n_pairs_over_threshold": int(len(pairs_df)),
        "corr_threshold": float(corr_threshold)
    }
    return ModuleResult(
        name="correlation",
        summary=summary,
        tables={"strong_pairs": pairs_df, "corr_matrix": corr, "excluded_accounts": excluded},
        figures={"heatmap": fig},
        evidences=[],
        warnings=([f"제외된 계정 {len(excluded)}개(변동없음/활동월 부족)."] if not excluded.empty else [])
    )


