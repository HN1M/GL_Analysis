from __future__ import annotations
import pandas as pd
import numpy as np
import plotly.express as px
from typing import List, Dict, Any, Tuple, Optional, Mapping, Sequence
import re
from analysis.contracts import LedgerFrame, ModuleResult, EvidenceDetail
from utils.helpers import find_column_by_keyword
from utils.viz import apply_corr_heatmap_theme
from config import (
    CORR_DEFAULT_METHOD, CORR_THRESHOLD_DEFAULT, CORR_MIN_ACTIVE_MONTHS_DEFAULT
)


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


def _normalize_cycles_map(df: pd.DataFrame, cycles_map):
    """
    cycles_map 입력 유연화:
    - {계정코드 -> 사이클코드} 형태면 그대로 사용
    - {사이클코드 -> [키워드]} 형태면 계정명 기반으로 추정 매핑 생성
    - None이면 빈 dict
    """
    if not cycles_map:
        return {}
    # code->cycle 형태 판별
    # 값이 문자열이면 사이클 코드라고 가정
    if isinstance(next(iter(cycles_map.values())), str):
        return {str(k): str(v) for k, v in cycles_map.items()}
    # cycle->keywords 형태면 계정명으로 유추
    try:
        name_map = (df.drop_duplicates("계정코드")
                      .assign(계정코드=lambda d: d["계정코드"].astype(str))
                      .set_index("계정코드")["계정명"].astype(str).to_dict())
    except Exception:
        name_map = {}
    out = {}
    for code, nm in name_map.items():
        cyc = _infer_cycle(nm, cycles_map)
        if cyc: out[code] = cyc
    return out


def friendly_correlation_explainer() -> str:
    return (
        "### 해석 가이드(요약)\n"
        "- **상관 ≠ 인과**: 함께 움직인다고 원인/결과는 아닙니다.\n"
        "- **표본 길이**와 **활동월 수**가 짧으면 수치가 흔들립니다.\n"
        "- **음(-)의 상관**은 한쪽이 오르면 다른 쪽이 내리는 동행입니다.\n"
        "- 고급 탭의 **시차** 결과가 크면, ‘선후’ 관계 단서가 될 수 있으나 인과 입증은 아닙니다.\n"
        "- **롤링 안정성**이 낮으면(변동성↑) 일시적 상관일 가능성이 큽니다.\n"
    )


def suggest_anchor_accounts(lf: LedgerFrame, *, cycles_codes: list[str] | None = None,
                            corr_threshold: float = CORR_THRESHOLD_DEFAULT, topn: int = 5) -> pd.DataFrame:
    df = lf.df.copy()
    acct_col = find_column_by_keyword(df.columns, '계정코드')
    if not acct_col or df.empty:
        return pd.DataFrame(columns=['계정코드','계정명','규모합계','표준편차','degree','score'])
    piv = _monthly_pivot(df, acct_col)
    if piv.empty:
        return pd.DataFrame()
    if cycles_codes:
        idx_keep = piv.index.astype(str).isin([str(x) for x in cycles_codes])
        piv = piv.loc[idx_keep]
    if piv.shape[0] < 1:
        return pd.DataFrame()
    abs_piv = piv.abs()
    size = abs_piv.sum(axis=1)
    vol  = abs_piv.std(axis=1)
    corr = piv.T.corr('pearson').fillna(0.0)
    deg  = (corr.abs() >= float(corr_threshold)).sum(axis=1) - 1
    nz = lambda s: (s - s.min()) / (s.max() - s.min() + 1e-12)
    score = 0.4*nz(size) + 0.4*nz(vol) + 0.2*nz(deg)
    try:
        name_map = (
            df.drop_duplicates('계정코드')
              .assign(계정코드=lambda d: d['계정코드'].astype(str))
              .set_index('계정코드')['계정명'].astype(str).to_dict()
        )
    except Exception:
        name_map = {}
    out = (
        pd.DataFrame({
            '계정코드': piv.index.astype(str),
            '계정명':  piv.index.astype(str).map(name_map),
            '규모합계': size.values, '표준편차': vol.values,
            'degree': deg.reindex(piv.index).values, 'score': score.values
        })
        .sort_values('score', ascending=False)
        .head(int(topn))
    )
    return out


def run_correlation_module(
    lf: LedgerFrame,
    accounts: List[str] | None = None,
    *,
    method: str = CORR_DEFAULT_METHOD,
    corr_threshold: float = CORR_THRESHOLD_DEFAULT,
    min_active_months: int = CORR_MIN_ACTIVE_MONTHS_DEFAULT,
    cycles_map: Mapping[str, Sequence[str]] | Mapping[str, str] | None = None,
    within_same_cycle: bool | None = None,
    emit_evidences: bool = False,
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

    corr = piv_f.T.corr(method=method)  # 계정×계정
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
        color_continuous_scale="Blues",
    )
    fig = apply_corr_heatmap_theme(fig)

    # 임계 상관쌍 테이블 (idempotent-safe)
    def build_strong_pairs(corr_matrix: pd.DataFrame, code_to_name: dict, threshold: float = 0.7) -> pd.DataFrame:
        import numpy as _np
        import pandas as _pd
        cm = corr_matrix.copy()
        # ① 대각선 제거
        _np.fill_diagonal(cm.values, _np.nan)
        # ② 상삼각 제거(중복 방지)
        mask = _np.triu(_np.ones_like(cm, dtype=bool), k=1)
        cm = cm.mask(mask)
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

    # === Evidence 생성: |r|≥thr 쌍을 구조화 (risk_score = |r|, financial_impact = min(두 계정의 월별 절대합))
    evidences = []
    try:
        # 계정별 규모(절대 흐름) 합계
        abs_sum = piv_f.abs().sum(axis=1).astype(float)  # index: 계정코드
        cyc_map_norm = _normalize_cycles_map(df, cycles_map)
        for _, row in pairs_df.iterrows():
            code_a = str(row["계정코드_A"]); code_b = str(row["계정코드_B"])
            same_cyc = None
            if cyc_map_norm:
                same_cyc = str(cyc_map_norm.get(code_a,"")) == str(cyc_map_norm.get(code_b,""))
            if within_same_cycle is True and same_cyc is not True:
                continue  # 동일 사이클만 남김
            r = float(row["상관계수"])
            fin = float(min(abs_sum.get(code_a, 0.0), abs_sum.get(code_b, 0.0)))
            evidences.append(EvidenceDetail(
                row_id=f"{code_a}|{code_b}",
                reason=f"corr={r:+.2f}" + (f" · same_cycle={bool(same_cyc)}" if same_cyc is not None else ""),
                anomaly_score=abs(r),           # 정규화(0~1)
                financial_impact=fin,           # 잠재 공동변동 규모의 보수적 근사
                risk_score=abs(r),              # r의 크기가 해석 복잡도/추적 필요도를 대변
                is_key_item=False,
                impacted_assertions=[],         # Assertions 비활성(훈님 방침)
                links={
                    "account_code_a": code_a, "account_code_b": code_b,
                    "account_name_a": row.get("계정명_A",""), "account_name_b": row.get("계정명_B",""),
                    "corr": r, "same_cycle": bool(same_cyc) if same_cyc is not None else None
                }
            ))
    except Exception:
        evidences = []

    # 사이클 매핑 요약(계정명 필요하므로 별도 표에서는 계정명 매핑 필요 시 upstream에서 처리)
    summary = {
        "n_accounts": int(corr.shape[0]),
        "n_pairs_over_threshold": int(len(pairs_df)),
        "corr_threshold": float(corr_threshold),
        "method": str(method)
    }
    return ModuleResult(
        name="correlation",
        summary=summary,
        tables={"strong_pairs": pairs_df, "corr_matrix": corr, "excluded_accounts": excluded},
        figures={"heatmap": fig},
        evidences=evidences,
        warnings=([f"제외된 계정 {len(excluded)}개(변동없음/활동월 부족)."] if not excluded.empty else [])
    )


# --- NEW: Focus 모듈 (단일 계정 vs 나머지) ---
def run_correlation_focus_module(
    lf: LedgerFrame,
    focus_account: str,                 # 계정코드 또는 계정명
    *,
    cycles_map: Mapping[str, Sequence[str]] | Mapping[str, str] | None = None,
    method: str = CORR_DEFAULT_METHOD,
    min_active_months: int = CORR_MIN_ACTIVE_MONTHS_DEFAULT,
    within_same_cycle: bool = True,
    corr_threshold: float = CORR_THRESHOLD_DEFAULT,
) -> ModuleResult:
    df = lf.df.copy()
    acct_col = find_column_by_keyword(df.columns, '계정코드')
    if not acct_col:
        return ModuleResult("correlation_focus", {}, {}, {}, [], ["계정코드 컬럼을 찾지 못했습니다."])
    # 코드/이름 방어
    fc = str(focus_account)
    mask = (df[acct_col].astype(str) == fc) | (df.get("계정명","").astype(str) == fc)
    if not mask.any():
        return ModuleResult("correlation_focus", {}, {}, {}, [], ["선택한 계정을 찾을 수 없습니다."])
    piv = _monthly_pivot(df, acct_col)
    piv_f, excluded = _filter_accounts_for_corr(piv, min_active_months=min_active_months)
    # focus 존재 보장
    # 이름→코드 매핑
    name_to_code = (
        df.drop_duplicates("계정명")
          .assign(계정코드=lambda d: d["계정코드"].astype(str))
          .set_index("계정명")["계정코드"].astype(str).to_dict()
    )
    code = fc if fc in piv_f.index else name_to_code.get(fc)
    if code not in piv_f.index:
        return ModuleResult("correlation_focus", {}, {"excluded_accounts": excluded}, {}, [], ["포커스 계정에 유효한 월별 변동이 없습니다."])
    # within_same_cycle 필터
    cyc_map_norm = _normalize_cycles_map(df, cycles_map)
    if within_same_cycle and cyc_map_norm:
        my_cycle = str(cyc_map_norm.get(str(code), ""))
        keep = [ix for ix in piv_f.index if str(cyc_map_norm.get(str(ix), "")) == my_cycle]
        piv_f = piv_f.loc[keep] if len(keep) >= 2 else piv_f
    if piv_f.shape[0] < 2:
        return ModuleResult("correlation_focus", {}, {"excluded_accounts": excluded}, {}, [], ["상관을 계산할 타 계정이 부족합니다."])
    rvec = piv_f.T.corr(method=method)[str(code)].drop(labels=[str(code)], errors="ignore").sort_values(key=lambda s: s.abs(), ascending=False)
    # 코드→이름 맵
    name_map = (df.drop_duplicates('계정코드').assign(계정코드=lambda d: d['계정코드'].astype(str))
                   .set_index('계정코드')['계정명'].astype(str).to_dict())
    tbl = pd.DataFrame({
        "상대계정코드": rvec.index.astype(str),
        "상대계정명": [name_map.get(c, c) for c in rvec.index.astype(str)],
        "상관계수": rvec.values
    })
    # 시각화(바 차트)
    fig = px.bar(tbl.head(30), x="상대계정명", y="상관계수", title=f"포커스: {name_map.get(str(code), str(code))} vs 타 계정")
    fig.update_yaxes(range=[-1,1])
    # evidence (임계 이상 Top-N)
    evid = []
    abs_sum = piv_f.abs().sum(axis=1).astype(float)
    for _, r in tbl.iterrows():
        v = float(r["상관계수"])
        if abs(v) < float(corr_threshold): break
        c2 = str(r["상대계정코드"])
        fin = float(min(abs_sum.get(str(code),0.0), abs_sum.get(c2,0.0)))
        same_cyc = None
        if cyc_map_norm:
            same_cyc = str(cyc_map_norm.get(str(code),"")) == str(cyc_map_norm.get(c2,""))
            if within_same_cycle and not same_cyc: 
                continue
        evid.append(EvidenceDetail(
            row_id=f"{code}|{c2}",
            reason=f"focus_corr={v:+.2f}" + (f" · same_cycle={bool(same_cyc)}" if same_cyc is not None else ""),
            anomaly_score=abs(v),
            financial_impact=fin,
            risk_score=abs(v),
            is_key_item=False,
            impacted_assertions=[],
            links={"focus_code": str(code), "other_code": c2, "focus_name": name_map.get(str(code), str(code)), "other_name": r["상대계정명"], "corr": v}
        ))
    summ = {"focus_code": str(code), "n_candidates": int(len(tbl))}
    return ModuleResult("correlation_focus", summ, {"focus_corr": tbl, "excluded_accounts": excluded}, {"bar": fig}, evid, [])
