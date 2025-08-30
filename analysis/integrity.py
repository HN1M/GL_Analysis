import pandas as pd
from typing import List, Dict, Any
try:
    from analysis.contracts import ModuleResult
except Exception:
    ModuleResult = None  # 타입 힌트/런타임 가드


def analyze_reconciliation(ledger_df: pd.DataFrame, master_df: pd.DataFrame):
    """Master와 Ledger 데이터 간의 정합성을 검증합니다.

    반환값:
    - overall_status: "Pass" | "Warning" | "Fail"
    - 결과 DataFrame
    """
    results, overall_status = [], "Pass"
    cy_ledger_df = ledger_df[ledger_df['연도'] == ledger_df['연도'].max()]
    for _, master_row in master_df.iterrows():
        account_code = master_row['계정코드']
        bspl = master_row.get('BS/PL', 'PL').upper()
        bop = master_row.get('전기말잔액', 0)
        eop_master = master_row.get('당기말잔액', 0)

        net_change_gl = cy_ledger_df[cy_ledger_df['계정코드'] == account_code]['거래금액'].sum()
        eop_gl = (bop + net_change_gl) if bspl == 'BS' else net_change_gl

        difference = eop_master - eop_gl
        diff_pct = abs(difference) / max(abs(eop_master), 1)
        status = "Fail" if diff_pct > 0.001 else "Warning" if abs(difference) > 0 else "Pass"
        if status == "Fail":
            overall_status = "Fail"
        elif status == "Warning" and overall_status == "Pass":
            overall_status = "Warning"

        results.append({
            '계정코드': account_code,
            '계정명': master_row.get('계정명', ''),
            '구분': bspl,
            '기초잔액(Master)': bop,
            '당기증감액(Ledger)': net_change_gl,
            '계산된 기말잔액(GL)': eop_gl,
            '기말잔액(Master)': eop_master,
            '차이': difference,
            '상태': status
        })
    return overall_status, pd.DataFrame(results)


# NEW: 표준 DTO(ModuleResult) 반환 래퍼
def run_integrity_module(ledger_df: pd.DataFrame, master_df: pd.DataFrame):
    """
    기존 analyze_reconciliation 결과를 표준 ModuleResult로 감쌉니다.
    - summary: 상태/계정 수/Fail·Warning 건수/최대 차이
    - tables: {"reconciliation": 결과 DF}
    - evidences: (MVP 단계) 비움
    """
    status, df = analyze_reconciliation(ledger_df, master_df)
    summary: Dict[str, Any] = {
        "status": str(status),
        "n_accounts": int(df["계정코드"].nunique()) if (df is not None and not df.empty and "계정코드" in df.columns) else 0,
        "n_fail": int((df["상태"] == "Fail").sum()) if (df is not None and "상태" in df.columns) else 0,
        "n_warn": int((df["상태"] == "Warning").sum()) if (df is not None and "상태" in df.columns) else 0,
        "max_abs_diff": float(df["차이"].abs().max()) if (df is not None and not df.empty and "차이" in df.columns) else 0.0,
    }
    warnings: List[str] = [] if status == "Pass" else [f"정합성 상태: {status}"]
    if ModuleResult is None:
        # contracts 미가용 환경 안전가드
        from typing import NamedTuple
        class _MR(NamedTuple):
            name: str; summary: Dict[str, Any]; tables: Dict[str, pd.DataFrame]; figures: Dict; evidences: List; warnings: List[str]
        return _MR("integrity", summary, {"reconciliation": df}, {}, [], warnings)
    return ModuleResult(
        name="integrity",
        summary=summary,
        tables={"reconciliation": df},
        figures={},
        evidences=[],
        warnings=warnings,
    )
