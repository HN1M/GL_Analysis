from __future__ import annotations
import pandas as pd
from typing import Iterable, Optional

def find_column_by_keyword(columns: Iterable[str], keyword: str) -> Optional[str]:
    """열 이름에서 keyword(부분일치, 대소문자 무시)를 우선 탐색."""
    keyword = str(keyword or "").lower()
    cols = [str(c) for c in columns]
    # 1) 완전 일치 우선
    for c in cols:
        if c.lower() == keyword:
            return c
    # 2) 부분 일치
    for c in cols:
        if keyword in c.lower():
            return c
    return None

def add_provenance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """업로드 출처 정보가 없어도 row_id를 강제로 부여."""
    out = df.copy()
    if "row_id" not in out.columns:
        out["row_id"] = out.reset_index().index.astype(str)
    return out

def add_period_tag(df: pd.DataFrame) -> pd.DataFrame:
    """연도 최대값 기준으로 CY/PY/Other 태그를 부여."""
    out = df.copy()
    if "연도" not in out.columns:
        out["period_tag"] = "Other"
        return out
    y_max = out["연도"].max()
    out["period_tag"] = out["연도"].apply(lambda y: "CY" if y == y_max else ("PY" if y == y_max - 1 else "Other"))
    return out


# --- NEW: 차트 전용 대변계정 판정(그래프에서만 부호 반전) ---
def is_credit_account(account_type: str | None, dc: str | None = None) -> bool:
    """
    계정 성격이 대변(Credit)인지 판정합니다.
    - dc가 주어지면 우선 사용(예: '차변'/'대변' 또는 'D'/'C')
    - 아니면 account_type으로 간접 판정: 부채/자본/수익 → Credit
    """
    try:
        if dc is not None:
            s = str(dc).strip().upper()
            return s.startswith("C") or ("대변" in s)
    except Exception:
        pass
    try:
        t = str(account_type or "").strip()
        return t in {"부채", "자본", "수익"}
    except Exception:
        return False
