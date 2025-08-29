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

