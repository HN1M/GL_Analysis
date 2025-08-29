from __future__ import annotations
import pandas as pd
from typing import Iterable, Optional


def ensure_rowid(df: pd.DataFrame, id_col: str = "row_id") -> pd.DataFrame:
    """row_id 컬럼이 없으면 생성하지 않고 그대로 반환(계약 준수는 상위 단계에서)."""
    return df if id_col in df.columns else df


def attach_customdata(df: pd.DataFrame, cols: Iterable[str], id_col: str = "row_id"):
    """
    Plotly에 올릴 customdata 배열 생성.
    반환: (df, customdata(ndarray), header_labels(list))
    """
    import numpy as np
    use_cols = [c for c in cols if c in df.columns]
    if id_col not in use_cols and id_col in df.columns:
        use_cols = [id_col] + use_cols
    arr = df[use_cols].to_numpy()
    return df, np.asarray(arr), use_cols


def fmt_money(x) -> str:
    try:
        return f"{float(x):,.0f}"
    except Exception:
        return str(x)


