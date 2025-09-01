import pandas as pd


def month_end_00(date_series: pd.Series) -> pd.Series:
    """
    월말을 '해당월 말일 00:00:00'로 정규화 (반올림/올림 없이 고정)
    예: 2025-06-30 00:00:00
    """
    ds = pd.to_datetime(date_series, errors="coerce")
    return ds.dt.to_period("M").dt.to_timestamp("M")  # 월말 00:00:00


def aggregate_monthly(df: pd.DataFrame, date_col: str, amount_col: str) -> pd.DataFrame:
    """
    월별 발생액 합계를 반환.
    - 날짜 컬럼은 반드시 month_end_00로 정규화
    """
    work = df[[date_col, amount_col]].copy()
    work[date_col] = month_end_00(work[date_col])
    out = (
        work.groupby(date_col, dropna=True, as_index=False)[amount_col]
        .sum()
        .sort_values(date_col)
    )
    out = out.rename(columns={date_col: "date", amount_col: "amount"})
    return out


