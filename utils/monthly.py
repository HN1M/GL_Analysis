import pandas as pd


def monthly_flow(df: pd.DataFrame, date_col: str, amt_col: str) -> pd.DataFrame:
    d = df[[date_col, amt_col]].copy()
    p = pd.to_datetime(d[date_col]).dt.to_period("M")
    g = d.assign(_p=p).groupby("_p", as_index=False)[amt_col].sum()
    g = g.rename(columns={"_p": "month"}).rename(columns={amt_col: "flow"})
    g["label"] = g["month"].astype(str)
    try:
        g["order"] = g["month"].astype(int)
    except Exception:
        # Fallback: use numeric YYYYMM from label
        g["order"] = g["label"].str.replace("-", "", regex=False).astype(int)
    return g[["month", "label", "order", "flow"]]


def monthly_balance_from_col(df: pd.DataFrame, date_col: str, bal_col: str) -> pd.DataFrame:
    d = df[[date_col, bal_col]].copy()
    p = pd.to_datetime(d[date_col]).dt.to_period("M")
    g = d.assign(_p=p).sort_values(date_col).groupby("_p", as_index=False).last()
    g = g.rename(columns={"_p": "month", bal_col: "balance"})
    g["label"] = g["month"].astype(str)
    try:
        g["order"] = g["month"].astype(int)
    except Exception:
        g["order"] = g["label"].str.replace("-", "", regex=False).astype(int)
    return g[["month", "label", "order", "balance"]]


def monthly_balance_from_flow(flow_df: pd.DataFrame, opening: float = 0.0) -> pd.DataFrame:
    g = flow_df[["month", "label", "order", "flow"]].copy()
    g["balance"] = float(opening) + g["flow"].astype(float).cumsum()
    return g[["month", "label", "order", "balance"]]


