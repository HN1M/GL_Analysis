# analysis/ts_v2.py
from __future__ import annotations
import numpy as np
import pandas as pd
from .aggregation import aggregate_monthly, month_end_00


def run_timeseries_minimal(df, *, account_name, date_col, amount_col, is_bs, opening=0.0, pm_value=0.0):
    x = df[[date_col, amount_col]].copy()
    x.rename(columns={date_col: "date", amount_col: "amount"}, inplace=True)
    x["date"] = month_end_00(pd.to_datetime(x["date"], errors="coerce"))
    x["amount"] = pd.to_numeric(x["amount"], errors="coerce").fillna(0.0)
    x = x.dropna(subset=["date"])

    g = (x.groupby("date", as_index=False)["amount"].sum()
           .sort_values("date").reset_index(drop=True))

    y = g["amount"].astype(float)
    span = max(3, min(6, len(y)))              # 짧은 시계열 가드
    yhat = y.ewm(span=span, adjust=False).mean()
    err  = y - yhat
    sig  = err.rolling(window=min(6, max(2, len(err))), min_periods=2).std(ddof=0)
    z    = err / sig.replace(0, np.nan)

    kit      = (y.abs() > float(pm_value)).astype(float)
    pm_ratio = np.minimum(1.0, err.abs() / float(pm_value) if pm_value else 0.0)
    risk     = np.minimum(1.0, 0.5 * (z.abs() / 3.0) + 0.3 * pm_ratio + 0.2 * kit)

    flow = g.assign(
        account=str(account_name),
        measure="flow",
        actual=y.values,
        predicted=yhat.values,
        error=err.values,
        z=z.values,
        risk=risk.values,
        model="EMA",
    )

    if is_bs:
        bal = flow.copy()
        bal["measure"]   = "balance"
        bal["actual"]    = float(opening) + bal["actual"].cumsum()
        bal["predicted"] = float(opening) + bal["predicted"].cumsum()
        bal["error"]     = bal["actual"] - bal["predicted"]
        bal_sig          = bal["error"].rolling(window=min(6, max(2, len(bal))), min_periods=2).std(ddof=0)
        bal["z"]         = bal["error"] / bal_sig.replace(0, np.nan)
        out = pd.concat([flow, bal], ignore_index=True)
    else:
        out = flow

    # === 형 강제: Index 누출 방지 ===
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for c in ["actual","predicted","error","z","risk"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["model"]   = out["model"].astype(str)
    out["measure"] = out["measure"].astype(str)
    out["account"] = out["account"].astype(str)
    return out.reset_index(drop=True)
