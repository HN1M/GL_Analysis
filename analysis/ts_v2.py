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


# --- BEGIN: TS Utilities (stats, anomaly table, future shading) ---

import numpy as np
import pandas as pd

def compute_series_stats(dfm: pd.DataFrame) -> dict:
    """
    dfm: 단일 계정×단일 measure(=flow/balance) 구간의 in-sample 결과 프레임
         (필수 컬럼: ['actual','predicted'])
    반환: {'MAE':..., 'MAPE':..., 'RMSE':..., 'N':...}
    """
    if dfm is None or dfm.empty:
        return {'MAE': np.nan, 'MAPE': np.nan, 'RMSE': np.nan, 'N': 0}
    s = dfm[['actual', 'predicted']].apply(pd.to_numeric, errors='coerce').dropna()
    n = int(len(s))
    if n == 0:
        return {'MAE': np.nan, 'MAPE': np.nan, 'RMSE': np.nan, 'N': 0}
    err = (s['actual'] - s['predicted']).to_numpy()
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    # 0 나눗셈 방지: |actual|<eps는 무시하여 MAPE를 NaN으로 처리 → 전체는 nanmean
    denom = s['actual'].to_numpy()
    denom = np.where(np.abs(denom) < 1e-9, np.nan, np.abs(denom))
    mape = float(np.nanmean(np.abs(err) / denom) * 100.0)
    return {'MAE': mae, 'MAPE': mape, 'RMSE': rmse, 'N': n}

def _ensure_z_with_rolling(dfm: pd.DataFrame, k: int = 6) -> pd.DataFrame:
    """
    z가 없거나 전부 결측이면 최근 k개월 롤링-표준편차(+expanding 보정)로 z를 계산.
    z = error / std_k  (초기 구간은 expanding std로 보강)
    """
    out = dfm.copy()
    if 'z' in out.columns and out['z'].notna().any():
        return out
    out['actual'] = pd.to_numeric(out.get('actual'), errors='coerce')
    out['predicted'] = pd.to_numeric(out.get('predicted'), errors='coerce')
    if 'error' not in out.columns:
        out['error'] = out['actual'] - out['predicted']
    out = out.sort_values('date')
    k = int(max(3, k))
    std_k = out['error'].rolling(window=k, min_periods=3).std()
    std_exp = out['error'].expanding(min_periods=3).std()
    std = std_k.fillna(std_exp).replace(0, np.nan)
    out['z'] = out['error'] / std
    return out

def build_anomaly_table(df_all: pd.DataFrame, *, topn: int = 10, k: int = 6, pm_value: float | None = None) -> pd.DataFrame:
    """
    단일 계정의 전체 measure(flow/balance)를 입력받아 Top-|z| 월 상위 N행을 반환.
    반환 컬럼: ['일자','실측','예측','잔차','z','PM대비','위험도','모델','기준','|z|']
    """
    frames = []
    for ms in ('flow', 'balance'):
        d = df_all[df_all['measure'].eq(ms)]
        if d.empty:
            continue
        d = _ensure_z_with_rolling(d, k=k)
        d['PM대비'] = (
            np.minimum(1.0, np.abs(d['error']) / float(pm_value))
            if pm_value and pm_value > 0 else np.nan
        )
        d['기준'] = np.where(d['measure'].eq('flow'), 'Flow', 'Balance')
        keep = ['date','actual','predicted','error','z','PM대비','risk','model','기준']
        for c in keep:
            if c not in d.columns:
                d[c] = np.nan
        frames.append(d[keep].copy())

    if not frames:
        cols = ['일자','실측','예측','잔차','z','PM대비','위험도','모델','기준','|z|']
        return pd.DataFrame(columns=cols)

    out = pd.concat(frames, ignore_index=True)
    out['|z|'] = np.abs(out['z'])
    out = out.sort_values(['|z|','date'], ascending=[False, False]).head(int(topn)).copy()
    out = out.rename(columns={
        'date':'일자','actual':'실측','predicted':'예측','error':'잔차','risk':'위험도','model':'모델'
    })
    return out

def add_future_shading(fig, last_date, horizon_months: int = 0):
    """
    미래 예측 시각 강조용 음영. horizon_months<=0이면 아무 것도 하지 않음.
    """
    if fig is None or not horizon_months or horizon_months <= 0:
        return fig
    last = pd.to_datetime(last_date)
    x0 = last + pd.offsets.MonthBegin(1)      # 다음 달 시작
    x1 = last + pd.offsets.MonthEnd(horizon_months)
    try:
        fig.add_vrect(
            x0=x0, x1=x1,
            fillcolor="LightGrey", opacity=0.15, line_width=0,
            layer="below"  # ← 데이터/경계선 아래로 깔기
        )
    except Exception:
        pass
    return fig

# --- END: TS Utilities ---