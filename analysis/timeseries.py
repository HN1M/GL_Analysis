# timeseries.py
# v3 — Compact TS module with PY+CY window, MoR(EMA/MA/ARIMA/Prophet), dual-basis(flow/balance)
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Callable
import math
import numpy as np
import pandas as pd

# -------- Optional config / anomaly imports with safe fallbacks --------
try:
    from config import PM_DEFAULT as _PM_DEFAULT
except Exception:
    # 실운영 기본값: 5억 (사용자 입력 미제공 시 최후의 안전값)
    _PM_DEFAULT = 500_000_000
try:
    from config import FORECAST_MIN_POINTS as _FORECAST_MIN_POINTS
except Exception:
    _FORECAST_MIN_POINTS = 8
try:
    from config import ARIMA_DEFAULT_ORDER as _ARIMA_DEFAULT_ORDER
except Exception:
    _ARIMA_DEFAULT_ORDER = (1, 1, 1)

def _risk_from_fallback(z_abs: float, amount: float, pm: float) -> float:
    # simple logistic mapping as a fallback
    return float(1.0 / (1.0 + math.exp(-abs(z_abs))))

try:
    from analysis.anomaly import _risk_from as _RISK_EXTERNAL  # type: ignore
    def _risk_score(z_abs: float, amount: float, pm: float) -> float:
        try:
            r = _RISK_EXTERNAL(z_abs, amount=amount, pm=float(pm))
            return float(r[-1] if isinstance(r, (list, tuple)) else r)
        except Exception:
            return _risk_from_fallback(z_abs, amount, pm)
except Exception:
    def _risk_score(z_abs: float, amount: float, pm: float) -> float:
        return _risk_from_fallback(z_abs, amount, pm)

# ----------------------------- Utilities ------------------------------
def _to_month_period_index(dates: pd.Series) -> pd.PeriodIndex:
    return pd.to_datetime(dates).dt.to_period("M")

def _longest_contiguous_month_run(periods: pd.PeriodIndex) -> pd.PeriodIndex:
    if len(periods) <= 1: return periods
    p = pd.PeriodIndex(np.unique(np.asarray(periods)), freq="M")
    best_s = best_e = cur_s = 0
    for i in range(1, len(p)):
        if (p[i] - p[i-1]).n != 1:
            if i-1 - cur_s > best_e - best_s:
                best_s, best_e = cur_s, i-1
            cur_s = i
    if len(p)-1 - cur_s > best_e - best_s:
        best_s, best_e = cur_s, len(p)-1
    return p[best_s:best_e+1]

def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = np.maximum(1e-12, np.abs(yt) + np.abs(yp))
    return float(np.mean(200.0 * np.abs(yt - yp) / denom))

def _std_last(x: np.ndarray, w: int = 6) -> float:
    if len(x) < 2: return 0.0
    s = np.std(x[-min(w, len(x)):], ddof=1) if len(x) > 1 else 0.0
    return float(s if math.isfinite(s) else 0.0)

def z_and_risk(residuals: np.ndarray, pm: float = _PM_DEFAULT) -> Tuple[np.ndarray, np.ndarray]:
    """잔차 시퀀스에 대해 표준화 z와 위험도 배열을 반환.
    테스트 호환을 위해 간단한 정규화와 |z|→risk 매핑을 사용.
    """
    r = np.asarray(residuals, dtype=float)
    if r.size <= 1:
        z = np.zeros_like(r)
    else:
        sd = float(np.std(r, ddof=1))
        z = (r / sd) if sd > 0 else np.zeros_like(r)
    risk_vals = np.array([_risk_score(abs(float(zi)), amount=1.0, pm=float(pm)) for zi in z], dtype=float)
    return z, risk_vals

def _has_seasonality(y: pd.Series) -> bool:
    y = pd.Series(y, dtype=float)
    if len(y) < 12: return False
    ac = np.abs(np.fft.rfft((y - y.mean()).values))
    core = ac[2:] if len(ac) > 2 else ac
    return bool(core.size and (core.max() / (core.mean() + 1e-9) > 5.0))

# --------------------------- Model backends ---------------------------
def _model_registry() -> Dict[str, bool]:
    ok_arima = ok_prophet = False
    try:
        import statsmodels.api as _  # noqa
        ok_arima = True
    except Exception:
        pass
    try:
        from prophet import Prophet as _  # noqa
        ok_prophet = True
    except Exception:
        pass
    return {"ema": True, "ma": True, "arima": ok_arima, "prophet": ok_prophet}

def model_registry() -> Dict[str, bool]:
    """공개 API: 사용 가능한 백엔드 레지스트리 반환."""
    return _model_registry()

# EMA
def _fit_ema(y: pd.Series, alpha: float = 0.3) -> Dict[str, Any]:
    return {"alpha": float(alpha), "y": y}

def _pred_ema(m: Dict[str, Any], steps: Optional[int] = None) -> np.ndarray:
    y: pd.Series = m["y"]
    alpha = float(m["alpha"])
    pred = y.ewm(alpha=alpha, adjust=False).mean().shift(1).fillna(y.iloc[:1].values[0]).values
    return pred if steps is None else np.repeat(pred[-1], int(steps))

# MA
def _fit_ma(y: pd.Series, window: int = 6) -> Dict[str, Any]:
    return {"window": int(window), "y": y}

def _pred_ma(m: Dict[str, Any], steps: Optional[int] = None) -> np.ndarray:
    y: pd.Series = m["y"]; w = int(m["window"])
    pred = y.rolling(w, min_periods=1).mean().shift(1).fillna(y.iloc[:1].values[0]).values
    return pred if steps is None else np.repeat(pred[-1], int(steps))

# ARIMA
def _fit_arima(y: pd.Series, order: Tuple[int,int,int] = _ARIMA_DEFAULT_ORDER):
    import statsmodels.api as sm
    return sm.tsa.ARIMA(y, order=tuple(order)).fit()

def _pred_arima(m, steps: Optional[int] = None) -> np.ndarray:
    if steps is None:
        fv = pd.Series(m.fittedvalues).shift(1).fillna(method="bfill")
        return fv.values
    return np.asarray(m.forecast(steps=int(steps)))

# Prophet
def _fit_prophet(y: pd.Series):
    from prophet import Prophet
    df = pd.DataFrame({"ds": y.index.to_timestamp(), "y": y.values})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(df)
    return {"m": m, "idx": y.index}

def _pred_prophet(m: Dict[str, Any], steps: Optional[int] = None) -> np.ndarray:
    model = m["m"]; idx: pd.PeriodIndex = m["idx"]
    if steps is None:
        fit = model.predict(pd.DataFrame({"ds": idx.to_timestamp()}))["yhat"].values
        return np.roll(fit, 1)  # 1-step ahead approx.
    last = idx[-1].to_timestamp()
    future = pd.date_range(last + pd.offsets.MonthBegin(1), periods=int(steps), freq="MS")
    return model.predict(pd.DataFrame({"ds": future}))["yhat"].values

# -------------------- Model selection (MoR) via rolling CV -------------
def _rolling_origin_cv(
    y: pd.Series,
    fit_fn, pred_fn,
    k: int = 3, min_train: int = 6
) -> float:
    y = y.dropna(); n = len(y)
    if n < max(min_train + k, 8):
        try:
            m = fit_fn(y); yhat = pred_fn(m)
        except Exception:
            return 999.0
        yhat = np.asarray(yhat)[:n] if yhat is not None else np.repeat(y.iloc[:1].values, n)
        return _smape(y.values, yhat)
    step = max((n - min_train) // (k + 1), 1)
    scores = []
    for i in range(min_train, n, step):
        tr = y.iloc[:i]; te = y.iloc[i:i+step]
        if te.empty: break
        try:
            m = fit_fn(tr); yh = pred_fn(m, steps=len(te))
            scores.append(_smape(te.values, np.asarray(yh)[:len(te)]))
        except Exception:
            scores.append(999.0)
    return float(np.mean(scores)) if scores else 999.0

def _choose_model(y: pd.Series, measure: str) -> Tuple[str, np.ndarray]:
    reg = _model_registry()
    cands: List[Tuple[str, np.ndarray, Any, Any]] = []
    # always EMA/MA
    m_ema = _fit_ema(y); yhat_ema = _pred_ema(m_ema); cands.append(("EMA", yhat_ema, _fit_ema, _pred_ema))
    m_ma  = _fit_ma(y);  yhat_ma  = _pred_ma(m_ma);   cands.append(("MA",  yhat_ma,  _fit_ma,  _pred_ma))
    # ARIMA
    if reg["arima"]:
        try:
            m = _fit_arima(y); yhat = _pred_arima(m); cands.append(("ARIMA", yhat, _fit_arima, _pred_arima))
        except Exception:
            pass
    # Prophet: only for flow, enough data & seasonal
    if measure == "flow" and reg["prophet"] and len(y) >= 12 and _has_seasonality(y):
        try:
            m = _fit_prophet(y); yhat = _pred_prophet(m); cands.append(("Prophet", yhat, _fit_prophet, _pred_prophet))
        except Exception:
            pass
    # pick by CV
    scores = [(nm, _rolling_origin_cv(y, fit, pred)) for (nm, _, fit, pred) in cands]
    best = min(scores, key=lambda x: x[1])[0] if scores else "EMA"
    # return best in-sample prediction
    if best == "EMA": return "EMA", yhat_ema
    if best == "MA":  return "MA",  yhat_ma
    if best == "ARIMA":
        try: return "ARIMA", _pred_arima(_fit_arima(y))
        except Exception: return "EMA", yhat_ema
    if best == "Prophet":
        try: return "Prophet", _pred_prophet(_fit_prophet(y))
        except Exception: return "EMA", yhat_ema
    return "EMA", yhat_ema

# --------------------------- Core predictors --------------------------
def _prepare_monthly(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=[date_col]).copy()
    p = _to_month_period_index(df[date_col])
    df2 = df.copy()
    df2["_p"] = p
    df2 = df2.dropna(subset=["_p"]).sort_values("_p")
    run = _longest_contiguous_month_run(df2["_p"])
    return df2[df2["_p"].isin(run)].reset_index(drop=True)

def _one_track_lastrow(
    monthly: pd.DataFrame,
    value_col: str,
    measure: str,
    pm_value: float
) -> Optional[Dict[str, Any]]:
    df = _prepare_monthly(monthly, "date")
    if df.empty or value_col not in df.columns: return None
    y = pd.Series(df[value_col].astype(float).values, index=pd.PeriodIndex(df["_p"], freq="M"))
    if len(y) < 2: return None
    model, yhat = _choose_model(y, measure=measure)
    resid = y.values - yhat
    error_last = float(resid[-1])
    sigma = _std_last(resid, w=6)
    z = float(error_last / sigma) if sigma > 0 else 0.0
    risk = _risk_score(abs(z), amount=float(y.iloc[-1]), pm=float(pm_value))
    # 날짜 앵커: flow=월초(how='start'), balance=월말(how='end')
    _how = 'start' if measure == 'flow' else 'end'
    return {
        "date": df["_p"].iloc[-1].to_timestamp(how=_how),
        "measure": measure,
        "actual": float(y.iloc[-1]),
        "predicted": float(yhat[-1]),
        "error": float(error_last),
        "z": float(z),
        "z_label": "resid_z",
        "risk": float(risk),
        "model": model,
    }

# ------------------------------- API ----------------------------------
def run_timeseries_for_account(
    monthly: pd.DataFrame,
    account: str,
    is_bs: bool,
    flow_col: str = "flow",
    balance_col: Optional[str] = None,
    allow_prophet: bool = True,   # kept for backward compatibility (no-op switch)
    pm_value: float = _PM_DEFAULT,
    **kwargs: Any,                # absorb legacy args safely
) -> pd.DataFrame:
    """
    단일 계정의 월별 데이터에서 마지막 포인트를 평가.
    - BS 계정: flow/balance 2행(해당 시 존재) 반환
    - PL 계정: flow 1행 반환
    반환 컬럼: ["date","account","measure","actual","predicted","error","z","risk","model"]
    """
    rows: List[Dict[str, Any]] = []
    # flow
    if flow_col in monthly.columns:
        r = _one_track_lastrow(monthly.rename(columns={flow_col: "val"}), "val", "flow", pm_value)
        if r: rows.append(r)
    # balance
    if is_bs:
        if balance_col and (balance_col in monthly.columns):
            r = _one_track_lastrow(monthly.rename(columns={balance_col: "val"}), "val", "balance", pm_value)
            if r: rows.append(r)
        else:
            if flow_col in monthly.columns:
                tmp = monthly[["date", flow_col]].copy()
                tmp["val"] = tmp[flow_col].astype(float).cumsum()
                r = _one_track_lastrow(tmp[["date","val"]], "val", "balance", pm_value)
                if r: rows.append(r)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["account"] = account
        out = out[["date","account","measure","actual","predicted","error","z","risk","model"]]
        out = out.sort_values(["account","measure","date"]).reset_index(drop=True)
    else:
        out = pd.DataFrame(columns=["date","account","measure","actual","predicted","error","z","risk","model"])
    return out

def run_timeseries_module(
    df: pd.DataFrame,
    *,
    account_col: str = "account",
    date_col: str = "date",
    amount_col: str = "amount",
    pm_value: float = _PM_DEFAULT,
    make_balance: bool = False,  # 기본값 False로 변경: 필요 시 balance 구성
    output: str = "all",        # "all" | "flow" | "balance"
    evidence_adapter: Optional[Callable[[Dict[str, Any]], Any]] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    집계형: 계정별 월합계(amount)만 주어진 경우.
    기본: flow만 계산. make_balance=True일 때 balance(누적합)도 함께 계산.
    output으로 최종 반환 필터링 가능("flow"/"balance").
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["account","date","measure","actual","predicted","error","z","risk","model"])
    work = df[[account_col, date_col, amount_col]].copy()
    work.columns = ["account","date","amount"]
    work = work.sort_values(["account","date"])
    all_rows: List[pd.DataFrame] = []
    for acc, g in work.groupby("account", dropna=False):
        mon = g[["date","amount"]].rename(columns={"amount":"flow"}).copy()
        if make_balance:
            mon["balance"] = mon["flow"].astype(float).cumsum()
        out = run_timeseries_for_account(mon, str(acc), is_bs=make_balance, flow_col="flow",
                                         balance_col=("balance" if make_balance else None),
                                         pm_value=float(pm_value))
        if not out.empty:
            # CEAVOP 제안(간단 규칙): error>0 → E(존재), error<=0 → C(완전성)
            try:
                out["assertion"] = out["error"].map(lambda e: "E" if float(e) > 0 else "C")
            except Exception:
                out["assertion"] = "E"
        if output in ("flow", "balance") and not out.empty:
            out = out[out["measure"] == output]
        all_rows.append(out)
    result = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(columns=["account","date","measure","actual","predicted","error","z","risk","model"])
    if evidence_adapter is not None and not result.empty:
        rows = []
        for r in result.to_dict(orient="records"):
            d = dict(r)
            if "amount" not in d:
                d["amount"] = float(d.get("actual", 0.0))
            if "z_abs" not in d:
                try:
                    d["z_abs"] = abs(float(d.get("z", 0.0)))
                except Exception:
                    d["z_abs"] = 0.0
            if "assertion" not in d:
                try:
                    d["assertion"] = "E" if float(d.get("error", 0.0)) > 0 else "C"
                except Exception:
                    d["assertion"] = "E"
            rows.append(evidence_adapter(d))
        return rows  # type: ignore[return-value]
    return result

def run_timeseries_module_with_flag(
    df: pd.DataFrame,
    *,
    account_col: str = "account",
    date_col: str = "date",
    amount_col: str = "amount",
    is_bs_col: str = "is_bs",
    pm_value: float = _PM_DEFAULT,
) -> pd.DataFrame:
    """
    혼합 데이터셋에서 계정별 BS 여부에 따라 듀얼(Flow+Balance) 또는 단일(Flow)로 처리.
    - is_bs=True: flow + balance(누적합) 계산
    - is_bs=False: flow만 계산
    반환 컬럼: ["account","date","measure","actual","predicted","error","z","risk","model"]
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["account","date","measure","actual","predicted","error","z","risk","model"])
    work = df[[account_col, date_col, amount_col, is_bs_col]].copy()
    work.columns = ["account","date","amount","is_bs"]
    work = work.sort_values(["account","date"])
    outs: List[pd.DataFrame] = []
    for acc, g in work.groupby("account", dropna=False):
        is_bs = bool(g["is_bs"].iloc[-1])
        mon = g[["date","amount"]].rename(columns={"amount":"flow"}).copy()
        balance_col = None
        if is_bs:
            mon["balance"] = mon["flow"].astype(float).cumsum()
            balance_col = "balance"
        out = run_timeseries_for_account(
            mon, str(acc), is_bs=is_bs, flow_col="flow",
            balance_col=balance_col, pm_value=float(pm_value)
        )
        if not out.empty:
            try:
                out["assertion"] = out["error"].map(lambda e: "E" if float(e) > 0 else "C")
            except Exception:
                out["assertion"] = "E"
        outs.append(out)
    return pd.concat(outs, ignore_index=True) if outs else pd.DataFrame(columns=["account","date","measure","actual","predicted","error","z","risk","model"])

# ----------------------- Helper: in-sample prediction -------------------
def insample_predict_df(
    monthly: pd.DataFrame,
    value_col: str,
    measure: str,
    pm_value: float = _PM_DEFAULT,
) -> pd.DataFrame:
    """
    월별 데이터(monthly: ['date', value_col])에 대해 MoR이 고른 모델의 in-sample 예측선을 반환.
    반환 컬럼: date, actual, predicted, model, train_months, data_span, sigma_win, measure, value_col
    """
    df = _prepare_monthly(monthly, "date")
    if df.empty or value_col not in df.columns:
        return pd.DataFrame(columns=["date","actual","predicted","model","train_months","data_span","sigma_win","measure","value_col"])
    y = pd.Series(df[value_col].astype(float).values, index=pd.PeriodIndex(df["_p"], freq="M"))
    if len(y) < 2:
        return pd.DataFrame(columns=["date","actual","predicted","model","train_months","data_span","sigma_win","measure","value_col"])
    model, yhat = _choose_model(y, measure=measure)
    span = f"{y.index[0].strftime('%Y-%m')} ~ {y.index[-1].strftime('%Y-%m')}"
    _how = 'start' if measure == 'flow' else 'end'
    out = pd.DataFrame({
        "date": y.index.to_timestamp(how=_how),
        "actual": y.values,
        "predicted": yhat,
        "model": model,
    })
    out["train_months"] = len(y)
    out["data_span"] = span
    out["sigma_win"] = 6  # z 계산에 쓰는 최근 분산 윈도우
    out["measure"] = str(measure)
    out["value_col"] = str(value_col)
    return out


# ------------------- Validation Builder (tidy helper) -------------------
def build_trend_validation_data(
    monthly: pd.DataFrame,
    *,
    flow_col: str = "flow",
    balance_col: Optional[str] = None,
    is_bs: bool = False,
    pm_value: float = _PM_DEFAULT,
) -> pd.DataFrame:
    """
    트렌드 검증용 tidy 데이터 생성:
      columns → ['date','actual','predicted','measure','model']
    - PL: flow만
    - BS: flow + balance(있으면 사용, 없으면 flow 누적합으로 생성)
    """
    rows: List[pd.DataFrame] = []

    # flow
    if flow_col in monthly.columns:
        df_flow = insample_predict_df(
            monthly[["date", flow_col]].rename(columns={flow_col: "val"}),
            value_col="val",
            measure="flow",
            pm_value=pm_value,
        )
        if not df_flow.empty:
            rows.append(df_flow[["date","actual","predicted","measure","model"]])

    # balance (BS만) — balance는 '월말' 시점 기준
    if is_bs:
        if balance_col and (balance_col in monthly.columns):
            base = monthly[["date", balance_col]].rename(columns={balance_col: "val"})
        else:
            # 누적합으로 balance 가상 생성(월말 시점으로 표시됨)
            if flow_col not in monthly.columns:
                base = None
            else:
                tmp = monthly[["date", flow_col]].copy()
                tmp["val"] = tmp[flow_col].astype(float).cumsum()
                base = tmp[["date","val"]]
        if base is not None:
            df_bal = insample_predict_df(base, value_col="val", measure="balance", pm_value=pm_value)
            if not df_bal.empty:
                rows.append(df_bal[["date","actual","predicted","measure","model"]])

    if not rows:
        return pd.DataFrame(columns=["date","actual","predicted","measure","model"])
    return pd.concat(rows, ignore_index=True)

# ------------------- Validation: reconcile with trend -------------------
def reconcile_with_trend(
    ts_flow: pd.Series,
    ts_bal: pd.Series,
    trend_flow: pd.Series,
    trend_bal: pd.Series,
    tol: int = 1,
) -> pd.DataFrame:
    """
    timeseries 입력(flow/balance)과 trend 산출치(flow/balance)를 월별 대조.
    tol 절대차(원) 초과인 행만 반환.
    """
    import pandas as _pd
    idx = sorted(set(_pd.to_datetime(getattr(ts_flow, 'index', [])).tolist()) |
                 set(_pd.to_datetime(getattr(trend_flow, 'index', [])).tolist()))
    rows = []
    for d in idx:
        af = int(_pd.Series(ts_flow).get(d, 0))
        tf = int(_pd.Series(trend_flow).get(d, 0))
        ab = int(_pd.Series(ts_bal).get(d, 0))
        tb = int(_pd.Series(trend_bal).get(d, 0))
        rows.append({
            "month": d,
            "flow(ts)": af, "flow(trend)": tf, "Δflow": af - tf,
            "bal(ts)": ab,  "bal(trend)": tb,  "Δbal": ab - tb,
        })
    df_chk = pd.DataFrame(rows).set_index("month")
    return df_chk[(df_chk["Δflow"].abs() > tol) | (df_chk["Δbal"].abs() > tol)]


# ------------------- Optional: lightweight validation summary -----------
def validation_summary(
    monthly: pd.DataFrame,
    *,
    date_col: str = "date",
    value_col: str = "amount",
    pm_value: float = _PM_DEFAULT,
    last_k: int = 6,
) -> Dict[str, Any]:
    """막대 대조 대신 쓰는 경량 숫자 진단 카드."""
    df = monthly[[date_col, value_col]].rename(columns={date_col: "date", value_col: "val"}).copy()
    df = _prepare_monthly(df, "date")
    if df.empty or "val" not in df.columns:
        return {"n_points": 0}
    y = pd.Series(df["val"].astype(float).values, index=pd.PeriodIndex(df["_p"], freq="M"))
    if len(y) < 2:
        return {"n_points": int(len(y))}
    scores = [("EMA", _rolling_origin_cv(y, _fit_ema, _pred_ema)),
              ("MA",  _rolling_origin_cv(y, _fit_ma,  _pred_ma))]
    reg = _model_registry()
    if reg["arima"]:
        try: scores.append(("ARIMA", _rolling_origin_cv(y, _fit_arima, _pred_arima)))
        except Exception: pass
    if reg["prophet"] and len(y) >= 12 and _has_seasonality(y):
        try: scores.append(("Prophet", _rolling_origin_cv(y, _fit_prophet, _pred_prophet)))
        except Exception: pass
    best_cv_name, best_cv_smape = min(scores, key=lambda x: x[1])
    model_name, yhat = _choose_model(y, measure="flow")
    resid = y.values - yhat
    k = min(last_k, len(y))
    smape_k = _smape(y.values[-k:], yhat[-k:])
    last_err = float(resid[-1])
    sigma = _std_last(resid, w=6)
    z = float(last_err / sigma) if sigma > 0 else 0.0
    pm_ratio = (abs(float(y[-1])) / float(pm_value)) if pm_value else 0.0
    return {
        "n_points": int(len(y)),
        "mor": model_name,
        "cv_smape": float(best_cv_smape),
        "smape_last_k": float(smape_k),
        "last_month": str(y.index[-1].to_timestamp(how='start').date()),
        "last_actual": float(y[-1]),
        "last_pred": float(yhat[-1]),
        "last_error": last_err,
        "last_z": z,
        "pm_ratio": pm_ratio,
    }
