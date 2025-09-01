# timeseries.py
# v3 — Compact TS module with PY+CY window, MoR(EMA/MA/ARIMA/Prophet), dual-basis(flow/balance)
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Callable, Mapping
import math
import numpy as np
STANDARD_COLS = ["date","account","measure","model","actual","predicted","error","z","risk"]

def _ensure_ts_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    시계열 결과 DF를 표준 스키마로 정규화한다.
    누락된 컬럼은 NaN으로 추가하고, date는 datetime으로 강제.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=STANDARD_COLS)
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for c in STANDARD_COLS:
        if c not in out.columns:
            out[c] = np.nan
    # 불필요 컬럼은 보존하되, 기준 컬럼 우선 반환
    extra = [c for c in out.columns if c not in STANDARD_COLS]
    return out[STANDARD_COLS + extra]
import pandas as pd
import plotly.graph_objects as go
from pandas.tseries.offsets import MonthEnd
def to_month_end_index(idx) -> pd.DatetimeIndex:
    """Convert a datetime-like or period-like index to month-end DatetimeIndex.
    NEW: Always normalize to month-end 00:00:00 (floor to day) for stable axes.
    """
    pidx = pd.PeriodIndex(idx, freq="M")
    _end = pidx.to_timestamp(how="end").floor("D")
    return pd.DatetimeIndex(_end)

# Attempt to import visualization and helper utilities (assuming they exist in the project structure)
try:
    from utils.viz import add_period_guides, add_materiality_threshold
except ImportError:
    # Fallbacks if utils.viz is not found
    def add_period_guides(fig, dates):
        return fig
    def add_materiality_threshold(fig, threshold):
        return fig
try:
    from utils.helpers import model_reason_text
except ImportError:
    # Fallback if utils.helpers is not found
    def model_reason_text(model_name, diagnostics):
        return f"Model {model_name} was selected based on cross-validation metrics."

# Optional contracts import for DTO outputs
try:
    from analysis.contracts import LedgerFrame, ModuleResult, EvidenceDetail
except Exception:  # pragma: no cover - keep loose coupling for tests
    LedgerFrame = None  # type: ignore
    ModuleResult = None  # type: ignore
    EvidenceDetail = None  # type: ignore

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
DATE_CANDIDATES = ['회계일자','전표일자','거래일자','일자','date','Date']
AMT_CANDIDATES  = ['거래금액','발생액','금액','금액(원)','거래금액_절대값','발생액_절대값','순액','순액(원)']

def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None
def _to_month_period_index(dates: pd.Series) -> pd.PeriodIndex:
    return pd.to_datetime(dates).dt.to_period("M")

def _to_month_end(ts: pd.Series) -> pd.Series:
    """Normalize datetimes to month-end and floor to seconds for clean display."""
    ts = pd.to_datetime(ts, errors="coerce")
    return (ts + MonthEnd(0)).dt.floor("S")

def _monthly_flow_and_balance(
    df: pd.DataFrame,
    date_col: str,
    amount_col: str,
    opening: float = 0.0,
) -> Tuple[pd.Series, pd.Series]:
    """월별 발생액 합계(flow)와 기초+누적발생액(balance) 반환.
    반환 Series는 month-end DatetimeIndex를 가지며 index.name="date"로 설정된다.
    """
    if df is None or df.empty:
        idx = pd.DatetimeIndex([], name="date")
        return pd.Series(dtype=float, index=idx), pd.Series(dtype=float, index=idx)
    p = pd.to_datetime(df[date_col], errors="coerce").dt.to_period("M")
    amt = pd.to_numeric(df[amount_col], errors="coerce").fillna(0.0)
    flow = amt.groupby(p).sum().astype(float)
    balance = (flow.cumsum() + float(opening)).astype(float)
    month_end = flow.index.to_timestamp(how="end").floor("D")
    flow_s = pd.Series(flow.values, index=month_end)
    bal_s = pd.Series(balance.values, index=month_end)
    flow_s.index.name = bal_s.index.name = "date"
    return flow_s, bal_s

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
    # 월말 기준으로 고정
    _dates = pd.to_datetime(df[date_col], errors="coerce")
    p = _to_month_period_index(_dates)
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
    return {
        "date": y.index[-1].to_timestamp(how='end'),
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
    return _ensure_ts_schema(out)

def run_timeseries_module(
    df: pd.DataFrame,
    *,
    account_col: str = "account",
    date_col: str = "date",
    amount_col: str = "amount",
    targets: Optional[List[str]] = None,
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
        return _ensure_ts_schema(pd.DataFrame(columns=["account","date","measure","actual","predicted","error","z","risk","model"]))
    work = df[[account_col, date_col, amount_col]].copy()
    work.columns = ["account","date","amount"]
    work = work.sort_values(["account","date"])
    if targets:
        try:
            tgt = set(map(str, targets))
            work["account"] = work["account"].astype(str)
            work = work[work["account"].isin(tgt)]
        except Exception:
            pass
    # 최소 포인트 가드: 계정별 date 유니크가 부족하면 빈 결과 반환
    MIN_POINTS = 6
    if work["date"].nunique() < MIN_POINTS:
        return _ensure_ts_schema(pd.DataFrame(columns=["account","date","measure","actual","predicted","error","z","risk","model"]))
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
    return _ensure_ts_schema(result)

def run_timeseries_module_with_flag(
    df: pd.DataFrame,
    account_col: str,
    date_col: str,
    amount_col: str,
    account_name: str,
    is_bs: bool,
    backend: str = "ema",
    *,
    opening_map: Mapping[str, float] | None = None,
    return_mode: str = "insample",
) -> pd.DataFrame:
    """
    기존 단일 출력에서 확장: BS 계정은 balance/flow dual 로직 적용.
    balance = opening(전기말잔액 등) + flow.cumsum()
    """
    if df is None or df.empty:
        return _ensure_ts_schema(pd.DataFrame(columns=["date","account","measure","actual","predicted","error","z","risk","model"]))

    # 월별 flow/balance 생성 (월말 00:00:00 보장)
    acct = str(df[account_col].iloc[0]) if (account_col in df.columns and not df.empty) else account_name
    opening = 0.0
    if opening_map:
        opening = float(opening_map.get(acct, opening_map.get(account_name, 0.0)))
    flow_s, bal_s = _monthly_flow_and_balance(df, date_col, amount_col, opening=opening)

    def _run(track: str, s: pd.Series) -> pd.DataFrame:
        base = pd.DataFrame({"date": s.index, track: s.values}).sort_values("date")
        if return_mode == "lastrow":
            ins = base.tail(1).rename(columns={track: "actual"})
            ins["predicted"] = np.nan; ins["error"] = np.nan; ins["z"] = np.nan; ins["risk"] = np.nan
            ins["model"] = backend.upper()
        else:
            ins = insample_predict_df(base, value_col=track, measure=track, pm_value=_PM_DEFAULT)
        ins["account"] = account_name
        ins["measure"] = track
        ins["date"] = to_month_end_index(ins["date"])  # 안전 보정
        return ins[["date","account","measure","actual","predicted","error","z","risk","model"]]

    res = [_run("flow", flow_s)]
    if is_bs:
        res.append(_run("balance", bal_s))
    final_df = pd.concat(res, ignore_index=True).sort_values(["measure","date"]) if res else pd.DataFrame(columns=["date","account","measure","actual","predicted","error","z","risk","model"])
    return _ensure_ts_schema(final_df)

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
    # 월말 고정 + 초 단위로 내림
    idx = to_month_end_index(y.index)
    out = pd.DataFrame({
        "date": idx,
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


# ========================================================================
# ============= NEW: Functions moved from app.py for Refactoring =========
# ========================================================================

# ----------------------- TS Diagnostics (moved from app.py) -------------

def _adf_stationary(y_vals: np.ndarray) -> Tuple[bool, float]:
    """ADF test for stationarity."""
    try:
        from statsmodels.tsa.stattools import adfuller
        y_clean = np.asarray(y_vals, dtype=float)
        # ADF requires finite values and sufficient length
        if not np.all(np.isfinite(y_clean)) or len(y_clean) < 3:  # ADF needs at least 3 points for default settings
            return (False, np.nan)

        p = float(adfuller(y_clean)[1])
        return (p < 0.05, p)  # True=정상성 확보
    except Exception:
        # 간단 폴백 (원본 로직 유지)
        y = np.asarray(y_vals, dtype=float)
        if len(y) < 6:
            return (False, np.nan)

        std_orig = np.nanstd(y)
        std_diff = np.nanstd(np.diff(y))

        if not math.isfinite(std_orig) or not math.isfinite(std_diff):
            return (False, np.nan)

        # Avoid division by zero if original std is 0
        if std_orig == 0:
            return (std_diff == 0, np.nan)

        return (std_diff < 0.9 * std_orig, np.nan)

def _has_seasonality_safe(y_vals: np.ndarray) -> bool:
    """Safe wrapper for seasonality check."""
    try:
        return bool(_has_seasonality(pd.Series(y_vals)))
    except Exception:
        return False

# ----------------------- TS Visualization (moved from app.py) -----------

def create_timeseries_figure(
    df_hist: pd.DataFrame,
    measure: str,
    title: str,
    pm_value: float,
    show_dividers: bool = False
) -> Tuple[Optional[go.Figure], Optional[Dict[str, Any]]]:
    """
    Creates a timeseries figure (actual vs predicted) and returns the figure and stats.
    (Refactored from app.py's _make_ts_fig_with_stats, removing Streamlit dependencies)
    """
    vcol = 'flow' if measure == 'flow' else 'balance'
    work = df_hist.copy()
    if "date" not in work.columns:
        return None, {"error": "Column 'date' not found in data."}
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date"]).sort_values("date")

    # 1) 모델링된 입력(actual/predicted)이면 그대로 사용
    if {"actual","predicted"}.issubset(work.columns):
        cols = ["date","actual","predicted"] + (["model"] if "model" in work.columns else [])
        ins = work[cols].copy()
    else:
        # 2) 값 컬럼을 잡아 in-sample 예측선 생성
        if vcol in work.columns:
            base = work[["date", vcol]].rename(columns={vcol: "val"})
        elif "actual" in work.columns:
            base = work[["date","actual"]].rename(columns={"actual":"val"})
        elif "value" in work.columns:
            base = work[["date","value"]].rename(columns={"value":"val"})
        else:
            return None, {"error": f"Neither '{vcol}' nor 'actual'/'value' column found."}
        base = base.dropna(subset=["val"])  # 안전 가드
        ins = insample_predict_df(
            base.rename(columns={"val": vcol}),
            value_col=vcol,
            measure=measure,
            pm_value=float(pm_value)
        )

    if ins.empty or len(ins) < 2:
        reason = "points<2" if (not ins.empty) else "empty"
        return None, {"error": f"Insufficient data for plot ({reason}).",
                      "diagnostics": {"n_months": int(len(ins))}}

    # --- 1. Figure Creation ---
    fig = go.Figure()
    # Using styles consistent with the original app.py
    fig.add_trace(go.Scatter(x=ins['date'], y=ins['actual'], mode='lines', name='actual'))
    fig.add_trace(go.Scatter(x=ins['date'], y=ins['predicted'], mode='lines', name='predicted', line=dict(dash='dot')))

    fig.update_layout(
        title=title,
        xaxis_title='month',
        yaxis_title='원'
    )
    fig.update_yaxes(tickformat=",.0f", separatethousands=True, ticksuffix="")

    # Add PM threshold
    try:
        fig = add_materiality_threshold(fig, float(pm_value))
    except Exception:
        pass  # Proceed without PM line if utility fails

    # Add time dividers (Year/Quarter) – 기본 OFF
    if show_dividers:
        try:
            fig = add_period_guides(fig, ins['date'])
        except Exception:
            pass  # Proceed without dividers if utility fails

    # --- 2. Statistics Calculation (Logic preserved from app.py) ---
    stats_output: Dict[str, Any] = {}
    y_vals = np.asarray(ins['actual'].values, dtype=float)
    n_months = int(np.isfinite(y_vals).sum())

    # Diagnostics (Seasonality, Stationarity)
    seas = _has_seasonality_safe(y_vals)
    stat_ok, pval = _adf_stationary(y_vals)

    stats_output["diagnostics"] = {
        "seasonality": seas,
        "stationary": stat_ok,
        "p_value": pval,
        "n_months": n_months,
        "is_short": n_months < 12,
    }

    # Model Metrics (MAE, MAPE, AIC/BIC)
    # Sticking to the original definitions used in app.py
    mae = float(np.mean(np.abs(ins['actual'] - ins['predicted'])))
    # Handle division by zero safely within the mean calculation
    actuals = ins['actual']
    predicted = ins['predicted']
    mape = float(np.mean(np.where(actuals != 0, np.abs((actuals - predicted) / actuals) * 100, 0)))

    aic = bic = np.nan
    best_model_name = str(ins['model'].iloc[-1]) if ('model' in ins.columns and not ins.empty) else "EMA"

    if best_model_name.upper() == "ARIMA":
        try:
            # Refit ARIMA to get AIC/BIC (as done in the original app.py)
            _y = ins['actual'].reset_index(drop=True)
            _ar = _fit_arima(_y)
            aic = float(getattr(_ar, "aic", np.nan))
            bic = float(getattr(_ar, "bic", np.nan))
        except Exception:
            pass

    # Trend analysis and Seasonality strength (logic moved exactly from app.py)
    recent_trend = False
    # Using y_vals (already defined as float array)
    if n_months >= 6:
        x = np.arange(len(y_vals))
        # Original logic was potentially unsafe with NaNs, but preserving it as requested
        try:
            slope = np.polyfit(x, y_vals, 1)[0]
            recent_trend = abs(slope) > 0.3 * (y_vals.std() + 1e-9)
        except Exception:
            pass  # Handle potential issues during polyfit

    try:
        # Calculate seasonality strength (logic moved exactly from app.py)
        ac = np.abs(np.fft.rfft((y_vals - y_vals.mean())))
        core = ac[2:] if ac.size > 2 else ac
        seas_strength_raw = float(core.max() / (core.mean() + 1e-9)) if core.size else 0.0
        seas_strength = max(0.0, min((seas_strength_raw - 1.0) / 4.0, 1.0))
    except Exception:
        seas_strength = 0.0

    # Prepare diagnostics dictionary for model_reason_text
    diagnostics_for_reasoning = {
        "n_points": n_months,
        "seasonality_strength": seas_strength,
        "stationary": bool(stat_ok),
        "recent_trend": bool(recent_trend),
        "cv_mape_rank": 1,  # Assuming this is the best model (Rank 1)
        "mae": mae, "mape": mape, "aic": aic, "bic": bic,
    }

    # Get model reasoning text
    reasoning_text = model_reason_text(best_model_name, diagnostics_for_reasoning)

    # Metadata
    train_months = int(ins['train_months'].iloc[-1]) if 'train_months' in ins.columns else int(len(ins))
    if 'data_span' in ins.columns and not ins['data_span'].empty:
        span_txt = str(ins['data_span'].iloc[-1])
    else:
        try:
            dmin = ins['date'].min(); dmax = ins['date'].max()
            span_txt = f"{dmin:%Y-%m} ~ {dmax:%Y-%m}"
        except Exception:
            span_txt = "-"
    sigma_window = int(ins['sigma_win'].iloc[-1]) if 'sigma_win' in ins.columns else 6

    stats_output["metrics"] = {"mae": mae, "mape": mape, "aic": aic, "bic": bic}
    stats_output["metadata"] = {
        "model": best_model_name, "train_months": train_months,
        "data_span": span_txt, "sigma_window": sigma_window,
        "reasoning": reasoning_text,
    }

    # Detailed stats for the "expander" view in UI
    stats_output["details"] = {
        "모델": best_model_name, "학습기간(월)": train_months, "데이터 구간": span_txt,
        "σ 윈도우(최근)": sigma_window, "CV(K)": 3,
    }

    return fig, stats_output


# --------------------------- Lightweight series API ---------------------------
def build_series(ledger, accounts: List[str]) -> Tuple[pd.DataFrame, str]:
    """
    원장(ledger.df)에서 날짜/금액 컬럼을 자동 탐색하여 월별 시계열을 생성.
    - 정상(ledger 모드): 원장에서 월별 합계를 산출하여 tidy 반환
    - 폴백(master 모드): meta.master_df의 잔액 3포인트(전전기말/전기말/당기말)를 반환

    반환: (df, mode)
      df columns → ['계정코드','계정명','month','value']
      mode → 'ledger' | 'master'
    """
    df = getattr(ledger, 'df', None)
    if df is None or df.empty:
        return pd.DataFrame(columns=['계정코드','계정명','month','value']), 'master'

    # 계정 필터 준비(문자열 통일)
    accounts = list(map(str, accounts or []))
    work = df.copy()
    if '계정코드' in work.columns:
        try:
            work['계정코드'] = work['계정코드'].astype(str)
        except Exception:
            pass

    date_col = _pick_col(work, DATE_CANDIDATES)
    amt_col  = _pick_col(work, AMT_CANDIDATES)

    if date_col and amt_col and ('계정코드' in work.columns) and ('계정명' in work.columns):
        # 날짜/금액 안전 정규화
        work[date_col] = pd.to_datetime(work[date_col], errors='coerce')
        work[amt_col] = pd.to_numeric(work[amt_col], errors='coerce')
        work = work.dropna(subset=[date_col, amt_col, '계정코드', '계정명'])

        # 대상 계정 필터 → 월말 고정
        tmp = work[work['계정코드'].astype(str).isin(accounts)].copy()
        if tmp.empty:
            return pd.DataFrame(columns=['계정코드','계정명','month','value']), 'ledger'
        tmp['_month'] = tmp[date_col].dt.to_period('M').dt.to_timestamp(how='end').floor('D')
        mon = (tmp.groupby(['계정코드','계정명','_month'], as_index=False)[amt_col].sum())
        mon = mon.rename(columns={'_month': 'month', amt_col: 'value'}).sort_values(['계정코드','month'])
        return mon, 'ledger'

    # --- 폴백: master 잔액 3포인트 ---
    m = getattr(ledger, 'meta', {}).get('master_df') if hasattr(ledger, 'meta') else None
    need = {'전전기말잔액','전기말잔액','당기말잔액'}
    if m is None or not need.issubset(set(m.columns)):
        # 완전 폴백 실패 시 빈 프레임 반환(호출측에서 메시지 처리)
        return pd.DataFrame(columns=['계정코드','계정명','month','value']), 'master'
    try:
        cy = int(getattr(ledger, 'meta', {}).get('CY', pd.Timestamp.today().year))
    except Exception:
        cy = pd.Timestamp.today().year
    dates = [pd.Timestamp(cy-2, 12, 31), pd.Timestamp(cy-1, 12, 31), pd.Timestamp(cy, 12, 31)]
    m2 = m.copy()
    try:
        m2['계정코드'] = m2['계정코드'].astype(str)
    except Exception:
        pass
    mm = m2[m2['계정코드'].isin(accounts)][['계정코드','계정명','전전기말잔액','전기말잔액','당기말잔액']]
    rows: List[pd.DataFrame] = []
    for _, r in mm.iterrows():
        vals = [r.get('전전기말잔액', 0), r.get('전기말잔액', 0), r.get('당기말잔액', 0)]
        rows.append(pd.DataFrame({
            '계정코드': r['계정코드'],
            '계정명' : r['계정명'],
            'month' : dates,
            'value' : vals
        }))
    g = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=['계정코드','계정명','month','value'])
    return g, 'master'
