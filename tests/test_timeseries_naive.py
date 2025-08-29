# 시계열 모듈의 간단 백엔드(EMA 등) 동작성 테스트
import pandas as pd
import numpy as np
from analysis.timeseries import run_timeseries_module, model_registry, z_and_risk, run_timeseries_for_account


def test_timeseries_naive_backend():
    df = pd.DataFrame({
        "account": ["A"]*7 + ["B"]*7,
        "date":    list(range(1,8)) + list(range(1,8)),
        "amount":  [10,11,10,12,11,10, 20] + [8,8,8,8,8,8, 5],
    })
    res = run_timeseries_module(df, account_col="account", date_col="date",
                                amount_col="amount", backend="ema", window=3)
    assert set(res["account"]) == {"A","B"}
    assert set(res["assertion"]).issubset({"E","C"})


def test_z_and_risk_basic():
    # 한글: 0 중심 대칭 잔차에 대해 |z|가 크면 risk가 커진다
    resid = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    z, r = z_and_risk(resid)
    assert len(z) == len(resid)
    assert len(r) == len(resid)
    assert float(r[0]) > float(r[1])
    assert float(r[-1]) == float(r[0])


def test_model_registry_keys_present():
    reg = model_registry()
    for k in ["ma","ema","arima","prophet"]:
        assert k in reg
    assert reg["ma"] is True and reg["ema"] is True


def test_run_timeseries_for_account_dual():
    # 한글: 12개월 샘플로 flow 누적 balance 생성하여 두 트랙 모두 반환
    dates = pd.period_range("2024-01", periods=12, freq="M").to_timestamp()
    flow = np.linspace(100, 210, num=12)
    df = pd.DataFrame({"date": dates, "flow": flow})
    df["balance"] = df["flow"].cumsum()
    out = run_timeseries_for_account(df, account="매출채권", is_bs=True, flow_col="flow", balance_col="balance")
    assert set(out["measure"]).issuperset({"flow","balance"})
    assert set(out.columns) == {"date","account","measure","actual","predicted","error","z","risk","model"}


