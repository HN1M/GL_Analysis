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


# --- NEW: 모델 선택 이유 설명 텍스트 ---
def model_reason_text(name: str, d: dict) -> str:
    """
    간단한 규칙 기반으로 MoR 선택 사유를 자연어로 요약합니다.
    기대 키: cv_mape_rank, seasonality_strength(0~1), stationary(bool), recent_trend(bool), n_points(int)
    """
    try:
        why = []
        nm = str(name or "").lower()
        if d.get("cv_mape_rank") == 1:
            why.append("교차검증에서 가장 낮은 MAPE를 기록했습니다.")
        if float(d.get("seasonality_strength", 0.0)) > 0.4 and nm.startswith("prophet"):
            why.append("연/분기 수준의 계절성이 강하게 관측되었습니다.")
        if bool(d.get("stationary")) and nm.startswith("arima"):
            why.append("차분 후 정상성이 확보되어 ARIMA 적합이 유리했습니다.")
        if bool(d.get("recent_trend")) and (nm.startswith("ema") or nm.startswith("holt") or nm.startswith("exp")):
            why.append("최근 추세 변화가 커서 최근값 가중 모델이 더 잘 맞았습니다.")
        if int(d.get("n_points", 0)) < 18 and (nm.startswith("ma") or nm.startswith("ema")):
            why.append("관측치가 짧아 단순 이동평균 계열이 과적합 위험이 낮았습니다.")
        if not why:
            why.append("오차지표(MAE/MAPE)와 정보량(AIC/BIC)을 종합해 최적 모델로 선택되었습니다.")
        return " / ".join(why)
    except Exception:
        return "오차지표(MAE/MAPE)와 정보량(AIC/BIC)을 종합해 최적 모델로 선택되었습니다."


# --- NEW: 시계열용 날짜/금액 컬럼 자동 탐색 ---
def guess_time_and_amount_cols(df: pd.DataFrame):
    """시계열용 날짜/금액 컬럼을 유연하게 탐색한다."""
    date_candidates = ["회계일자", "전표일자", "거래일자", "일자", "date", "Date"]
    amt_candidates  = [
        "거래금액", "발생액", "금액", "금액(원)", "거래금액_절대값",
        "발생액_절대값", "순액", "순액(원)"
    ]
    cols = list(df.columns) if df is not None else []
    date_col = next((c for c in date_candidates if c in cols), None)
    amt_col  = next((c for c in amt_candidates  if c in cols), None)
    return date_col, amt_col


# --- NEW: 공용 add_or_replace (df.insert 대체) ---
def add_or_replace(df: pd.DataFrame, loc: int, col: str, values):
    """df.insert 대체: 이미 있으면 교체, 없으면 지정 위치에 추가."""
    import pandas as pd
    if col in df.columns:
        df[col] = values
        return df
    df.insert(loc, col, values)
    return df