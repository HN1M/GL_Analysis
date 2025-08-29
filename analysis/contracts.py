from dataclasses import dataclass, field
import pandas as pd
from typing import Dict, List, Any, Optional, Literal
# --- New: Measure 타입 힌트("flow" 또는 "balance") ---
Measure = Literal["flow", "balance"]


@dataclass(frozen=True)
class LedgerFrame:
    df: pd.DataFrame
    meta: Dict[str, Any]  # 예: {"company": "...", "file_name": "...", "uploaded_at": ...}

# CEAVOP assertions
ASSERTIONS = ["C","E","A","V","O","P"]

@dataclass(frozen=True)
class EvidenceDetail:
    row_id: str
    reason: str                  # e.g., "|Z|=3.1 (CY group mean-based)"
    anomaly_score: float         # 0~1 normalized
    financial_impact: float      # KRW absolute amount
    risk_score: float            # integrated score
    is_key_item: bool            # PM exceed flag
    # --- NEW: measurement basis and sign rule ---
    measure: Measure = "flow"     # "flow"(월별 발생액, Δ잔액/순액) 또는 "balance"
    sign_rule: str = "assets/expenses↑=+, liabilities/equity↑=-"
    # --- NEW: 시계열 예측 메타 (옵셔널) ---
    model: Optional[str] = None           # 사용된 모델명 (예: EMA/MA/ARIMA/Prophet)
    window_policy: Optional[str] = None   # 예: "PY+CY"
    data_span: Optional[str] = None       # 예: "YYYY-MM ~ YYYY-MM"
    train_months: Optional[int] = None    # 학습 월 수
    horizon: Optional[int] = None         # 예측 수평(월)
    basis_note: Optional[str] = None      # 예: "BS는 잔액·발생액 병렬 계산"
    extra: Optional[Dict[str, Any]] = field(default_factory=dict)
    impacted_assertions: List[str] = field(default_factory=list)  # e.g., ["A","C"]
    links: Dict[str, Any] = field(default_factory=dict)           # e.g., {"account_code": "...", "account_name": "..."}

@dataclass(frozen=True)
class ModuleResult:
    name: str
    summary: Dict[str, Any]             # LLM 입력용 핵심 수치/지표
    tables: Dict[str, pd.DataFrame]
    figures: Dict[str, Any]             # plotly Figure
    evidences: List[EvidenceDetail]     # structured evidences
    warnings: List[str]


# 공개 API 명시(스키마 고정에 도움)
__all__ = [
    "LedgerFrame", "EvidenceDetail", "ModuleResult", "ASSERTIONS"
]

