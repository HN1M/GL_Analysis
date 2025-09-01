LLM_MODEL = "gpt-4o"
LLM_TEMPERATURE = 0.2
LLM_JSON_MODE = True
PM_DEFAULT = 500_000_000  # Project-wide Performance Materiality (KRW)

EMBED_BATCH_SIZE = 256
EMBED_CACHE_DIR = ".cache/embeddings"

# 훈님 결정 반영 ✅
SHAP_TOP_N_PER_ACCOUNT_DEFAULT = 25   # 사용자 UI에서 20~30 범위 선택 가능
CYCLE_RECOMMENDER = "llm_only"        # LLM 100% 자동 추천
PM_DEFAULT = PM_DEFAULT              # (kept above; single source of truth)

# --- Correlation defaults ---
# New canonical names
CORR_DEFAULT_METHOD = "pearson"
CORR_THRESHOLD_DEFAULT = 0.70
CORR_MIN_ACTIVE_MONTHS_DEFAULT = 6
CORR_MAX_LAG_DEFAULT = 6
CORR_ROLLING_WINDOW_DEFAULT = 6

# Backward-compatible aliases (kept for existing imports)
CORR_METHOD_DEFAULT = CORR_DEFAULT_METHOD
CORR_MIN_ACTIVE_MONTHS = CORR_MIN_ACTIVE_MONTHS_DEFAULT
CORR_ROLLWIN_DEFAULT = CORR_ROLLING_WINDOW_DEFAULT

# ---- NEW: Embedding & Clustering defaults ----
# Embedding model switch (Small by default; Large improves semantics at higher cost)
EMB_MODEL_SMALL = "text-embedding-3-small"
EMB_MODEL_LARGE = "text-embedding-3-large"
EMB_USE_LARGE_DEFAULT = False           # UI/auto-upscale can override per run

# UMAP threshold (apply UMAP → HDBSCAN only when N is large)
UMAP_APPLY_THRESHOLD = 8000             # set 0/None to disable
UMAP_N_COMPONENTS = 20
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.0

# HDBSCAN noise-rescue cosine threshold
HDBSCAN_RESCUE_TAU = 0.75               # 0.72~0.78 usually works well

# Adaptive clustering knobs (computed per run, not static)
# min_cluster_size = max(8, int(sqrt(N))); min_samples = max(2, int(0.5 * min_cluster_size))

# ===== NEW: Materiality & Risk Weights =====
# === 통합 리스크 가중치(고정값; v2.0-RC 동결) ===
# 점수 = 0.5*A(|Z|정규화) + 0.4*F(PM 대비 비율 capped 1) + 0.1*K(PM 초과=1)
RISK_WEIGHT_A = 0.5
RISK_WEIGHT_F = 0.4
RISK_WEIGHT_K = 0.1

# --- NEW: Z-Score → sigmoid 스케일 조정 (로드맵 호환)
# anomaly_score = sigmoid(|Z| / Z_SIGMOID_DIVISOR)
# 로드맵 권고: 3.0 (과도 포화 완화)
Z_SIGMOID_DIVISOR = 3.0
Z_SIGMOID_SCALE = Z_SIGMOID_DIVISOR  # 하위호환

# --- 표준 회계 사이클 (STANDARD_ACCOUNTING_CYCLES) ---
# 키: 사이클 식별자, 값: 해당 사이클에 매핑될 가능성이 높은 계정명 키워드(부분일치)
# *한국어/영문 혼용. 필요 시 프로젝트 도메인에 맞춰 보강하세요.
STANDARD_ACCOUNTING_CYCLES = {
    "Cash": ["현금", "예금", "단기금융", "Cash", "Bank"],
    "Revenue": ["매출", "판매수익", "Sales", "Revenue"],
    "Receivables": ["매출채권", "외상매출금", "미수금", "Receivable", "A/R"],
    "Inventory": ["재고", "상품", "제품", "원재료", "재공품", "Inventory"],
    "Payables": ["매입채무", "외상매입금", "미지급금", "Payable", "A/P"],
    "Expenses": ["복리후생비", "급여", "임차료", "접대비", "감가상각비", "비용", "Expense"],
    "FixedAssets": ["유형자산", "감가상각누계", "기계장치", "건물", "비품", "PPE", "Fixed Asset"],
    "Equity": ["자본금", "이익잉여금", "자본잉여금", "Equity", "Capital"],
}

# --- NEW: Anomaly (Semantic & Isolation Forest) defaults ---
IFOREST_ENABLED_DEFAULT = True
IFOREST_N_ESTIMATORS = 256
IFOREST_MAX_SAMPLES = "auto"
IFOREST_CONTAM_DEFAULT = 0.03
IFOREST_RANDOM_STATE = 42

# Semantic outlier thresholds
SEMANTIC_Z_THRESHOLD = 2.5
SEMANTIC_MIN_RECORDS = 12
ANOMALY_IFOREST_SCORE_THRESHOLD = 0.70

# --- Provisional rule naming (도메인 합의 전) ---
PROVISIONAL_RULE_VERSION = "v1.0"
PROVISIONAL_RULE_NAME = f"잠정 기준({PROVISIONAL_RULE_VERSION})"

def provisional_risk_formula_str() -> str:
    """UI/리포트 안내문에 쓰일 가중치 요약 문자열을 동적으로 생성"""
    a = int(RISK_WEIGHT_A * 100)
    f = int(RISK_WEIGHT_F * 100)
    k = int(RISK_WEIGHT_K * 100)
    return f"통계적 이상({a}%) + 재무적 영향({f}%) + KIT 여부({k}%)"

# 리포트(최종본) 포함 조건 노브 (기본: 포함 안 함)
INCLUDE_RISK_MATRIX_SUMMARY_IN_FINAL = False
# ‘상위 N’ 결과가 이 값 미만이면 최종본에 생략 (근거 컨텍스트엔 유지)
RISK_MATRIX_SECTION_MIN_ITEMS = 3

# --- TimeSeries forecast knobs ---
FORECAST_MIN_POINTS = 8         # Prophet/ARIMA 사용 권장 최소 길이(권고치)
ARIMA_DEFAULT_ORDER = (1,1,1)

# --- User overrides for STANDARD_ACCOUNTING_CYCLES ---
CYCLES_USER_OVERRIDES_PATH = ".cache/cycles_overrides.json"
