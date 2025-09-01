from __future__ import annotations
import numpy as np, math
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple
from utils.helpers import find_column_by_keyword
from analysis.embedding import ensure_rich_embedding_text, perform_embedding_only  # ← services 주입식 임베딩 사용
from config import (
    IFOREST_ENABLED_DEFAULT, IFOREST_N_ESTIMATORS, IFOREST_MAX_SAMPLES,
    IFOREST_CONTAM_DEFAULT, IFOREST_RANDOM_STATE,
    SEMANTIC_Z_THRESHOLD, SEMANTIC_MIN_RECORDS, ANOMALY_IFOREST_SCORE_THRESHOLD
)


def compute_amount_columns(df: pd.DataFrame) -> pd.DataFrame:
    """발생액(절대 규모) / 순액(차-대) 계산."""
    dcol = find_column_by_keyword(df.columns, '차변')
    ccol = find_column_by_keyword(df.columns, '대변')
    df = df.copy()
    if not dcol or not ccol:
        df['발생액'] = 0.0; df['순액'] = 0.0; df['거래금액'] = 0.0
        return df
    d = pd.to_numeric(df[dcol], errors='coerce').fillna(0.0)
    c = pd.to_numeric(df[ccol], errors='coerce').fillna(0.0)
    row_amt = np.where((d > 0) & (c == 0), d,
              np.where((c > 0) & (d == 0), c,
              np.where((d == 0) & (c == 0), 0.0, np.abs(d - c))))
    df['발생액'] = row_amt
    df['순액']  = d - c
    df['거래금액'] = df['순액']
    return df


def calculate_grouped_stats_and_zscore(df: pd.DataFrame, target_accounts: List[str], data_type: str = "당기") -> pd.DataFrame:
    """선택 계정 그룹의 발생액 분포 기준 Z-Score 산출."""
    acct_col = find_column_by_keyword(df.columns, '계정코드')
    df = compute_amount_columns(df.copy())
    if not acct_col:
        df['Z-Score'] = 0.0
        return df
    is_target = df[acct_col].astype(str).isin([str(x) for x in target_accounts])
    tgt = df.loc[is_target, '발생액'].astype(float)
    df['Z-Score'] = 0.0
    if tgt.empty:
        return df
    mu = float(tgt.mean()); std = float(tgt.std(ddof=1))
    if std and std > 0:
        df.loc[is_target, 'Z-Score'] = (df.loc[is_target, '발생액'] - mu) / std
    else:
        med = float(tgt.median()); mad = float((np.abs(tgt - med)).median())
        df.loc[is_target, 'Z-Score'] = 0.0 if mad == 0 else 0.6745 * (df.loc[is_target, '발생액'] - med) / mad
    return df

# ---------------------- NEW: Semantic features -------------------------
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    da = float(np.linalg.norm(a)); db = float(np.linalg.norm(b))
    if da == 0.0 or db == 0.0: return 0.0
    return float(np.dot(a, b) / (da * db))

def _zseries(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    mu, sd = float(x.mean()), float(x.std(ddof=1))
    if sd and sd > 0: return (x - mu) / sd
    # MAD fallback
    med = float(x.median()); mad = float((x.sub(med).abs()).median())
    return pd.Series(0.0, index=x.index) if mad == 0 else 0.6745 * (x - med) / mad

def _maybe_subcluster_vectors(X: np.ndarray) -> np.ndarray:
    """Return labels for vectors (auto-k KMeans via KDMeans shim)."""
    try:
        from analysis.kdmeans_shim import HDBSCAN
        model = HDBSCAN(n_clusters=None, random_state=42)
        return model.fit_predict(X).astype(int)
    except Exception:
        return np.zeros(len(X), dtype=int)

def _add_semantic_features(
    df: pd.DataFrame,
    *,
    acct_col: str,
    embed_client: Any,
    embed_texts_fn,              # injected (e.g., services.cache.get_or_embed_texts)
    use_large: Optional[bool] = None,
    subcluster: bool = False
) -> pd.DataFrame:
    """임베딩 벡터, 계정/클러스터 센트로이드, semantic_z(코사인 거리 z) 생성."""
    if df is None or df.empty: return df
    base = ensure_rich_embedding_text(df.copy())  # desc+vendor+월+규모+성격 조합 텍스트 생성
    base = perform_embedding_only(
        base, client=embed_client, text_col="embedding_text",
        use_large=use_large, embed_texts_fn=embed_texts_fn
    )
    if 'vector' not in base.columns or base['vector'].isna().any():
        return base
    # 벡터 행렬
    V = np.vstack(base['vector'].values).astype(float)
    # 선택: 계정 내 서브클러스터
    if subcluster:
        labels = pd.Series(index=base.index, dtype=int)
        for code, sub in base.groupby(base[acct_col].astype(str)):
            idx = sub.index
            Xi = np.vstack(sub['vector'].values)
            if len(Xi) < max(SEMANTIC_MIN_RECORDS, 4):
                labels.loc[idx] = 0
            else:
                labels.loc[idx] = _maybe_subcluster_vectors(Xi)
        base['cluster_id'] = labels.astype(int)
    else:
        base['cluster_id'] = 0
    # 그룹(계정×클러스터) 센트로이드 & 코사인 거리
    dists = []
    for (acct, cid), sub in base.groupby([base[acct_col].astype(str), 'cluster_id']):
        vecs = np.vstack(sub['vector'].values)
        c = vecs.mean(axis=0)
        # 1 - cosine sim → semantic distance
        dd = [1.0 - _cosine(v, c) for v in vecs]
        dists.append(pd.Series(dd, index=sub.index))
    base['semantic_dist'] = pd.concat(dists).sort_index()
    # z-표준화(계정×클러스터별)
    base['semantic_z'] = (
        base.groupby([base[acct_col].astype(str), 'cluster_id'])['semantic_dist']
            .transform(_zseries)
            .astype(float)
    )
    return base

# ---------------------- NEW: Isolation Forest --------------------------
def _fit_iforest_and_score(F: pd.DataFrame, *, contamination: float) -> np.ndarray:
    """Return anomaly scores in [0,1]."""
    try:
        from sklearn.ensemble import IsolationForest
    except Exception:
        return np.zeros(len(F), dtype=float)
    # NaN 방어 및 스케일링 간단 적용
    X = F.fillna(0.0).astype(float).values
    iso = IsolationForest(
        n_estimators=int(IFOREST_N_ESTIMATORS),
        max_samples=IFOREST_MAX_SAMPLES,
        contamination=float(contamination),
        random_state=int(IFOREST_RANDOM_STATE),
        n_jobs=-1
    ).fit(X)
    raw = -iso.score_samples(X)              # 더 클수록 이상
    lo, hi = float(np.min(raw)), float(np.max(raw))
    s = (raw - lo) / (hi - lo + 1e-12)      # [0,1]
    return s

# --- NEW: ensure_zscore ---
def ensure_zscore(df: pd.DataFrame, account_codes: List[str]):
    """
    Recompute Z-Score for the given account subset and return (df, ok).
    ok=True only if Z-Score column exists and has at least one non-null value.
    """
    df2 = calculate_grouped_stats_and_zscore(df.copy(), target_accounts=[str(x) for x in account_codes] if account_codes else [])
    z = df2.get('Z-Score')
    ok = (z is not None) and (z.notna().any())
    return df2, bool(ok)




# === (ADD) v0.18: ModuleResult 러너 ===
from analysis.contracts import ModuleResult, EvidenceDetail
from config import PM_DEFAULT, RISK_WEIGHT_A, RISK_WEIGHT_F, RISK_WEIGHT_K, Z_SIGMOID_SCALE, Z_SIGMOID_DIVISOR
import plotly.express as px
import numpy as np
import pandas as pd


def _z_bins_025_sigma(series: pd.Series):
    """0.25σ 간격 bin (±3σ 테일 포함)."""
    # 경계에 +3.0 포함(+inf 테일) → 총 bin 수 = 24(코어) + 2(테일) = 26
    edges = [-np.inf] + [round(x, 2) for x in np.arange(-3.0, 3.0 + 0.25, 0.25)] + [np.inf]
    core_lefts = [x for x in np.arange(-3.0, 3.0, 0.25)]  # 24개
    labels_mid = [f"{a:.2f}~{a+0.25:.2f}σ" for a in core_lefts]
    labels = ["≤-3σ"] + labels_mid + ["≥3σ"]               # 26개
    cats = pd.cut(
        series.astype(float),
        bins=edges,
        labels=labels,
        right=False,
        include_lowest=True,
    )
    # 빈 구간도 0으로 채워 순서 유지
    counts = cats.value_counts(sort=False).reindex(labels, fill_value=0)
    out = pd.DataFrame({"구간": labels, "건수": counts.values})
    order = labels
    return out, order


def _sigmoid(x: float) -> float:
    import math
    return 1.0 / (1.0 + math.exp(-x))


def _risk_from(z_abs: float, amount: float, pm: float):
    """리스크 점수 구성요소 계산.
    - a: 시그모이드 정규화된 이탈 강도(|Z|/scale). scale은 설정값.
    - f: PM 대비 금액비율(0~1로 캡). PM이 0/음수면 0으로 강제.
    - k: Key Item 플래그(PM 초과시 1). PM이 0/음수면 0으로 강제.
    """
    # 우선순위: Z_SIGMOID_DIVISOR(신규 노브) > Z_SIGMOID_SCALE(구명). 기본 1.0
    div = None
    try:
        div = float(Z_SIGMOID_DIVISOR)
    except Exception:
        div = None
    if not div or div <= 0:
        try:
            div = float(Z_SIGMOID_SCALE)
        except Exception:
            div = 1.0
    if not div or div <= 0:
        div = 1.0
    a = _sigmoid(float(abs(z_abs)) / float(div))      # anomaly_score
    # PM 가드: pm<=0이면 f=0, k=0
    if pm is None or float(pm) <= 0:
        f = 0.0
        k = 0.0
    else:
        f = min(1.0, abs(float(amount)) / float(pm))  # PM ratio (capped at 1)
        k = 1.0 if abs(float(amount)) >= float(pm) else 0.0
    score = RISK_WEIGHT_A * a + RISK_WEIGHT_F * f + RISK_WEIGHT_K * k
    return a, f, k, score


def _assertions_for_row(z_val: float) -> List[str]:
    # 기본 규칙: A는 항상 포함. 음의 큰 이탈(C), 양의 큰 이탈(E)을 보강.
    out = {"A"}
    try:
        if float(z_val) <= -2.0:
            out.add("C")
        if float(z_val) >=  2.0:
            out.add("E")
    except Exception:
        pass
    return sorted(out)


def run_anomaly_module(
    lf,
    target_accounts=None,
    topn=20,
    pm_value: Optional[float] = None,
    *,
    # --- NEW: injection knobs (analysis 레이어는 services에 직접 의존 금지) ---
    embed_client: Any = None,
    embed_texts_fn=None,
    use_large_embedding: Optional[bool] = None,
    semantic_enabled: bool = True,
    subcluster_enabled: bool = False,
    iforest_enabled: Optional[bool] = None,
    iforest_contamination: Optional[float] = None,
):
    df = lf.df.copy()
    acct_col = find_column_by_keyword(df.columns, '계정코드')
    if not acct_col:
        return ModuleResult("anomaly", {}, {}, {}, [], ["계정코드 컬럼을 찾지 못했습니다."])

    # 대상 계정 서브셋
    if target_accounts:
        codes = [str(x) for x in target_accounts]
        df = df[df[acct_col].astype(str).isin(codes)].copy()

    # Z-Score 계산
    df = calculate_grouped_stats_and_zscore(df, target_accounts=df[acct_col].astype(str).unique().tolist())
    if '회계일자' in df.columns:
        df['연월'] = df['회계일자'].dt.to_period('M').astype(str)

    # 이상치 플래그 (±3σ)
    df['is_outlier'] = df['Z-Score'].abs() >= 3

    # === (NEW) 의미피처/IForest 생성 ===
    if semantic_enabled and (embed_client is not None) and (embed_texts_fn is not None):
        try:
            df = _add_semantic_features(
                df, acct_col=acct_col, embed_client=embed_client,
                embed_texts_fn=embed_texts_fn, use_large=use_large_embedding,
                subcluster=subcluster_enabled
            )
        except Exception:
            # 의미피처 실패해도 기본 Z-Score 흐름은 유지
            pass
    # Isolation Forest (의미피처가 있든 없든 수치특징만으로도 동작)
    if iforest_enabled is None:
        iforest_enabled = bool(IFOREST_ENABLED_DEFAULT)
    if iforest_enabled:
        try:
            feats: Dict[str, Any] = {}
            feats['amt']      = pd.to_numeric(df.get('발생액', 0.0), errors='coerce').abs()
            feats['amt_log']  = np.log1p(feats['amt'])
            feats['z_abs']    = df.get('Z-Score', 0.0).abs()
            feats['sem_abs']  = df.get('semantic_z', 0.0).abs() if 'semantic_z' in df.columns else 0.0
            if '연월' in df.columns:
                # 간단 월 인덱스(모델의 시퀀스 surrogate)
                feats['month_idx'] = pd.Categorical(df['연월']).codes.astype(float)
            F = pd.DataFrame(feats, index=df.index)
            contam = float(iforest_contamination) if iforest_contamination is not None else float(IFOREST_CONTAM_DEFAULT)
            df['iforest_score'] = _fit_iforest_and_score(F, contamination=contam)
        except Exception:
            pass

    # 이상치 후보 테이블 (절댓값 기준 상위)
    out_cols = [c for c in ['row_id','회계일자','연월','계정코드','계정명','거래처','적요','발생액','Z-Score'] if c in df.columns]
    # (NEW) 테이블에 신호 컬럼 노출
    for extra in ['semantic_z','iforest_score','cluster_id']:
        if extra in df.columns and extra not in out_cols:
            out_cols.append(extra)
    cand = (df.assign(absz=df['Z-Score'].abs())
              .sort_values('absz', ascending=False)
              .drop(columns=['absz'])
              .head(int(topn)))
    table = cand[out_cols + (['is_outlier'] if 'is_outlier' in cand.columns and 'is_outlier' not in out_cols else [])] if out_cols else cand

    # === EvidenceDetail 생성 (KIT + |Z| 기준) ===
    pm = float(pm_value) if pm_value is not None else float(PM_DEFAULT)
    ev_rows: List[EvidenceDetail] = []
    # 증거 채집 대상: (1) PM 초과 or (2) |Z|>=2.5 or (3) 상위 topn
    #               + (4) semantic_z 과대 or (5) iforest_score 과대
    mask_key = df['발생액'].abs() >= pm if '발생액' in df.columns else pd.Series(False, index=df.index)
    mask_z   = df['Z-Score'].abs() >= 2.5 if 'Z-Score' in df.columns else pd.Series(False, index=df.index)
    mask_sem = df['semantic_z'].abs() >= float(SEMANTIC_Z_THRESHOLD) if 'semantic_z' in df.columns else pd.Series(False, index=df.index)
    thr_ifo  = float(ANOMALY_IFOREST_SCORE_THRESHOLD)
    mask_ifo = df['iforest_score'] >= thr_ifo if 'iforest_score' in df.columns else pd.Series(False, index=df.index)
    idx_sel  = set(df.index[mask_key | mask_z | mask_sem | mask_ifo].tolist()) | set(table.index.tolist())
    sub = df.loc[sorted(idx_sel)].copy() if len(idx_sel)>0 else df.head(0).copy()
    for _, r in sub.iterrows():
        z  = float(r.get('Z-Score', 0.0)) if pd.notna(r.get('Z-Score', np.nan)) else 0.0
        za = abs(z)
        amt = float(r.get('발생액', 0.0))
        a, f, k, score = _risk_from(za, amt, pm)   # (기존) 통합 위험 점수는 PM/|Z| 기반 유지
        # (NEW) anomaly_score에 의미/IForest 신호를 반영해 탐색 우선순위 개선
        semz = float(abs(r.get('semantic_z', 0.0))) if pd.notna(r.get('semantic_z', np.nan)) else 0.0
        ifo  = float(r.get('iforest_score', 0.0)) if pd.notna(r.get('iforest_score', np.nan)) else 0.0
        try:
            div = float(Z_SIGMOID_DIVISOR) if float(Z_SIGMOID_DIVISOR) > 0 else 3.0
        except Exception:
            div = 3.0
        sem_a = 1.0 / (1.0 + math.exp(-(semz/div))) if semz > 0 else 0.0
        anomaly_score = float(max(a, sem_a, ifo))
        ev_rows.append(EvidenceDetail(
            row_id=str(r.get('row_id','')),
            reason="; ".join(filter(None, [
                f"amt_z={z:+.2f}",
                (f"sem_z={r.get('semantic_z'):+.2f}" if 'semantic_z' in r and pd.notna(r['semantic_z']) else ""),
                (f"iforest={ifo:.2f}" if 'iforest_score' in r and pd.notna(r['iforest_score']) else "")
            ])),
            anomaly_score=anomaly_score,
            financial_impact=abs(amt),
            risk_score=float(score),
            is_key_item=bool(abs(amt) >= pm),
            impacted_assertions=_assertions_for_row(z),
            links={
                "account_code": str(r.get('계정코드','')),
                "account_name": str(r.get('계정명','')),
                "vendor":      str(r.get('거래처','')),
                "narration":   str(r.get('적요','')),
                "cluster_name": str(r.get('cluster_name','')) if 'cluster_name' in r.index else "",
                "cluster_group": str(r.get('cluster_group','')) if 'cluster_group' in r.index else "",
                "month":       str(r.get('연월','')) if '연월' in r.index else "",
                "period_tag": str(r.get('period_tag','')),
            }
        ))

    # step-σ bin 분포 막대
    figures = {}
    try:
        dist_df, order = _z_bins_025_sigma(df['Z-Score'])
        total_n = int(len(df))
        outlier_rate = float((df['Z-Score'].abs() >= 3).mean() * 100) if total_n else 0.0
        title = f"Z-Score 분포 (0.25σ bin, ±3σ 집계) — N={total_n:,}, outlier≈{outlier_rate:.1f}%"
        fig = px.bar(dist_df, x='구간', y='건수', title=title)
        fig.update_yaxes(separatethousands=True)
        fig.update_layout(bargap=0.10)
        figures = {"zscore_hist": fig}
    except Exception:
        pass

    summary = {
        "n_rows": int(len(df)),
        "n_candidates": int(len(table)),
        "accounts": sorted(df[acct_col].astype(str).unique().tolist()),
        "period_tag_coverage": dict(df.get('period_tag', pd.Series(dtype=str)).value_counts()) if 'period_tag' in df.columns else {}
    }
    # Evidence 미리보기 테이블(선택)
    try:
        import pandas as _pd
        ev_tbl = _pd.DataFrame([{
            "row_id": e.row_id,
            "계정코드": e.links.get("account_code",""),
            "계정명":   e.links.get("account_name",""),
            "risk_score": e.risk_score,
            "is_key_item": e.is_key_item,
            "impacted": ",".join(e.impacted_assertions),
            "reason": e.reason,
        } for e in ev_rows]).sort_values("risk_score", ascending=False).head(100)
    except Exception:
        ev_tbl = None

    return ModuleResult(
        name="anomaly",
        summary=summary,
        tables={"anomaly_top": table, **({"evidence_preview": ev_tbl} if ev_tbl is not None else {})},
        figures=figures,
        evidences=ev_rows,
        warnings=[]
    )