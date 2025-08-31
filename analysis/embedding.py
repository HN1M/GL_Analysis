from __future__ import annotations
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Optional, Tuple, Callable, Sequence, Any
# --- KDMeans 기반 HDBSCAN 대체 사용(의미상 HDBSCAN과 유사 동작) ---
from analysis.kdmeans_shim import HDBSCAN   # (주의) 내부적으로 KMeans 기반 구현
_HAS_HDBSCAN = True
# ---------------------------------------

from utils.helpers import find_column_by_keyword
from config import (
    EMB_MODEL_SMALL, EMB_MODEL_LARGE, EMB_USE_LARGE_DEFAULT,
    UMAP_APPLY_THRESHOLD, UMAP_N_COMPONENTS, UMAP_N_NEIGHBORS, UMAP_MIN_DIST,
    HDBSCAN_RESCUE_TAU,
)

# Embedding call defaults (can be overridden via pick_emb_model / params)
EMB_BATCH_SIZE = 128
EMB_TIMEOUT = 60
EMB_MAX_RETRY = 4
EMB_TRUNC_CHARS = 2000


def embed_texts_batched(
    texts: Sequence[str],
    *,
    embed_texts_fn: Callable[..., Any],
    client,
    model: str,
    batch_size: int = EMB_BATCH_SIZE,
    timeout: int = EMB_TIMEOUT,
    max_retry: int = EMB_MAX_RETRY,
    trunc_chars: int = EMB_TRUNC_CHARS,
) -> Dict[str, List[float]]:
    """배치 임베딩 유틸. {원본문자열: 벡터} 반환.
    services 레이어에 직접 의존하지 않고, 호출자가 임베딩 함수(embed_texts_fn)를 주입한다.
    """
    if not texts:
        return {}
    san: List[str] = []
    for t in texts:
        s = t if isinstance(t, str) else str(t)
        san.append(s[:trunc_chars] if trunc_chars and len(s) > trunc_chars else s)

    # 호출자로부터 주입받은 함수 사용(예: services.cache.get_or_embed_texts)
    return embed_texts_fn(
        san, client=client, model=model, batch_size=batch_size, timeout=timeout, max_retry=max_retry
    )


def _clean_text_series(s: pd.Series) -> pd.Series:
    """Lightweight denoising: collapse long numbers, squeeze spaces, trim."""
    s = s.astype(str)
    s = s.str.replace(r"\d{8,}", "#NUM", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s

def ensure_embedding_text(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df['embedding_text'] exists (desc+vendor) and is cleaned."""
    if 'embedding_text' not in df.columns:
        desc = df['적요'].fillna('').astype(str) if '적요' in df.columns else ''
        cp   = df['거래처'].fillna('').astype(str) if '거래처' in df.columns else ''
        df['embedding_text'] = desc + " (거래처: " + cp + ")"
    df['embedding_text'] = _clean_text_series(df['embedding_text'])
    return df


def _amount_bucket(a: float) -> str:
    a = float(abs(a))
    if a < 1_000_000:   return "1백만 미만"
    if a < 10_000_000:  return "1천만 미만"
    if a < 100_000_000: return "1억원 미만"
    if a < 500_000_000: return "5억원 미만"
    if a < 1_000_000_000:return "10억원 미만"
    if a < 5_000_000_000:return "50억원 미만"
    return "50억원 이상"


def ensure_rich_embedding_text(df: pd.DataFrame) -> pd.DataFrame:
    """적요+거래처+월+금액구간+차/대 성격을 조합해 임베딩 텍스트 생성."""
    # 발생액/순액은 anomaly.compute_amount_columns를 쓰면 순환 import가 생김 → 최소 필드만 계산
    def _compute_amount_cols(_df: pd.DataFrame) -> pd.DataFrame:
        dcol = find_column_by_keyword(_df.columns, '차변')
        ccol = find_column_by_keyword(_df.columns, '대변')
        if not dcol or not ccol:
            _df['발생액'] = 0.0; _df['순액'] = 0.0
            return _df
        d = pd.to_numeric(_df[dcol], errors='coerce').fillna(0.0)
        c = pd.to_numeric(_df[ccol], errors='coerce').fillna(0.0)
        row_amt = np.where((d > 0) & (c == 0), d,
                  np.where((c > 0) & (d == 0), c,
                  np.where((d == 0) & (c == 0), 0.0, np.abs(d - c))))
        _df['발생액'] = row_amt
        _df['순액']  = d - c
        return _df

    df = _compute_amount_cols(df.copy())
    month = df['회계일자'].dt.month.fillna(0).astype(int).astype(str).str.zfill(2) if '회계일자' in df.columns else "00"
    amtbin = df['발생액'].apply(_amount_bucket)
    sign   = np.where(df['순액'] >= 0, "차변성", "대변성")
    desc = df['적요'].fillna('').astype(str) if '적요' in df.columns else ''
    cp   = df['거래처'].fillna('').astype(str) if '거래처' in df.columns else ''
    df['embedding_text'] = desc + " | 거래처:" + cp + " | 월:" + month + " | 금액구간:" + amtbin + " | 성격:" + sign
    df['embedding_text'] = _clean_text_series(df['embedding_text'])
    return df


def perform_embedding_only(
    df: pd.DataFrame,
    client,
    text_col: str = 'embedding_text',
    *,
    use_large: bool|None=None,
    embed_texts_fn: Callable[..., Any],
) -> pd.DataFrame:
    """df[text_col]을 배치 임베딩해서 df['vector'] 추가"""
    if df.empty: return df
    if text_col not in df.columns:
        raise ValueError(f"임베딩 텍스트 컬럼 '{text_col}'이 없습니다.")
    uniq = df[text_col].astype(str).unique().tolist()
    model = pick_emb_model(use_large=use_large)
    mapping = embed_texts_batched(
        uniq,
        embed_texts_fn=embed_texts_fn,
        client=client,
        model=model,
    )
    df = df.copy()
    df['vector'] = df[text_col].astype(str).map(mapping)
    # Guard embedding failures
    if df is None or df.empty:
        return df
    # 누락 보강 시도
    if df['vector'].isna().any():
        miss = df.loc[df['vector'].isna(), text_col].astype(str).unique().tolist()
        if miss:
            fb = embed_texts_batched(
                miss,
                embed_texts_fn=embed_texts_fn,
                client=client,
                model=model,
            )
            df.loc[df['vector'].isna(), 'vector'] = df.loc[df['vector'].isna(), text_col].astype(str).map(fb)
    return df


def _l2_normalize(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

def _adaptive_hdbscan(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = int(X.shape[0])
    model = HDBSCAN(
        n_clusters=None,           # 자동 k 선택(실루엣 기반)
        min_cluster_size=max(8, int(np.sqrt(max(2, n)))),  # 너무 많은 군집 방지
        max_k=None,                # 필요시 상한 지정 가능
        k_search="silhouette",     # 휴리스틱 대신 실루엣 기반
        sample_size=2000,
        random_state=42,
        n_init="auto",
    )
    model.fit(X)
    labels = model.labels_.astype(int)
    try:
        probs = model.probabilities_.astype(float)
    except Exception:
        probs = np.ones(shape=(n,), dtype=float)
    return labels, probs

def _optional_umap(X: np.ndarray, enabled: Optional[bool] = None) -> Tuple[np.ndarray, bool]:
    """Dimensionality reduction control.
    - enabled=True: always try UMAP; on failure return (X, False)
    - enabled=False: skip → (X, False)
    - enabled=None: apply only if dataset size >= UMAP_APPLY_THRESHOLD
    Returns (X_or_reduced, used_flag).
    """
    if enabled is False:
        return X, False
    force = enabled is True
    try:
        thr = int(UMAP_APPLY_THRESHOLD) if UMAP_APPLY_THRESHOLD else None
    except Exception:
        thr = None
    if force or (thr and X.shape[0] >= thr):
        try:
            import umap
            reducer = umap.UMAP(
                n_components=int(UMAP_N_COMPONENTS),
                n_neighbors=int(UMAP_N_NEIGHBORS),
                min_dist=float(UMAP_MIN_DIST),
                random_state=42,
                metric="euclidean",
            )
            return reducer.fit_transform(X), True
        except Exception:
            return X, False
    return X, False

def _rescue_noise(df: pd.DataFrame, tau: float = HDBSCAN_RESCUE_TAU) -> pd.DataFrame:
    # KDMeans는 노이즈(-1) 라벨이 없으므로 구조적 리스큐 불필요
    return df

def pick_emb_model(use_large: bool|None=None) -> str:
    """Select embedding model (small/large)."""
    flag = EMB_USE_LARGE_DEFAULT if use_large is None else bool(use_large)
    return EMB_MODEL_LARGE if flag else EMB_MODEL_SMALL


def postprocess_cluster_names(df: pd.DataFrame) -> pd.DataFrame:
    """(간소화) LLM이 준 cluster_name을 그대로 유지한다. 태그/접미사 미부여."""
    return df


def perform_embedding_and_clustering(
    df: pd.DataFrame,
    client,
    *,
    name_with_llm: bool = True,
    must_name_with_llm: bool = False,
    naming_fn: Optional[Callable[[list[str], list[str]], Optional[str]]] = None,
    use_large: bool|None = None,
    rescue_tau: float = HDBSCAN_RESCUE_TAU,
    umap_enabled: bool|None = None,   # None => use config threshold
    embed_texts_fn: Callable[..., Any],
):
    """
    Embedding + (optional UMAP) + L2-normalized Euclidean HDBSCAN + noise rescue + (LLM naming).
    Returns: (df, ok)
    ok=False if: no vectors, or LLM naming required but missing/failed.
    """
    df = ensure_embedding_text(df.copy())
    uniq = df['embedding_text'].astype(str).unique().tolist()
    model = pick_emb_model(use_large=use_large)
    mapping = embed_texts_batched(
        uniq,
        embed_texts_fn=embed_texts_fn,
        client=client,
        model=model,
    )
    df['vector'] = df['embedding_text'].astype(str).map(mapping)
    # Guard: embedding may fail and return None vectors
    if df is None or df.empty:
        return None, False
    # keep only valid vectors
    mask = df['vector'].apply(lambda v: isinstance(v, (list, tuple)) and len(v) > 0)
    df = df.loc[mask].copy()
    if df.empty:
        return None, False

    X = np.vstack(df['vector'].values).astype(float)
    # Optional UMAP if dataset large (threshold controlled by config)
    X, umap_used = _optional_umap(X, enabled=umap_enabled)
    # L2 normalize and cluster with Euclidean (≈ cosine)
    Xn = _l2_normalize(X)
    labels, probs = _adaptive_hdbscan(Xn)
    df['cluster_id'] = labels
    df['cluster_prob'] = probs
    # telemetry attrs
    try:
        df.attrs['embedding_model'] = model
        df.attrs['umap_used'] = bool(umap_used)
        df.attrs['rescue_tau'] = float(rescue_tau) if rescue_tau is not None else None
    except Exception:
        pass

    # --- Cluster naming via injected LLM callback (with graceful fallback) ---
    labels_uniq = sorted(pd.Series(labels).unique())
    names = {}
    if name_with_llm and (naming_fn is not None):
        for cid in labels_uniq:
            if cid == -1:
                names[cid] = "클러스터 노이즈(-1)"
                continue
            sub = df[df['cluster_id'] == cid]
            descs = sub['적요'].dropna().astype(str).unique().tolist()[:5] if '적요' in sub.columns else []
            vendors = sub['거래처'].dropna().astype(str).unique().tolist()[:5] if '거래처' in sub.columns else []
            # 콜백 사용(services.cluster_naming에서 생성)
            try:
                cand = naming_fn(descs, vendors)
            except Exception:
                cand = None
            # fallback rule-based name if LLM failed
            if not cand or cand == "이름 생성 실패":
                # heuristic: frequent vendor or keyword + amount tag
                amt_tag = "규모 중간"
                try:
                    abs_amt = sub.get('발생액', pd.Series(dtype=float)).abs().median()
                    if float(abs_amt) >= 1e8: amt_tag = "1억원 이상"
                    elif float(abs_amt) >= 1e7: amt_tag = "1천만~1억"
                except Exception:
                    pass
                top_vendor = sub.get('거래처', pd.Series(dtype=str)).value_counts().index.tolist()
                vname = top_vendor[0] if top_vendor else "일반"
                cand = f"{vname} 중심({amt_tag})"
            names[cid] = cand
    else:
        for cid in labels_uniq:
            names[cid] = "클러스터 노이즈(-1)" if cid == -1 else "이름 생성 실패"

    df['cluster_name'] = df['cluster_id'].map(names)
    df = postprocess_cluster_names(df)

    # --- Noise rescue: reassign -1 to nearest centroid if cosine >= tau ---
    if rescue_tau and float(rescue_tau) > 0:
        df = _rescue_noise(df, tau=float(rescue_tau))

    # gate: if must_name_with_llm, all non-noise clusters must have valid names
    if must_name_with_llm:
        non_noise = df[df['cluster_id'] != -1]
        has_any = not non_noise.empty
        invalid = non_noise['cluster_name'].isna() | non_noise['cluster_name'].astype(str).str.contains("^이름 생성 실패|^클러스터\\s", regex=True)
        if (not has_any) or bool(invalid.any()):
            return df, False

    # default reporting group equals the (validated) cluster_name; may be unified later
    df['cluster_group'] = df['cluster_name']
    return df, True



# --- NEW: LLM synonym grouping for cluster names ---
def _cosine_sim_matrix(vecs: list[list[float]]):
    import numpy as np
    V = np.asarray(vecs, dtype=float)
    if V.ndim != 2 or V.shape[0] == 0:
        return np.zeros((0, 0))
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    return Vn @ Vn.T


def unify_cluster_names_with_llm(
    df: pd.DataFrame,
    sim_threshold: float = 0.90,
    emb_model: str = EMB_MODEL_SMALL,
    *,
    embed_texts_fn: Callable[..., Any],
    confirm_pair_fn: Optional[Callable[[str, str], bool]] = None,
):
    """
    Collapse clusters with effectively identical names.
    Strategy:
      1) Embed unique names (excluding noise), preselect candidate pairs via cosine >= sim_threshold.
      2) Ask LLM YES/NO if two names are synonyms for accounting transaction categories.
      3) Union-Find merge; choose canonical = most frequent name in df (fallback shortest).
    Returns: (df_with_cluster_group, mapping{name->canonical})
    """
    import numpy as np
    import itertools
    base = df.copy()
    if 'cluster_name' not in base.columns:
        base['cluster_group'] = base.get('cluster_name', None)
        return base, {}
    names = (
        base.loc[base['cluster_id'] != -1, 'cluster_name']
        .dropna().astype(str).unique().tolist()
    )
    if not names:
        base['cluster_group'] = base['cluster_name']
        return base, {}

    # Embedding prefilter
    name2vec = embed_texts_batched(
        names,
        embed_texts_fn=embed_texts_fn,
        client=None,
        model=emb_model,
    )
    ordered = [n for n in names if n in name2vec]
    vecs = [name2vec[n] for n in ordered]
    S = _cosine_sim_matrix(vecs)

    # Union-Find
    parent = {n: n for n in ordered}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Pair confirmation (LLM or 다른 정책) — 반드시 주입 받은 confirm_pair_fn을 사용
    for i, j in itertools.combinations(range(len(ordered)), 2):
        if S[i, j] < float(sim_threshold):
            continue
        a, b = ordered[i], ordered[j]
        if confirm_pair_fn is None:
            # 콜백이 없으면 보수적으로 merge 생략(아키텍처 준수)
            continue
        try:
            if confirm_pair_fn(a, b):
                union(a, b)
        except Exception:
            continue

    # Build groups
    groups = {}
    for n in ordered:
        r = find(n)
        groups.setdefault(r, []).append(n)

    # Choose canonical per group
    freq = base['cluster_name'].value_counts().to_dict()
    mapping = {}
    for root, members in groups.items():
        cand = sorted(members, key=lambda x: (-freq.get(x, 0), len(x)))[0]
        for m in members:
            mapping[m] = cand

    base['cluster_group'] = base['cluster_name'].map(lambda x: mapping.get(x, x))
    return base, mapping


# --- NEW: Utilities for PY→CY mapping and label unification ---
def _cosine(a, b):
    import numpy as np
    if a is None or b is None:
        return np.nan
    a = np.asarray(a)
    b = np.asarray(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else np.nan


def map_previous_to_current_clusters(df_cur: pd.DataFrame, df_prev: pd.DataFrame) -> pd.DataFrame:
    """
    전기 전표를 당기 클러스터 센트로이드에 최근접 배정하여 (mapped_cluster_id/name, mapped_sim) 부여.
    - 노이즈(-1) 센트로이드는 제외
    - 반환: prev_df(with mapped_cluster_id, mapped_cluster_name, mapped_sim)
    """
    import numpy as np
    import pandas as pd
    need_cols = ['cluster_id', 'cluster_name', 'vector']
    if any(c not in df_cur.columns for c in need_cols) or 'vector' not in df_prev.columns:
        return df_prev.copy()
    cur = df_cur[df_cur['cluster_id'] != -1].copy()
    if cur.empty:
        return df_prev.copy()
    # 센트로이드 계산
    cents = (
        cur.groupby(['cluster_id', 'cluster_name'])['vector']
           .apply(lambda s: np.mean(np.vstack(list(s)), axis=0))
           .reset_index()
    )
    prev = df_prev.copy()

    def _pick(row: pd.Series) -> pd.Series:
        v = row.get('vector', None)
        if v is None:
            return pd.Series({'mapped_cluster_id': np.nan, 'mapped_cluster_name': None, 'mapped_sim': np.nan})
        sims = cents['vector'].apply(lambda c: _cosine(v, c))
        if len(sims) == 0 or sims.isna().all():
            return pd.Series({'mapped_cluster_id': np.nan, 'mapped_cluster_name': None, 'mapped_sim': np.nan})
        idx = int(sims.idxmax())
        return pd.Series({
            'mapped_cluster_id': int(cents.loc[idx, 'cluster_id']),
            'mapped_cluster_name': cents.loc[idx, 'cluster_name'],
            'mapped_sim': float(sims.max()) if not np.isnan(sims.max()) else np.nan,
        })

    prev[['mapped_cluster_id', 'mapped_cluster_name', 'mapped_sim']] = prev.apply(_pick, axis=1)
    return prev


def unify_cluster_labels_llm(*_args, **_kwargs) -> dict:
    """Deprecated in analysis layer. Use services.cluster_naming.unify_cluster_labels_llm instead."""
    return {}


# --- NEW: Yearly clustering helpers and alignment ---
def cluster_year(df: pd.DataFrame, client, *, embed_texts_fn: Callable[..., Any]) -> pd.DataFrame:
    """
    당기/전기 등 입력 df에 대해 풍부 임베딩 텍스트를 보장하고 HDBSCAN+LLM 네이밍을 실행.
    반환: ['row_id','cluster_id','cluster_name','cluster_prob','vector']가 포함된 DataFrame(부분집합 가능).
    입력이 비어있으면 빈 DataFrame 반환.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    from .embedding import ensure_rich_embedding_text, perform_embedding_and_clustering
    df_in = ensure_rich_embedding_text(df.copy())
    df_out, ok = perform_embedding_and_clustering(
        df_in,
        client,
        name_with_llm=True,
        must_name_with_llm=False,
        embed_texts_fn=embed_texts_fn,
    )
    if not ok or df_out is None:
        return pd.DataFrame()
    keep = [c for c in ['row_id','cluster_id','cluster_name','cluster_prob','vector'] if c in df_out.columns]
    return df_out[keep].copy()


def compute_centroids(df: pd.DataFrame) -> pd.DataFrame:
    """
    ['cluster_id','vector']를 갖는 df에서 클러스터별 센트로이드 계산(-1 제외).
    'cluster_name'이 있으면 함께 유지.
    반환: columns=['cluster_id','cluster_name','vector']
    """
    import numpy as np
    import pandas as pd
    need = ['cluster_id','vector']
    if df is None or df.empty or any(c not in df.columns for c in need):
        return pd.DataFrame(columns=['cluster_id','cluster_name','vector'])
    base = df[df['cluster_id'] != -1].copy()
    if base.empty:
        return pd.DataFrame(columns=['cluster_id','cluster_name','vector'])
    def _mean_stack(s):
        try:
            return np.mean(np.vstack(list(s)), axis=0)
        except Exception:
            return None
    cents = base.groupby('cluster_id')['vector'].apply(_mean_stack).reset_index()
    if 'cluster_name' in base.columns:
        name_map = base.drop_duplicates('cluster_id').set_index('cluster_id')['cluster_name']
        cents['cluster_name'] = cents['cluster_id'].map(name_map)
    else:
        cents['cluster_name'] = None
    # re-order columns
    cents = cents[['cluster_id','cluster_name','vector']]
    # drop rows with invalid vectors
    cents = cents[cents['vector'].apply(lambda v: isinstance(v, (list, tuple)) and len(v) > 0)]
    return cents.reset_index(drop=True)


def align_yearly_clusters(df_cy: pd.DataFrame, df_py: pd.DataFrame, sim_threshold: float = 0.70) -> dict:
    """
    CY/PY 센트로이드 코사인 유사도 행렬 기반 Hungarian 매칭(cost=1-sim).
    반환: {py_cluster_id: (cy_cluster_id, sim)} (임계치 미만은 값 None)
    """
    import numpy as np
    py_c = compute_centroids(df_py)
    cy_c = compute_centroids(df_cy)
    if py_c.empty or cy_c.empty:
        return {}
    # build similarity matrix
    py_vecs = list(py_c['vector'].values)
    cy_vecs = list(cy_c['vector'].values)
    S_py = _cosine_sim_matrix(py_vecs)
    S_cy = _cosine_sim_matrix(cy_vecs)
    # We need PY x CY sims; compute directly
    # Efficient: normalize and dot
    import numpy as np
    def _norm(V):
        V = np.asarray([np.asarray(v, dtype=float) for v in V], dtype=float)
        return V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    Npy = _norm(py_vecs)
    Ncy = _norm(cy_vecs)
    sim = Npy @ Ncy.T  # shape: [n_py, n_cy]
    # Hungarian matching on cost = 1 - sim
    try:
        from scipy.optimize import linear_sum_assignment
        cost = 1.0 - sim
        row_ind, col_ind = linear_sum_assignment(cost)
    except Exception:
        # Fallback: greedy matching by highest sim without replacement
        pairs = []
        used_py = set(); used_cy = set()
        # flatten and sort
        flat = [
            (i, j, float(sim[i, j]))
            for i in range(sim.shape[0])
            for j in range(sim.shape[1])
        ]
        flat.sort(key=lambda x: x[2], reverse=True)
        for i, j, s in flat:
            if i in used_py or j in used_cy:
                continue
            pairs.append((i, j))
            used_py.add(i); used_cy.add(j)
        row_ind = np.array([p[0] for p in pairs], dtype=int)
        col_ind = np.array([p[1] for p in pairs], dtype=int)
    mapping: dict[int, tuple[int, float] | None] = {}
    for k in range(len(row_ind)):
        i = int(row_ind[k]); j = int(col_ind[k])
        s = float(sim[i, j])
        py_id = int(py_c.loc[i, 'cluster_id'])
        cy_id = int(cy_c.loc[j, 'cluster_id'])
        if s >= float(sim_threshold):
            mapping[py_id] = (cy_id, s)
        else:
            mapping[py_id] = None
    # Ensure all PY clusters are present in mapping
    for py_id in py_c['cluster_id'].tolist():
        if py_id not in mapping:
            mapping[py_id] = None
    return mapping

