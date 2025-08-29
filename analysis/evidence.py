from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional


def build_knn_index(prev_df: pd.DataFrame):
    """prev_df['vector']로 KNN 생성."""
    if 'vector' not in prev_df.columns or prev_df['vector'].isna().any():
        raise ValueError("build_knn_index: prev_df에 'vector' 필요")
    from sklearn.neighbors import NearestNeighbors
    X = np.vstack(prev_df['vector'].values)
    knn = NearestNeighbors(metric='cosine', n_neighbors=min(10, len(X))).fit(X)
    return knn, X


def cluster_centroid_vector(cluster_df: pd.DataFrame):
    if 'vector' not in cluster_df.columns or cluster_df.empty:
        return None
    return np.mean(np.vstack(cluster_df['vector'].values), axis=0)


def retrieve_similar_from_previous(prev_df, prev_knn, prev_X, query_vec, topk=5, dedup_by_vendor=True, min_sim=0.7):
    if query_vec is None or prev_X is None or len(prev_X) == 0:
        return pd.DataFrame()
    dist, idx = prev_knn.kneighbors([query_vec], n_neighbors=min(max(10, topk*3), len(prev_X)))
    cands = prev_df.iloc[idx[0]].copy()
    cands['similarity'] = (1 - dist[0])
    cands = cands[cands['similarity'] >= min_sim]
    if dedup_by_vendor and '거래처' in cands.columns:
        cands = cands.sort_values('similarity', ascending=False).drop_duplicates('거래처', keep='first')
    cands = cands.sort_values('similarity', ascending=False).head(topk)
    cols = ['회계일자','계정코드','거래처','적요','발생액','similarity']
    for c in cols:
        if c not in cands.columns: cands[c] = np.nan
    return cands[cols]


def build_cluster_evidence_block(current_df: pd.DataFrame, previous_df: pd.DataFrame,
                                 topk: int = 3, restrict_same_months: bool = True, min_sim: float = 0.7,
                                 dedup_by_vendor: bool = True) -> str:
    if any(col not in current_df.columns for col in ['cluster_id','vector']):
        return "\n\n## 근거 인용(전기 유사 거래)\n- 현재 데이터에 클러스터/벡터가 없어 근거를 생성할 수 없습니다."
    if previous_df.empty or 'vector' not in previous_df.columns:
        return "\n\n## 근거 인용(전기 유사 거래)\n- 전기 데이터 임베딩이 없어 근거를 생성할 수 없습니다."
    def _ok_vec(v):
        return v is not None and isinstance(v, (list, tuple, np.ndarray)) and len(v) > 0
    lines = ["\n\n## 근거 인용(전기 유사 거래)"]
    for cid in sorted(current_df['cluster_id'].unique()):
        cur_c = current_df[current_df['cluster_id'] == cid]
        if cur_c.empty: continue
        cname = cur_c['cluster_name'].iloc[0] if 'cluster_name' in cur_c.columns else str(cid)
        lines.append(f"[클러스터 #{cid} | {cname}]")
        prev_subset = previous_df.copy()
        if restrict_same_months and '회계일자' in cur_c.columns and cur_c['회계일자'].notna().any():
            months = set(cur_c['회계일자'].dt.month.dropna().unique().tolist())
            filtered = previous_df[previous_df['회계일자'].dt.month.isin(months)]
            prev_subset = filtered if not filtered.empty else previous_df
        if 'vector' in prev_subset.columns:
            prev_subset = prev_subset[prev_subset['vector'].apply(_ok_vec)].copy()
        if prev_subset.empty:
            lines.append("    └ 전기 유사 벡터 없음"); continue
        try:
            knn, X = build_knn_index(prev_subset)
        except Exception as e:
            lines.append(f"    └ 인덱스 생성 실패: {e}"); continue
        qv = cluster_centroid_vector(cur_c)
        ev = retrieve_similar_from_previous(prev_subset, knn, X, qv, topk=topk, dedup_by_vendor=dedup_by_vendor, min_sim=min_sim)
        if ev.empty:
            lines.append("    └ 유사 전표: 없음")
        else:
            def _fmt_date(x): 
                try: return x.strftime('%Y-%m-%d') if pd.notna(x) else ""
                except: return ""
            def _fmt_money(x):
                try: return f"{int(x):,}원"
                except: return str(x)
            def _fmt_sim(s):
                try: return f"{float(s):.2f}"
                except: return "N/A"
            for rank, (_, r) in enumerate(ev.sort_values('similarity', ascending=False).iterrows(), 1):
                lines.append(f"    {rank}) {_fmt_date(r['회계일자'])} | {str(r['거래처'])} | {_fmt_money(r['발생액'])} | sim {_fmt_sim(r['similarity'])}")
    return "\n".join(lines)



def build_transaction_evidence_block(current_df, previous_df, topn=10, per_tx_topk=3, min_sim=0.8):
    import numpy as np, pandas as pd
    def _ok_vec(v): return isinstance(v, (list, tuple, np.ndarray)) and len(v)>0
    if current_df.empty or 'vector' not in current_df.columns: 
        return "\n\n## 거래별 근거\n- 현재 데이터에 벡터가 없어 근거를 생성할 수 없습니다."
    if previous_df.empty or 'vector' not in previous_df.columns:
        return "\n\n## 거래별 근거\n- 전기 데이터 임베딩이 없어 근거를 생성할 수 없습니다."

    cur = current_df.copy()
    if 'Z-Score' in cur.columns and cur['Z-Score'].notna().any():
        order_idx = cur['Z-Score'].abs().sort_values(ascending=False).index
    else:
        # Z-Score 미시행 시 발생액 상위
        amt = cur.get('발생액', pd.Series(dtype=float))
        order_idx = amt.sort_values(ascending=False).index
    cur = cur.reindex(order_idx).head(int(topn))

    # 전기 벡터 유효성 필터
    prev = previous_df.copy()
    prev = prev[prev['vector'].apply(_ok_vec)]
    if prev.empty:
        return "\n\n## 거래별 근거\n- 전기 데이터 벡터가 유효하지 않습니다."

    from .evidence import build_knn_index, retrieve_similar_from_previous
    try:
        knn, X = build_knn_index(prev)
    except Exception:
        return "\n\n## 거래별 근거\n- 전기 KNN 인덱스 생성 실패."

    lines = [f"\n\n## 거래별 근거 (상위 {len(cur)}건)"]
    for i, (_, r) in enumerate(cur.iterrows(), 1):
        qv = r.get('vector', None)
        if qv is None: continue
        # 동월 우선
        psub = prev
        if '회계일자' in r and pd.notna(r['회계일자']):
            m = r['회계일자'].month
            cand = prev[prev['회계일자'].dt.month == m]
            if not cand.empty: psub = cand
            knn, X = build_knn_index(psub)
        ev = retrieve_similar_from_previous(psub, knn, X, qv, topk=int(per_tx_topk), dedup_by_vendor=True, min_sim=float(min_sim))
        dt = r['회계일자'].strftime('%Y-%m-%d') if '회계일자' in r and pd.notna(r['회계일자']) else ''
        amt = r.get('발생액', 0.0); z = r.get('Z-Score', np.nan)
        ztxt = f" | Z={z:+.2f}" if not pd.isna(z) else ""
        lines.append(f"[{i}] {dt} | 거래처:{r.get('거래처','')} | 금액:{int(amt):,}원{ztxt}")
        if ev.empty:
            lines.append("    └ 유사 전표: 없음")
        else:
            lines.append(f"    └ 전기 유사 Top {len(ev)}")
            for _, rr in ev.iterrows():
                d2 = rr['회계일자'].strftime('%Y-%m-%d') if pd.notna(rr['회계일자']) else ''
                lines.append(f"       • {d2} | {rr['거래처']} | {int(rr['발생액']):,}원 | sim {rr['similarity']:.2f}")
    return "\n".join(lines)

# --- NEW: Structured evidence blocks for the redesigned context ---
def build_current_cluster_block(current_df: pd.DataFrame) -> str:
    """
    ## 당기 클러스터 및 금액
    - One bullet per cluster_group: total absolute amount, count, and ONE example voucher.
    """
    import pandas as pd
    if current_df.empty or 'cluster_group' not in current_df.columns:
        return "\n\n## 당기 클러스터 및 금액\n- (클러스터 결과 없음)"
    lines = ["\n\n## 당기 클러스터 및 금액"]
    grp = current_df.copy()
    grp['abs_amt'] = grp.get('발생액', pd.Series(dtype=float)).abs()
    for name, cdf in grp.groupby('cluster_group', dropna=False):
        tot = cdf['abs_amt'].sum()
        cnt = len(cdf)
        ex = cdf.sort_values('abs_amt', ascending=False).head(1).iloc[0]
        dt = ex['회계일자'].strftime('%Y-%m-%d') if '회계일자' in ex and pd.notna(ex['회계일자']) else ''
        vend = ex.get('거래처', '')
        amt = int(ex.get('발생액', 0.0))
        lines.append(f"- [{name}] 건수 {cnt}건, 규모(절대값) {tot:,.0f}원")
        lines.append(f"  · 예시: {dt} | {vend} | {amt:,.0f}원")
    return "\n".join(lines)

def build_previous_projection_block(current_df: pd.DataFrame, previous_df: pd.DataFrame, min_sim: float = 0.70) -> str:
    """
    ## 전기 클러스터 및 금액
    Project PY vouchers onto CY cluster centroids; report total abs amount, avg similarity, and ONE example.
    """
    import pandas as pd
    import numpy as np
    from .evidence import build_knn_index, retrieve_similar_from_previous, cluster_centroid_vector
    if current_df.empty or previous_df.empty or 'vector' not in previous_df.columns or 'cluster_group' not in current_df.columns:
        return "\n\n## 전기 클러스터 및 금액\n- (전기 데이터/벡터/클러스터 정보 없음)"
    lines = ["\n\n## 전기 클러스터 및 금액"]
    prev_ok = previous_df[previous_df['vector'].apply(lambda v: isinstance(v, (list, tuple, np.ndarray)) and len(v)>0)]
    if prev_ok.empty:
        return "\n\n## 전기 클러스터 및 금액\n- (전기 유효 벡터 없음)"
    try:
        knn, X = build_knn_index(prev_ok)
    except Exception:
        return "\n\n## 전기 클러스터 및 금액\n- (전기 KNN 인덱스 생성 실패)"
    for name, cur_c in current_df.groupby('cluster_group', dropna=False):
        qv = cluster_centroid_vector(cur_c)
        ev = retrieve_similar_from_previous(prev_ok, knn, X, qv, topk=10, dedup_by_vendor=True, min_sim=float(min_sim))
        if ev.empty:
            lines.append(f"- [{name}] 유사 전표 없음")
            continue
        ev['abs_amt'] = ev.get('발생액', pd.Series(dtype=float)).abs()
        tot = ev['abs_amt'].sum()
        avg_sim = ev['similarity'].mean()
        ex = ev.sort_values('similarity', ascending=False).head(1).iloc[0]
        dt = ex['회계일자'].strftime('%Y-%m-%d') if pd.notna(ex['회계일자']) else ''
        lines.append(f"- [{name}] 규모(절대값) {tot:,.0f}원, 평균 유사도 {avg_sim:.2f}")
        lines.append(f"  · 예시: {dt} | {ex['거래처']} | {int(ex['발생액']):,}원 | sim {ex['similarity']:.2f}")
    return "\n".join(lines)

def build_zscore_top5_block(current_df: pd.DataFrame, previous_df: pd.DataFrame, topn: int = 5, min_sim: float = 0.70) -> str:
    """
    ## Z-score 기준 TOP5 전표
    List top |Z| vouchers with one counterpart from PY (same-month preferred), no row-id.
    """
    import pandas as pd, numpy as np
    from .evidence import build_knn_index, retrieve_similar_from_previous
    if current_df.empty or 'Z-Score' not in current_df.columns:
        return "\n\n## Z-score 기준 TOP5 전표\n- (Z-Score 미계산)"
    cur = current_df.copy()
    order = cur['Z-Score'].abs().sort_values(ascending=False).index
    cur = cur.reindex(order).head(int(topn))
    lines = [f"\n\n## Z-score 기준 TOP5 전표"]
    if previous_df.empty or 'vector' not in previous_df.columns:
        for i, (_, r) in enumerate(cur.iterrows(), 1):
            dt = r['회계일자'].strftime('%Y-%m-%d') if '회계일자' in r and pd.notna(r['회계일자']) else ''
            lines.append(f"- [{i}] {dt} | {r.get('거래처','')} | {int(r.get('발생액',0)):,.0f}원 | Z={float(r.get('Z-Score',0)):+.2f}")
        return "\n".join(lines)
    # KNN on PY (same-month preferred)
    prev = previous_df[previous_df['vector'].apply(lambda v: isinstance(v, (list, tuple, np.ndarray)) and len(v)>0)].copy()
    if prev.empty:
        for i, (_, r) in enumerate(cur.iterrows(), 1):
            dt = r['회계일자'].strftime('%Y-%m-%d') if '회계일자' in r and pd.notna(r['회계일자']) else ''
            lines.append(f"- [{i}] {dt} | {r.get('거래처','')} | {int(r.get('발생액',0)):,.0f}원 | Z={float(r.get('Z-Score',0)):+.2f}")
        return "\n".join(lines)
    knn_all, X_all = build_knn_index(prev)
    for i, (_, r) in enumerate(cur.iterrows(), 1):
        qv = r.get('vector', None)
        dt = r['회계일자'].strftime('%Y-%m-%d') if '회계일자' in r and pd.notna(r['회계일자']) else ''
        head = f"- [{i}] {dt} | {r.get('거래처','')} | {int(r.get('발생액',0)):,.0f}원 | Z={float(r.get('Z-Score',0)):+.2f}"
        if qv is None:
            lines.append(head)
            continue
        psub = prev
        if '회계일자' in r and pd.notna(r['회계일자']):
            m = r['회계일자'].month
            cand = prev[prev['회계일자'].dt.month == m]
            if not cand.empty:
                psub = cand
        try:
            knn, X = build_knn_index(psub)
        except Exception:
            knn, X = knn_all, X_all
        ev = retrieve_similar_from_previous(psub, knn, X, qv, topk=1, dedup_by_vendor=True, min_sim=float(min_sim))
        lines.append(head)
        if ev.empty:
            lines.append("  · 전기 대응: 없음")
        else:
            rr = ev.iloc[0]
            d2 = rr['회계일자'].strftime('%Y-%m-%d') if pd.notna(rr['회계일자']) else ''
            lines.append(f"  · 전기 대응: {d2} | {rr['거래처']} | {int(rr['발생액']):,}원 | sim {rr['similarity']:.2f}")
    return "\n".join(lines)


# --- NEW: 전기 기준 TOP5 블록 ---
def build_zscore_top5_block_for_py(previous_df: pd.DataFrame, current_df: pd.DataFrame, topn: int = 5, min_sim: float = 0.70) -> str:
    """
    ## 전기 Z-score 기준 TOP5 전표
    전기 데이터를 기준으로 |Z| 상위 5건을 나열하고, 가능한 경우 당기 대응 1건을 함께 표시.
    previous_df에 Z-Score가 있어야 한다.
    """
    import pandas as pd, numpy as np
    from .evidence import build_knn_index, retrieve_similar_from_previous

    if previous_df.empty or 'Z-Score' not in previous_df.columns:
        return "\n\n## 전기 Z-score 기준 TOP5 전표\n- (전기 Z-Score 미계산)"

    prev = previous_df.copy()
    order = prev['Z-Score'].abs().sort_values(ascending=False).index
    prev = prev.reindex(order).head(int(topn))

    lines = [f"\n\n## 전기 Z-score 기준 TOP5 전표"]

    if current_df.empty or 'vector' not in current_df.columns:
        for i, (_, r) in enumerate(prev.iterrows(), 1):
            dt = r['회계일자'].strftime('%Y-%m-%d') if '회계일자' in r and pd.notna(r['회계일자']) else ''
            lines.append(f"- [{i}] {dt} | {r.get('거래처','')} | {int(r.get('발생액',0)):,.0f}원 | Z={float(r.get('Z-Score',0)):+.2f}")
        return "\n".join(lines)

    cur_ok = current_df[current_df['vector'].apply(lambda v: isinstance(v, (list, tuple, np.ndarray)) and len(v)>0)].copy()
    if cur_ok.empty:
        for i, (_, r) in enumerate(prev.iterrows(), 1):
            dt = r['회계일자'].strftime('%Y-%m-%d') if '회계일자' in r and pd.notna(r['회계일자']) else ''
            lines.append(f"- [{i}] {dt} | {r.get('거래처','')} | {int(r.get('발생액',0)):,.0f}원 | Z={float(r.get('Z-Score',0)):+.2f}")
        return "\n".join(lines)

    knn_all, X_all = build_knn_index(cur_ok)

    for i, (_, r) in enumerate(prev.iterrows(), 1):
        dt = r['회계일자'].strftime('%Y-%m-%d') if '회계일자' in r and pd.notna(r['회계일자']) else ''
        head = f"- [{i}] {dt} | {r.get('거래처','')} | {int(r.get('발생액',0)):,.0f}원 | Z={float(r.get('Z-Score',0)):+.2f}"

        qv = r.get('vector', None)
        if qv is None:
            lines.append(head); continue

        psub = cur_ok
        if '회계일자' in r and pd.notna(r['회계일자']):
            m = r['회계일자'].month
            cand = cur_ok[cur_ok['회계일자'].dt.month == m]
            if not cand.empty: psub = cand
        try:
            knn, X = build_knn_index(psub)
        except Exception:
            knn, X = knn_all, X_all

        ev = retrieve_similar_from_previous(psub, knn, X, qv, topk=1, dedup_by_vendor=True, min_sim=float(min_sim))
        lines.append(head)
        if ev.empty:
            lines.append("  · 당기 대응: 없음")
        else:
            rr = ev.iloc[0]
            d2 = rr['회계일자'].strftime('%Y-%m-%d') if pd.notna(rr['회계일자']) else ''
            lines.append(f"  · 당기 대응: {d2} | {rr['거래처']} | {int(rr['발생액']):,}원 | sim {rr['similarity']:.2f}")

    return "\n".join(lines)