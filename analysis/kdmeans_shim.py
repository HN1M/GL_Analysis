from __future__ import annotations
from typing import Optional, Sequence
import numpy as np

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
except Exception as e:
    raise ImportError("scikit-learn이 필요합니다. `pip install scikit-learn`") from e


class HDBSCAN:
    """
    KDMeans: KMeans를 사용하되 HDBSCAN의 최소 속성 인터페이스를 흉내냄.
    - fit(X): labels_, probabilities_ 설정
    - labels_: np.ndarray[int], [0..k-1]
    - probabilities_: np.ndarray[float], 0~1 (KDMeans에서는 전부 1.0로 설정)
    매개변수:
      - n_clusters: 고정 k (None이면 자동 선택)
      - min_cluster_size: k 상한을 계산하기 위한 힌트(너무 많은 군집 방지)
      - max_k: 자동 선택 시 k 상한(기본: 데이터 크기와 min_cluster_size로 유도)
      - k_search: "silhouette" | "heuristic"
      - sample_size: 자동 선택 시 실루엣 계산에 사용할 샘플 크기(기본 2000)
      - random_state: 재현성
      - n_init: KMeans 초기화 횟수(또는 "auto")
    """
    def __init__(
        self,
        n_clusters: Optional[int] = None,
        min_cluster_size: int = 8,
        max_k: Optional[int] = None,
        k_search: str = "silhouette",
        sample_size: int = 2000,
        random_state: int = 42,
        n_init: str | int = "auto",
    ):
        self.n_clusters = n_clusters
        self.min_cluster_size = max(2, int(min_cluster_size))
        self.max_k = max_k
        self.k_search = k_search
        self.sample_size = int(sample_size)
        self.random_state = int(random_state)
        self.n_init = n_init

        # 학습 후 속성(HDBSCAN 호환)
        self.labels_: Optional[np.ndarray] = None
        self.probabilities_: Optional[np.ndarray] = None
        # 추가 텔레메트리
        self.chosen_k_: Optional[int] = None
        self.silhouette_: Optional[float] = None

    # --- 내부: k 후보 산정 ---
    def _candidate_ks(self, n: int) -> Sequence[int]:
        if n < 2:
            return [1]
        base = max(2, int(np.sqrt(n)))
        # 최소 크기 제약 기반 상한
        max_by_min = max(2, n // self.min_cluster_size)
        # 외부 상한 적용
        if self.max_k is not None:
            max_by_min = min(max_by_min, int(self.max_k))
        # 지나치게 큰 k는 계산 비용 이슈 → 실무적으로 캡
        hard_cap = 24 if n >= 1200 else 12
        k_hi = max(2, min(max_by_min, hard_cap))

        ks = {2, 3, 5, base - 1, base, base + 1, int(np.log2(n)) + 1, k_hi}
        ks = {int(k) for k in ks if 2 <= int(k) <= k_hi}
        return sorted(ks)

    # --- 내부: 샘플링 ---
    def _sample(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        if n <= self.sample_size:
            return X
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(n, size=self.sample_size, replace=False)
        return X[idx]

    # --- 내부: k 자동 선택 (실루엣) ---
    def _choose_k(self, X: np.ndarray) -> int:
        n = X.shape[0]
        if n < 2:
            return 1
        if self.n_clusters is not None:
            return max(1, int(self.n_clusters))

        # 후보 목록
        ks = self._candidate_ks(n)
        if len(ks) == 0:
            return max(2, int(np.sqrt(n)))

        if self.k_search != "silhouette":
            # 휴리스틱: √n에 가장 가까운 값
            base = max(2, int(np.sqrt(n)))
            return min(ks, key=lambda k: abs(k - base))

        Xs = self._sample(X)
        best_k, best_s = None, -1.0

        for k in ks:
            if k >= len(Xs):   # 샘플보다 큰 k 불가
                continue
            try:
                km = KMeans(n_clusters=int(k), n_init=self.n_init, random_state=self.random_state)
                labels = km.fit_predict(Xs)
                # 모든 라벨이 하나면 실루엣 계산 불가
                if len(set(labels)) < 2:
                    continue
                s = silhouette_score(Xs, labels, metric="euclidean")
                if s > best_s:
                    best_k, best_s = int(k), float(s)
            except Exception:
                continue

        if best_k is None:
            # 폴백: √n 인근
            base = max(2, int(np.sqrt(n)))
            best_k = min(ks, key=lambda k: abs(k - base))
            best_s = float("nan")

        self.silhouette_ = best_s
        return int(best_k)

    # --- 공개 API ---
    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] < 1:
            raise ValueError("X must be 2D array with at least 1 row")

        k = self._choose_k(X)
        self.chosen_k_ = k

        km = KMeans(n_clusters=int(k), n_init=self.n_init, random_state=self.random_state)
        labels = km.fit_predict(X)

        # HDBSCAN 호환 속성 부여
        self.labels_ = labels.astype(int)
        self.probabilities_ = np.ones(shape=(X.shape[0],), dtype=float)
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_


