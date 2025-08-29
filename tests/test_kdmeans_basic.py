# 간단 단위테스트: KDMeans가 잘 학습되고 속성들이 채워지는지 확인
import numpy as np
from analysis.kdmeans_shim import HDBSCAN

def test_kdmeans_fixed_k():
    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.normal(loc=[0,0], scale=0.1, size=(25,2)),
        rng.normal(loc=[3,3], scale=0.1, size=(25,2)),
    ])
    model = HDBSCAN(n_clusters=2, random_state=0).fit(X)
    assert model.labels_.shape[0] == X.shape[0]
    assert model.chosen_k_ == 2
    assert set(model.labels_) == {0,1}
    assert np.allclose(model.probabilities_, 1.0)

def test_kdmeans_auto_k_runs():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 4))
    model = HDBSCAN(n_clusters=None, random_state=1).fit(X)
    assert model.labels_.shape[0] == X.shape[0]
    assert model.chosen_k_ is not None

