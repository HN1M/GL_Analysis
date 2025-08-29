import numpy as np, pandas as pd
from analysis.anomaly import _z_bins_025_sigma


def test_zbins_label_and_count():
    s = pd.Series(np.linspace(-4, 4, 101))
    df, order = _z_bins_025_sigma(s)
    assert len(df) == len(order) == 26      # 테일 포함 26개
    assert int(df["건수"].sum()) == 101     # 총합 보존

