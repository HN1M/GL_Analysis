from __future__ import annotations
from typing import Any, Dict
import pandas as pd
# Assuming contracts are available in the environment
try:
    from analysis.contracts import ModuleResult
except ImportError:
    # Define a simple fallback if contracts are missing
    from typing import NamedTuple, List, Dict
    class ModuleResult(NamedTuple):
        name: str; summary: Dict; tables: Dict; figures: Dict; evidences: List; warnings: List

def wrap_dfs_as_module_result(df_cy: pd.DataFrame, df_py: pd.DataFrame, name: str = "report_ctx") -> ModuleResult:
    """Wraps legacy df_cy/df_py into a standard ModuleResult for unified processing."""
    df_cy = (df_cy.copy() if df_cy is not None else pd.DataFrame())
    df_py = (df_py.copy() if df_py is not None else pd.DataFrame())
    summary: Dict[str, Any] = {
        "rows_cy": int(len(df_cy)),
        "rows_py": int(len(df_py)),
        "accounts_cy": int(df_cy["계정코드"].nunique()) if "계정코드" in df_cy.columns else 0,
        "accounts_py": int(df_py["계정코드"].nunique()) if "계정코드" in df_py.columns else 0,
    }
    return ModuleResult(
        name=name,
        summary=summary,
        tables={"current": df_cy, "previous": df_py},
        figures={},
        evidences=[],   # Adapter only wraps DFs, does not generate new evidences
        warnings=[]
    )
