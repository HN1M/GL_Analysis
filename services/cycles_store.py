from __future__ import annotations
import json, os
from typing import Dict, List
from config import STANDARD_ACCOUNTING_CYCLES, CYCLES_USER_OVERRIDES_PATH


def _load_overrides() -> Dict[str, List[str]]:
    p = CYCLES_USER_OVERRIDES_PATH
    if not p or not os.path.exists(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {str(k): [str(x) for x in v] for k, v in data.items() if isinstance(v, list)}
    except Exception:
        # 잘못된 파일/JSON은 무시
        return {}


def get_effective_cycles() -> Dict[str, List[str]]:
    # 프리셋을 복사하고, 동일 키의 항목은 사용자 설정으로 덮어씀
    base = {k: list(v) for k, v in STANDARD_ACCOUNTING_CYCLES.items()}
    ov = _load_overrides()
    for k, v in ov.items():
        base[k] = list(v)
    return base

