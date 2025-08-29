from __future__ import annotations
import os, sys
from typing import Optional


def _read_kv_file(path: str) -> dict:
    # 단순 .env 파서 (python-dotenv 없이도 작동)
    data = {}
    if not os.path.exists(path):
        return data
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip("\"'")  # 양쪽 따옴표 제거
            data[k] = v
    return data


def _maybe_import_dotenv():
    try:
        from dotenv import load_dotenv  # type: ignore
        return load_dotenv
    except Exception:
        return None


# 동의어 키 -> 표준 키 정규화
_OPENAI_ALIASES = [
    "OPENAI_API_KEY", "OPENAI_KEY", "OPENAI_TOKEN", "OPENAIAPIKEY", "OPENAI_APIKEY",
    # Azure/OpenAI 변형들 (있으면 그대로도 허용)
    "AZURE_OPENAI_API_KEY",
]


def _normalize_env(d: dict) -> None:
    # 표준 키가 없고, 동의어가 있으면 끌어와서 OPENAI_API_KEY 세팅
    if not d.get("OPENAI_API_KEY"):
        for k in _OPENAI_ALIASES:
            if k in d and d[k]:
                d["OPENAI_API_KEY"] = d[k]
                break


def ensure_api_keys_loaded() -> bool:
    # 1) python-dotenv가 있으면 먼저 시도
    load_dotenv = _maybe_import_dotenv()
    if load_dotenv:
        # 두 파일을 모두 시도(존재하는 것만 적용)
        for p in (".env", "API_KEY.env"):
            try:
                load_dotenv(dotenv_path=p, override=False)
            except Exception:
                pass

    # 2) 수동 파싱 (dotenv가 없거나, 못 읽은 경우 대비)
    merged = {}
    for p in (".env", "API_KEY.env"):
        try:
            merged.update(_read_kv_file(p))
        except Exception:
            pass

    # 3) 동의어 정규화 → OPENAI_API_KEY
    _normalize_env(merged)

    # 4) 환경변수에 반영(존재하지 않는 경우에만 세팅)
    for k, v in merged.items():
        if k not in os.environ and v:
            os.environ[k] = v

    ok = bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("AZURE_OPENAI_API_KEY"))
    # 가시적 플래그
    os.environ["LLM_AVAILABLE"] = "1" if ok else "0"
    return ok


def is_llm_ready() -> bool:
    return os.environ.get("LLM_AVAILABLE", "0") == "1"


def log_llm_status(logger=None):
    # 한국어 상태 로그
    if is_llm_ready():
        msg = "🔌 OpenAI Key 감지: 온라인 LLM 모드로 생성합니다. (클러스터/요약 LLM 사용)"
    else:
        msg = "🔌 OpenAI Key 없음: 오프라인 리포트 모드로 생성합니다. (클러스터/요약 LLM 미사용)"
    if logger:
        try:
            logger.info(msg)
            return
        except Exception:
            pass
    print(msg, file=sys.stderr)


# 앱 부팅 시 사용할 진입점 (import만으로 부팅초기화하고 싶을 때)
def boot():
    ensure_api_keys_loaded()
    log_llm_status()


