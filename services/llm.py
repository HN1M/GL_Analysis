"""
# services/llm.py
# - 하이브리드 클라이언트: 키가 있으면 OpenAI 온라인 모드, 없으면 오프라인 스텁
"""

import os
from functools import lru_cache


def openai_available() -> bool:
    try:
        # 부팅 시 env_loader가 LLM_AVAILABLE 플래그를 셋업
        from infra.env_loader import ensure_api_keys_loaded, is_llm_ready
        ensure_api_keys_loaded()
        return bool(is_llm_ready())
    except Exception:
        # 직접 환경변수 확인 (폴백)
        return bool(os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_KEY"))


class _DummyEmbeddings:
    def create(self, *args, **kwargs):
        raise RuntimeError("Embeddings API not available in offline stub.")


class _DummyChat:
    class _Resp:
        class Choice:
            class Msg:
                content = ""
            message = Msg()
        choices = [Choice()]

    def completions(self, *args, **kwargs):
        raise RuntimeError("Chat completions not available in offline stub.")

    def completions_create(self, *args, **kwargs):
        raise RuntimeError("Chat completions not available in offline stub.")


class _DummyClient:
    embeddings = _DummyEmbeddings()
    chat = _DummyChat()


@lru_cache(maxsize=1)
def _openai_client():
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        return None
    # 키는 env_loader가 이미 주입함
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
    try:
        return OpenAI(api_key=api_key) if api_key else OpenAI()
    except Exception:
        return None


class LLMClient:
    def __init__(self, model: str | None = None, temperature: float | None = None, json_mode: bool | None = None):
        self._online = openai_available() and (_openai_client() is not None)
        self.client = _openai_client() if self._online else _DummyClient()
        # 호출 편의 파라미터 저장(online generate에서 사용)
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o")
        try:
            self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.2" if temperature is None else str(temperature)))
        except Exception:
            self.temperature = 0.2
        self.json_mode = True if (os.getenv("LLM_JSON_MODE", "true").lower() in ("1","true","yes")) else False

    def generate(self, system: str, user: str, tools=None, *, model: str | None = None, max_tokens: int | None = None, force_json: bool | None = None) -> str:
        if not self._online or self.client is None:
            raise RuntimeError("LLM not available: no API key or client. Run in offline mode.")
        try:
            use_model = model or self.model
            kwargs = dict(
                model=use_model,
                temperature=float(self.temperature),
                messages=[{"role":"system","content":system},{"role":"user","content":user}],
            )
            if max_tokens is not None:
                kwargs["max_tokens"] = int(max_tokens)
            if tools:
                kwargs["tools"] = tools
                try:
                    first_tool = tools[0]["function"]["name"]
                    if first_tool:
                        kwargs["tool_choice"] = {"type": "function", "function": {"name": first_tool}}
                except Exception:
                    pass
            else:
                use_force_json = self.json_mode if force_json is None else bool(force_json)
                if use_force_json:
                    kwargs["response_format"] = {"type": "json_object"}

            try:
                resp = self.client.chat.completions.create(**kwargs, timeout=60)
            except TypeError:
                resp = self.client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message
            try:
                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    call = tool_calls[0]
                    args = getattr(call.function, "arguments", None)
                    return (args or "").strip()
            except Exception:
                pass
            return (msg.content or "").strip()
        except Exception as e:
            raise


