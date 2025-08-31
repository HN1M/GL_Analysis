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

    def name_cluster(self, samples_desc: list[str], samples_vendor: list[str]) -> str | None:
        """거래 샘플을 기반으로 클러스터의 이름을 생성합니다."""
        import textwrap
        if not self._online or self.client is None:
            return None
        prompt = textwrap.dedent(f"""
        너는 회계 감사 보조 AI다. 아래 거래 샘플을 보고 이 그룹의 성격을 가장 잘 드러내는
        한국어 **클러스터 이름 1개만**, 10자 내외로 제시해라.
        숫자/기호/따옴표/접두사 금지. 예: 직원 급여, 통신비, 임원 복지.

        [거래 적요 샘플]
        - {'\n- '.join(samples_desc) if samples_desc else '(샘플 부족)'}

        [주요 거래처 샘플]
        - {'\n- '.join(samples_vendor) if samples_vendor else '(샘플 부족)'}

        정답:
        """).strip()
        try:
            resp = self.client.chat.completions.create(
                model=self.model, messages=[{"role":"user","content":prompt}], temperature=0
            )
            cand = (resp.choices[0].message.content or "").strip().splitlines()[0].strip(" \"'[]()")
            if not cand:
                return None
            bad = {"클러스터","unknown","이름없음","미정","기타"}
            if cand.lower() in bad or cand.startswith("클러스터"):
                return None
            return cand[:20]
        except Exception:
            return None


import json, re
from typing import Dict

LLM_ENABLED = True

CYCLE_DEFS = {
    "CLOSE":"결산·조정, 충당금/평가손익/환산 등 기말조정",
    "REV":"매출·수익 관련",
    "PUR":"매입·구매·비용(광고, 수수료, 임차료 등 포함)",
    "HR":"급여·상여·퇴직·복리후생",
    "TREASURY_INVEST":"예금·금융상품·투자·파생·이자/배당수익",
    "TREASURY_FINANCE":"차입·사채·어음·이자비용",
    "TAX":"부가세·법인세·원천 등 세무",
    "PPE":"유형자산·감가상각·건설중",
    "INTANG":"무형자산·상각",
    "LEASE":"리스·사용권자산·리스부채",
    "OTHER":"기타(정말 분류가 불가할 때만)",
}

def _normalize_cycle_label(label: str) -> str:
    t = str(label).strip().upper()
    aliases = {
        "결산": "CLOSE", "매출": "REV", "매입": "PUR", "매입·비용":"PUR", "비용":"PUR",
        "인사":"HR", "급여":"HR", "복리후생":"HR",
        "자금운용":"TREASURY_INVEST", "투자":"TREASURY_INVEST", "파생":"TREASURY_INVEST",
        "자금조달":"TREASURY_FINANCE", "차입":"TREASURY_FINANCE", "사채":"TREASURY_FINANCE",
        "세무":"TAX", "유형자산":"PPE", "무형자산":"INTANG", "리스":"LEASE",
        "기타":"OTHER"
    }
    return aliases.get(t, t if t in CYCLE_DEFS else "OTHER")

def suggest_cycles_for_accounts(account_names: Dict[str, str]) -> Dict[str, str]:
    if not LLM_ENABLED or not account_names:
        return {}
    try:
        client = LLMClient()
        if not client._online or client.client is None:
            return {}
        def short_defs():
            items = [f"- {k}: {v}" for k, v in CYCLE_DEFS.items() if k != "OTHER"]
            items.append("- OTHER: 위 어떤 범주에도 명확히 속하지 않을 때만")
            return "\n".join(items)
        examples = (
            '{"1000":"현금및현금성자산","1130":"매출채권","5110":"광고선전비","2210":"단기차입금","8100":"법인세비용"}'
        )
        expected = (
            '{"1000":"TREASURY_INVEST","1130":"REV","5110":"PUR","2210":"TREASURY_FINANCE","8100":"TAX"}'
        )
        prompt = (
            "아래 '허용 라벨' 중 하나로 각 계정코드를 분류하세요. 한 항목은 오직 하나의 라벨.\n"
            "'OTHER'는 최후수단입니다. 애매하면 가장 가까운 라벨을 고르세요.\n"
            "반드시 JSON(object)만 출력하세요. 추가 텍스트/설명 금지.\n\n"
            f"허용 라벨(정의):\n{short_defs()}\n\n"
            f"예시 입력:\n{examples}\n\n예시 출력(JSON만):\n{expected}\n\n"
            f"입력(JSON):\n{json.dumps(account_names, ensure_ascii=False)}"
        )
        raw = client.generate("You are an expert accountant.", prompt, force_json=True, max_tokens=2000)
        data = json.loads(raw)
        out: Dict[str, str] = {}
        if isinstance(data, dict):
            for code, lab in data.items():
                out[str(code)] = _normalize_cycle_label(lab)
        return out
    except Exception:
        return {}

