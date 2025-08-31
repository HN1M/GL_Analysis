from __future__ import annotations
from typing import Optional, Callable, List, Dict

# --- Factories: analysis 계층에 넘겨줄 콜백 생성 ---

def make_llm_name_fn(client, model: str = "gpt-4o-mini") -> Callable[[List[str], List[str]], Optional[str]]:
    """(services 전용) 클러스터 이름을 생성하는 콜백을 만들어 반환."""
    def _name_fn(samples_desc: List[str], samples_vendor: List[str]) -> Optional[str]:
        import textwrap
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
            resp = client.chat.completions.create(
                model=model, messages=[{"role":"user","content":prompt}], temperature=0
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
    return _name_fn


def make_synonym_confirm_fn(client, model: str = "gpt-4o-mini") -> Callable[[str, str], bool]:
    """(services 전용) 두 이름이 사실상 같은 의미인지 YES/NO로 확인하는 콜백."""
    def _confirm(a: str, b: str) -> bool:
        try:
            q = (
                "너는 회계 감사 보조 AI다. 다음 두 표현이 '회계 거래 카테고리' 이름으로서 "
                "사실상 같은 의미인지 YES/NO로만 답하라.\n"
                f"A: {a}\nB: {b}\n정답:"
            )
            resp = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": q}], temperature=0
            )
            ans = (resp.choices[0].message.content or "").strip().split()[0].upper()
            return ans.startswith("Y")
        except Exception:
            return False
    return _confirm


def unify_cluster_labels_llm(names: List[str], client, model: str = "gpt-4o-mini") -> Dict[str, str]:
    """
    유사 의미의 한글 클러스터명을 LLM으로 묶어 canonical name 매핑을 리턴.
    입력: ["경비 관리","경비 처리","관리 경비", ...]
    출력: {"경비 처리":"경비 관리", ...}
    """
    uniq = sorted([n for n in set([str(x) for x in names]) if n and n.lower() != 'nan'])
    if not uniq:
        return {}
    prompt = (
        "다음 한국어 클러스터 이름들을 의미가 같은 것끼리 묶어 하나의 대표명으로 통합하세요.\n"
        "규칙: 1) 가장 일반적/짧은 표현을 대표명으로, 2) JSON 객체로만 응답, 3) 형식: {원래명:대표명, ...}.\n"
        f"목록: {uniq}"
    )
    try:
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], temperature=0
        )
        import json
        txt = (resp.choices[0].message.content or "").strip()
        mapping = json.loads(txt)
        if isinstance(mapping, dict):
            return mapping
    except Exception:
        pass
    return {n: n for n in uniq}


