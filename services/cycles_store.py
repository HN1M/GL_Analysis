# services/cycles_store.py
from __future__ import annotations
from typing import Dict, Iterable, List

# 내부 표준 코드(영문) — 저장/로직용
ALLOWED_CYCLES = [
    "CLOSE",            # 결산/조정
    "REV",              # 매출
    "PUR",              # 매입/구매/비용
    "HR",               # 인사(급여/복리후생/퇴직)
    "TREASURY_INVEST",  # 자금운용(예금/금융상품/투자/파생자산/이자·배당수익)
    "TREASURY_FINANCE", # 자금조달(차입금/사채/이자비용)
    "TAX",              # 세무(부가세/법인세/원천 등)
    "PPE",              # 유형자산
    "INTANG",           # 무형자산
    "LEASE",            # 리스(사용권자산/리스부채/리스료)
    "OTHER",            # 기타
]

# 화면용 한글 라벨 매핑
CYCLE_KO: Dict[str, str] = {
    "CLOSE":            "결산",
    "REV":              "매출",
    "PUR":              "매입·비용",
    "HR":               "인사",
    "TREASURY_INVEST":  "자금운용",
    "TREASURY_FINANCE": "자금조달",
    "TAX":              "세무",
    "PPE":              "유형자산",
    "INTANG":           "무형자산",
    "LEASE":            "리스",
    "OTHER":            "기타",
}
KO_TO_CODE = {v: k for k, v in CYCLE_KO.items()}

def code_to_ko(code: str) -> str:
    return CYCLE_KO.get(str(code).upper(), "기타")

def ko_to_code(label: str) -> str:
    return KO_TO_CODE.get(str(label), "OTHER")

# 업로드 단위 저장소(메모리). 필요하면 파일/DB 저장으로 교체 가능.
_MEM: Dict[str, Dict[str, str]] = {}   # {upload_id: {account_code: cycle_name}}

def set_cycles_map(upload_id: str, mapping: Dict[str, str]) -> None:
    """업로드 식별자 별 계정→사이클 매핑 저장."""
    if not upload_id:
        upload_id = "_default"
    cleaned: Dict[str, str] = {}
    for k, v in (mapping or {}).items():
        lab = (str(v).strip().upper() if v is not None else "OTHER")
        cleaned[str(k)] = lab if lab in ALLOWED_CYCLES else "OTHER"
    _MEM[upload_id] = cleaned

def get_cycles_map(upload_id: str) -> Dict[str, str]:
    """해당 업로드의 매핑 조회(없으면 빈 dict)."""
    if not upload_id:
        upload_id = "_default"
    return dict(_MEM.get(upload_id, {}))

def get_effective_cycles(upload_id: str | None = None) -> Dict[str, str]:
    """UI 편의용: 업로드 id 없으면 빈 dict 반환(기본값)."""
    if upload_id and upload_id in _MEM:
        return dict(_MEM[upload_id])
    return {}

def rule_based_guess(code_to_name: Dict[str, str]) -> Dict[str, str]:
    """계정명 규칙기반 사이클 추정(가벼운 기본기능)."""
    KW = {
        "CLOSE":  ["결산", "조정", "대손", "충당금", "평가손실", "평가이익", "외화환산"],
        "REV":    ["매출", "상품매출", "용역수익", "수익", "매출채권", "외상매출"],
        "PUR":    ["매입", "구매", "원재료", "상품매입", "외주", "용역비", "지급수수료", "운반비", "광고선전", "접대", "임차료", "수수료", "수도광열", "통신비"],
        "HR":     ["급여", "임금", "상여", "퇴직", "복리후생", "연금", "식대", "경조", "의료비"],
        "TREASURY_INVEST": ["예금", "CMA", "단기금융상품", "유가증권", "투자", "파생", "이자수익", "배당수익", "금융수익"],
        "TREASURY_FINANCE":["차입금", "대출", "사채", "어음", "이자비용", "금융비용"],
        "TAX":    ["부가세", "법인세", "원천", "지방세", "가산세", "세금과공과"],
        "PPE":    ["유형자산", "건설중", "기계장치", "비품", "차량", "건물", "토지", "감가상각", "처분손익"],
        "INTANG": ["무형자산", "개발비", "소프트웨어", "상표권", "영업권", "상각비"],
        "LEASE":  ["리스", "사용권자산", "리스부채", "리스료"],
    }
    out: Dict[str, str] = {}
    for code, nm in (code_to_name or {}).items():
        name = str(nm or "")
        label = "OTHER"
        for cyc, kws in KW.items():
            if any(kw in name for kw in kws):
                label = cyc
                break
        out[str(code)] = label
    return out

def accounts_for_cycles(mapping: Dict[str, str], cycles: Iterable[str]) -> List[str]:
    """선택한 사이클에 해당하는 계정코드 리스트."""
    want = set(map(str, cycles or []))
    return [code for code, cyc in (mapping or {}).items() if str(cyc) in want]

def accounts_for_cycles_ko(mapping: Dict[str, str], cycles_ko: Iterable[str]) -> List[str]:
    codes = [ko_to_code(x) for x in (cycles_ko or [])]
    return accounts_for_cycles(mapping, codes)

def merge_cycles(base: Dict[str, str], override: Dict[str, str]) -> Dict[str, str]:
    out = dict(base)
    for k, v in (override or {}).items():
        if v:
            out[str(k)] = str(v)
    return out

def build_cycles_preset(upload_id: str, code_to_name: Dict[str, str], use_llm: bool = False) -> Dict[str, str]:
    """룰베이스 1차 → (선택) LLM 보조로 병합 → 저장."""
    base = rule_based_guess(code_to_name)
    if use_llm:
        try:
            from services.llm import suggest_cycles_for_accounts
            llm_map = suggest_cycles_for_accounts(code_to_name)
            base = merge_cycles(base, llm_map)
        except Exception:
            pass
    set_cycles_map(upload_id, base)
    return base

__all__ = [
    "ALLOWED_CYCLES",
    "CYCLE_KO",
    "code_to_ko",
    "ko_to_code",
    "set_cycles_map",
    "get_cycles_map",
    "get_effective_cycles",
    "rule_based_guess",
    "accounts_for_cycles",
    "accounts_for_cycles_ko",
    "merge_cycles",
    "build_cycles_preset",
]
