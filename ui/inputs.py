import re
import streamlit as st

# KRW 입력 위젯(천단위 쉼표) - 안정형
# - 사용자는 자유롭게 타이핑(쉼표/공백/문자 섞여도 무시)하고,
#   포커스 아웃/엔터 시에만 정규화(숫자만 유지 → 쉼표 포맷)합니다.
# - 실제 숫자값은 session_state[key] (int)로 보관합니다.
# - 표시용 문자열은 session_state[f"{key}__txt"] 로 관리합니다.

def _parse_krw_text(s: str) -> int:
    """문자열에서 숫자만 추출하여 안전하게 int로 변환(음수 방어, 공란=0)."""
    if s is None:
        return 0
    s = str(s).replace(",", "").strip()
    s = re.sub(r"[^\d]", "", s)  # 숫자 이외 제거
    if s == "":
        return 0
    try:
        return max(0, int(s))
    except Exception:
        return 0

def _fmt_krw(n: int) -> str:
    """정수를 천단위 쉼표 문자열로 포맷."""
    try:
        return f"{int(n):,}"
    except Exception:
        return "0"

def krw_input(label: str, key: str, default_value: int = 0, help_text: str = "") -> int:
    """
    KRW 입력(천단위 쉼표) 통합 위젯.
    - 숫자 상태: st.session_state[key] (int)
    - 표시 상태: st.session_state[f"{key}__txt"] (str, '1,234,567')
    - on_change 시에만 정규화하여 잔고장(500,00 등) 방지
    """
    # 초기 상태 보정
    if key not in st.session_state:
        st.session_state[key] = int(default_value)
    if f"{key}__txt" not in st.session_state:
        st.session_state[f"{key}__txt"] = _fmt_krw(st.session_state[key])

    def _commit():
        """사용자 입력 완료(포커스 아웃/엔터) 시 숫자/문자 상태 동기화."""
        raw = st.session_state.get(f"{key}__txt", "")
        val = _parse_krw_text(raw)
        st.session_state[key] = val
        st.session_state[f"{key}__txt"] = _fmt_krw(val)
        # Streamlit은 on_change 후 자동 rerun → 그래프/표 갱신에 충분

    # 표시 입력창(타이핑 중에는 포맷 강제하지 않음)
    st.text_input(
        label,
        key=f"{key}__txt",
        help=help_text,
        placeholder="예: 500,000,000",
        on_change=_commit,
    )
    return int(st.session_state.get(key, int(default_value)))


