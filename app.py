# app_v0.17.py (거래처 상세 분석 오류 수정)
# --- BEGIN: LLM 키 부팅 보장 ---
try:
    from infra.env_loader import boot as _llm_boot
    _llm_boot()  # 키 로드 + 상태 로그
except Exception as _e:
    # 최악의 경우에도 앱은 뜨게 하고, 상태를 stderr로만 알림
    import sys
    print(f"[env_loader] 초기화 실패: {_e}", file=sys.stderr)
# --- END: LLM 키 부팅 보장 ---

import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path
import plotly.graph_objects as go
from utils.helpers import find_column_by_keyword, add_provenance_columns, add_period_tag
from analysis.integrity import analyze_reconciliation, run_integrity_module
from analysis.contracts import LedgerFrame, ModuleResult
from analysis.trend import create_monthly_trend_figure, run_trend_module
from analysis.timeseries import (
    run_timeseries_module,          # ← 보고서 탭에서 계속 사용
    create_timeseries_figure        # ← 그래프 렌더 그대로 사용
)
from analysis.ts_v2 import (
    run_timeseries_minimal,
    compute_series_stats,     # ← NEW
    build_anomaly_table,      # ← NEW
    add_future_shading        # ← NEW (시각 음영)
)
from analysis.aggregation import aggregate_monthly, month_end_00
from analysis.anomaly import run_anomaly_module, compute_amount_columns
from analysis.correlation import run_correlation_module
from analysis.vendor import (
    create_pareto_figure,
    create_vendor_heatmap,
    create_vendor_detail_figure,
    run_vendor_module,
)
from analysis.report import run_final_analysis, build_methodology_note
from analysis.embedding import (
    ensure_rich_embedding_text,
    perform_embedding_and_clustering,
    perform_embedding_only,
    unify_cluster_names_with_llm,
)
from analysis.anomaly import calculate_grouped_stats_and_zscore
from services.llm import LLMClient
from services.cache import get_or_embed_texts
import services.cycles_store as cyc
from config import EMB_USE_LARGE_DEFAULT, HDBSCAN_RESCUE_TAU, EMB_MODEL_SMALL
try:
    from config import PM_DEFAULT
except Exception:
    PM_DEFAULT = 500_000_000
from utils.viz import add_materiality_threshold, add_pm_badge
from services.cluster_naming import (
    make_synonym_confirm_fn,
    unify_cluster_labels_llm,
)

# --- KRW 입력(천단위 콤마) 유틸: 콜백 기반으로 안정화 ---
def _krw_input(label: str, key: str, *, default_value: int, help_text: str = "") -> int:
    """
    한국 원화 입력 위젯(천단위 콤마). 핵심 규칙:
    1) 위젯 키(pm_value__txt 등)를 런 루프에서 직접 대입하지 않는다.
    2) 콤마 재포맷은 on_change 콜백 안에서만 수행한다.
    3) 분석에 쓰는 정수 값은 st.session_state[key]에 보관한다.
    """
    txt_key = f"{key}__txt"  # 실제 text_input 위젯이 바인딩되는 키

    # 초기 셋업: 숫자/문자 상태를 위젯 생성 전에 준비
    if key not in st.session_state:
        st.session_state[key] = int(default_value)
    if txt_key not in st.session_state:
        st.session_state[txt_key] = f"{int(st.session_state[key]):,}"

    # 콜백: 포커스 아웃/Enter 시 콤마 포맷을 적용하고 숫자 상태를 동기화
    def _on_blur_format():
        raw_now = st.session_state.get(txt_key, "")
        digits = re.sub(r"[^\d]", "", str(raw_now or ""))
        val = int(digits) if digits else 0
        if val < 0:
            val = 0
        st.session_state[key] = int(val)            # 분석에 쓰는 정수 상태
        st.session_state[txt_key] = f"{int(val):,}"  # 위젯 표시 텍스트(콤마)

    # 위젯 생성
    raw = st.text_input(
        label,
        value=st.session_state[txt_key],
        key=txt_key,
        help=help_text,
        placeholder="예: 500,000,000",
        on_change=_on_blur_format,
    )

    # 라이브 타이핑 동안에도 그래프가 즉시 반영되도록 정수 상태만 업데이트(위젯 키는 건드리지 않음)
    digits_live = re.sub(r"[^\d]", "", str(raw or ""))
    live_val = int(digits_live) if digits_live else 0
    if live_val < 0:
        live_val = 0
    st.session_state[key] = int(live_val)

    return int(st.session_state[key])


# 사이클 프리셋을 계정 선택기로 주입하는 헬퍼
def _apply_cycles_to_picker(*, upload_id: str, cycles_state_key: str, accounts_state_key: str, master_df: pd.DataFrame):
    """선택된 사이클의 계정들을 계정 멀티셀렉트에 합쳐 넣어준다."""
    cycles_map = cyc.get_effective_cycles(upload_id)
    chosen_cycles = st.session_state.get(cycles_state_key, []) or []
    # 지원: KO 라벨 또는 코드 라벨 — 공식 KO 라벨 집합 기준으로 판별
    KO_LABELS = set(cyc.CYCLE_KO.values())
    if chosen_cycles and all(lbl in KO_LABELS for lbl in chosen_cycles):
        codes = cyc.accounts_for_cycles_ko(cycles_map, chosen_cycles)
    else:
        codes = cyc.accounts_for_cycles(cycles_map, chosen_cycles)
    names = (master_df[master_df['계정코드'].astype(str).isin(codes)]['계정명']
                .dropna().astype(str).unique().tolist())
    cur = set(st.session_state.get(accounts_state_key, []) or [])
    st.session_state[accounts_state_key] = sorted(cur.union(names))


# --- NEW: Correlation UI helpers (DRY) ---
from analysis.correlation import run_correlation_module  # 표준(기본) 상관 모듈

def _render_corr_basic_tab(*, upload_id: str):
    """
    기본 상관관계 분석(히트맵/강한 상관쌍/제외계정)을 렌더합니다.
    - 기존 '데이터 무결성 및 흐름' 탭의 구현을 그대로 옮겨, 상관 탭의 '기본' 서브탭에서 사용.
    - state key는 'corr_basic_*' 네임스페이스로 충돌 방지.
    """
    import services.cycles_store as cyc
    mdf = st.session_state.master_df
    acct_names = sorted(mdf['계정명'].dropna().astype(str).unique().tolist())
    st.subheader("계정 간 상관 히트맵(기본)")
    colA, colB = st.columns([2,1])
    with colA:
        picked_accounts = st.multiselect(
            "상관 분석 대상 계정(2개 이상 선택)",
            acct_names,
            default=[],
            help="선택한 계정들 간 월별 흐름의 피어슨 상관을 계산합니다.",
            key="corr_basic_accounts"
        )
    with colB:
        cycles_map_now = cyc.get_effective_cycles(upload_id)
        if cycles_map_now:
            picked_cycles = st.multiselect(
                "사이클 프리셋 선택", list(cyc.CYCLE_KO.values()),
                default=[], key="corr_basic_cycles"
            )
            st.button("➕ 프리셋 적용", key="btn_apply_cycles_corr_basic", on_click=_apply_cycles_to_picker,
                      kwargs=dict(upload_id=upload_id,
                                  cycles_state_key="corr_basic_cycles",
                                  accounts_state_key="corr_basic_accounts",
                                  master_df=st.session_state.master_df))
    corr_thr = st.slider(
        "상관 임계치(강한 상관쌍 표 전용)",
        min_value=0.50, max_value=0.95, step=0.05, value=0.70,
        help="절대값 기준 임계치 이상인 계정쌍만 표에 표시합니다.",
        key="corr_basic_thr"
    )

    if len(picked_accounts) < 2:
        st.info("계정을 **2개 이상** 선택하면 히트맵이 표시됩니다.")
        return

    # 스코프 적용된 LedgerFrame을 재사용
    lf_use = _lf_by_scope()

    # 계정명 → 코드
    codes = (
        mdf[mdf['계정명'].isin(picked_accounts)]['계정코드']
        .astype(str).tolist()
    )
    cmod = run_correlation_module(
        lf_use,
        accounts=codes,
        corr_threshold=float(corr_thr),
        cycles_map=cyc.get_effective_cycles(upload_id),
    )
    _push_module(cmod)
    for w in cmod.warnings:
        st.warning(w)

    # 히트맵(+호버 계정명)
    if 'heatmap' in cmod.figures:
        fig = cmod.figures['heatmap']
        try:
            name_map = dict(zip(
                mdf["계정코드"].astype(str),
                mdf["계정명"].astype(str)
            ))
            tr = fig.data[0]
            x_codes = list(map(str, getattr(tr, 'x', [])))
            y_codes = list(map(str, getattr(tr, 'y', [])))
            x_names = [name_map.get(c, c) for c in x_codes]
            y_names = [name_map.get(c, c) for c in y_codes]
            tr.update(x=x_names, y=y_names)
            fig.update_traces(hovertemplate="계정: %{y} × %{x}<br>상관계수: %{z:.3f}<extra></extra>")
        except Exception:
            pass
        st.plotly_chart(fig, use_container_width=True, key=f"corr_basic_heatmap_{'_'.join(codes)}_{int(corr_thr*100)}")

    # 임계치 이상 상관쌍
    if 'strong_pairs' in cmod.tables and not cmod.tables['strong_pairs'].empty:
        st.markdown("**임계치 이상 상관쌍**")
        st.dataframe(cmod.tables['strong_pairs'], use_container_width=True, height=320)

    # 제외된 계정
    if 'excluded_accounts' in cmod.tables and not cmod.tables['excluded_accounts'].empty:
        with st.expander("제외된 계정 보기(변동없음/활동월 부족)", expanded=False):
            exc = cmod.tables['excluded_accounts'].copy()
            if '계정코드' in exc.columns:
                name_map = dict(zip(
                    mdf["계정코드"].astype(str),
                    mdf["계정명"].astype(str)
                ))
                exc['계정코드'] = exc['계정코드'].astype(str)
                exc['계정명'] = exc['계정코드'].map(name_map)
                cols = ['계정명', '계정코드'] + [c for c in exc.columns if c not in ('계정명','계정코드')]
                exc = exc[cols]
            st.dataframe(exc, use_container_width=True)


def _render_corr_advanced_tab(*, upload_id: str):
    """
    고급 상관관계 분석(방법/시차/롤링 안정성 등)을 렌더합니다.
    - 기존 '상관관계(고급)' 탭의 코드를 서브탭용 함수로 모듈화.
    - state key는 기존 'corr_adv_*' 유지(호환).
    """
    import services.cycles_store as cyc
    st.subheader("고급 상관관계")
    lf_adv = _lf_by_scope()  # 스코프 일관성 유지
    if lf_adv is None:
        st.info("원장을 먼저 업로드해 주세요.")
        return

    mdf_adv = st.session_state.master_df
    acct_names_adv = sorted(mdf_adv['계정명'].dropna().astype(str).unique().tolist())
    colA, colB = st.columns(2)
    with colA:
        picked_accounts_adv = st.multiselect("분석 계정(다중 선택)", options=acct_names_adv, key="corr_adv_accounts")
    with colB:
        picked_cycles_adv = st.multiselect("사이클 프리셋(선택 시 계정 자동 반영)", options=list(cyc.CYCLE_KO.values()), key="corr_adv_cycles")
        if st.button("프리셋 적용", key="btn_apply_preset_corr_adv"):
            mapping = cyc.get_effective_cycles(upload_id)
            codes = cyc.accounts_for_cycles_ko(mapping, picked_cycles_adv)
            code_to_name = (
                mdf_adv[['계정코드','계정명']].assign(계정코드=lambda d: d['계정코드'].astype(str)).drop_duplicates()
                    .set_index('계정코드')['계정명'].astype(str).to_dict()
            )
            cur_set = set(st.session_state.get("corr_adv_accounts", []))
            cur_set.update({code_to_name.get(c, c) for c in codes})
            st.session_state["corr_adv_accounts"] = sorted(cur_set)

    method = st.selectbox("상관 방식", ["pearson", "spearman", "kendall"], index=0, key="corr_adv_method")
    corr_threshold = st.slider("임계치(|r|)", 0.1, 0.95, 0.70, 0.05, key="corr_adv_thr")
    c1, c2 = st.columns(2)
    with c1:
        max_lag = st.slider("최대 시차(개월)", 0, 12, 6, 1, key="corr_adv_maxlag")
    with c2:
        rolling_window = st.slider("롤링 윈도우(개월)", 3, 24, 6, 1, key="corr_adv_rollwin")

    if st.button("분석 실행", key="run_corr_adv"):
        try:
            from analysis.corr_advanced import run_corr_advanced as run_corr_adv
            # ✅ UI(계정명) → 코드 변환
            _names = st.session_state.get("corr_adv_accounts", picked_accounts_adv) or []
            _codes = (
                mdf_adv[mdf_adv['계정명'].isin(_names)]['계정코드']
                .astype(str).drop_duplicates().tolist()
            )
            mr = run_corr_adv(
                lf_adv,
                _codes,
                method=st.session_state.get("corr_adv_method", "pearson"),
                corr_threshold=float(st.session_state.get("corr_adv_thr", 0.70)),
                max_lag=int(st.session_state.get("corr_adv_maxlag", 6)),
                rolling_window=int(st.session_state.get("corr_adv_rollwin", 6)),
            )
            st.subheader("히트맵")
            if "heatmap" in mr.figures:
                st.plotly_chart(mr.figures["heatmap"], use_container_width=True)
            if "strong_pairs" in mr.tables:
                st.subheader("임계치 이상 상관쌍")
                st.dataframe(mr.tables["strong_pairs"], use_container_width=True)
            if "lagged_pairs" in mr.tables:
                st.subheader("최적 시차 상관(Top)")
                st.dataframe(mr.tables["lagged_pairs"], use_container_width=True)
            if "rolling_stability" in mr.tables:
                st.subheader("롤링 안정성(변동성 낮은 순)")
                st.dataframe(mr.tables["rolling_stability"], use_container_width=True)
        except Exception as _e:
            st.warning(f"고급 상관 분석 실패: {_e}")


# --- 3. UI 부분 ---
st.set_page_config(page_title="AI 분석 시스템 v0.18", layout="wide")
st.title("훈's GL분석 시스템")
st.markdown("---")

for key in ['mapping_confirmed', 'analysis_done']:
    if key not in st.session_state:
        st.session_state[key] = False

# --- NEW: 모듈 결과 수집용 컨테이너 ---
if 'modules' not in st.session_state:
    st.session_state['modules'] = {}

def _push_module(mod: ModuleResult):
    """ModuleResult를 세션에 수집(동명 모듈은 최신으로 교체)."""
    try:
        if mod and getattr(mod, "name", None):
            st.session_state['modules'][str(mod.name)] = mod
    except Exception:
        pass


# (removed) number_input 기반 대체 구현: 쉼표 미표시·키 충돌 유발 가능성 → 단일 구현으로 통일

with st.sidebar:
    st.header("1. 데이터 준비")
    uploaded_file = st.file_uploader("분석할 엑셀 파일을 올려주세요.", type=["xlsx", "xlsm"])
    if 'last_file' not in st.session_state or st.session_state.last_file != uploaded_file:
        st.session_state.mapping_confirmed = False
        st.session_state.analysis_done = False
        st.session_state.last_file = uploaded_file

    st.markdown("---")
    st.header("2. 분석 기간")
    default_scope = st.session_state.get("period_scope", "당기")
    st.session_state.period_scope = st.radio(
        "분석 스코프(트렌드 제외):",
        options=["당기", "당기+전기"],
        index=["당기","당기+전기"].index(default_scope),
        horizontal=True,
        help="상관/거래처/이상치 모듈에 적용됩니다. 트렌드는 설계상 CY vs PY 비교 유지."
    )
    st.markdown("---")
    st.header("3. Embedding / Clustering")
    st.session_state.use_large_embedding = st.toggle(
        "Use Large Embedding (cost ↑)",
        value=st.session_state.get("use_large_embedding", EMB_USE_LARGE_DEFAULT),
        help="Large model improves semantics but is slower and more expensive."
    )
    st.session_state.rescue_tau = st.slider(
        "Noise rescue τ (cosine)",
        min_value=0.60, max_value=0.90, step=0.01,
        value=float(st.session_state.get("rescue_tau", HDBSCAN_RESCUE_TAU)),
        help="Reassign -1 (noise) to nearest cluster if similarity ≥ τ."
    )
    st.markdown("---")
    st.header("4. Materiality")
    pm_val = _krw_input(
        "Performance Materiality (KRW)",
        key="pm_value",
        default_value=PM_DEFAULT,
        help_text="Used for KIT (PM exceed) and integrated risk scoring."
    )
    st.caption("ⓘ The PM threshold is drawn as a red dotted line on applicable charts. "
               "Y-axis scaling may change to accommodate this line.")

    # 🧹 캐시 관리
    with st.expander("🧹 캐시 관리", expanded=False):
        if st.button("임베딩 캐시 비우기"):
            import shutil
            from services.cache import _model_dir
            for m in ["text-embedding-3-small", "text-embedding-3-large"]:
                try:
                    shutil.rmtree(_model_dir(m), ignore_errors=True)
                except Exception as e:
                    st.warning(f"{m} 삭제 실패: {e}")
            st.success("임베딩 캐시 삭제 완료")

        if st.button("데이터 캐시 비우기"):
            st.cache_data.clear()
            st.success("Streamlit 데이터 캐시 삭제 완료")

        if st.button("캐시 정보 보기"):
            from services.cache import get_cache_info
            try:
                st.write(get_cache_info("text-embedding-3-small"))
                st.write(get_cache_info("text-embedding-3-large"))
            except Exception as e:
                st.info(f"정보 조회 실패: {e}")


@st.cache_data(show_spinner=False)
def _read_excel(_file, sheet_name=None):
    return pd.read_excel(_file, sheet_name=sheet_name)


@st.cache_data(show_spinner=False)
def _read_xls(_file):
    # pickle 직렬화 가능한 타입만 캐시 → 시트명 리스트로 반환
    return pd.ExcelFile(_file).sheet_names

# (removed duplicated definition) _krw_input — 위의 단일 버전만 유지

def _apply_scope(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    """스코프 적용 시 결측 컬럼 방어: 'period_tag' 미존재면 원본 반환.
    df.get('period_tag','')가 문자열을 반환할 경우 .eq 호출 AttributeError를 방지한다.
    """
    if df is None or df.empty or 'period_tag' not in df.columns:
        return df
    if scope == "당기":
        return df[df['period_tag'].eq('CY')]
    if scope == "당기+전기":
        return df[df['period_tag'].isin(['CY', 'PY'])]
    return df

# === 공용: 표 높이 자동 제한(행 수 기반) ===
def _auto_table_height(df: pd.DataFrame, max_rows: int = 8,
                       row_px: int = 28, header_px: int = 38, pad_px: int = 12) -> int:
    """
    표 높이를 '표시 행 수' 기준으로 계산해 넘기기 위한 유틸.
    - max_rows: 최대 표시 행수
    - 실패 시 300px로 폴백
    """
    try:
        n = int(min(max(len(df), 1), max_rows))
        return int(header_px + n * row_px + pad_px)
    except Exception:
        return 300

def _lf_by_scope() -> LedgerFrame:
    """상관/거래처/이상치에서 사용할 스코프 적용 LedgerFrame."""
    hist = st.session_state.get('lf_hist')
    scope = st.session_state.get('period_scope', '당기')
    if hist is None:
        return None
    return LedgerFrame(df=_apply_scope(hist.df, scope), meta=hist.meta)

# (removed) 구버전 텍스트입력 + ±step / ✖reset 변형들 — 사용자 요청으로 버튼류 삭제 및 단일화


if uploaded_file is not None:
    if not st.session_state.mapping_confirmed:
        # ... 컬럼 매핑 UI ...
        try:
            st.info("2단계: 엑셀의 컬럼을 분석 표준 필드에 맞게 지정해주세요.")
            sheet_names = _read_xls(uploaded_file)
            first_ledger_sheet = next((s for s in sheet_names if 'ledger' in s.lower()), None)
            if first_ledger_sheet is None:
                st.error("오류: 'Ledger' 시트를 찾을 수 없습니다.")
                st.stop()
            ledger_cols = _read_excel(uploaded_file, sheet_name=first_ledger_sheet).columns.tolist()
            ledger_map = {}
            st.markdown("#### **Ledger 시트** 항목 매핑")
            cols = st.columns(6)
            ledger_fields = {'회계일자': '일자', '계정코드': '계정코드', '거래처': '거래처', '적요': '적요', '차변': '차변', '대변': '대변'}
            for i, (key, keyword) in enumerate(ledger_fields.items()):
                with cols[i]:
                    is_optional = key == '거래처'
                    default_col = find_column_by_keyword(ledger_cols, keyword)
                    options = ['선택 안 함'] + ledger_cols if is_optional else ledger_cols
                    default_index = options.index(default_col) if default_col in options else 0
                    ledger_map[key] = st.selectbox(f"**'{key}'** 필드 선택", options, index=default_index, key=f"map_ledger_{key}")
            st.markdown("---")
            st.markdown("#### **Master 시트** 항목 매핑")
            master_cols = _read_excel(uploaded_file, sheet_name='Master').columns.tolist()
            master_map = {}
            cols = st.columns(7)
            master_fields = {'계정코드': '계정코드', '계정명': '계정명', 'BS/PL': 'BS/PL', '차변/대변': '차변/대변', '당기말잔액': '당기말', '전기말잔액': '전기말', '전전기말잔액': '전전기말'}
            for i, (key, keyword) in enumerate(master_fields.items()):
                with cols[i]:
                    default_col = find_column_by_keyword(master_cols, keyword)
                    default_index = master_cols.index(default_col) if default_col in master_cols else 0
                    master_map[key] = st.selectbox(f"**'{key}'** 필드 선택", master_cols, index=default_index, key=f"map_master_{key}")
            if st.button("✅ 매핑 확인 및 데이터 처리", type="primary"):
                st.session_state.ledger_map = ledger_map
                st.session_state.master_map = master_map
                st.session_state.mapping_confirmed = True
                st.rerun()
        except Exception as e:
            st.error(f"엑셀 파일의 컬럼을 읽는 중 오류가 발생했습니다: {e}")

    else:  # 매핑 확인 후
        try:
            ledger_map, master_map = st.session_state.ledger_map, st.session_state.master_map
            master_df = _read_excel(uploaded_file, sheet_name='Master')
            sheet_names = _read_xls(uploaded_file)
            ledger_sheets = [s for s in sheet_names if 'ledger' in s.lower()]
            if not ledger_sheets:
                st.error("오류: 'Ledger' 시트를 찾을 수 없습니다.")
                st.stop()
            all_parts = []
            for s in ledger_sheets:
                part = _read_excel(uploaded_file, sheet_name=s)
                part['source_sheet'] = s
                part = add_provenance_columns(part)
                all_parts.append(part)
            ledger_df = pd.concat(all_parts, ignore_index=True)
            # row_id: 파일명|시트:행  (세션/재실행에도 안정)
            try:
                base = Path(getattr(uploaded_file, "name", "uploaded.xlsx")).stem
                if 'row_id' in ledger_df.columns:
                    ledger_df['row_id'] = base + "|" + ledger_df['row_id'].astype(str)
            except Exception:
                pass
            ledger_df.rename(columns={v: k for k, v in ledger_map.items() if v != '선택 안 함'}, inplace=True)
            master_df.rename(columns={v: k for k, v in master_map.items()}, inplace=True)

            # 🔧 병합 전에 타입/포맷을 먼저 통일
            for df_ in [ledger_df, master_df]:
                if '계정코드' in df_.columns:
                    df_['계정코드'] = (
                        df_['계정코드']
                        .astype(str)
                        .str.replace(r'\.0$', '', regex=True)
                        .str.strip()
                    )

            master_essentials = master_df[['계정코드', '계정명']].drop_duplicates()
            ledger_df = pd.merge(ledger_df, master_essentials, on='계정코드', how='left')
            ledger_df['계정명'] = ledger_df['계정명'].fillna('미지정 계정')

            ledger_df['회계일자'] = pd.to_datetime(ledger_df['회계일자'], errors='coerce')
            ledger_df.dropna(subset=['회계일자'], inplace=True)
            for col in ['차변', '대변']:
                ledger_df[col] = pd.to_numeric(ledger_df[col], errors='coerce').fillna(0)
            for col in ['당기말잔액', '전기말잔액', '전전기말잔액']:
                if col in master_df.columns:
                    master_df[col] = pd.to_numeric(master_df[col], errors='coerce').fillna(0)
                else:
                    master_df[col] = 0
            ledger_df['거래금액'] = ledger_df['차변'] - ledger_df['대변']
            ledger_df['거래금액_절대값'] = abs(ledger_df['거래금액'])
            ledger_df['연도'] = ledger_df['회계일자'].dt.year
            ledger_df['월'] = ledger_df['회계일자'].dt.month
            # ✅ 분석 규칙: 계정 서브셋 분석 시에도 전체 히스토리를 확보하기 위한 편의 파생
            ledger_df['연월'] = ledger_df['회계일자'].dt.to_period('M').astype(str)
            # ✅ period_tag 추가(CY/PY/Other)
            ledger_df = add_period_tag(ledger_df)
            if '거래처' not in ledger_df.columns:
                ledger_df['거래처'] = '정보 없음'
            ledger_df['거래처'] = ledger_df['거래처'].fillna('정보 없음').astype(str)

            if st.button("🚀 전체 분석 실행", type="primary"):
                with st.spinner('데이터를 분석 중입니다...'):
                    # ✅ 정합성은 사용자 기간 선택과 무관하게 전체 기준으로 계산
                    st.session_state.recon_status, st.session_state.recon_df = analyze_reconciliation(ledger_df, master_df)
                    # ✅ 표준 LedgerFrame 구성(정합성은 항상 전체 기준: DF_hist)
                    lf_hist = LedgerFrame(df=ledger_df, meta={
                        "file_name": getattr(uploaded_file, "name", "uploaded.xlsx"),
                        "master_df": master_df,
                    })
                    # 초기엔 focus=hist (후속 단계에서 사용자 필터 연결)
                    lf_focus = lf_hist

                    st.session_state.master_df = master_df
                    st.session_state.ledger_df = ledger_df
                    st.session_state.lf_hist = lf_hist
                    st.session_state.lf_focus = lf_focus
                    st.session_state.analysis_done = True
                st.rerun()

            if st.session_state.analysis_done:
                st.success("✅ 분석이 완료되었습니다. 아래 탭에서 결과를 확인하세요.")
                # --- 계정→사이클 매핑 검토/수정 ---
                upload_id = getattr(uploaded_file, "name", "uploaded.xlsx")
                # 업로드 직후 1회: 프리셋 없으면 룰베이스 자동 생성
                names_dict = (
                    master_df[['계정코드','계정명']]
                        .drop_duplicates()
                        .assign(계정코드=lambda d: d['계정코드'].astype(str))
                        .set_index('계정코드')['계정명'].astype(str).to_dict()
                )
                if not cyc.get_effective_cycles(upload_id):
                    cyc.build_cycles_preset(upload_id, names_dict, use_llm=False)

                with st.expander("🧭 계정 → 사이클 매핑 검토/수정", expanded=False):
                    cur_map = cyc.get_effective_cycles(upload_id)
                    map_df = master_df[['계정코드','계정명']].drop_duplicates().copy()
                    map_df['계정코드'] = map_df['계정코드'].astype(str)
                    map_df['사이클(표시)'] = map_df['계정코드'].map(lambda c: cyc.code_to_ko(cur_map.get(c, 'Other')))

                    st.caption("사이클 라벨을 수정한 뒤 저장을 누르세요. (표시는 한글, 내부는 코드로 저장됩니다)")
                    edited = st.data_editor(
                        map_df, hide_index=True, use_container_width=True,
                        column_config={
                            "사이클(표시)": st.column_config.SelectboxColumn(
                                options=list(cyc.CYCLE_KO.values()), required=True
                            )
                        }
                    )

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        if st.button("💾 매핑 저장", type="primary", key="btn_save_cycles_map"):
                            new_map_codes = {
                                str(r['계정코드']): cyc.ko_to_code(r['사이클(표시)'])
                                for _, r in edited.iterrows()
                            }
                            cyc.set_cycles_map(upload_id, new_map_codes)
                            st.success(f"저장됨: {len(new_map_codes):,}개 계정")
                    with c2:
                        if st.button("🤖 LLM 추천 병합", help="룰베이스 결과 위에 LLM 제안을 덮어씌웁니다", key="btn_merge_llm_cycles"):
                            cyc.build_cycles_preset(upload_id, names_dict, use_llm=True)
                            st.success("LLM 추천을 병합했습니다.")
                            st.rerun()
                    with c3:
                        if st.button("↺ 룰베이스로 초기화", key="btn_reset_rule_cycles"):
                            cyc.build_cycles_preset(upload_id, names_dict, use_llm=False)
                            st.success("룰베이스로 초기화했습니다.")
                            st.rerun()
                with st.expander("🔍 빠른 진단(데이터 품질 체크)", expanded=False):
                    df = st.session_state.ledger_df.copy()
                    issues = []

                    invalid_date = int(df['회계일자'].isna().sum())
                    if invalid_date > 0:
                        issues.append(f"❗ 유효하지 않은 날짜(NaT): {invalid_date:,}건")

                    if '거래처' in df.columns:
                        missing_vendor = int((df['거래처'].isna() | (df['거래처'] == '정보 없음')).sum())
                        if missing_vendor > 0:
                            issues.append(f"ℹ️ 거래처 정보 없음/결측: {missing_vendor:,}건")

                    zero_abs = int((df['거래금액_절대값'] == 0).sum())
                    issues.append(f"ℹ️ 금액 절대값이 0인 전표: {zero_abs:,}건")

                    unlinked = int(df['계정명'].eq('미지정 계정').sum())
                    if unlinked > 0:
                        issues.append(f"❗ Master와 매칭되지 않은 전표(계정명 미지정): {unlinked:,}건")

                    st.write("**체크 결과**")
                    if issues:
                        for line in issues:
                            st.write("- " + line)
                    else:
                        st.success("문제 없이 깔끔합니다!")
                tab_integrity, tab_vendor, tab_anomaly, tab_ts, tab_report, tab_corr = st.tabs(["🌊 데이터 무결성 및 흐름", "🏢 거래처 심층 분석", "🔬 이상 패턴 탐지", "📉 시계열 예측", "🧠 분석 종합 대시보드", "📊 상관관계"])

                # (이전 버전) 대시보드 탭은 사용자 요청으로 제거됨
                with tab_integrity:  # ...
                    st.header("데이터 무결성 및 흐름")
                    st.caption(f"🔎 현재 스코프: {st.session_state.get('period_scope','당기')}")
                    st.subheader("1. 데이터 정합성 검증 결과")
                    mod = st.session_state.get('modules', {}).get('integrity')
                    status = (getattr(mod, 'summary', {}) or {}).get('overall_status', 'Pass') if mod else 'Pass'
                    result_df = (getattr(mod, 'tables', {}) or {}).get('reconciliation') if mod else st.session_state.get('recon_df')
                    if status == "Pass":
                        st.success("✅ 모든 계정의 데이터가 일치합니다.")
                    elif status == "Warning":
                        st.warning("⚠️ 일부 계정에서 사소한 차이가 발견되었습니다.")
                    else:
                        st.error("🚨 일부 계정에서 중대한 차이가 발견되었습니다.")

                    def highlight_status(row):
                        if row.상태 == 'Fail':
                            return ['background-color: #ffcccc'] * len(row)
                        elif row.상태 == 'Warning':
                            return ['background-color: #fff0cc'] * len(row)
                        return [''] * len(row)

                    format_dict = {col: '{:,.0f}' for col in result_df.select_dtypes(include=np.number).columns}
                    st.dataframe(result_df.style.apply(highlight_status, axis=1).format(format_dict), use_container_width=True)
                    st.markdown("---")
                    st.subheader("2. 계정별 월별 추이 (PY vs CY)")
                    # ✅ 자동 추천 제거: 사용자가 계정을 선택한 경우에만 그래프 렌더
                    account_list = st.session_state.master_df['계정명'].unique()
                    selected_accounts = st.multiselect(
                        "분석할 계정을 선택하세요 (1개 이상 필수)",
                        account_list, default=[],
                        key="trend_accounts_pick"
                    )
                    # ▼ 사이클 프리셋(선택 시 위 멀티셀렉트에 계정 자동 반영)
                    cycles_map_now = cyc.get_effective_cycles(upload_id)
                    if cycles_map_now:
                        picked_cycles = st.multiselect(
                            "사이클 프리셋 선택(선택하면 위 계정 목록에 자동 반영)",
                            list(cyc.CYCLE_KO.values()),
                            default=[], key="trend_cycles_pick"
                        )
                        st.button("➕ 프리셋 적용", key="btn_apply_cycles_trend", on_click=_apply_cycles_to_picker,
                                  kwargs=dict(upload_id=upload_id,
                                              cycles_state_key="trend_cycles_pick",
                                              accounts_state_key="trend_accounts_pick",
                                              master_df=st.session_state.master_df))
                    if not selected_accounts:
                        st.info("계정을 1개 이상 선택하면 월별 추이 그래프가 표시됩니다.")
                    else:
                        lf_use = st.session_state.get('lf_focus') or st.session_state.get('lf_hist')
                        # 선택된 계정명을 계정코드로 변환
                        mdf = st.session_state.master_df
                        accounts_codes = (
                            mdf[mdf['계정명'].isin(selected_accounts)]['계정코드']
                            .astype(str)
                            .tolist()
                        )
                        mod = run_trend_module(lf_use, accounts=accounts_codes)
                        _push_module(mod)
                        for w in mod.warnings:
                            st.warning(w)
                        if mod.figures:
                            for title, fig in mod.figures.items():
                                # PM 임계선(항상 표시; 범위 밖이면 자동 확장)
                                st.plotly_chart(
                                    add_materiality_threshold(fig, float(st.session_state.get("pm_value", PM_DEFAULT))),
                                    use_container_width=True,
                                    key=f"trend_{title}"
                                )
                        else:
                            st.info("표시할 추이 그래프가 없습니다.")

                    st.markdown("---")
                    st.subheader("3. 계정 간 상관 히트맵")
                    st.info("이 기능은 상단의 **📊 상관관계 → '기본' 서브탭**으로 이동했습니다.")

                with tab_vendor:
                    st.header("거래처 심층 분석")
                    st.caption(f"🔎 현재 스코프: {st.session_state.get('period_scope','당기')}")

                    st.subheader("거래처 집중도 및 활동성 (계정별)")
                    master_df_res = st.session_state.master_df
                    account_list_vendor = master_df_res['계정명'].unique()
                    selected_accounts_vendor = st.multiselect("분석할 계정(들)을 선택하세요.", account_list_vendor, default=[], key="vendor_accounts_pick")
                    cycles_map_now = cyc.get_effective_cycles(upload_id)
                    if cycles_map_now:
                        picked_cycles_vendor = st.multiselect(
                            "사이클 프리셋 선택", list(cyc.CYCLE_KO.values()),
                            default=[], key="vendor_cycles_pick"
                        )
                        st.button("➕ 프리셋 적용", key="btn_apply_cycles_vendor", on_click=_apply_cycles_to_picker,
                                  kwargs=dict(upload_id=upload_id,
                                              cycles_state_key="vendor_cycles_pick",
                                              accounts_state_key="vendor_accounts_pick",
                                              master_df=st.session_state.master_df))

                    # 🔧 최소 거래금액(연간, CY) 필터 — KRW 입력(커밋 시 쉼표 정규화)
                    min_amount_vendor = _krw_input(
                        "최소 거래금액(연간, CY) 필터",
                        key="vendor_min_amount",
                        default_value=0,
                        help_text="CY 기준 거래금액 합계가 이 값 미만인 거래처는 '기타'로 합산됩니다."
                    )
                    include_others_vendor = st.checkbox("나머지는 '기타'로 합산", value=True)

                    if selected_accounts_vendor:
                        selected_codes = (
                            master_df_res[master_df_res['계정명'].isin(selected_accounts_vendor)]['계정코드']
                            .astype(str)
                            .tolist()
                        )
                        lf_use = _lf_by_scope()
                        vmod = run_vendor_module(
                            lf_use,
                            account_codes=selected_codes,
                            min_amount=float(min_amount_vendor),
                            include_others=bool(include_others_vendor),
                        )
                        _push_module(vmod)
                        if vmod.figures:
                            col1, col2 = st.columns(2)
                            with col1:
                                if 'pareto' in vmod.figures:
                                    figp = vmod.figures['pareto']
                                    figp = add_materiality_threshold(figp, float(st.session_state.get("pm_value", PM_DEFAULT)))
                                    st.plotly_chart(figp, use_container_width=True, key=f"vendor_pareto_{'_'.join(selected_accounts_vendor) or 'all'}")
                            with col2:
                                if 'heatmap' in vmod.figures:
                                    figh = add_pm_badge(vmod.figures['heatmap'], float(st.session_state.get("pm_value", PM_DEFAULT)))
                                    st.plotly_chart(figh, use_container_width=True, key=f"vendor_heatmap_{'_'.join(selected_accounts_vendor) or 'all'}")
                        else:
                            st.warning("선택하신 계정에는 분석할 거래처 데이터가 부족합니다.")
                        for w in vmod.warnings:
                            st.warning(w)
                    else:
                        st.info("계정을 선택하면 해당 계정의 거래처 집중도 및 활동성 분석을 볼 수 있습니다.")

                    st.markdown("---")
                    st.subheader("거래처별 세부 분석 (전체 계정)")
                    full_ledger_df = st.session_state.ledger_df
                    vendor_list = sorted(full_ledger_df[full_ledger_df['거래처'] != '정보 없음']['거래처'].unique())

                    if len(vendor_list) > 0:
                        options = ['선택하세요...'] + vendor_list
                        selected_vendor = st.selectbox("상세 분석할 거래처를 선택하세요.", options, index=0)

                        if selected_vendor != '선택하세요...':
                            all_months_in_data = pd.period_range(
                                start=full_ledger_df['회계일자'].min(),
                                end=full_ledger_df['회계일자'].max(),
                                freq='M'
                            ).strftime('%Y-%m').tolist()
                            detail_fig = create_vendor_detail_figure(full_ledger_df, selected_vendor, all_months_in_data)
                            # PM line on vendor detail (stacked bars)
                            try:
                                detail_fig = add_materiality_threshold(detail_fig, float(st.session_state.get("pm_value", PM_DEFAULT)))
                            except Exception:
                                pass
                            if detail_fig:
                                st.plotly_chart(detail_fig, use_container_width=True, key=f"vendor_detail_{selected_vendor}")
                    else:
                        st.info("분석할 거래처 데이터가 없습니다.")

                with tab_anomaly:
                    st.header("이상 패턴 탐지")
                    st.caption(f"🔎 현재 스코프: {st.session_state.get('period_scope','당기')}")
                    mdf = st.session_state.master_df
                    acct_names = mdf['계정명'].unique()
                    pick = st.multiselect("대상 계정 선택(미선택 시 자동 추천)", acct_names, default=[])
                    topn = st.slider("표시 개수(상위 |Z|)", min_value=10, max_value=500, value=20, step=10)
                    if st.button("이상치 분석 실행"):
                        lf_use = _lf_by_scope()
                        codes = None
                        if pick:
                            codes = mdf[mdf['계정명'].isin(pick)]['계정코드'].astype(str).tolist()
                        amod = run_anomaly_module(lf_use, target_accounts=codes, topn=topn, pm_value=float(st.session_state.get("pm_value", PM_DEFAULT)))
                        _push_module(amod)
                        for w in amod.warnings: st.warning(w)
                        if 'anomaly_top' in amod.tables:
                            _tbl = amod.tables['anomaly_top'].copy()
                            fmt = {}
                            if '발생액' in _tbl.columns: fmt['발생액'] = '{:,.0f}'
                            if 'Z-Score' in _tbl.columns: fmt['Z-Score'] = '{:.2f}'
                            st.dataframe(_tbl.style.format(fmt), use_container_width=True)
                        if 'zscore_hist' in amod.figures:
                            st.plotly_chart(amod.figures['zscore_hist'], use_container_width=True, key="anomaly_hist")

                with tab_ts:
                    st.header("시계열 예측")
                    with st.expander("🧭 해석 가이드", expanded=False, icon=":material/help:"):
                        st.markdown(
                            """
### 용어
- **z(표준화 지수)**: `z = (실측 − 예측) / σ`  
  - 월별 예측 잔차(실측-예측)를 표준화한 지수입니다. **이상 패턴 탐지의 Z-Score와 다른 개념입니다.**
  - |z|≈2는 **이례적**, |z|≈3은 **매우 이례적**입니다.  
- **σ(표준편차) 집계**: 최근 *k=6개월* 잔차의 표준편차로 표준화합니다. 데이터가 짧으면 시작~현재까지의 **expanding σ**를 사용합니다.  
- **위험도(0~1)** = `min(1, 0.5·|z|/3 + 0.3·PM대비 + 0.2·KIT)`  
  - PM대비 = `min(1, |실측−예측| / PM)`,  **KIT** = PM 초과 여부(True/False)
- **Flow / Balance**: *Flow*는 **월 발생액(Δ잔액)**, *Balance*는 **월말 잔액**입니다. *(BS 계정은 Balance 기준도 병행 계산합니다.)*
- **정상성**: 시계열의 평균/분산이 시간에 따라 **안 변함**(ARIMA가 특히 선호).
- **MAE**: 평균 절대 오차(원 단위). **작을수록 정확**.
- **MAPE**: 상대 오차(%). **규모 다른 계정 비교**에 유용.
- **AIC/BIC**: 모델 복잡도까지 고려한 **정보량 지표**. **작을수록 우수**.

### 차트 읽기
- 실선=실측, 점선=예측(**MoR**: EMA/MA/ARIMA/Prophet 중 자동 선택)  
- 회색 점선: **연(굵게)** / **분기(얇게)** 경계선, 붉은 점선: **PM 기준선**

### 사용한 예측모델
- **MA(이동평균)**: 최근 *n*개월 **단순 평균**. **짧은 데이터/변동 완만**할 때 안정적.
- **EMA(지수이동평균)**: **최근값 가중** 평균. **최근 추세 반영**이 필요할 때 유리.
- **ARIMA(p,d,q)**: **자기상관** 기반. **계절성이 약(또는 제거 가능)**하고 **데이터가 충분**할 때 강함.
- **Prophet**: **연/분기 계절성·휴일효과**가 뚜렷할 때 적합(이상치에 비교적 견고).

> :blue[**모델은 계정×기준(Flow/Balance)별로 교차검증 오차(MAPE/MAE)와 (가능하면) 정보량(AIC/BIC)을 종합해 자동 선택됩니다.**]
"""
                        )

                    # 0) 원장/세션 확보
                    master_df: pd.DataFrame = st.session_state.get("master_df", pd.DataFrame())
                    if master_df.empty:
                        st.info("원장 데이터가 없습니다.")
                        st.stop()

                    # --- state bootstrap --- (위젯 생성 전에 실행)
                    st.session_state.setdefault("ts_accounts_names", [])
                    st.session_state.setdefault("ts_cycles_ko", [])
                    st.session_state.setdefault("ts_acc_buffer", None)
                    st.session_state.setdefault("ts_acc_needs_update", False)

                    # --- preset 주입 훅: rerun 직후, 멀티셀렉트 생성 '이전'에 1회 주입 ---
                    if st.session_state.ts_acc_needs_update and st.session_state.ts_acc_buffer is not None:
                        st.session_state.ts_accounts_names = st.session_state.ts_acc_buffer
                        st.session_state.ts_acc_needs_update = False

                    upload_id = getattr(uploaded_file, 'name', '_default')

                    # 두 컨테이너로 시각 순서는 유지(계정 위, 프리셋 아래) + 코드 순서 제어
                    box_accounts = st.container()
                    box_preset = st.container()

                    # Helper: 프리셋(KO 라벨) → 계정명 리스트로 확장
                    def expand_cycles_to_account_names(*, upload_id: str, cycles_ko: list[str], master_df: pd.DataFrame) -> list[str]:
                        try:
                            mapping = cyc.get_effective_cycles(upload_id)
                            codes = cyc.accounts_for_cycles_ko(mapping, cycles_ko)
                            df_map = master_df[["계정코드","계정명"]].dropna().copy()
                            df_map["계정코드"] = df_map["계정코드"].astype(str)
                            code_to_name = df_map.drop_duplicates("계정코드").set_index("계정코드")["계정명"].astype(str).to_dict()
                            names = [code_to_name.get(str(c), str(c)) for c in codes]
                            # 유니크+순서보존
                            return list(dict.fromkeys([n for n in names if n]))
                        except Exception:
                            return []

                    # (아래) 프리셋 영역: 버튼으로 버퍼만 갱신
                    with box_preset:
                        st.markdown("#### 사이클 프리셋 선택(선택 시 위 계정 목록에 **적용 버튼**으로 반영)")
                        chosen_cycles = st.multiselect(
                            "사이클 프리셋",
                            options=list(cyc.CYCLE_KO.values()),
                            key="ts_cycles_ko",
                        )
                        if st.button("➕ 프리셋 적용", key="ts_apply_preset"):
                            names_from_cycles = expand_cycles_to_account_names(
                                upload_id=upload_id,
                                cycles_ko=st.session_state.ts_cycles_ko,
                                master_df=master_df,
                            )
                            merged = list(dict.fromkeys([
                                *st.session_state.ts_accounts_names,
                                *names_from_cycles,
                            ]))
                            st.session_state.ts_acc_buffer = merged
                            st.session_state.ts_acc_needs_update = True
                            st.rerun()

                    # (위) 계정 영역: 멀티셀렉트 그리기(값 주입은 상단 훅이 담당)
                    with box_accounts:
                        all_account_names = (
                            master_df["계정명"].dropna().astype(str).sort_values().unique().tolist()
                        )
                        picked_names = st.multiselect(
                            "대상 계정(복수 선택 가능)",
                            options=all_account_names,
                            key="ts_accounts_names",
                            help="선택한 계정에 대해서만 예측 테이블/그래프를 생성합니다."
                        )

                    # 미래 예측 개월 수 슬라이더 추가
                    forecast_horizon = st.slider(
                        "미래 예측 개월 수(시각화용)", min_value=0, max_value=12, value=0, step=1,
                        help="표본 N<6이면 자동으로 0으로 비활성화됩니다."
                        )

                    if not picked_names:
                        st.info("시계열 결과가 없습니다. (선택한 계정/기간에 데이터 없음)")
                        st.stop()

                    # 2) 계정명 → 계정코드
                    name_to_code = (
                        master_df.dropna(subset=["계정명","계정코드"]).astype({"계정명":"string","계정코드":"string"})
                                 .drop_duplicates(subset=["계정명"]).set_index("계정명")["계정코드"].to_dict()
                    )
                    want_codes = [name_to_code.get(n) for n in picked_names if n in name_to_code]

                    # 3) 정식 시계열 파이프라인: ledger → 월별집계(flow) → balance(opening+누적) → 예측/진단/그림
                    lf_use = st.session_state.get('lf_hist')
                    st.caption("ⓘ 시계열 분석은 좌측 스코프 설정과 무관하게 전체 히스토리를 사용합니다.")
                    if lf_use is None:
                        st.info("원장을 먼저 업로드해 주세요.")
                        st.stop()

                    # (1) 분석 대상 슬라이스
                    ldf = lf_use.df.copy()
                    ldf = ldf[ldf['계정코드'].astype(str).isin([str(x) for x in want_codes])]

                    # (2) 필수 파생: 발생액/순액 보장
                    from analysis.anomaly import compute_amount_columns
                    ldf = compute_amount_columns(ldf)

                    # (3) 날짜/금액 컬럼 픽업(없으면 안전 종료)
                    from analysis.timeseries import DATE_CANDIDATES, AMT_CANDIDATES
                    date_col = next((c for c in DATE_CANDIDATES if c in ldf.columns), None)
                    amount_col = next((c for c in AMT_CANDIDATES if c in ldf.columns), None)
                    if not date_col or not amount_col:
                        st.error(
                            "필수 컬럼을 찾지 못했습니다.\n"
                            f"- 날짜 후보: {DATE_CANDIDATES}\n- 금액 후보: {AMT_CANDIDATES}\n\n"
                            f"현재 컬럼: {list(ldf.columns)}"
                        )
                        st.stop()

                    # (4) opening(전기말잔액) 맵 구성
                    opening_map = {}
                    if "전기말잔액" in master_df.columns and "계정코드" in master_df.columns:
                        opening_map = (
                            master_df[["계정코드","전기말잔액"]]
                            .dropna(subset=["계정코드"])
                            .assign(전기말잔액=lambda d: pd.to_numeric(d["전기말잔액"], errors="coerce").fillna(0.0))
                            .groupby("계정코드")["전기말잔액"].first().to_dict()
                        )

                    # (5) BS/PL 플래그
                    is_bs_map = {}
                    if "BS/PL" in master_df.columns and "계정코드" in master_df.columns:
                        is_bs_map = (
                            master_df.dropna(subset=["계정코드","BS/PL"])
                            .astype({"계정코드":"string","BS/PL":"string"})
                            .drop_duplicates(subset=["계정코드"])
                            .assign(is_bs=lambda d: d["BS/PL"].str.upper().eq("BS"))
                            .set_index("계정코드")["is_bs"].to_dict()
                        )

                    # (5.5) 차변/대변 플래그(대변 계정은 부호 반전)
                    is_credit_map = {}
                    if "차변/대변" in master_df.columns and "계정코드" in master_df.columns:
                        is_credit_map = (
                            master_df.dropna(subset=["계정코드","차변/대변"])
                                     .assign(계정코드=lambda d: d["계정코드"].astype(str),
                                             credit=lambda d: d["차변/대변"].astype(str).str.contains("대변"))
                                     .drop_duplicates(subset=["계정코드"])
                                     .set_index("계정코드")["credit"].to_dict()
                        )

                    # (6) 모델 선택(레지스트리)
                    st.caption("모형: EMA(고정). 복잡 러너는 비활성화되었습니다.")
                    backend = "ema"

                    PM = float(st.session_state.get("pm_value", PM_DEFAULT))
                    # (7) 계정별 실행: 결과 수집용 버퍼
                    gathered_flow = []
                    gathered_balance = []
                    results_per_account = {}

                    for code in want_codes:
                        sub = ldf[ldf["계정코드"].astype(str) == str(code)].copy()
                        if sub.empty:
                            continue
                        acc_name = (master_df[master_df["계정코드"].astype(str)==str(code)]["계정명"].dropna().astype(str).head(1).tolist() or [str(code)])[0]
                        is_bs = bool(is_bs_map.get(str(code), False))

                        PM = float(st.session_state.get("pm_value", PM_DEFAULT))
                        # ✨ 대변계정이면 부호 반전
                        sign = -1.0 if bool(is_credit_map.get(str(code), False)) else 1.0
                        # 모델 입력 컬럼을 수치화 + 반전
                        try:
                            sub[amount_col] = pd.to_numeric(sub[amount_col], errors="coerce").fillna(0.0) * float(sign)
                        except Exception:
                            sub[amount_col] = pd.to_numeric(sub.get(amount_col, 0.0), errors="coerce").fillna(0.0) * float(sign)
                        # BS 잔액용 opening도 동일 기준으로 반전
                        opening = 0.0
                        if isinstance(opening_map, dict):
                            opening = float(opening_map.get(str(code), 0.0)) * float(sign)

                        out = run_timeseries_minimal(
                            sub,
                            account_name=acc_name,
                            date_col=date_col,
                            amount_col=amount_col,
                            is_bs=bool(is_bs),
                            opening=opening,
                            pm_value=PM
                        )

                        # (수집) 통합 요약표(1행) + 그래프(다포인트) 분리
                        if not out.empty:
                            tmp = out.copy()
                            tmp.insert(0, "계정", acc_name)

                            # 그래프/진단용(전 구간)
                            results_per_account[acc_name] = tmp

                            # === 요약행(마지막 1행) + 통계열 추가 ===
                            for ms in ("flow", "balance"):
                                dfm = tmp[tmp["measure"].eq(ms)]
                                if dfm.empty:
                                    continue
                                stats = compute_series_stats(dfm)
                                last_row = dfm.tail(1).copy()
                                last_row["MAE"]  = stats["MAE"]
                                last_row["MAPE"] = stats["MAPE"]
                                last_row["RMSE"] = stats["RMSE"]
                                last_row["N"]    = stats["N"]

                                if ms == "flow":
                                    gathered_flow.append(last_row)
                                else:
                                    gathered_balance.append(last_row)

                    # === 공용: 표 높이 자동 계산 ===
                    def _auto_table_height(df: pd.DataFrame, *, min_rows=3, max_rows=10, row_px=32, header_px=40, padding_px=16) -> int:
                        n = 0 if df is None else int(len(df))
                        n = max(min_rows, min(max_rows, n))
                        return header_px + n * row_px + padding_px

                    # === NEW: 통합 테이블(그래프보다 위에 한 번만) ===
                    def _render_table(blocks, title):
                        if not blocks:
                            return
                        tbl = pd.concat(blocks, ignore_index=True)

                        show_cols = ["계정","date","actual","predicted","error","z","risk","model",
                                     "MAE","MAPE","RMSE","N"]  # ← 통계 열 추가
                        for c in show_cols:
                            if c not in tbl.columns:
                                tbl[c] = np.nan

                        # 사용자 친화 라벨/정렬
                        tbl = (tbl.rename(columns={
                            "date":"일자","actual":"실측","predicted":"예측",
                            "error":"잔차","risk":"위험도","model":"모델(MoR)"
                        })
                            .sort_values(["계정","일자"])
                        )

                        st.subheader(title)
                        fmt = {}
                        for c in ["실측","예측","잔차","MAE","RMSE"]:
                            if c in tbl.columns: fmt[c] = "{:,.0f}"
                        if "MAPE" in tbl.columns: fmt["MAPE"] = "{:.2f}%"
                        if "z" in tbl.columns: fmt["z"] = "{:.2f}"
                        if "위험도" in tbl.columns: fmt["위험도"] = "{:.2f}"

                        # 표 높이: 행수 기반으로 자동(최대 420px)
                        rows = max(1, len(tbl))
                        height = min(420, 42 + rows * 28)

                        st.dataframe(tbl[["계정","일자","실측","예측","잔차","z","위험도","모델(MoR)","MAE","MAPE","RMSE","N"]]
                                     .style.format(fmt),
                                     use_container_width=True, height=height)

                        st.download_button(
                            "CSV 다운로드", data=tbl.to_csv(index=False).encode("utf-8-sig"),
                            file_name=f"timeseries_summary_{'flow' if 'Flow' in title else 'balance'}.csv",
                            mime="text/csv"
                        )

                    def _auto_height(df: pd.DataFrame, max_rows: int = 12) -> int:
                        rows = int(min(len(df), max_rows))
                        base_row = 34  # 체감값
                        header = 38
                        pad = 8
                        return header + rows * base_row + pad

                    def _render_table_combined(flow_blocks, balance_blocks, title="선택계정 요약 (Flow+Balance)"):
                        import pandas as pd
                        import numpy as np
                        blocks = []

                        def _prep(df, label):
                            if df is None or len(df) == 0:
                                return None
                            x = pd.concat(df, ignore_index=True)
                            
                            # 먼저 rename을 해서 컬럼명을 통일
                            rename_map = {
                                "account": "계정", "date": "일자",
                                "actual": "실측", "predicted": "예측",
                                "error": "잔차", "risk": "위험도",
                                "model": "모델(MoR)"
                            }
                            for k, v in rename_map.items():
                                if k in x.columns and v not in x.columns:  # 중복 방지
                                    x.rename(columns={k: v}, inplace=True)
                            
                            # 그 다음 '기준' 컬럼 추가
                            x.insert(0, "기준", label)
                            return x

                        f = _prep(flow_blocks, "발생액(Flow)")
                        b = _prep(balance_blocks, "잔액(Balance)")
                        if f is not None: blocks.append(f)
                        if b is not None: blocks.append(b)
                        if not blocks:
                                return

                        tbl = pd.concat(blocks, ignore_index=True)

                        # 중복 컬럼 제거 (혹시 있다면)
                        tbl = tbl.loc[:, ~tbl.columns.duplicated()]

                        # ✅ z 라벨 변경(표에서만)
                        col_map = {
                            "date":"일자","actual":"실측","predicted":"예측","error":"잔차",
                            "z":"z(시계열)","risk":"위험도","model":"모델(MoR)"
                        }
                        for k, v in col_map.items():
                            if k in tbl.columns:
                                tbl.rename(columns={k: v}, inplace=True)

                        want_cols = ["기준", "계정", "일자", "실측", "예측", "잔차", "z(시계열)", "위험도", "모델(MoR)"]
                        show_cols = [c for c in want_cols if c in tbl.columns]
                        tbl = tbl[show_cols].copy()

                        # 포맷
                        fmt = {"실측":"{:,.0f}","예측":"{:,.0f}","잔차":"{:,.0f}","위험도":"{:.2f}","z(시계열)":"{:.2f}"}

                        st.subheader(title)

                        # 인덱스 숨김 + 통일된 높이
                        tbl = tbl.reset_index(drop=True)
                        try:
                            st.dataframe(
                                tbl.style.format(fmt),
                                use_container_width=True,
                                hide_index=True,
                                height=_auto_height(tbl)
                            )
                        except TypeError:
                            st.dataframe(
                                tbl.style.format(fmt),
                                use_container_width=True,
                                height=_auto_height(tbl)
                            )

                        # CSV
                            st.download_button(
                            "CSV 다운로드",
                            data=tbl.to_csv(index=False).encode("utf-8-sig"),
                            file_name="timeseries_summary_all.csv",
                                mime="text/csv"
                            )

                    def _render_outlier_alert(results_per_account: dict, *, topn: int = 10, z_thr: float = 2.0):
                        """
                        results_per_account: {계정명 -> DataFrame}, DataFrame은 최소 컬럼
                          ['date','actual','predicted','error','z','risk','model','measure'] 가정
                          measure ∈ {'flow','balance'}
                        """
                        import pandas as pd
                        import numpy as np
                        rows = []
                        for acc_name, df in (results_per_account or {}).items():
                            if df is None or df.empty or "z" not in df.columns:
                                continue
                            dfx = df.copy()
                            # 계정명 보강 (혹시 누락 대비)
                            if "계정" not in dfx.columns:
                                dfx["계정"] = acc_name
                            # 기준 라벨(발생액/잔액)
                            dfx["기준"] = dfx.get("measure", "").map(
                                {"flow": "발생액(Flow)", "balance": "잔액(Balance)"}
                            ).fillna("발생액(Flow)")
                            # |z| 필터
                            dfx = dfx[dfx["z"].abs() >= float(z_thr)]
                            if not dfx.empty:
                                rows.append(dfx)

                        # 헤더 + 테이블
                        import streamlit as st
                        st.subheader(f"이상월 알림 (상위 {topn}건, 기준 |z| ≥ {z_thr:.1f})")

                        if not rows:
                            st.info(f"이상월 없음(기준 |z| ≥ {z_thr:.1f})")
                            return

                        out = pd.concat(rows, ignore_index=True)

                        # 정렬: |z| 내림차순 → |잔차| 보조
                        out = out.sort_values(
                            by=["z", "error"],
                            key=lambda s: s.abs() if s.name in ("z", "error") else s,
                            ascending=[False, False]
                        ).head(int(topn))

                        # 표시 컬럼/한글명 + z 라벨 변경
                        rename = {"date": "일자", "actual": "실측", "predicted": "예측",
                                  "error": "잔차", "z": "z(시계열)", "risk": "위험도", "model": "모델"}
                        for k, v in rename.items():
                            if k in out.columns:
                                out.rename(columns={k: v}, inplace=True)

                        show_cols = [c for c in ["계정", "일자", "실측", "예측", "잔차", "z(시계열)", "위험도", "모델", "기준"] if c in out.columns]
                        out = out[show_cols]

                        fmt = {"실측":"{:,.0f}","예측":"{:,.0f}","잔차":"{:,.0f}","위험도":"{:.2f}","z(시계열)":"{:.2f}"}

                        try:
                            st.dataframe(
                                out.style.format(fmt),
                                use_container_width=True,
                                hide_index=True,
                                height=_auto_height(out)
                            )
                        except TypeError:
                            # Streamlit 구버전 호환
                            st.dataframe(
                                out.reset_index(drop=True).style.format(fmt),
                                use_container_width=True,
                                height=_auto_height(out)
                            )

                    # === NEW: 선택계정 통계 및 이상월 리스트 렌더링 함수 정의 ===
                    def _safe_div(a, b):
                        try:
                            b = np.where(np.abs(b) < 1e-9, 1.0, b)
                            return a / b
                        except Exception:
                            return np.nan

                    _render_table_combined(gathered_flow, gathered_balance, title="선택계정 요약 (Flow+Balance)")

                    # 통합 이상월 알림 (|z| ≥ 2.0 고정)
                    _render_outlier_alert(results_per_account, topn=10, z_thr=2.0)

                    # ============ 🔎 시계열 파이프라인 진단(현황판) ============ #
                    with st.expander("🔎 시계열 파이프라인 진단(현황판)", expanded=False):
                        st.caption("각 단계별로 포인트 수/타입/정규화 상태를 집계합니다. 그래프가 안 뜨면 어디서 끊겼는지 여기서 확인하세요.")
                        # 0) 원본 슬라이스 요약
                        st.markdown("**0) 입력(원장 슬라이스) 요약**")
                        try:
                            st.write({
                                "선택계정 수": len(want_codes),
                                "원장 행수(선택계정)": int(len(ldf)),
                                "date_col": date_col,
                                "amount_col": amount_col,
                                "date_dtype": str(ldf[date_col].dtype),
                                "amount_dtype": str(ldf[amount_col].dtype),
                                "NaT(날짜)": int(pd.to_datetime(ldf[date_col], errors="coerce").isna().sum()),
                                "NaN(금액)": int(pd.to_numeric(ldf[amount_col], errors="coerce").isna().sum()),
                                "기간": f"{pd.to_datetime(ldf[date_col], errors='coerce').min()} ~ {pd.to_datetime(ldf[date_col], errors='coerce').max()}",
                            })
                            st.dataframe(
                                ldf[[date_col, "계정코드", "계정명", amount_col]].head(5),
                                use_container_width=True,
                                height=_auto_table_height(ldf.head(5))
                            )
                        except Exception as _e:
                            st.warning(f"입력 요약 실패: {_e}")

                        # 1) 계정×월 집계 확인
                        st.markdown("**1) 월별 집계 상태**")
                        try:
                            _tmp = ldf[[date_col, amount_col]].copy()
                            _tmp = _tmp.rename(columns={date_col: '회계일자', amount_col: '거래금액'})
                            _grp = aggregate_monthly(_tmp, date_col='회계일자', amount_col='거래금액').rename(columns={"amount":"flow"})
                            _grp["date"] = pd.to_datetime(_grp["date"], errors="coerce")
                            _norm_ok = int((_grp["date"].dt.hour.eq(0) & _grp["date"].dt.minute.eq(0)).sum())
                            st.write({
                                "집계 포인트 수": int(len(_grp)),
                                "월말 00:00:00 비율": f"{_norm_ok}/{len(_grp)}",
                                "예: 첫 3행": None
                            })
                            st.dataframe(
                                _grp.head(3),
                                use_container_width=True,
                                height=_auto_table_height(_grp.head(3))
                            )
                            # (보너스) 경계선 예상 개수
                            try:
                                rng = pd.date_range(pd.to_datetime(ldf[date_col]).min(), pd.to_datetime(ldf[date_col]).max(), freq="M")
                                q_ends = [m for m in rng if m.month in (3,6,9,12)]
                                y_ends = [m for m in rng if m.month == 12]
                                st.write({"경계선(분기말) 예상 개수": len(q_ends), "경계선(연말) 예상 개수": len(y_ends)})
                            except Exception as _ee:
                                st.info(f"경계선 개수 계산 실패: {_ee}")
                        except Exception as _e:
                            st.warning(f"월별 집계 상태 계산 실패: {_e}")

                        # 2) 러너 결과 요약
                        st.markdown("**2) 모델 입력/출력 상태(run_timeseries_minimal · EMA)**")
                        try:
                            if not (gathered_flow or gathered_balance):
                                st.warning("러너 출력(gathered_*)가 비어 있습니다. 상단 입력/집계 단계 확인 필요.")
                            else:
                                parts = []
                                if gathered_flow: parts += gathered_flow
                                if gathered_balance: parts += gathered_balance
                                parts = [p for p in parts if isinstance(p, pd.DataFrame) and not p.empty]
                                if not parts:
                                    st.warning("러너 출력이 비어 있습니다.(유효한 DataFrame 없음)")
                                else:
                                    _all = pd.concat(parts, ignore_index=True)
                                    st.write({
                                        "계정×기준(measure) 개수": int(_all[["계정","measure"]].drop_duplicates().shape[0]) if set(["계정","measure"]).issubset(_all.columns) else 0,
                                        "actual 존재": bool("actual" in _all.columns),
                                        "predicted 존재": bool("predicted" in _all.columns),
                                        "flow 포인트": int(_all[_all.get("measure","flow").eq("flow")].shape[0]) if "measure" in _all.columns else int(_all.shape[0]),
                                        "balance 포인트": int(_all[_all.get("measure","flow").eq("balance")].shape[0]) if "measure" in _all.columns else 0,
                                    })
                                    st.dataframe(
                                        _all.head(5),
                                        use_container_width=True,
                                        height=_auto_table_height(_all.head(5))
                                    )
                        except Exception as _e:
                            st.warning(f"러너 출력 요약 실패: {type(_e).__name__}: {_e}")

                        # 부호 보정 가드 표시
                        st.markdown("**부호 보정 가드**")
                        try:
                            pipeline_norm = any(c in ldf.columns for c in ["발생액_norm", "amount_norm", "__sign", "sign"])
                            plot_sign_flip = False  # 플롯 레벨 반전은 하지 않음
                            st.write({
                                "pipeline_norm": bool(pipeline_norm),
                                "plot_sign_flip": bool(plot_sign_flip),
                                "guard_ok": bool(pipeline_norm and not plot_sign_flip)
                            })
                            if pipeline_norm and plot_sign_flip:
                                st.warning("경고: 파이프라인과 플롯에서 모두 부호를 만지면 이중 반전 위험이 있습니다.")
                        except Exception as _e:
                            st.info(f"부호 보정 가드 점검 실패: {_e}")

                        # 3) 그림 입력 전 점검(계정별)
                        st.markdown("**3) 그림 입력 사전 점검(create_timeseries_figure 직전)**")
                        try:
                            for acc_name, df_all in results_per_account.items():
                                for ms in (["flow","balance"] if df_all["measure"].eq("balance").any() else ["flow"]):
                                    dfx = df_all[df_all["measure"].eq(ms)]
                                    st.write(f"- {acc_name} / {ms}: N={len(dfx)} · 컬럼={list(dfx.columns)} · 날짜범위={pd.to_datetime(dfx['date']).min()}~{pd.to_datetime(dfx['date']).max()}")
                                    st.dataframe(
                                        dfx[["date","actual","predicted"]].head(3),
                                        use_container_width=True,
                                        height=_auto_table_height(dfx.head(3))
                                    )
                        except Exception as _e:
                            st.warning(f"그림 입력 점검 실패: {_e}")

                    # === 연/분기 경계선 기본 표시 ===
                    show_dividers = True

                    # === 계정×기준 통계 요약 ===
                    def _safe_stats_block(df_in: pd.DataFrame) -> dict:
                        s = {}
                        try:
                            a = pd.to_numeric(df_in.get("actual"), errors="coerce")
                            p = pd.to_numeric(df_in.get("predicted"), errors="coerce")
                            e = a - p
                            s["N"] = int(len(df_in))
                            s["MAE"] = float(np.nanmean(np.abs(e)))
                            denom = a.replace(0, np.nan)
                            s["MAPE(%)"] = float(np.nanmean(np.abs(e / denom)) * 100.0)
                            s["RMSE"] = float(np.sqrt(np.nanmean((e**2))))
                            z = pd.to_numeric(df_in.get("z"), errors="coerce")
                            s["|z|_max"] = float(np.nanmax(np.abs(z))) if z is not None else np.nan
                            s["last_z"] = float(z.iloc[-1]) if (z is not None and len(z) > 0) else np.nan
                            s["last_err"] = float(e.iloc[-1]) if len(e) > 0 else np.nan
                            s["model"] = str(df_in.get("model").iloc[-1]) if "model" in df_in.columns and len(df_in) > 0 else ""
                        except Exception:
                            pass
                        return s

                    stats_rows = []
                    for acc_name, df_all in results_per_account.items():
                        for ms in (["flow","balance"] if df_all["measure"].eq("balance").any() else ["flow"]):
                            d = df_all[df_all["measure"].eq(ms)]
                            if d.empty:
                                continue
                            row = {"계정": acc_name, "기준": ("발생액(Flow)" if ms=="flow" else "잔액(Balance)")}
                            row.update(_safe_stats_block(d))
                            stats_rows.append(row)

                    if stats_rows:
                        stats_df = pd.DataFrame(stats_rows)[
                            ["계정","기준","N","MAE","MAPE(%)","RMSE","|z|_max","last_z","last_err","model"]
                        ]
                        fmt = {"MAE":"{:,.0f}","MAPE(%)":"{:.2f}","RMSE":"{:,.0f}","|z|_max":"{:.2f}","last_z":"{:.2f}","last_err":"{:,.0f}"}
                        st.subheader("계정별 통계 요약")
                        st.dataframe(
                            stats_df.style.format(fmt),
                            use_container_width=True,
                            height=_auto_table_height(stats_df)
                        )

                    # === 그래프 렌더(아래): 계정별로 표시 ===
                    for acc_name, df_all in results_per_account.items():
                        for measure in (["flow","balance"] if (df_all["measure"].eq("balance").any()) else ["flow"]):
                            dfm = df_all[df_all["measure"]==measure].rename(columns={"account":"계정"})
                            title = f"{acc_name} — {'발생액(Flow)' if measure=='flow' else '잔액(Balance)'}"

                            # 표본수 게이트: N<6이면 미래 음영 비활성화
                            stats = compute_series_stats(dfm)
                            _hz = int(forecast_horizon or 0)
                            if stats["N"] < 6:
                                _hz = 0

                            fig, stats_d = create_timeseries_figure(
                                dfm, measure=measure, title=title,
                                pm_value=PM, show_dividers=True   # ← 분기/연 경계선 켜기
                            )

                            # 미래 구간 음영(시각화 전용)
                            if _hz > 0 and not dfm.empty:
                                last_date = pd.to_datetime(dfm["date"]).max()
                                fig = add_future_shading(fig, last_date, horizon_months=_hz)

                            st.subheader(title)
                            if fig is not None:
                                st.plotly_chart(fig, use_container_width=True)

                            # MoR 로그 표시
                            log = (results_per_account.get(acc_name, pd.DataFrame())).attrs.get("mor_log", {})
                            if stats_d or log:
                                st.caption(
                                    f"모형:{dfm['model'].iloc[-1] if not dfm.empty else '-'} · "
                                    f"선정근거:{log.get('metric','-')} "
                                    f"(MAPE={log.get('mape_best',''):g}%, MAE={log.get('mae_best',''):,.0f}) · "
                                    f"표본월:{log.get('n_months','-')}"
                                    + ("" if _hz == forecast_horizon else " · (표본 부족으로 미래음영 비활성)")
                                )
                # ⚠️ 기존 tab5(위험평가) 블록 전체 삭제됨
                
                with tab_corr:
                    st.header("상관관계")
                    upload_id = getattr(uploaded_file, 'name', '_default')
                    # 한 탭 내 순차 렌더(서브탭 사용 금지)
                    st.subheader("기본 상관관계")
                    _render_corr_basic_tab(upload_id=upload_id)
                    st.markdown("---")
                    st.subheader("고급 상관관계")
                    _render_corr_advanced_tab(upload_id=upload_id)
                with tab_report:
                    st.header("🧠 분석 종합 대시보드")
                    # --- Preview: modules session quick view ---
                    modules_list_preview = list(st.session_state.get('modules', {}).values())
                    with st.expander("🔎 모듈별 요약/증거 미리보기", expanded=False):
                        if not modules_list_preview:
                            st.info("모듈 결과가 비어 있습니다. 먼저 각 모듈을 실행하세요.")
                        else:
                            for mr in modules_list_preview:
                                try:
                                    st.subheader(f"• {getattr(mr, 'name', 'module')}")
                                    if getattr(mr, 'summary', None):
                                        st.json(mr.summary)
                                    evs = list(getattr(mr, 'evidences', []))
                                    if evs:
                                        st.write("Evidence 샘플 (상위 3)")
                                        for ev in evs[:3]:
                                            try:
                                                st.write(f"- reason={ev.reason} | risk={float(ev.risk_score):.2f} | amount={float(ev.financial_impact):,.0f}")
                                            except Exception:
                                                st.write("- (표시 실패)")
                                    if getattr(mr, 'tables', None):
                                        try: st.caption(f"tables: {list(mr.tables.keys())}")
                                        except Exception: pass
                                    if getattr(mr, 'figures', None):
                                        try: st.caption(f"figures: {list(mr.figures.keys())}")
                                        except Exception: pass
                                except Exception:
                                    st.caption("(미리보기 실패)")
                    # LLM 키 미가용이어도 오프라인 리포트 모드로 생성 가능
                    LLM_OK = False
                    try:
                        from services.llm import openai_available
                        LLM_OK = bool(openai_available())
                    except Exception:
                        LLM_OK = False
                    if not LLM_OK:
                        st.info("🔌 OpenAI Key 없음: 오프라인 리포트 모드로 생성합니다. (클러스터/요약 LLM 미사용)")
                    rendered_report = False

                    # === 모델/토큰/컨텍스트 옵션 UI ===
                    colm1, colm2, colm3 = st.columns([1,1,1])
                    with colm1:
                        llm_model_choice = st.selectbox(
                            "LLM 모델", options=["gpt-5", "gpt-4o"], index=1,
                            help="gpt-5 미가용 시 자동으로 gpt-4o로 대체하세요(코드에서 예외 처리)."
                        )
                    with colm2:
                        desired_tokens = st.number_input(
                            "보고서 최대 출력 토큰", min_value=512, max_value=32000, value=16000, step=512,
                            help="실제 전송값은 모델 컨텍스트와 입력 토큰을 고려해 안전 클램프됩니다."
                        )
                    with colm3:
                        ctx_topk = st.number_input("컨텍스트 Evidence Top-K(모듈별)", min_value=5, max_value=100, value=20, step=5)
                        st.caption("요약/도표는 최소화하고 증거는 상위 Top-K만 사용합니다.")
                        st.session_state['ctx_topk'] = int(ctx_topk)

                    # 선택한 모델/토큰을 세션에 저장하여 하단 호출부에서 실제 사용
                    st.session_state['llm_model'] = llm_model_choice
                    st.session_state['llm_max_tokens'] = int(desired_tokens)

                    # --- 입력 영역 ---
                    mdf = st.session_state.master_df
                    ldf = st.session_state.ledger_df

                    # ① 계정 선택(필수) — 자동 추천 제거
                    acct_names_all = sorted(mdf['계정명'].dropna().unique().tolist())
                    pick_accounts = st.multiselect(
                        "보고서 대상 계정(들)을 선택하세요. (최소 1개)",
                        options=acct_names_all,
                        default=[],
                        key="report_accounts_pick"
                    )
                    cycles_map_now = cyc.get_effective_cycles(upload_id)
                    if cycles_map_now:
                        picked_cycles_report = st.multiselect(
                            "사이클 프리셋 선택", list(cyc.CYCLE_KO.values()),
                            default=[], key="report_cycles_pick"
                        )
                        st.button("➕ 프리셋 적용", key="btn_apply_cycles_report", on_click=_apply_cycles_to_picker,
                                  kwargs=dict(upload_id=upload_id,
                                              cycles_state_key="report_cycles_pick",
                                              accounts_state_key="report_accounts_pick",
                                              master_df=st.session_state.master_df))
                    # ② 옵션 제거: 항상 수행 플래그
                    opt_knn_evidence = True
                    opt_patterns = True
                    opt_patterns_py = True

                    # ③ 사용자 메모(선택)
                    manual_ctx = st.text_area(
                        "보고서에 추가할 메모/주의사항(선택)",
                        placeholder="예: 5~7월 대형 캠페인 집행 영향, 3분기부터 단가 인상 예정 등"
                    )

                    # ④ 선택 계정코드 매핑
                    pick_codes = (
                        mdf[mdf['계정명'].isin(st.session_state['report_accounts_pick'])]['계정코드']
                        .astype(str).tolist()
                    )

                    colA, colB, colC = st.columns([1,1,1])
                    with colA: st.write("선택 계정코드:", ", ".join(pick_codes) if pick_codes else "-")
                    with colB: st.write("기준 연도(CY):", int(ldf['연도'].max()))
                    with colC: st.write("보고서 기준:", "Current Year GL")

                    # 버튼은 계정 미선택 시 비활성화
                    btn = st.button("📝 보고서 생성", type="primary", disabled=(len(pick_codes) == 0))
                    if len(pick_codes) == 0:
                        st.info("계정 1개 이상 선택 시 버튼이 활성화됩니다.")

                    if btn:
                        import time
                        from analysis.anomaly import compute_amount_columns
                        from analysis.embedding import ensure_rich_embedding_text, perform_embedding_and_clustering
                        from analysis.report import build_report_context, run_final_analysis, build_methodology_note, run_offline_fallback_report
                        from services.llm import LLMClient
                        from analysis.anomaly import ensure_zscore

                        t0 = time.perf_counter()
                        with st.status("보고서 준비 중...", expanded=True) as s:
                            # Step 1) 데이터 슬라이싱
                            s.write("① 스코프 적용 및 데이터 슬라이싱(CY/PY)…")
                            cur_year = ldf['연도'].max()
                            df_cy = ldf[(ldf['period_tag'] == 'CY') & (ldf['계정코드'].astype(str).isin(pick_codes))].copy()
                            df_py = ldf[(ldf['period_tag'] == 'PY') & (ldf['계정코드'].astype(str).isin(pick_codes))].copy()
                            s.write(f"    └ CY {len(df_cy):,}건 / PY {len(df_py):,}건")

                            # Step 2) 필수 파생(발생액/순액)
                            s.write("② 금액 파생 컬럼 생성(발생액/순액)…")
                            df_cy = compute_amount_columns(df_cy)
                            df_py = compute_amount_columns(df_py)

                            # Step 3) (선택) 패턴요약: 임베딩/클러스터링 (LLM 사용 가능 시에만)
                            cl_ok = False
                            if LLM_OK and opt_patterns and not df_cy.empty:
                                s.write("③ 임베딩·클러스터링 실행(선택)…")
                                # 입력 텍스트 풍부화 + 임베딩 + HDBSCAN (최대 N 제한으로 안전가드)
                                df_cy_small = df_cy.copy()
                                max_rows = 8000
                                if len(df_cy_small) > max_rows:
                                    df_cy_small = df_cy_small.sample(max_rows, random_state=42)
                                    s.write(f"    └ 데이터가 많아 {max_rows:,}건으로 샘플링")
                                df_cy_small = ensure_rich_embedding_text(df_cy_small)
                                try:
                                    llm_service = LLMClient(model=st.session_state.get('llm_model', 'gpt-4o'))
                                    emb_client = llm_service.client  # OpenAI 클라이언트 객체
                                    naming_function = llm_service.name_cluster
                                    # 보고서 생성을 위해 LLM 기반 클러스터 네이밍을 필수로 요구
                                    df_clu, ok = perform_embedding_and_clustering(
                                        df_cy_small, emb_client,
                                        name_with_llm=True, must_name_with_llm=True,
                                        use_large=bool(st.session_state.get("use_large_embedding", False)),
                                        rescue_tau=float(st.session_state.get("rescue_tau", HDBSCAN_RESCUE_TAU)),
                                        embed_texts_fn=get_or_embed_texts,
                                        naming_fn=naming_function,
                                    )
                                    if ok:
                                        # 유사한 클러스터 이름을 LLM으로 통합
                                        df_clu, name_map = unify_cluster_names_with_llm(
                                            df_clu,
                                            sim_threshold=0.90,
                                            emb_model=st.session_state.get('embedding_model', None) or EMB_MODEL_SMALL,
                                            embed_texts_fn=get_or_embed_texts,
                                            confirm_pair_fn=make_synonym_confirm_fn(emb_client, st.session_state.get('llm_model', 'gpt-4o')),
                                        )
                                        # 추가 LLM 라벨 통합(JSON 매핑 방식) — CY의 cluster_group은 유지
                                        try:
                                            raw_map = unify_cluster_labels_llm(df_clu['cluster_name'].dropna().unique().tolist(), emb_client)
                                            if raw_map:
                                                df_clu['cluster_name'] = df_clu['cluster_name'].map(lambda x: raw_map.get(str(x), x))
                                                # ❗ cluster_group는 unify_cluster_names_with_llm()이 정한 canonical을 유지
                                        except Exception:
                                            pass
                                        # 간단 요약(상위 5개)
                                        topc = (df_clu.groupby('cluster_group')['발생액']
                                                .agg(['count','sum']).sort_values('sum', ascending=False).head(5))
                                        s.write("    └ 클러스터 상위 5개 요약:")
                                        st.dataframe(
                                            topc.rename(columns={'count':'건수','sum':'발생액합계'})
                                                .style.format({'발생액합계':'{:,.0f}'}),
                                            use_container_width=True
                                        )
                                        # 품질 지표(노이즈율·클러스터 수 등) 기록
                                        try:
                                            n = int(len(df_clu))
                                            noise_rate = float((df_clu['cluster_id'] == -1).mean()) if n else 0.0
                                            n_clusters = int(df_clu.loc[df_clu['cluster_id'] != -1, 'cluster_id'].nunique())
                                            if n_clusters > 0:
                                                avg_size = float(df_clu[df_clu['cluster_id'] != -1].groupby('cluster_id').size().mean())
                                            else:
                                                avg_size = 0.0
                                            rescue_rate = float(df_clu.get('rescued', False).mean()) if 'rescued' in df_clu.columns else 0.0
                                            model_used = df_clu.attrs.get('embedding_model', 'unknown')
                                            umap_on = bool(df_clu.attrs.get('umap_used', False))
                                            s.write(
                                                f"    └ Quality: N={n:,}, noise={noise_rate*100:.1f}%, "
                                                f"clusters={n_clusters}, avg_size={avg_size:.1f}, rescued={rescue_rate*100:.1f}%"
                                            )
                                            s.write(
                                                f"    └ Model/UMAP: {model_used} | UMAP={'on' if umap_on else 'off'} | τ={float(st.session_state.get('rescue_tau', HDBSCAN_RESCUE_TAU)):.2f}"
                                            )
                                            # 대시보드 카드용 품질 지표 저장
                                            st.session_state['cluster_quality'] = {
                                                "N": n,
                                                "noise_rate": noise_rate,
                                                "n_clusters": n_clusters,
                                                "avg_size": avg_size,
                                                "rescued_rate": rescue_rate,
                                                "model": model_used,
                                                "umap": umap_on,
                                                "tau": float(st.session_state.get('rescue_tau', HDBSCAN_RESCUE_TAU)),
                                            }
                                        except Exception:
                                            pass
                                        # 보고서 컨텍스트에 반영: group/label 동시 부착
                                        df_cy = df_cy.merge(
                                            df_clu[['row_id','cluster_id','cluster_name','cluster_group']],
                                            on='row_id', how='left'
                                        )
                                        # 필요 시 vector도 함께 병합 가능:
                                        # df_cy = df_cy.merge(df_clu[['row_id','vector']], on='row_id', how='left')
                                        # (현재는 perform_embedding_only 단계에서 CY/PY df에 vector가 직접 부여됨)
                                        # --- PY clustering and alignment (optional) ---
                                        if opt_patterns_py and not df_py.empty:
                                            try:
                                                from analysis.embedding import cluster_year, align_yearly_clusters, unify_cluster_labels_llm
                                                # sampling guard similar to CY
                                                df_py_small = df_py.copy()
                                                max_rows = 8000
                                                if len(df_py_small) > max_rows:
                                                    df_py_small = df_py_small.sample(max_rows, random_state=42)
                                                    s.write(f"    └ PY 데이터가 많아 {max_rows:,}건으로 샘플링")
                                                df_py_clu = cluster_year(
                                                    df_py_small, emb_client, embed_texts_fn=get_or_embed_texts
                                                )
                                                # 가능한 경우 row_id 기준으로 PY 결과 컬럼을 df_py에 병합
                                                if not df_py_clu.empty and 'row_id' in df_py.columns:
                                                    df_py = df_py.merge(df_py_clu, on='row_id', how='left', suffixes=("", "_pyclu"))
                                                # 정렬: PY 클러스터를 CY 클러스터에 매핑
                                                if 'cluster_id' in df_py_clu.columns:
                                                    mapping = align_yearly_clusters(df_clu, df_py_clu, sim_threshold=0.70)
                                                    # cluster_id → (aligned_cy_cluster, aligned_sim)
                                                    cy_id_to_name = df_clu.drop_duplicates('cluster_id').set_index('cluster_id')['cluster_name'].to_dict()
                                                    def _get_pair(cid):
                                                        try:
                                                            if pd.isna(cid):
                                                                return (np.nan, np.nan)
                                                            cid_int = int(cid)
                                                            return mapping.get(cid_int, (np.nan, np.nan))
                                                        except Exception:
                                                            return (np.nan, np.nan)
                                                    if 'cluster_id' in df_py.columns:
                                                        pairs = df_py['cluster_id'].map(_get_pair)
                                                        df_py[['aligned_cy_cluster', 'aligned_sim']] = pd.DataFrame(pairs.tolist(), index=df_py.index)
                                                        # 이름은 CY의 이름으로 정렬(가능한 경우)
                                                        df_py['cluster_name'] = df_py['aligned_cy_cluster'].map(cy_id_to_name).fillna(df_py.get('cluster_name'))
                                                # 최종 라벨 정합: 전체 이름 집합 기준으로 통합; CY의 cluster_group은 유지, PY는 canonical로 정렬
                                                try:
                                                    all_names = pd.Series([], dtype=object)
                                                    if 'cluster_name' in df_cy.columns:
                                                        all_names = pd.concat([all_names, df_cy['cluster_name'].dropna().astype(str)], ignore_index=True)
                                                    if 'cluster_name' in df_py.columns:
                                                        all_names = pd.concat([all_names, df_py['cluster_name'].dropna().astype(str)], ignore_index=True)
                                                    all_names = all_names.dropna().unique().tolist()
                                                    canon = unify_cluster_labels_llm(all_names, emb_client)
                                                    if canon:
                                                        if 'cluster_name' in df_cy.columns:
                                                            df_cy['cluster_name'] = df_cy['cluster_name'].map(lambda x: canon.get(str(x), x))
                                                        if 'cluster_name' in df_py.columns:
                                                            df_py['cluster_name'] = df_py['cluster_name'].map(lambda x: canon.get(str(x), x))
                                                        if 'cluster_group' in df_py.columns:
                                                            df_py['cluster_group'] = df_py['cluster_name']
                                                except Exception:
                                                    pass
                                            except Exception as e:
                                                s.write(f"    └ PY 클러스터링/정렬 스킵: {e}")
                                        # 컨텍스트에 별도 노트는 추가하지 않음
                                        cl_ok = True
                                    else:
                                        s.write("    └ LLM 클러스터 이름 생성 실패 또는 결과 없음 → 보고서 생성 요건 미충족")
                                except Exception as e:
                                    s.write(f"    └ 임베딩/클러스터링 실패: {e}")
                            else:
                                s.write("③ 임베딩·클러스터링: LLM 미가용 또는 옵션 비활성 → 스킵")

                            # Step 3-1) (옵션 A) 근거 인용(KNN)용 임베딩만 수행 (LLM 가능 시)
                            if LLM_OK and opt_knn_evidence:
                                s.write("③-1 근거 인용용 임베딩(CY/PY)…")
                                from analysis.embedding import perform_embedding_only, ensure_rich_embedding_text
                                emb_client2 = LLMClient().client
                                df_cy = ensure_rich_embedding_text(df_cy)
                                df_py = ensure_rich_embedding_text(df_py)
                                df_cy = perform_embedding_only(
                                    df_cy, client=emb_client2,
                                    use_large=bool(st.session_state.get("use_large_embedding", False)),
                                    embed_texts_fn=get_or_embed_texts,
                                )
                                df_py = perform_embedding_only(
                                    df_py, client=emb_client2,
                                    use_large=bool(st.session_state.get("use_large_embedding", False)),
                                    embed_texts_fn=get_or_embed_texts,
                                )
                            elif not LLM_OK:
                                s.write("③-1 근거 인용 임베딩: LLM 미가용 → 스킵")

                            # Step 3-2) Z-Score: 반드시 존재해야 함
                            s.write("③-2 Z-Score 계산/검증…")
                            df_cy, z_ok = ensure_zscore(df_cy, pick_codes)
                            df_py, _    = ensure_zscore(df_py, pick_codes)  # 전기에도 Z-Score 계산(컨텍스트에서 사용)
                            if not z_ok:
                                s.write("    └ Z-Score 미계산 또는 전부 결측")

                            # ✅ 게이트 완화: Z-Score만 확보되면 보고서 진행.
                            #    (클러스터 실패 시 관련 섹션은 자동 축약/생략)
                            if not z_ok:
                                st.error("보고서 생성 중단: Z-Score 없음.")
                                s.update(label="보고서 요건 미충족", state="error")
                                st.stop()
                            if not cl_ok:
                                s.write("    └ 클러스터링 결과 없음 → 리포트에서 클러스터 섹션은 생략/축약됩니다.")

                            # Step 4) 컨텍스트 생성(전 모듈 포함) + 방법론 노트
                            s.write("④ 컨텍스트 텍스트 구성(전 모듈)…")
                            from analysis.report_adapter import wrap_dfs_as_module_result
                            from analysis.report import generate_rag_context_from_modules
                            from analysis.integrity import run_integrity_module
                            from analysis.timeseries import run_timeseries_module

                            # (1) 세션 초기화 및 공통 값 준비
                            st.session_state['modules'] = {}
                            lf_use = _lf_by_scope()
                            pm_use = float(st.session_state.get('pm_value', PM_DEFAULT))

                            # (2) 주요 모듈 실행 및 수집
                            if lf_use is not None:
                                # 이상치
                                try:
                                    amod = run_anomaly_module(lf_use, target_accounts=pick_codes or None,
                                                              topn=int(st.session_state.get('ctx_topk', 20)), pm_value=pm_use)
                                    _push_module(amod)
                                except Exception as _e:
                                    st.warning(f"anomaly 모듈 실패: {_e}")
                                # 추세(선택 계정 필요)
                                try:
                                    if pick_codes:
                                        _push_module(run_trend_module(lf_use, accounts=pick_codes))
                                except Exception as _e:
                                    st.warning(f"trend 모듈 실패: {_e}")
                                # 거래처
                                try:
                                    if pick_codes:
                                        _push_module(run_vendor_module(lf_use, account_codes=pick_codes,
                                                                       min_amount=0.0, include_others=True))
                                except Exception as _e:
                                    st.warning(f"vendor 모듈 실패: {_e}")
                                # 상관(2개 이상일 때만)
                                try:
                                    if len(pick_codes) >= 2:
                                        _push_module(run_correlation_module(lf_use, accounts=pick_codes,
                                                                            corr_threshold=0.70,
                                                                            cycles_map=cyc.get_effective_cycles()))
                                except Exception as _e:
                                    st.warning(f"correlation 모듈 실패: {_e}")
                                # 정합성(ModuleResult) — 선택 계정 필터 적용
                                try:
                                    _push_module(run_integrity_module(lf_use, accounts=pick_codes))
                                except Exception as _e:
                                    st.warning(f"integrity 모듈 실패: {_e}")
                                # NEW: 시계열 포함(집계→DTO 래핑)
                                try:
                                    if not df_cy.empty:
                                        ts = pd.concat([df_cy, df_py], ignore_index=True)
                                        ts["date"] = month_end_00(ts["회계일자"])  # 월말 00:00:00 정규화
                                        ts["account"] = ts["계정코드"].astype(str)
                                        ts["amount"] = ts.get("발생액", 0.0).astype(float)
                                        ts_in = ts.groupby(["account","date"], as_index=False)["amount"].sum()
                                        df_ts = run_timeseries_module(ts_in, account_col="account", date_col="date", amount_col="amount",
                                                                      pm_value=pm_use, output="flow", make_balance=False)
                                        summ_ts = {
                                            "n_series": int(df_ts["account"].nunique()) if not df_ts.empty else 0,
                                            "n_points": int(len(df_ts)),
                                            "max_abs_z": float(df_ts["z"].abs().max()) if ("z" in df_ts.columns and not df_ts.empty) else 0.0,
                                        }
                                        _push_module(ModuleResult(name="timeseries", summary=summ_ts,
                                                                  tables={"ts": df_ts}, figures={}, evidences=[], warnings=([] if not df_ts.empty else ["insufficient_points"])))
                                except Exception as _e:
                                    st.warning(f"timeseries 모듈 실패: {_e}")

                            # (3) 레거시 DF도 어댑터로 함께 포함(경량 컨텍스트용)
                            mr_ctx = wrap_dfs_as_module_result(df_cy, df_py, name="report_ctx")
                            modules_list = list(st.session_state.get('modules', {}).values()) + [mr_ctx]
                            # (4) 최종 컨텍스트 생성(Top-K 적용) — 신규 경로만 사용 + 메모 주입
                            ctx = generate_rag_context_from_modules(
                                modules_list,
                                pm_value=pm_use,
                                topk=int(st.session_state.get('ctx_topk', 20)),
                                manual_note=(manual_ctx or "")
                            )

                            # (상단 공통 미리보기로 대체)
                            note = build_methodology_note(report_accounts=pick_codes)

                            # Step 5) LLM 호출 전 점검(길이/토큰)
                            s.write("⑤ LLM 프롬프트 점검…")
                            prompt_len = len(ctx) + len(note)
                            s.write(f"    └ 컨텍스트 길이: {prompt_len:,} chars")
                            try:
                                import tiktoken
                                enc = tiktoken.get_encoding("cl100k_base")
                                est_tokens = len(enc.encode(ctx)) + len(enc.encode(note))
                                s.write(f"    └ 예상 토큰 수: ~{est_tokens:,} tokens")
                            except Exception:
                                s.write("    └ tiktoken 미설치: 토큰 추정 생략")

                            # Step 6) 보고서 생성: LLM 가능하면 시도, 실패/불가 시 오프라인 폴백
                            final_report = None
                            if LLM_OK:
                                s.write("⑥ LLM 요약 생성 호출…")
                                try:
                                    t_llm0 = time.perf_counter()
                                    llm_client = LLMClient(model=st.session_state.get('llm_model'))
                                    final_report = run_final_analysis(
                                        context=ctx + "\n" + note,
                                        account_codes=pick_codes,
                                        model=st.session_state.get('llm_model'),
                                        max_tokens=int(st.session_state.get('llm_max_tokens', 16000)),
                                        generate_fn=llm_client.generate,
                                    )
                                    s.write(f"    └ LLM 완료 (경과 {time.perf_counter()-t_llm0:.1f}s)")
                                except Exception as e:
                                    s.write(f"    └ LLM 실패: {e} → 오프라인 폴백으로 전환")

                            if final_report is None:
                                s.write("⑥-폴백: 오프라인 리포트 생성…")
                                final_report = run_offline_fallback_report(
                                    current_df=df_cy,
                                    previous_df=df_py,
                                    account_codes=pick_codes,
                                    pm_value=float(st.session_state.get('pm_value', PM_DEFAULT))
                                )

                            s.update(label="보고서 준비 완료", state="complete")

                            # 결과 출력 및 세션 보존
                            st.session_state['last_report'] = final_report
                            st.session_state['last_context'] = ctx + "\n" + note
                            st.session_state['last_dfcy'] = df_cy
                            st.session_state['last_dfpy'] = df_py

                            st.success("보고서가 생성되었습니다.")
                            st.markdown("### 📄 AI 요약 보고서")
                            st.markdown(final_report)

                        with st.expander("🔎 근거 컨텍스트(LLM 입력)", expanded=False):
                            st.text(st.session_state['last_context'])

                        # ZIP 단일 다운로드 + RAW 미리보기
                        import io, zipfile
                        def _build_raw_evidence(df_cy_in):
                            keep = [c for c in ['회계일자','계정코드','계정명','거래처','적요','발생액','순액','Z-Score','cluster_group','cluster_name'] if c in df_cy_in.columns]
                            return df_cy_in[keep].copy() if keep else pd.DataFrame()
                        def _make_zip_blob(report_txt: str, context_txt: str, raw_df: pd.DataFrame) -> bytes:
                            mem = io.BytesIO()
                            with zipfile.ZipFile(mem, 'w', zipfile.ZIP_DEFLATED) as z:
                                z.writestr('report.txt', report_txt)
                                z.writestr('context.txt', context_txt)
                                z.writestr('evidence_raw.csv', raw_df.to_csv(index=False, encoding='utf-8-sig'))
                            mem.seek(0)
                            return mem.getvalue()

                        raw_df = _build_raw_evidence(st.session_state['last_dfcy'])
                        st.markdown("#### 📑 근거: 선택 계정 원장(RAW) + 클러스터")
                        if not raw_df.empty:
                            st.dataframe(
                                raw_df.head(100).style.format({'발생액':'{:,.0f}','순액':'{:,.0f}','Z-Score':'{:.2f}'}),
                                use_container_width=True, height=350
                            )
                        else:
                            st.info("표시할 RAW가 없습니다.")

                        zip_bytes = _make_zip_blob(
                            report_txt=st.session_state['last_report'],
                            context_txt=st.session_state['last_context'],
                            raw_df=raw_df
                        )
                        st.download_button(
                            "📥 보고서+근거 다운로드(ZIP)",
                            data=zip_bytes,
                            file_name="ai_report_with_evidence.zip",
                            mime="application/zip",
                            key="zip_dl_current"  # 고유 키(현재 결과)
                        )

                        st.caption(f"⏱ 총 소요: {time.perf_counter()-t0:.1f}s")
                        rendered_report = True

                    # === 캐시된 이전 결과 렌더(버튼 미클릭 시에만) ===
                    if st.session_state.get('last_report') and not btn:
                        st.success("보고서가 준비되어 있습니다.")
                        st.markdown("### 📄 AI 요약 보고서")
                        st.markdown(st.session_state['last_report'])
                        with st.expander("🔎 근거 컨텍스트(LLM 입력)", expanded=False):
                            st.text(st.session_state['last_context'])
                        # RAW 미리보기 + ZIP 버튼 재출력
                        import io, zipfile
                        def _build_raw_evidence(df_cy_in):
                            keep = [c for c in ['회계일자','계정코드','계정명','거래처','적요','발생액','순액','Z-Score','cluster_group','cluster_name'] if c in df_cy_in.columns]
                            return df_cy_in[keep].copy() if keep else pd.DataFrame()
                        def _make_zip_blob(report_txt: str, context_txt: str, raw_df: pd.DataFrame) -> bytes:
                            mem = io.BytesIO()
                            with zipfile.ZipFile(mem, 'w', zipfile.ZIP_DEFLATED) as z:
                                z.writestr('report.txt', report_txt)
                                z.writestr('context.txt', context_txt)
                                z.writestr('evidence_raw.csv', raw_df.to_csv(index=False, encoding='utf-8-sig'))
                            mem.seek(0)
                            return mem.getvalue()
                        raw_df = _build_raw_evidence(st.session_state.get('last_dfcy', pd.DataFrame()))
                        st.markdown("#### 📑 근거: 선택 계정 원장(RAW) + 클러스터")
                        if not raw_df.empty:
                            st.dataframe(
                                raw_df.head(100).style.format({'발생액':'{:,.0f}','순액':'{:,.0f}','Z-Score':'{:.2f}'}),
                                use_container_width=True, height=350
                            )
                        else:
                            st.info("표시할 RAW가 없습니다.")
                        zip_bytes = _make_zip_blob(
                            report_txt=st.session_state['last_report'],
                            context_txt=st.session_state['last_context'],
                            raw_df=raw_df
                        )
                        st.download_button(
                            "📥 보고서+근거 다운로드(ZIP)",
                            data=zip_bytes,
                            file_name="ai_report_with_evidence.zip",
                            mime="application/zip",
                            key="zip_dl_cached"  # 고유 키(캐시 결과)
                        )
                        # (가능 시) 클러스터 품질 카드 표시
                        cq = st.session_state.get("cluster_quality")
                        if cq:
                            st.markdown("---")
                            st.subheader("클러스터 품질 요약")
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Noise rate", f"{cq['noise_rate']*100:.1f}%")
                            c2.metric("#Clusters", f"{cq['n_clusters']}")
                            c3.metric("Avg size", f"{cq['avg_size']:.1f}")
                            c4.metric("Rescued", f"{cq['rescued_rate']*100:.1f}%")
                            st.caption(f"Model: {cq['model']} | UMAP: {'on' if cq['umap'] else 'off'} | τ={cq['tau']:.2f} | N={cq['N']:,}")
        except Exception as e:
            st.error(f"데이터 처리 중 오류가 발생했습니다: {e}")
            if st.button("매핑 단계로 돌아가기"):
                st.session_state.mapping_confirmed = False
                st.rerun()
else:
    st.info("⬅️ 왼쪽 사이드바에서 분석할 엑셀 파일을 업로드해주세요.")


