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
from utils.helpers import find_column_by_keyword, add_provenance_columns, add_period_tag
from analysis.integrity import analyze_reconciliation, run_integrity_module
from analysis.contracts import LedgerFrame, ModuleResult
from analysis.trend import create_monthly_trend_figure, run_trend_module
from analysis.timeseries import (
    model_registry,
    run_timeseries_module_with_flag,
    create_timeseries_figure
)
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
)
from analysis.anomaly import calculate_grouped_stats_and_zscore
from services.llm import LLMClient
from services.cache import get_or_embed_texts
from services.cycles_store import get_effective_cycles
from config import EMB_USE_LARGE_DEFAULT, HDBSCAN_RESCUE_TAU
try:
    from config import PM_DEFAULT
except Exception:
    PM_DEFAULT = 500_000_000
from utils.viz import add_materiality_threshold, add_pm_badge

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
                tab_integrity, tab_vendor, tab_anomaly, tab_ts, tab_report = st.tabs(["🌊 데이터 무결성 및 흐름", "🏢 거래처 심층 분석", "🔬 이상 패턴 탐지", "📉 시계열 예측", "🧠 분석 종합 대시보드"])

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
                        account_list, default=[]
                    )
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
                    # ✅ 버튼 없이 즉시 렌더: 계정 2개 이상 선택 + 임계치 슬라이더 제공
                    corr_accounts = st.multiselect(
                        "상관 분석 대상 계정(2개 이상 선택)",
                        account_list,
                        default=selected_accounts,
                        help="선택한 계정들 간 월별 흐름의 피어슨 상관을 계산합니다."
                    )
                    corr_thr = st.slider(
                        "상관 임계치(강한 상관쌍 표 전용)",
                        min_value=0.50, max_value=0.95, step=0.05, value=0.70,
                        help="절대값 기준 임계치 이상인 계정쌍만 표에 표시합니다."
                    )
                    if len(corr_accounts) < 2:
                        st.info("계정을 **2개 이상** 선택하면 히트맵이 표시됩니다.")
                    else:
                        lf_use = _lf_by_scope()
                        mdf = st.session_state.master_df
                        codes = mdf[mdf['계정명'].isin(corr_accounts)]['계정코드'].astype(str).tolist()
                        cmod = run_correlation_module(
                            lf_use,
                            accounts=codes,
                            corr_threshold=float(corr_thr),
                            cycles_map=get_effective_cycles(),
                        )
                        _push_module(cmod)
                        for w in cmod.warnings:
                            st.warning(w)
                        if cmod.figures:
                            stable_codes = "_".join(map(str, codes)) or "all"
                            stable_thr = str(int(corr_thr*100))
                            st.plotly_chart(
                                cmod.figures['heatmap'],
                                use_container_width=True,
                                key=f"corr_heatmap_{stable_codes}_{stable_thr}"
                            )
                        if 'strong_pairs' in cmod.tables and not cmod.tables['strong_pairs'].empty:
                            st.markdown("**임계치 이상 상관쌍**")
                            st.dataframe(cmod.tables['strong_pairs'], use_container_width=True)
                        if 'excluded_accounts' in cmod.tables and not cmod.tables['excluded_accounts'].empty:
                            with st.expander("제외된 계정 보기(변동없음/활동월 부족)", expanded=False):
                                st.dataframe(cmod.tables['excluded_accounts'], use_container_width=True)

                with tab_vendor:
                    st.header("거래처 심층 분석")
                    st.caption(f"🔎 현재 스코프: {st.session_state.get('period_scope','당기')}")

                    st.subheader("거래처 집중도 및 활동성 (계정별)")
                    master_df_res = st.session_state.master_df
                    account_list_vendor = master_df_res['계정명'].unique()
                    selected_accounts_vendor = st.multiselect("분석할 계정(들)을 선택하세요.", account_list_vendor, default=[])

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
                    with st.expander("🧭 해석 가이드", expanded=False):
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
                    # 모델 가용 배지(디버깅 겸 사용자 안내)
                    try:
                        # model_registry is imported at the top
                        _reg = model_registry()
                        st.caption(f"지원 모델: EMA ✓ · MA ✓ · ARIMA {'✓' if _reg['arima'] else '—'} · Prophet {'✓' if _reg['prophet'] else '—'}")
                    except Exception:
                        pass
                    # (중복 가이드 제거됨)
                    lf_use = _lf_by_scope()
                    mdf = st.session_state.master_df
                    dfm = lf_use.df.copy()
                    dfm['연월'] = dfm['회계일자'].dt.to_period('M').dt.to_timestamp('M')
                    agg = (dfm.groupby(['계정명','연월'])['거래금액'].sum()
                               .reset_index().rename(columns={'계정명':'account','연월':'date','거래금액':'amount'}))
                    pick_accounts_ts = st.multiselect("대상 계정", sorted(agg['account'].unique()), default=[], key="ts_accounts")
                    use_ts = agg if not pick_accounts_ts else agg[agg['account'].isin(pick_accounts_ts)]
                    # BS 여부를 반영해 balance 기준도 병행 계산
                    try:
                        bs_map = st.session_state.master_df[['계정명','BS/PL']].drop_duplicates()
                        _bs_flag = bs_map.set_index('계정명')['BS/PL'].map(lambda x: str(x).upper()== 'BS').to_dict()
                    except Exception:
                        _bs_flag = {}
                    work_ts = use_ts.copy()
                    work_ts['is_bs'] = work_ts['account'].map(lambda name: bool(_bs_flag.get(str(name), False)))

                    res = run_timeseries_module_with_flag(work_ts,
                                       account_col='account', date_col='date', amount_col='amount', is_bs_col='is_bs',
                                       pm_value=float(st.session_state.get("pm_value", PM_DEFAULT)))
                    if not res.empty:
                        out = res.copy()
                        out = out.rename(columns={'account':'계정'})
                        for c in ['actual','predicted','error','z','risk']:
                            out[c] = pd.to_numeric(out[c], errors='coerce')
                        # 사용자 친화적 표기(기준): 발생액/잔액
                        try:
                            out['measure'] = out['measure'].map(lambda m: '발생액(flow)' if str(m)=='flow' else ('잔액(balance)' if str(m)=='balance' else str(m)))
                        except Exception:
                            pass
                        _disp = out[['date','계정','measure','model','actual','predicted','error','z','risk']].rename(columns={
                            'date': '월',
                            'measure': '기준(Measure)',
                            'model': '모델(MoR)',
                            'actual': '실제(월 합계)',
                            'predicted': '예측(월 합계)',
                            'error': '차이(실제-예측)',
                            'z': '표준화지수(z)',
                            'risk': '위험도(0~1)'
                        })
                        st.caption("MoR(최적 모델) 기준. BS 계정은 balance 기준도 함께 표시합니다.")
                        st.dataframe(_disp.style.format({
                            '실제(월 합계)':'{:,.0f}', '예측(월 합계)':'{:,.0f}', '차이(실제-예측)':'{:,.0f}', '표준화지수(z)':'{:+.2f}', '위험도(0~1)':'{:.2f}'
                        }), use_container_width=True)

                        # === 라인차트 ===
                        st.markdown("#### 라인차트")
                        # 월별 집계에서 flow/balance 히스토리 구성
                        hist_base = use_ts.rename(columns={'amount':'flow'}).sort_values('date').copy()
                        hist_base['balance'] = hist_base['flow']
                        # 계정별 opening (=전기말잔액) 맵
                        _open = st.session_state.master_df[['계정명','전기말잔액']].drop_duplicates()
                        opening_map = _open.set_index('계정명')['전기말잔액'].to_dict()

                        def _apply_opening(g):
                            acc_name = str(g['account'].iloc[0])
                            opn = float(opening_map.get(acc_name, 0.0))
                            g = g.copy()
                            g['balance'] = opn + g['flow'].astype(float).cumsum()
                            return g

                        hist_base = hist_base.groupby('account', group_keys=False).apply(_apply_opening)

                        # 계정 선택
                        sel_acc = st.selectbox("계정 선택(라인차트)", sorted(hist_base['account'].unique()), key="ts_plot_acc_main")

                        # BS/PL 판단
                        _mdf = st.session_state.master_df[['계정코드','계정명','BS/PL','차변/대변']].drop_duplicates()
                        is_bs = bool(_mdf[_mdf['계정명'] == sel_acc]['BS/PL'].astype(str).str.upper().eq('BS').any())

                        cur_hist = hist_base[hist_base['account'] == sel_acc].copy()
                        if cur_hist.empty:
                            st.info("선택 계정의 월별 데이터가 없습니다.")
                        else:
                            # (기존 학습/모델 표기 로직 제거 — create_timeseries_figure에서 메타와 지표 제공)

                            # 대변계정(부채·자본·수익)인 경우 그래프 부호 반전
                            try:
                                from utils.helpers import is_credit_account
                                # Master에서 해당 계정의 속성 조회
                                _row = _mdf[_mdf['계정명'] == sel_acc].iloc[0] if not _mdf[_mdf['계정명'] == sel_acc].empty else None
                                acc_type = _row.get('BS/PL', 'PL') if _row is not None else 'PL'
                                dc_flag = _row.get('차변/대변') if _row is not None else None
                                if is_credit_account(acc_type if acc_type in ['부채','자본','수익'] else None, dc_flag):
                                    cur_hist = cur_hist.copy()
                                    cur_hist['flow'] = -cur_hist['flow']
                                    cur_hist['balance'] = -cur_hist['balance']
                            except Exception:
                                pass

                            # UI: 공통 옵션 설정
                            show_dividers = st.toggle("연/분기 구분선 표시", value=True, key=f"ts_dividers_toggle_{sel_acc}")
                            pm_val_current = float(st.session_state.get("pm_value", PM_DEFAULT))

                            # Helper to render figure and stats
                            def _render_fig_and_stats(fig, stats, key_suffix):
                                if fig and stats:
                                    try:
                                        diag = stats.get("diagnostics", {})
                                        pval = diag.get("p_value")
                                        b1, b2, b3 = st.columns(3)
                                        b1.caption(f"계절성: {'강함' if diag.get('seasonality') else '약함'}")
                                        ptxt = "" if pval is None or (isinstance(pval, float) and np.isnan(pval)) else f" (p={pval:.3f})"
                                        b2.caption(f"정상성: {'확보' if diag.get('stationary') else '미확보'}" + ptxt)
                                        b3.caption(f"데이터: {diag.get('n_months')}개월 — {'충분' if not diag.get('is_short') else '짧음'}")
                                    except Exception:
                                        pass
                                    st.plotly_chart(fig, use_container_width=True, key=f"ts_line_{sel_acc}_{key_suffix}")
                                    try:
                                        meta = stats.get("metadata", {})
                                        metrics = stats.get("metrics", {})
                                        mae, mape = metrics.get('mae'), metrics.get('mape')
                                        aic, bic = metrics.get('aic'), metrics.get('bic')
                                        st.caption(
                                            f"선택모델: **{meta.get('model')}** · 학습기간: {meta.get('data_span')} ({meta.get('train_months')}개월) · "
                                            f"σ윈도우: {meta.get('sigma_window')}개월 · MAE: {mae:,.0f}원 · MAPE: {mape:.1f}% · "
                                            f"AIC: {aic if isinstance(aic, float) and np.isfinite(aic) else '—'} · BIC: {bic if isinstance(bic, float) and np.isfinite(bic) else '—'}"
                                        )
                                        if meta.get('reasoning'):
                                            st.info(meta.get('reasoning'))
                                    except Exception:
                                        pass
                                    # 4. Detailed Stats Expander
                                    with st.expander("이 차트의 통계 설정 보기", expanded=False):
                                        st.write(stats.get("details"))
                                elif stats and "error" in stats:
                                    st.warning(stats["error"])
                                else:
                                    st.info("차트를 생성할 데이터가 부족합니다.")

                            if is_bs:
                                pair = st.toggle("쌍차트 보기(Flow+Balance)", value=True)
                                if pair:
                                    c1, c2 = st.columns(2)
                                    with c1:
                                        f1, st1 = create_timeseries_figure(
                                            cur_hist, 'flow', f"{sel_acc} — Flow (actual vs MoR)",
                                            pm_value=pm_val_current,
                                            show_dividers=show_dividers
                                        )
                                        _render_fig_and_stats(f1, st1, "flow")
                                    with c2:
                                        f2, st2 = create_timeseries_figure(
                                            cur_hist, 'balance', f"{sel_acc} — Balance (actual vs MoR)",
                                            pm_value=pm_val_current,
                                            show_dividers=show_dividers
                                        )
                                        _render_fig_and_stats(f2, st2, "balance")
                                else:
                                    fig, stx = create_timeseries_figure(
                                        cur_hist, 'flow', f"{sel_acc} — Flow (actual vs MoR)",
                                        pm_value=pm_val_current,
                                        show_dividers=show_dividers
                                    )
                                    _render_fig_and_stats(fig, stx, "flow_single")
                            else:
                                fig, stx = create_timeseries_figure(
                                    cur_hist, 'flow', f"{sel_acc} — Flow (actual vs MoR)",
                                    pm_value=pm_val_current,
                                    show_dividers=show_dividers
                                )
                                _render_fig_and_stats(fig, stx, "flow_only")

                        # (삭제됨) 막대 대조 UI — 오류 원인 경로 차단

                    else:
                        st.info("예측을 표시할 충분한 월별 데이터가 없습니다.")
                # ⚠️ 기존 tab5(위험평가) 블록 전체 삭제됨
                
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
                        default=[]
                    )
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
                        mdf[mdf['계정명'].isin(pick_accounts)]['계정코드']
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
                                    emb_client = LLMClient(model=st.session_state.get('llm_model')).client  # OpenAI 클라이언트 객체
                                    # 보고서 생성을 위해 LLM 기반 클러스터 네이밍을 필수로 요구
                                    df_clu, ok = perform_embedding_and_clustering(
                                        df_cy_small, emb_client,
                                        name_with_llm=True, must_name_with_llm=True,
                                        use_large=bool(st.session_state.get("use_large_embedding", False)),
                                        rescue_tau=float(st.session_state.get("rescue_tau", HDBSCAN_RESCUE_TAU)),
                                        llm_model=st.session_state.get('llm_model', 'gpt-4o'),
                                        embed_texts_fn=get_or_embed_texts,
                                    )
                                    if ok:
                                        # 유사한 클러스터 이름을 LLM으로 통합
                                        from analysis.embedding import unify_cluster_names_with_llm, unify_cluster_labels_llm
                                        df_clu, name_map = unify_cluster_names_with_llm(
                                            df_clu, emb_client,
                                            llm_model=st.session_state.get('llm_model', 'gpt-4o'),
                                            embed_texts_fn=get_or_embed_texts
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
                                                                            cycles_map=get_effective_cycles()))
                                except Exception as _e:
                                    st.warning(f"correlation 모듈 실패: {_e}")
                                # 정합성(레거시→DTO)
                                try:
                                    _push_module(run_integrity_module(ldf, mdf))
                                except Exception as _e:
                                    st.warning(f"integrity 모듈 실패: {_e}")
                                # NEW: 시계열 포함(집계→DTO 래핑)
                                try:
                                    if not df_cy.empty:
                                        ts = df_cy.copy()
                                        ts["date"] = pd.to_datetime(ts["회계일자"], errors="coerce").dt.to_period("M").dt.to_timestamp()
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
                                                                  tables={"ts": df_ts}, figures={}, evidences=[], warnings=[]))
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


