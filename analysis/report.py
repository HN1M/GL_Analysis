from __future__ import annotations
import pandas as pd
from typing import List, Callable, Optional, Any
from .evidence import (
    build_current_cluster_block,
    # build_previous_projection_block,  # 파일 하단 로컬 정의 사용
    build_zscore_top5_block,
    build_zscore_top5_block_for_py,
)
from .timeseries import run_timeseries_module
from .embedding import map_previous_to_current_clusters
import numpy as np
from config import PM_DEFAULT
import re
import json
import time


def _fmt_money(x):
    try:
        return f"{float(x):,.0f}원"
    except Exception:
        return str(x)


# --- 단위 강제 후처리: 억/만 → 원 단위 ---
_NUM = r'(?:\d{1,3}(?:,\d{3})*|\d+)'


def _to_int(s):
    return int(str(s).replace(',', ''))


def _replace_korean_units(m):
    # 케이스: "3억 5,072만 원" / "54억 1,444만 원" / "2억 원" / "370만 원"
    eok = m.group('eok')
    man = m.group('man')
    won = m.group('won')
    total = 0
    if eok:
        total += _to_int(eok) * 100_000_000
    if man:
        total += _to_int(man) * 10_000
    if won:
        total += _to_int(won)
    return f"{total:,.0f}원"


def _enforce_won_units(text: str) -> str:
    # 1) 억/만/원 혼합을 원 단위로 치환
    pat = re.compile(
        rf'(?:(?P<eok>{_NUM})\s*억)?\s*(?:(?P<man>{_NUM})\s*만)?\s*(?:(?P<won>{_NUM})\s*원)?'
        r'(?!\s*단위)', flags=re.IGNORECASE)

    def _smart_sub(s):
        out = []
        last = 0
        for m in pat.finditer(s):
            # 의미 없는 빈 매칭 방지: 억/만이 없으면 스킵(이미 원 단위일 가능성)
            if not any(m.group(g) for g in ('eok', 'man')):
                continue
            out.append(s[last:m.start()])
            out.append(_replace_korean_units(m))
            last = m.end()
        out.append(s[last:])
        return ''.join(out)

    return _smart_sub(text)


def _boldify_bracket_headers(text: str) -> str:
    # [요약], [주요 거래], [결론], [용어 설명] → **[...]**\n
    text = re.sub(r'^\[(요약|주요 거래|결론|용어 설명)\]\s*', r'**[\1]**\n', text, flags=re.MULTILINE)
    return text
def _strip_control(s: str) -> str:
    # 탭/개행 제외 모든 제어문자 제거 (0x00-0x1F, 0x7F)
    return re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", s or "")


# --- ModuleResult 기반 컨텍스트(경량): 정렬·금액·메모 지원 + Top-K 일관화 ---
def build_report_context_from_modules(
    modules: List["ModuleResult"],
    pm_value: float,
    topk: int = 20,
    manual_note: str = ""
) -> str:
    """
    여러 ModuleResult에서 summary/상위 evidences를 뽑아 간단한 텍스트 컨텍스트를 생성.
    - Evidence 정렬: risk_score → financial_impact 내림차순
    - 금액 표기: financial_impact를 금액으로 표기
    - 감사 메모(manual_note) 주입
    - Top-K 일관 적용
    """
    lines: List[str] = []
    lines.append(f"[PM] {pm_value:,.0f} KRW")
    for m in modules or []:
        try:
            lines.append(f"\n## Module: {getattr(m, 'name', 'module')}")
            summ = getattr(m, "summary", None)
            if summ:
                lines.append(f"- summary: {summ}")

            # Evidence 정렬(내림차순): 위험도 → 금액
            evs = list(getattr(m, "evidences", []))
            def _key(e):
                try:
                    return (
                        float(getattr(e, "risk_score", 0.0)),
                        float(getattr(e, "financial_impact", 0.0)),
                    )
                except Exception:
                    return (0.0, 0.0)
            evs = sorted(evs, key=_key, reverse=True)
            k = max(0, int(topk))

            if evs and k > 0:
                lines.append(f"- evidences(top{k}):")
                for e in evs[:k]:
                    try:
                        lnk = getattr(e, "links", {}) or {}
                        acct_nm = lnk.get("account_name", "")
                        acct_cd = lnk.get("account_code", "")
                        risk = float(getattr(e, "risk_score", 0.0))
                        kit  = bool(getattr(e, "is_key_item", False))
                        amt  = getattr(e, "financial_impact", None)
                        amt_txt = f" amount={amt:,.0f}" if isinstance(amt, (int, float)) else ""
                        measure = getattr(e, "measure", None)
                        model = getattr(e, "model", None)
                        tag = ""
                        if measure: tag += f"[{measure}]"
                        if model:   tag += f"[{model}]"
                        rsn  = str(getattr(e, "reason", ""))
                        lines.append(
                            _strip_control(
                                f"  - {tag} {acct_nm}({acct_cd}) risk={risk:.2f} KIT={kit}{amt_txt} reason={rsn}"
                            )
                        )
                    except Exception:
                        continue

            # 표 크기 요약(유지)
            tbls = getattr(m, "tables", None)
            if tbls:
                for nm, df in (tbls or {}).items():
                    try:
                        lines.append(f"- table[{nm}]: rows={len(df)} cols={len(df.columns)}")
                    except Exception:
                        pass
        except Exception:
            continue

    if manual_note:
        lines.append("\n## Auditor Note\n" + manual_note.strip())

    return _strip_control("\n".join(lines)).strip()


# Backward-compatible alias with explicit name used in app layer
def generate_rag_context_from_modules(
    modules: List["ModuleResult"],
    pm_value: float,
    topk: int = 20,
    manual_note: str = ""
) -> str:
    return build_report_context_from_modules(modules, pm_value, topk=topk, manual_note=manual_note)


def _safe_load(s: str):
    """엄격한 JSON 로더: 코드 펜스 제거 후 strict json.loads.
    주변 텍스트/마크다운 허용하지 않음.
    """
    text = (s or "").strip()
    # 시작 펜스 제거
    text = re.sub(r"^\s*```(?:json|JSON)?\s*\n", "", text)
    # 끝 펜스 제거
    text = re.sub(r"\n\s*```\s*$", "", text)
    text = text.strip()
    return json.loads(text)


def build_report_context(master_df: pd.DataFrame, current_df: pd.DataFrame, previous_df: pd.DataFrame,
                         account_codes: List[str], manual_context: str = "",
                         include_risk_summary: bool = False, pm_value: float | None = None) -> str:
    acc_info = master_df[master_df['계정코드'].astype(str).isin(account_codes)]
    acc_names = ", ".join(acc_info['계정명'].unique().tolist())
    master_summary = f"- 분석 대상 계정 그룹: {acc_names} ({', '.join(account_codes)})"
    if not acc_info.empty:
        acct_type = acc_info.iloc[0].get('BS/PL', 'PL')
        has_dates_cur = ('회계일자' in current_df.columns) and current_df['회계일자'].notna().any()
        has_dates_prev = ('회계일자' in previous_df.columns) and previous_df['회계일자'].notna().any()
        if str(acct_type).upper() == 'PL' and has_dates_cur:
            min_date = current_df['회계일자'].min(); max_date = current_df['회계일자'].max()
            if has_dates_prev:
                # 래핑 구간(예: 11~2월) 오판 방지: 연-월 Period로 비교
                cur_months = current_df['회계일자'].dt.to_period('M')
                prev_months = previous_df['회계일자'].dt.to_period('M')
                mask = prev_months.isin(cur_months.unique())
                prev_f = previous_df.loc[mask]
            else:
                prev_f = previous_df.copy()
            # Net first (순액: 차-대), absolute as reference (규모(절대값))
            cur_net = current_df.get('순액', pd.Series(dtype=float)).sum()
            prev_net = prev_f.get('순액', pd.Series(dtype=float)).sum()
            cur_abs = current_df.get('발생액', pd.Series(dtype=float)).sum()
            prev_abs = prev_f.get('발생액', pd.Series(dtype=float)).sum()
            var = cur_net - prev_net
            var_pct = (var / prev_net * 100) if prev_net not in (0, 0.0) else float('inf')
            period = f"{min_date.strftime('%m월')}~{max_date.strftime('%m월')}"
            master_summary += (
                f"\n- 당기 **순액(차-대)** 합계 ({period}): {cur_net:,.0f}원"
                f" | 전기 동기간 순액: {prev_net:,.0f}원 | 순액 증감: {var:,.0f}원 ({var_pct:+.2f}%)"
                f"\n- (참고) **규모(절대값)** 발생액: 당기 {cur_abs:,.0f}원 | 전기 {prev_abs:,.0f}원"
                f" | 차이: {cur_abs - prev_abs:,.0f}원"
            )
        else:
            cur_bal = acc_info.get('당기말잔액', pd.Series(dtype=float)).sum()
            prior_bal = acc_info.get('전기말잔액', pd.Series(dtype=float)).sum()
            var = cur_bal - prior_bal
            var_pct = (var / prior_bal * 100) if prior_bal not in (0, 0.0) else float('inf')
            master_summary += f"\n- 당기말 잔액(합산): {cur_bal:,.0f}원 | 전기말 잔액(합산): {prior_bal:,.0f}원 | 증감: {var:,.0f}원 ({var_pct:+.2f}%)"

    manual_summary = f"\n\n## 사용자 제공 추가 정보\n{manual_context}" if manual_context and not manual_context.isspace() else ""

    # --- New context layout ---
    sec_info = f"## 분석대상 계정정보\n{master_summary}{manual_summary}"
    sec_cur = build_current_cluster_block(current_df)
    # Prior-year: 전표 전체를 CY 센트로이드에 최근접 매핑하여 합산하는 증거 블록 사용
    sec_prev = build_previous_projection_block(current_df, previous_df)
    sec_top5_cy = build_zscore_top5_block(current_df, previous_df, topn=5)
    sec_top5_py = build_zscore_top5_block_for_py(previous_df, current_df, topn=5)
    sec_ts = build_timeseries_summary_block(current_df)

    # --- 위험 매트릭스 요약 제거(경영진주장/매트릭스 비활성화) ---
    # sec_risk = ""
    # if include_risk_summary:
    #     try:
    #         sec_risk = _build_risk_matrix_section(current_df, pm_value=pm_value)
    #     except Exception:
    #         sec_risk = ""

    parts = [sec_info, sec_cur, sec_prev, sec_ts, sec_top5_cy, sec_top5_py]
    # if sec_risk:
    #     parts.insert(2, sec_risk)  # 정보→(위험)→클러스터 순
    return "\n".join(parts)


def build_timeseries_summary_block(current_df: pd.DataFrame, topn: int = 5) -> str:
    """
    ## 예측 이탈 요약
    계정별 월별 합계를 기반으로 마지막 포인트의 예측 대비 이탈을 요약.
    """
    if current_df is None or current_df.empty or '회계일자' not in current_df.columns:
        return "\n\n## 예측 이탈 요약\n- (데이터 없음)"
    df = current_df.copy()
    # 날짜/계정 가드
    try:
        df['회계일자'] = pd.to_datetime(df['회계일자'], errors='coerce')
    except Exception:
        return "\n\n## 예측 이탈 요약\n- (날짜 형식 오류)"
    if '계정명' not in df.columns:
        return "\n\n## 예측 이탈 요약\n- (계정명이 필요합니다)"
    df['연월'] = df['회계일자'].dt.to_period('M')
    # 금액 컬럼 유연 인식
    cand = ['거래금액', '발생액', '거래금액_절대값', 'amount', '금액']
    val_col = next((c for c in cand if c in df.columns), None)
    if not val_col:
        return "\n\n## 예측 이탈 요약\n- (금액 컬럼을 찾지 못했습니다)"

    m = (df.groupby(['계정명','연월'], as_index=False)[val_col].sum())
    m['account'] = m['계정명']
    # Flow는 월 ‘집계’ 개념이므로 내부 앵커는 월초(start)로 통일
    m['date'] = m['연월'].dt.to_timestamp(how='start')
    m['amount'] = m[val_col]

    rows = run_timeseries_module(m[['account','date','amount']])
    if rows is None or rows.empty:
        return "\n\n## 예측 이탈 요약\n- (유의미한 이탈 없음)"

    rows = rows.sort_values('risk', ascending=False).head(int(topn))
    def _fmt_dt(x):
        try:
            import pandas as _pd
            return x.strftime('%Y-%m-%d') if _pd.notna(x) else ""
        except:
            return ""
    lines = [
        "\n\n## 예측 이탈 요약",
        "※ 기본은 '월별 발생액(Δ잔액/flow)'. BS 계정은 **balance** 기준도 내부 평가하며, 아래 표기는 MoR과 z·risk를 함께 보여줍니다."
    ]
    for _, r in rows.iterrows():
        _m = str(r.get('measure','flow'))
        _when = "월합계" if _m == "flow" else "월말"
        lines.append(
            f"- [{_fmt_dt(r['date'])}·{_when}] {r['account']} ({_m}, MoR={r.get('model','-')})"
            f" | 실제 {r['actual']:,.0f}원 vs 예측 {r['predicted']:,.0f}원"
            f" → {'상회' if float(r['error'])>0 else '하회'} | z={float(r['z']):+.2f} | risk={float(r['risk']):.2f}"
        )
    return "\n".join(lines)


def _build_risk_matrix_section(*_args, **_kwargs) -> str:
    # [Removed] CEAVOP/위험 매트릭스 섹션은 2025-08-30 기준 비활성화.
    # 재도입 전까지 빈 문자열을 반환하여 호출부를 깨지 않음.
    return ""


def build_methodology_note(report_accounts=None) -> str:
    lines = [
        "\n\n## 분석 기준(알림)",
        "- 이번 분석은 UI에서 선택된 계정 기준으로 산출되었습니다.",
        "- 요약 수치: **순액(차-대)** 기준. (발생액=규모(절대값)은 참고용)",
        "- Z-Score: 선택 계정들의 **발생액(절대값)** 분포 기준.",
        "- 유사도/근거: **적요+거래처** 임베딩 후 코사인 유사도(전기 동월 우선).",
        "- '클러스터 노이즈(-1)'는 의미가 충분히 모이지 않아 자동으로 묶이지 않은 산발적 거래 묶음입니다.",
    ]
    return "\n".join(lines)


def _format_from_json(obj: dict) -> str:
    """
    단순 스키마(JSON) → 최종 마크다운.
    - key_transactions: LLM이 작성한 전체 섹션 마크다운을 그대로 사용
    - glossary: 필수 항목 보강(없을 경우 기본 정의 추가)
    """
    summary = (obj.get("summary") or "").strip()
    kt_val = obj.get("key_transactions")
    # 과거 호환(오브젝트가 오면 텍스트로 변환 시도)
    if isinstance(kt_val, dict):
        parts = []
        for k, v in kt_val.items():
            if isinstance(v, str):
                parts.append(v.strip())
        key_tx_md = "\n\n".join(p for p in parts if p)
    else:
        key_tx_md = (kt_val or "").strip()

    conclusion = (obj.get("conclusion") or "").strip()

    md = (
        f"**[요약]**\n{summary}\n\n"
        f"**[주요 거래]**\n{key_tx_md}\n\n"
        f"**[결론]**\n{conclusion}"
    )
    return md


# --- 전기 전체 매핑 합산 방식으로 교체: 이전 전표를 CY 클러스터에 최근접 매핑 후 합산 ---
def build_previous_projection_block(current_df: pd.DataFrame, previous_df: pd.DataFrame, min_sim: float | None = None) -> str:
    """
    Project all PY vouchers onto CY cluster centroids and aggregate absolute amounts by the CY cluster_group.
    - No similarity computation is shown or used for filtering.
    - Output contains only total absolute amount and ONE example voucher (no sim).
    """
    import pandas as pd
    if current_df is None or previous_df is None or current_df.empty or previous_df.empty:
        return "\n\n## 전기 클러스터 및 금액\n- (전기 데이터 없음)"
    need_cur = {'cluster_id','cluster_name','vector'}
    if not need_cur.issubset(current_df.columns) or 'vector' not in previous_df.columns:
        return "\n\n## 전기 클러스터 및 금액\n- (클러스터/벡터 정보 부족)"

    prev_m = map_previous_to_current_clusters(current_df, previous_df)
    if prev_m is None or prev_m.empty or 'mapped_cluster_id' not in prev_m.columns:
        return "\n\n## 전기 클러스터 및 금액\n- (매핑 실패)"

    if 'cluster_group' in current_df.columns:
        id2group = current_df.drop_duplicates('cluster_id').set_index('cluster_id')['cluster_group'].to_dict()
    else:
        id2group = current_df.drop_duplicates('cluster_id').set_index('cluster_id')['cluster_name'].to_dict()

    prev_m = prev_m.copy()
    prev_m['mapped_group'] = prev_m['mapped_cluster_id'].map(id2group)
    prev_m['abs_amt'] = prev_m.get('발생액', pd.Series(dtype=float)).abs()

    agg = (
        prev_m.groupby('mapped_group', dropna=False)
              .agg(규모=('abs_amt','sum'))
              .reset_index()
              .sort_values('규모', ascending=False)
    )

    lines = ["\n\n## 전기 클러스터 및 금액"]
    for _, row in agg.iterrows():
        g = row['mapped_group'] if pd.notna(row['mapped_group']) else '(미매핑)'
        tot = row['규모']
        sub = prev_m[prev_m['mapped_group'] == row['mapped_group']]
        if not sub.empty:
            ex = sub.sort_values('abs_amt', ascending=False).head(1).iloc[0]
            raw_dt = ex.get('회계일자', None)
            if pd.notna(raw_dt):
                try:
                    _dt = pd.to_datetime(raw_dt, errors='coerce')
                    dt = _dt.strftime('%Y-%m-%d') if pd.notna(_dt) else ''
                except Exception:
                    dt = ''
            else:
                dt = ''
            lines.append(f"- [{g}] 규모(절대값) {tot:,.0f}원")
            lines.append(f"  · 예시: {dt} | {ex.get('거래처','')} | {int(ex.get('발생액',0)):,.0f}원")
        else:
            lines.append(f"- [{g}] 규모(절대값) {tot:,.0f}원")
    return "\n".join(lines)


def run_final_analysis(
    context: str,
    account_codes: list[str],
    *,
    model: str | None = None,
    max_tokens: int | None = 16000,
    generate_fn: Optional[Callable[..., str]] = None,
) -> str:
    system = (
        "You are a CPA. Do all hidden reasoning internally and output ONLY the JSON object in the EXACT schema below. "
        "Language: Korean (ko-KR) for every natural-language value.\n"
        "Schema: {"
        '"summary": str,'
        '"key_transactions": str,'
        '"conclusion": str,'
        '"glossary": [str]'
        "}\n"
        "Authoring rules:\n"
        "• Monetary values MUST be formatted in KRW like '1,234원' (never 억/만원).\n"
        "• [요약]은 계정군 수준의 변동과 규모를 한 문장으로 명료히.\n"
        "• [주요 거래]는 전체 서술을 네가 설계하되, 컨텍스트(CY/PY 클러스터, 매핑, Z-score 상위 항목)를 근거로 구성 비중/이상치/전기 대응관계를 자연스럽게 녹여라. 필요하면 불릿·소제목을 임의로 사용해 가독성을 높여라.\n"
        "• [결론]은 원인·리스크·통제·액션아이템 중심으로 실무적 제안 위주로 작성한다.\n"
        "• [용어 설명]에는 반드시 다음 두 항목이 포함되도록 한다:   1) '클러스터 노이즈(-1)' 정의, 2) 'Z-Score'가 ‘평균에서 몇 표준편차’인지의 직관적 의미.\n"
        "Compliance: Output MUST be the JSON object itself, with no markdown/code-fences or extra text."
    )
    user = (
        f"Target accounts: {', '.join(account_codes)}\n"
        f"{context}\n"
        "Return ONLY the JSON per schema via function call."
    )

    tool_schema = {
        "type": "function",
        "function": {
            "name": "emit_report",
            "description": "Return the report strictly in the fixed JSON schema.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "key_transactions": {"type": "string"},
                    "conclusion": {"type": "string"},
                    "glossary": {"type": "array","items":{"type":"string"}}
                },
                "required": ["summary","key_transactions","conclusion","glossary"]
            }
        }
    }

    max_retries = 2
    last_err = None

    for attempt in range(max_retries + 1):
        try:
            if generate_fn is None:
                raise RuntimeError("generate_fn not provided (LLM dependency must be injected)")
            raw = generate_fn(
                system=system, user=user, model=model,
                max_tokens=max_tokens, tools=[tool_schema], force_json=False
            )
            obj = _safe_load(raw)
            text = _format_from_json(obj)
            return _enforce_won_units(text)
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(1.0)
            continue

    raise ValueError(f"LLM failed to produce valid JSON report after retries. Details: {last_err}")


# --- NEW: LLM 미사용/실패 시 폴백 리포트 (순수 로컬 계산) ---
def run_offline_fallback_report(current_df: pd.DataFrame,
                                previous_df: pd.DataFrame,
                                account_codes: list[str],
                                pm_value: float | None = None) -> str:
    """
    외부 LLM을 전혀 사용하지 않고 간단 보고서를 생성한다.
    - 요약: CY/PY 순액/규모 비교
    - 주요 거래: |Z| Top 5 (가능하면 Z 기준, 없으면 발생액 상위)
    - 결론: KIT(≥PM) 건수/비중 및 리스크 주의
    - 용어: Z-Score, Key Item 기본 정의
    """
    pm = float(pm_value) if pm_value is not None else float(PM_DEFAULT)
    cur = current_df.copy()
    prev = previous_df.copy()

    def _safe_sum(df, col): 
        return float(df.get(col, pd.Series(dtype=float)).sum()) if not df.empty else 0.0

    cur_net  = _safe_sum(cur, "순액")
    prev_net = _safe_sum(prev, "순액")
    cur_abs  = _safe_sum(cur, "발생액")
    prev_abs = _safe_sum(prev, "발생액")

    var_net = cur_net - prev_net
    var_abs = cur_abs - prev_abs
    var_pct = (var_net / prev_net * 100.0) if prev_net not in (0, 0.0) else float("inf")

    # Top 5: Z-Score 우선, 없으면 발생액 상위
    top_df = cur.copy()
    if "Z-Score" in top_df.columns and top_df["Z-Score"].notna().any():
        top_df = top_df.reindex(top_df["Z-Score"].abs().sort_values(ascending=False).index)
    else:
        top_df = top_df.reindex(top_df.get("발생액", pd.Series(dtype=float)).abs().sort_values(ascending=False).index)
    top_df = top_df.head(5)

    # KIT 집계(절대발생액 기준)
    kit_mask = top_df.get("발생액", pd.Series(dtype=float)).abs() >= pm if not top_df.empty else pd.Series([], dtype=bool)
    kit_cnt  = int(kit_mask.sum()) if not top_df.empty else 0

    def _fmt_dt(x):
        try:
            import pandas as _pd
            return x.strftime("%Y-%m-%d") if _pd.notna(x) else ""
        except Exception:
            return ""

    # Compose sections (간단 Markdown)
    summary = (
        f"선택 계정({', '.join(account_codes)}) 기준으로 당기 **순액** {cur_net:,.0f}원,"
        f" 전기 {prev_net:,.0f}원 → 증감 {var_net:,.0f}원 ({var_pct:+.2f}%).\n"
        f"(참고) **규모(발생액 절대값)** 당기 {cur_abs:,.0f}원, 전기 {prev_abs:,.0f}원 → 차이 {var_abs:,.0f}원."
    )

    kt_lines = []
    if not top_df.empty:
        for i, (_, r) in enumerate(top_df.iterrows(), 1):
            dt = _fmt_dt(r.get("회계일자"))
            vend = str(r.get("거래처", "") or "")
            amt = float(r.get("발생액", 0.0))
            z   = r.get("Z-Score", np.nan)
            ztxt = f" | Z={float(z):+.2f}" if not pd.isna(z) else ""
            kt_lines.append(f"- [{i}] {dt} | {vend} | {amt:,.0f}원{ztxt}")
    key_tx = "\n".join(kt_lines) if kt_lines else "- 상위 항목을 산출할 데이터가 없습니다."

    conclusion = (
        f"PM {pm:,.0f}원 기준 **Key Item(KIT)** 후보는 상위 리스트 중 {kit_cnt}건입니다. "
        "Z-Score가 큰 항목은 적요·거래처 등 근거 확인과 원인 파악이 필요합니다. "
        "주요 변동은 월별 추이/상관 분석과 함께 교차검토하는 것을 권장합니다."
    )
    return (
        f"**[요약]**\n{summary}\n\n"
        f"**[주요 거래]**\n{key_tx}\n\n"
        f"**[결론]**\n{conclusion}"
    )

