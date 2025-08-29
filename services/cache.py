# services/cache.py  (전체 교체본)
# - 임베딩 캐시(SQLite)
# - LLM 매핑 캐시(승인/버전)
# - 캐시 정보 헬퍼

from __future__ import annotations
import os, sqlite3, json, hashlib, threading, time
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from config import EMBED_CACHE_DIR

# 모델별 DB 파일 접근 시 레이스를 막기 위한 락 맵
_LOCKS: Dict[str, threading.Lock] = {}

def _model_dir(model: str) -> Path:
    # 모델명을 안전한 폴더명으로 변환
    safe = model.replace("/", "_").replace(":", "_")
    d = Path(EMBED_CACHE_DIR) / safe
    d.mkdir(parents=True, exist_ok=True)
    return d

def _db_path(model: str) -> str:
    return str(_model_dir(model) / "embeddings.sqlite3")

def _sha1(s: str) -> str:
    # 보안용이 아닌 키 해싱(캐시 키용)
    return hashlib.sha1(s.encode("utf-8"), usedforsecurity=False).hexdigest()

def _get_lock(model: str) -> threading.Lock:
    if model not in _LOCKS:
        _LOCKS[model] = threading.Lock()
    return _LOCKS[model]

def _ensure_db(conn: sqlite3.Connection):
    # 기본 테이블 스키마 보장
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS emb (
            k TEXT PRIMARY KEY,
            text TEXT,
            vec TEXT
        )
    """)
    conn.commit()

def _open(model: str) -> sqlite3.Connection:
    p = _db_path(model)
    conn = sqlite3.connect(p, timeout=30)
    _ensure_db(conn)
    return conn

def cache_get_many(model: str, texts: List[str]) -> Dict[str, List[float]]:
    # 여러 텍스트에 대한 캐시 조회
    if not texts:
        return {}
    keys = [(t, _sha1(f"{model}|{t}")) for t in texts]
    with _get_lock(model):
        conn = _open(model)
        try:
            cur = conn.cursor()
            qmarks = ",".join("?" for _ in keys)
            cur.execute(f"SELECT k, vec FROM emb WHERE k IN ({qmarks})", [k for _, k in keys])
            rows = {k: json.loads(vec) for (k, vec) in cur.fetchall()}
        finally:
            conn.close()
    out: Dict[str, List[float]] = {}
    for t, k in keys:
        if k in rows:
            out[t] = rows[k]
    return out

def cache_put_many(model: str, pairs: List[Tuple[str, List[float]]]) -> None:
    # 여러 텍스트-벡터 쌍을 캐시에 저장
    if not pairs:
        return
    records = [(_sha1(f"{model}|{t}"), t, json.dumps(vec)) for (t, vec) in pairs]
    with _get_lock(model):
        conn = _open(model)
        try:
            conn.executemany("INSERT OR REPLACE INTO emb(k,text,vec) VALUES (?,?,?)", records)
            conn.commit()
        finally:
            conn.close()

def get_or_embed_texts(
    texts: List[str],
    client,
    *,
    model: str,
    batch_size: int,
    timeout: int,
    max_retry: int,
) -> Dict[str, List[float]]:
    # 텍스트 목록에 대해 캐시를 우선 조회하고, 누락분만 임베딩 API 호출
    texts = list(dict.fromkeys([str(t) for t in texts]))
    cached = cache_get_many(model, texts)
    missing = [t for t in texts if t not in cached]
    if not missing:
        return cached
    out = dict(cached)
    for s in range(0, len(missing), batch_size):
        sub = missing[s:s+batch_size]
        last_err = None
        for attempt in range(max_retry):
            try:
                try:
                    resp = client.embeddings.create(model=model, input=sub, timeout=timeout)
                except TypeError:
                    # 일부 클라이언트는 timeout 파라미터를 지원하지 않음
                    resp = client.embeddings.create(model=model, input=sub)
                vecs = [d.embedding for d in resp.data]
                pairs = list(zip(sub, vecs))
                cache_put_many(model, pairs)
                out.update({sub[i]: vecs[i] for i in range(len(sub))})
                last_err = None
                break
            except Exception as e:
                last_err = e
        if last_err:
            # 재시도 실패 시, 마지막 예외를 전파
            raise last_err
    return out

# ===== LLM 매핑 캐시 (승인/버전 고정) =====
DEFAULT_CACHE_PATH = os.path.join(".cache", "llm_mappings.json")

class LLMMappingCache:
    # 간단한 파일 기반 승인/버전 관리
    def __init__(self, path: str = DEFAULT_CACHE_PATH):
        self.path = path
        self._lock = threading.Lock()
        self._state: Dict[str, Any] = {"approved": {}, "proposed": {}, "versions": {}}
        self._load()

    def _load(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._state = json.load(f)
            except Exception:
                # 파손 파일은 조용히 무시
                pass

    def _save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._state, f, ensure_ascii=False, indent=2)

    def get_approved(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            item = self._state["approved"].get(key)
            return dict(item) if item else None

    def get_proposed(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            item = self._state["proposed"].get(key)
            return dict(item) if item else None

    def propose(self, key: str, value: Any, *, model: str, meta: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            self._state["proposed"][key] = {
                "value": value,
                "model": model,
                "meta": meta or {},
                "ts": time.time(),
            }
            self._save()

    def approve(self, key: str, *, value_override: Any | None = None, note: str = "") -> Dict[str, Any]:
        # 초안을 승인하여 버전을 올리고 확정
        with self._lock:
            src = self._state["proposed"].get(key) or self._state["approved"].get(key)
            if not src:
                raise KeyError(f"No proposed/approved entry for key={key!r}")
            prev_ver = int(self._state["versions"].get(key, 0))
            new_ver = prev_ver + 1
            final_value = src["value"] if value_override is None else value_override
            rec = {
                "value": final_value,
                "version": f"v{new_ver}",
                "note": note,
                "approved_ts": time.time(),
                "model": src.get("model"),
                "meta": src.get("meta", {}),
            }
            self._state["approved"][key] = rec
            self._state["versions"][key] = new_ver
            if key in self._state["proposed"]:
                del self._state["proposed"][key]
            self._save()
            return dict(rec)

# ===== 임베딩 캐시 정보 헬퍼 (app.py 사이드바 등에서 사용) =====
def get_cache_info(model: str) -> Dict[str, Any]:
    p = _db_path(model)
    info = {"model": model, "path": p, "exists": os.path.exists(p)}
    if not os.path.exists(p):
        return info
    try:
        conn = sqlite3.connect(p, timeout=10)
        _ensure_db(conn)
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM emb")
            nrows = int(cur.fetchone()[0])
            size = os.path.getsize(p)
            info.update({"rows": nrows, "size_bytes": size})
        finally:
            conn.close()
    except Exception as e:
        info["error"] = str(e)
    return info
