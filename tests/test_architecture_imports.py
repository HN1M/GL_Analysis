from pathlib import Path

FORBIDDEN = ("from services", "import services")


def test_analysis_not_import_services():
    bad = []
    for p in Path("analysis").rglob("*.py"):
        try:
            t = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if any(x in t for x in FORBIDDEN):
            bad.append(str(p))
    assert not bad, f"analysis must not import services: {bad}"


