# 리포트 내 원(₩) 단위 강제 변환 테스트
from analysis.report import _enforce_won_units

def test_unit_enforcement():
    s = "총액은 3억 5,072만 원이며 이전에는 2억 원이었다."
    out = _enforce_won_units(s)
    assert "350,720,000원" in out
    assert "200,000,000원" in out


