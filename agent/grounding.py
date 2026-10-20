"""[개선 4] 답변 근거 검사 — validator가 '형식'뿐 아니라 '사실'도 보게 한다.

원본 validate_node의 합격 기준은 "30자 이상 + '1)' 포함"뿐이었다. 실측에서 이런 답이 1회 만에
합격했다.
  - "최근 로그에서 반복 호출 또는 배치 작업 패턴 없음"  ← 로그를 조회하는 도구가 없다(지어냄)
  - "web-01의 CPU 사용률이 급증한 것으로 확인됨"        ← web-01을 조사하지 않았다
  - "cache-service도 동일한 리소스를 공유 … 추정"       ← cache-service를 조사하지 않았다

LLM-as-judge(qwen3:8b로 채점)도 시험해 봤지만 문제 있는 답 3개에 모두 5점을 줄 만큼 관대했다.
그래서 결정론적으로 확인할 수 있는 세 가지는 코드로 검사한다(W6·W8 정신 — 검증은 코드가).
  ① 조사하지 않은 시스템을 언급했는가
  ② 수치(%, s, ms)가 수집된 사실(모니터링 값·알림 문구·runbook 본문)에 있는가
  ③ 수집하지 않은 '로그 분석 결과'를 단정했는가
"""
import re

from .systems import KNOWN_SYSTEMS, SYSTEM_LIKE

# 숫자 + 단위("92%", "2.4s", "1.0s")만 검사한다. "504"(상태코드)처럼 단위 없는 숫자는 오탐을 줄이려고 제외.
NUMBER_WITH_UNIT = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s?(%|ms|s|초)(?![a-zA-Z])")
# "로그 … 결과/없음/확인됨" 같은 '단정' 표현만 잡는다. "로그를 확인하세요"(권고)는 통과.
# "로그인"은 로그가 아니다 — (?!인)으로 제외 ("로그인 응답이 … 진단 결과"가 오탐되던 것 수정).
LOG_CLAIM = re.compile(r"로그(?!인)[^.\n]{0,25}(?:결과|없음|없었|확인됨|확인되었|발견되지|나타나지)")


def _fact_numbers(collected: list[dict]) -> set[str]:
    """수집된 사실 안의 모든 숫자를 문자열로 모은다. 92 / 92.0 처럼 표기가 달라도 맞도록 정규화."""
    text_parts = []
    for c in collected:
        data = c.get("data")
        if isinstance(data, dict):
            # [8단계 수정] 조회 시각(last_checked)은 사실 수치가 아니다 — 원래는 "10:41"의 41이 사실로
            # 들어가 지어낸 "41%"가 통과할 수 있었다.
            text_parts += [str(v) for k, v in data.items() if k != "last_checked"]
        else:
            text_parts.append(str(data))
    nums = set()
    for n in re.findall(r"\d+(?:\.\d+)?", " ".join(text_parts)):
        nums.add(_norm(n))
    return nums


def _norm(n: str) -> str:
    return str(float(n)).rstrip("0").rstrip(".")


def check_grounding(draft: str, collected: list[dict], allowed_systems: list[str] | None = None) -> list[str]:
    """문제 목록을 돌려준다(빈 목록 = 통과). allowed_systems: 조사했거나 '조사 안 함'을 안내해야 하는 시스템."""
    issues: list[str] = []
    probed = {c["data"]["system_name"] for c in collected
              if c.get("source") == "monitoring" and isinstance(c.get("data"), dict)}
    allowed = probed | set(allowed_systems or [])

    for name in dict.fromkeys(SYSTEM_LIKE.findall(draft)):
        if name not in allowed:
            what = "조사하지 않은" if name in KNOWN_SYSTEMS else "모니터링 대상이 아닌"
            issues.append(f"{what} 시스템 {name}에 대해 단정했습니다. 수집된 사실에 있는 시스템만 언급하세요.")

    facts = _fact_numbers(collected)
    bad = [f"{n}{u}" for n, u in NUMBER_WITH_UNIT.findall(draft) if _norm(n) not in facts]
    if bad:
        issues.append(f"수집된 사실에 없는 수치가 있습니다({', '.join(dict.fromkeys(bad))}). "
                      f"수치는 수집된 값을 그대로 인용하세요.")

    has_log_source = any(c.get("source") == "log" for c in collected)
    if not has_log_source and LOG_CLAIM.search(draft):
        issues.append("로그는 수집하지 않았습니다. 로그 분석 '결과'를 단정하지 말고 '로그 확인 필요'로 쓰세요.")
    return issues
