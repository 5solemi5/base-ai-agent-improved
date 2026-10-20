"""[개선 2·5] 시스템 이름 처리의 단일 출처 (Single Source of Truth).

원본에서는 graph.py / ollama_llm.py / openai_llm.py가 각자 _load_known_systems()·
_guess_target()을 복사해 두고 있었고, 그 정규화 규칙 때문에 실측에서 세 가지 문제가 났다.

  ① 다중 시스템 누락 — 모델이 target을 "check auth-service connection pool"처럼 문장으로
     내면 KNOWN_SYSTEMS에 없으니 _guess_target()이 쿼리의 '첫 번째' 시스템으로 덮어쓰고,
     두 번째 task는 중복이라 버려졌다 (실측: web-01만 조사).
  ② 미등록 시스템 바꿔치기 — billing-service처럼 모니터링 대상이 아닌 이름은 조용히
     auth-service로 바뀌어, 다른 서비스의 진단서가 나갔다.
  ③ 과잉 조사 — thinking을 끄면 소형 모델이 쿼리에 없는 시스템까지 4~5개씩 계획했다.

그래서 "사용자가 직접 말한 시스템"을 계획의 기준으로 삼고, 모델 출력은 그 범위 안에서만
받아들이도록 규칙을 한 곳에 모은다(W6 권위 게이트 정신 — 결정은 코드가 한다).
"""
import json
import re
from pathlib import Path

MONITORING_JSON = Path(__file__).resolve().parent.parent / "data" / "monitoring.json"
DEFAULT_SYSTEM = "auth-service"   # 시스템을 전혀 특정할 수 없을 때만 쓰는 기본값


def load_known_systems(path: Path = MONITORING_JSON) -> frozenset[str]:
    """data/monitoring.json의 키 = 모니터링 대상 시스템 목록.

    파일을 못 읽으면 빈 목록이 아니라 예외를 그대로 올린다 — 원본은 조용히 2개짜리
    폴백 목록으로 바뀌어서, 경로가 틀어져도(예: 노드를 하위 폴더로 옮김) 아무도
    눈치채지 못했다. 데이터가 없으면 서버가 뜨지 않는 편이 낫다."""
    return frozenset(json.loads(path.read_text(encoding="utf-8")).keys())


KNOWN_SYSTEMS = load_known_systems()

# 시스템 이름 없이 증상만 말하는 경우를 위한 한국어 별칭.
# ("로그인 응답이 느려요" → auth-service). 순서대로 검사해 처음 걸린 것만 쓴다.
ALIASES: list[tuple[str, str]] = [
    (r"로그인|인증|토큰|auth", "auth-service"),
    (r"결제|payment", "payments-service"),
    (r"캐시|cache|redis", "cache-service"),
    (r"게이트웨이|gateway|api ?gw", "gateway-service"),
    (r"로그 ?수집|디스크|log-service", "log-service"),
    (r"웹 ?서버|web-01", "web-01"),
]

# "xxx-service", "xxx-01" 모양의 단어 — 모니터링 목록에 없는 이름(billing-service)을 찾을 때 쓴다.
# \b 대신 영문·숫자 기준 경계를 쓴다 — 한국어 조사가 바로 붙으면("payments-service랑") \b가 안 걸린다.
SYSTEM_LIKE = re.compile(r"(?<![a-z0-9-])([a-z][a-z0-9]*-(?:service|\d+))(?![a-z0-9])")

# 장애 신고로 볼 단어들. fake 라우터와 ollama/openai 라우터의 JSON 파싱 실패 폴백이 함께 쓴다.
# 원본 정규식(장애|진단|다운|지연|급증|incident|티켓)은 "느려/타임아웃/저하/치솟"을 몰라서
# fake 백엔드에서 장애 쿼리 6개 중 5개가 chat으로 빠졌다.
INCIDENT_PATTERN = re.compile(
    r"장애|진단|다운|지연|급증|incident|티켓|느려|느림|느린|타임아웃|timeout|저하|치솟|"
    r"오류|에러|error|실패|안 ?돼|안 ?됨|응답이|원인|cpu|메모리|디스크|커넥션|5\d\d",
    re.IGNORECASE)


def systems_in_text(text: str) -> list[str]:
    """text에 '그대로' 등장하는 모니터링 대상 시스템을 등장 순서대로 (중복 제거)."""
    found = sorted((text.index(s), s) for s in KNOWN_SYSTEMS if s in text)
    return list(dict.fromkeys(s for _, s in found))


def unknown_systems_in_text(text: str) -> list[str]:
    """시스템 이름처럼 생겼지만 모니터링 목록에 없는 것 (예: billing-service)."""
    return [s for s in dict.fromkeys(SYSTEM_LIKE.findall(text)) if s not in KNOWN_SYSTEMS]


def alias_system(text: str) -> str | None:
    lowered = text.lower()
    for pattern, system in ALIASES:
        if re.search(pattern, lowered):
            return system
    return None


def resolve_query_systems(query: str) -> dict:
    """쿼리에서 조사 대상을 결정한다. 결과는 세 갈래 중 하나다.

    - mentioned: 쿼리에 시스템 이름이 직접 나옴 → 그 시스템들만 조사 (source="query")
    - unknown만 있음: 모니터링 대상이 아닌 이름만 나옴 → 조사하지 않고 정직하게 안내
    - 아무것도 없음: 별칭(로그인→auth-service)으로 추정, 그것도 없으면 DEFAULT_SYSTEM
    """
    mentioned = systems_in_text(query)
    unknown = unknown_systems_in_text(query)
    if mentioned:
        return {"systems": mentioned, "unknown": unknown, "source": "query"}
    if unknown:
        return {"systems": [], "unknown": unknown, "source": "unknown-only"}
    alias = alias_system(query)
    if alias:
        return {"systems": [alias], "unknown": [], "source": "alias"}
    return {"systems": [DEFAULT_SYSTEM], "unknown": [], "source": "default"}


def normalize_plan(raw: dict, query: str) -> dict:
    """LLM이 낸 계획(raw)을 실행 가능한 tasks로 정규화한다. ollama/openai가 같이 쓴다.

    원칙: 조사 대상 시스템은 '쿼리가 결정'하고, 모델에게서는 analysis 필요 여부만 받는다.
      - 쿼리에 시스템이 있으면 그 시스템 전부를 조사한다(모델이 빠뜨려도 채움 → 누락 방지).
      - 모델이 쿼리에 없는 시스템을 넣어도 버린다(과잉 조사 방지).
      - 별칭("로그인"→auth-service)으로 정해졌으면 그대로 쓴다.
      - 아무 단서도 없을 때(default)만 모델 target 문자열 '안'에서 찾은 첫 시스템을 쓴다.
        (모델이 시스템을 전부 나열해도 하나만 — 단위 테스트가 잡아낸 과잉 조사 경로)
    """
    if not isinstance(raw, dict):
        raw = {}
    raw_tasks = raw.get("tasks") if isinstance(raw.get("tasks"), list) else []
    resolved = resolve_query_systems(query)
    targets = list(resolved["systems"])

    if resolved["source"] == "default":
        # 쿼리에 아무 단서가 없을 때만 — 모델이 target 문장 안에서 실제 시스템을 짚었다면 첫 번째 하나를 존중
        from_model = [s for t in raw_tasks if isinstance(t, dict)
                      for s in systems_in_text(str(t.get("target", "")))]
        if from_model:
            targets = from_model[:1]

    tasks = [{"id": f"task-{i}", "agent": "research", "kind": "monitoring",
              "target": s, "status": "pending"} for i, s in enumerate(targets)]
    wants_analysis = any(isinstance(t, dict) and t.get("agent") == "analysis" for t in raw_tasks)
    # 비교 분석은 조사 대상이 2개 이상일 때만 의미가 있다 — 1개짜리 분석은 LLM 호출 낭비
    if wants_analysis and len(targets) >= 2:
        tasks.append({"id": "task-analysis", "agent": "analysis",
                      "kind": "pattern-analysis", "status": "pending"})
    return {"needs_analysis": as_bool(raw.get("needs_analysis")) and len(targets) >= 2,
            "tasks": tasks, "unknown_systems": resolved["unknown"],
            "target_source": resolved["source"]}


def as_bool(value) -> bool:
    """bool("false") == True 함정 방지 — 모델이 문자열로 답해도 올바르게 해석한다."""
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1", "예")
    return bool(value)
