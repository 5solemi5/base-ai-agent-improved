"""[개선 3] RAG 기반 incident_type 검색 — 키워드 규칙(_classify_incident_type)을 대체한다.

원본 replan_check는 모니터링 알림 문구에서 "latency → pool → cpu → memory …" 순서로
키워드를 찾아 처음 걸린 것을 runbook으로 골랐다. 실측 문제:
  - healthy 시스템(web-01)만 조사하면 알림이 비어 아무것도 안 걸리고 폴백 db-down이 붙었다.
  - 여러 시스템의 알림을 한 문자열로 합쳐 규칙 순서로만 고르기 때문에, 사용자가 말한
    증상(타임아웃)보다 다른 시스템의 알림(CPU)이 우선되는 경우가 있었다.
  - 사용자 질문 자체("결제가 느려요", "타임아웃 나요")는 아예 보지 않았다.

이 모듈은 runbook 본문을 '단계(줄) 단위 조각'으로 나누고, 조각마다 incident_type을
메타데이터로 붙여 벡터 인덱스를 만든다(PDF 과제 예시 1번: "벡터 DB에는 본문에 해당하는
Incident_type을 Meta 정보로 미리 가지고 있음"). 질의 = 사용자 질문 + 저하된(degraded)
시스템의 알림 문구이고, 가장 비슷한 조각들의 incident_type으로 runbook을 고른다.

벡터화 방식(RUNBOOK_RETRIEVER 환경변수):
  - "tfidf"(기본): 순수 파이썬 TF-IDF. 한국어는 띄어쓰기·조사 때문에 단어 단위가 약해서
    문자 2~3-gram을 함께 쓴다. 외부 서버·모델·패키지가 필요 없고 결정론적이다.
  - "embed": Ollama 임베딩 API(/api/embed, 기본 모델 nomic-embed-text). 의미 유사도가
    더 좋지만 `ollama pull nomic-embed-text`가 필요하다. 호출이 실패하면 tfidf로 자동 폴백.
    (qwen3:0.6b/8b는 임베딩을 지원하지 않는다 — 실측: "This server does not support embeddings")
"""
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import httpx

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RUNBOOK_DIR = DATA_DIR / "runbooks"
META_PATH = DATA_DIR / "runbook_meta.json"
FALLBACK_TYPE = "high-latency"   # 정말 아무 근거가 없을 때 — "가장 흔한 증상(지연)" 절차


def _tokens(text: str) -> list[str]:
    """단어 토큰 + 문자 2·3-gram. 'payments-service랑'의 조사처럼 붙은 글자에도 부분 일치가 걸린다."""
    text = text.lower()
    words = re.findall(r"[a-z0-9.%/]+|[가-힣]+", text)
    grams = []
    for w in words:
        if re.fullmatch(r"[가-힣]+", w):
            grams += [w[i:i + n] for n in (2, 3) for i in range(len(w) - n + 1)]
    return words + grams


class _TfidfIndex:
    def __init__(self, docs: list[str]):
        self.doc_tokens = [Counter(_tokens(d)) for d in docs]
        df = Counter(t for toks in self.doc_tokens for t in toks)
        n = len(docs)
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}
        self.doc_vecs = [self._vec(toks) for toks in self.doc_tokens]

    def _vec(self, counts: Counter) -> dict[str, float]:
        vec = {t: (1 + math.log(c)) * self.idf.get(t, 0.0) for t, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def search(self, query: str) -> list[float]:
        q = self._vec(Counter(_tokens(query)))
        return [sum(w * d.get(t, 0.0) for t, w in q.items()) for d in self.doc_vecs]


class _EmbedIndex:
    def __init__(self, docs: list[str], model: str, url: str):
        self.model, self.url = model, url
        self.doc_vecs = self._embed(docs)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        resp = httpx.post(self.url, json={"model": self.model, "input": texts}, timeout=60)
        resp.raise_for_status()
        return resp.json()["embeddings"]

    def search(self, query: str) -> list[float]:
        q = self._embed([query])[0]
        qn = math.sqrt(sum(x * x for x in q)) or 1.0
        out = []
        for d in self.doc_vecs:
            dn = math.sqrt(sum(x * x for x in d)) or 1.0
            out.append(sum(a * b for a, b in zip(q, d)) / (qn * dn))
        return out


class RunbookRetriever:
    """runbook 조각 인덱스. 서버 수명 동안 한 번만 만든다(graph.py 모듈 로드 시)."""

    def __init__(self, method: str | None = None):
        meta = {k: v for k, v in json.loads(META_PATH.read_text(encoding="utf-8")).items()
                if not k.startswith("_")}
        self.chunks: list[dict] = []   # {"incident_type": ..., "text": ...}  ← 메타데이터 + 본문
        for path in sorted(RUNBOOK_DIR.glob("*.md")):
            itype = path.stem
            lines = [l.strip("#-0123456789. ").strip() for l in path.read_text(encoding="utf-8").splitlines()]
            for line in filter(None, lines):
                self.chunks.append({"incident_type": itype, "text": line})
            for phrase in meta.get(itype, []):
                self.chunks.append({"incident_type": itype, "text": phrase})
        self.types = sorted({c["incident_type"] for c in self.chunks})

        requested = (method or os.environ.get("RUNBOOK_RETRIEVER", "tfidf")).lower()
        self.method = "tfidf"
        self.index = None
        if requested == "embed":
            try:
                self.index = _EmbedIndex(
                    [c["text"] for c in self.chunks],
                    os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
                    os.environ.get("OLLAMA_EMBED_URL", "http://127.0.0.1:11434/api/embed"))
                self.method = "embed"
            except Exception as e:   # 모델 미설치·서버 미기동 → 조용히 죽지 않고 이유를 남기고 폴백
                self.fallback_reason = repr(e)
        if self.index is None:
            self.index = _TfidfIndex([c["text"] for c in self.chunks])

    def rank(self, query: str, top_k_chunks: int = 3) -> list[tuple[str, float]]:
        """incident_type별 점수 = 그 유형 조각 중 상위 top_k_chunks개 유사도의 합."""
        sims = self.index.search(query)
        per_type: dict[str, list[float]] = {t: [] for t in self.types}
        for chunk, s in zip(self.chunks, sims):
            per_type[chunk["incident_type"]].append(s)
        scores = {t: sum(sorted(v, reverse=True)[:top_k_chunks]) for t, v in per_type.items()}
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    def classify(self, query: str, collected: list[dict]) -> dict:
        """사용자 질문 + 저하된 시스템의 알림으로 incident_type 하나를 고른다.

        healthy 시스템의 정보는 질의에서 뺀다 — 원본은 web-01(healthy)만 보고 db-down을
        골랐다. 사용자 질문은 항상 넣는다(사용자가 말한 증상이 1차 근거)."""
        signals = []
        for c in collected:
            if c.get("source") != "monitoring":
                continue
            d = c["data"]
            if d.get("status") not in ("healthy", "unknown"):
                signals.append(" ".join(d.get("alerts", [])))
        text = query + " " + " ".join(signals)
        ranked = self.rank(text)
        best, score = ranked[0]
        if score <= 0:
            best = FALLBACK_TYPE
        return {"incident_type": best, "method": self.method,
                "scores": [(t, round(s, 3)) for t, s in ranked[:3]], "query_text": text}
