"""graph_db.search()가 실제로 호출될 때마다(MCP의 search_graph 포함) 그 쿼리와
탐색 결과(Query+Traversal)를 logs/traversal_log.jsonl에 한 줄씩 남긴다.

왜 여기 있나: mcp_server.py는 비즈니스 로직을 두지 않고 app.py의 REST API를
그대로 감싸서 쓰기만 하므로(mcp_server.py 파일 docstring 참고), Claude가 실제
대화 중 search_graph를 호출하든 static/search_flow_visualizer.html이 테스트
쿼리를 날리든 결국 둘 다 app.py의 /api/graph-search 라우트 하나를 거친다.
그래서 로깅도 그 라우트 한 곳(app.py)에서 이 모듈을 호출하는 것만으로 두
경로를 전부 잡아낸다 - graph_db.py나 mcp_server.py를 따로 건드릴 필요가 없다.

실사용 캡처가 목적이므로 어떤 요청이 "진짜 Claude가 쓴 것"인지 구분할 수 있게,
mcp_server.py의 httpx 클라이언트가 모든 요청에 X-AutoNote-Source: mcp 헤더를
붙이고(app.py에서 읽어 source로 저장), 그 헤더가 없는 나머지(브라우저에서 연
호출 - 시각화 툴 테스트 포함)는 source="web"으로 남는다.

로깅 실패가 실제 검색 응답을 망가뜨리면 안 되므로 log_traversal()은 내부에서
예외를 흡수한다 - 호출부에서 따로 try/except로 감쌀 필요가 없다.
"""
from __future__ import annotations

import json
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from paper_notes.node_store import NODE_STORE_ROOT

LOG_PATH = Path(NODE_STORE_ROOT) / "logs" / "traversal_log.jsonl"

_LOCK = threading.Lock()


def log_traversal(
    *,
    query: str,
    mode: str,
    top_k: int,
    brain_id: str | None,
    relation_types_requested: list[str] | None,
    neighbor_cap: int,
    hop2_top_n: int,
    results: list[dict],
    source: str,
    duration_ms: float,
    reasoning: dict | None = None,
) -> None:
    """검색 호출 하나를 JSONL 한 줄로 append한다. results는 graph_db.search()가
    반환한 그 리스트를 그대로 넣는다 - mode="all"이면 시드마다 flat neighbors[],
    mode="routed"면 provenance_neighbors/semantic_neighbors/routed_types/
    semantic_total_before_cap(+선택적 hop2)까지 그대로 들어가므로, 나중에
    static/search_flow_visualizer.html이 이 로그를 읽을 때 실시간 검색 결과를
    그리는 것과 완전히 같은 렌더링 함수를 그대로 재사용할 수 있다.

    reasoning은 graph_db.search()가 함께 반환하는 "사고 과정"(별칭/이름
    매칭, 의도 분류, 경로 탐색, 한 줄 요약)이다 - 이 기능 추가 전에 쌓인
    로그에는 없는 필드라, 읽는 쪽(read_traversal_log 사용처)은 없을 수도
    있다는 걸 감안해야 한다(.get("reasoning")). None이면 그냥 기록 안 한다."""
    try:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "query": query,
            "mode": mode,
            "top_k": top_k,
            "brain_id": brain_id,
            "relation_types_requested": relation_types_requested,
            "neighbor_cap": neighbor_cap,
            "hop2_top_n": hop2_top_n,
            "seed_count": len(results),
            "duration_ms": round(duration_ms, 1),
            "results": results,
            "reasoning": reasoning,
        }
        line = json.dumps(entry, ensure_ascii=False)
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK, LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as exc:  # pragma: no cover - 로깅은 최선 노력, 검색 자체를 막으면 안 됨
        print(f"[traversal_log] 기록 실패: {exc}", file=sys.stderr)


def delete_traversal_log_entry(ts: str) -> bool:
    """static/search_flow_visualizer.html의 "실사용 로그 히스토리"에서 항목 하나를
    지울 때 쓴다. JSONL은 append-only라 "그 줄만 지우기"가 원래 없으므로, 파일
    전체를 읽어서 ts(각 엔트리의 timestamp - datetime.now(timezone.utc).isoformat()
    라 마이크로초까지 있어 사실상 고유하다)가 일치하는 첫 줄만 빼고 통째로 다시
    쓴다. 로그가 테스트/실사용 검색 기록용이라 파일이 아주 커질 일이 없다는
    전제하에 단순하게 갔다 - 로그가 수만 줄 이상으로 커지면 인덱스 기반 접근으로
    바꿔야 한다. 매칭되는 엔트리가 있어서 지웠으면 True, 없었으면(이미 지워졌거나
    ts가 틀림) False를 반환한다."""
    if not LOG_PATH.is_file():
        return False
    with _LOCK:
        lines = LOG_PATH.read_text(encoding="utf-8").splitlines()
        kept: list[str] = []
        removed = False
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if not removed:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    kept.append(line)
                    continue
                if entry.get("ts") == ts:
                    removed = True
                    continue
            kept.append(line)
        if removed:
            LOG_PATH.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return removed


def read_traversal_log(
    limit: int = 50,
    source: str | None = None,
    mode: str | None = None,
) -> list[dict]:
    """최신순으로 최대 limit개의 로그 엔트리를 반환한다. source("mcp"/"web")나
    mode("all"/"routed")를 주면 그 값과 일치하는 엔트리만 남긴 뒤 최신 limit개를
    돌려준다. 파일이 없거나 손상된 줄은 조용히 건너뛴다(로그 조회 실패로 시각화
    툴 자체가 죽으면 안 되므로)."""
    if not LOG_PATH.is_file():
        return []
    matched: list[dict] = []
    with LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if source and entry.get("source") != source:
                continue
            if mode and entry.get("mode") != mode:
                continue
            matched.append(entry)
    limit = max(1, limit)
    return list(reversed(matched[-limit:]))
