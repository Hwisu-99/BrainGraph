"""BrainGraph -> Local(md 파일) -> Neo4j, 세 단계 파이프라인의 진행 상황을
디스크에 남기는 모듈(logs/sync_queue.json). static/sync_dashboard.html이 이걸
주기적으로 읽어 실시간 대시보드를 그린다.

왜 필요한가: 지금까지 Neo4j 동기화는 (a) 완전히 조용히 실패를 삼키거나
(paper_notes/graph_db.py의 대부분 호출) (b) FastAPI BackgroundTasks로 돌리되
"마지막 실패 하나"만 프로세스 메모리에 들고 있었다(Brain 재태깅). 두 경우 다
① 사용자가 지금 무슨 일이 진행 중인지 알 방법이 없고 ② 메모리에만 있던
정보는 앱이 재시작되면(정상 종료든 크래시든) 사라진다 - 로컬(md 파일)은 이미
바뀌었는데 Neo4j 반영은 실패한 채로 그 사실 자체를 잊어버리면, 사용자가 눈치
못 챈 드리프트가 영구히 남는다. 이 모듈은 각 동기화 작업을 "pending"으로
기록한 시점부터(로컬 쓰기가 이미 끝난 뒤, 실제 Neo4j 호출을 큐에 넣기 직전)
파일에 남겨서, 앱이 그 사이 죽어도 다음 실행에서 list_unresolved()로 "이거
아직 Neo4j에 안 갔을 수도 있다"를 알 수 있게 한다.

파일 하나에 최근 작업 목록(최대 MAX_ENTRIES개, 넘으면 오래된 것부터 버림)을
JSON 배열로 저장한다 - 개인 도구 규모에서 매번 파일 전체를 읽고 다시 쓰는
비용은 무시할 만하고, 상태를 그때그때 갱신해야 해서(pending -> syncing ->
done/error) append-only 로그(traversal_log.py)보다 이 방식이 더 맞는다.
"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from paper_notes.node_store import NODE_STORE_ROOT

LOG_PATH = Path(NODE_STORE_ROOT) / "logs" / "sync_queue.json"
MAX_ENTRIES = 300

_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load() -> list[dict]:
    if not LOG_PATH.is_file():
        return []
    try:
        data = json.loads(LOG_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save(entries: list[dict]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    trimmed = entries[-MAX_ENTRIES:]
    LOG_PATH.write_text(json.dumps(trimmed, ensure_ascii=False, indent=2), encoding="utf-8")


def _update(op_id: str, **fields) -> None:
    with _LOCK:
        entries = _load()
        for entry in entries:
            if entry["id"] == op_id:
                entry.update(fields)
                entry["updated_at"] = _now()
                break
        _save(entries)


def start(op_type: str, label: str, detail: dict | None = None) -> str:
    """새 작업을 "neo4j: pending" 상태로 기록한다. 이 시점엔 로컬(md 파일)
    쓰기는 이미 끝나 있어야 한다 - brain_graph/local 단계는 그래서 항상
    "done"으로 시작하고, 그 뒤로는 neo4j 단계만 pending -> syncing ->
    done/error/skipped로 바뀐다. 반환된 id로 이후 mark_*()를 호출한다."""
    op_id = uuid.uuid4().hex
    entry = {
        "id": op_id,
        "op_type": op_type,
        "label": label,
        "detail": detail or {},
        "brain_graph_status": "done",
        "local_status": "done",
        "neo4j_status": "pending",
        "neo4j_error": None,
        "started_at": _now(),
        "updated_at": _now(),
    }
    with _LOCK:
        entries = _load()
        entries.append(entry)
        _save(entries)
    return op_id


def mark_syncing(op_id: str) -> None:
    _update(op_id, neo4j_status="syncing")


def mark_done(op_id: str) -> None:
    _update(op_id, neo4j_status="done", neo4j_error=None)


def mark_error(op_id: str, message: str) -> None:
    _update(op_id, neo4j_status="error", neo4j_error=message)


def mark_skipped(op_id: str, message: str) -> None:
    """Neo4jNotConfigured처럼 "설정 자체가 안 됐다"는 실패 - 사용자가 당장
    조치할 수 있는 에러가 아니므로(대시보드에서 error와 시각적으로 구분해
    경고성을 낮춘다) 별도 상태를 둔다."""
    _update(op_id, neo4j_status="skipped", neo4j_error=message)


def list_recent(limit: int = 50) -> list[dict]:
    """최신순(방금 시작된 것부터)으로 최대 limit개."""
    entries = _load()
    return list(reversed(entries[-limit:]))


def list_unresolved() -> list[dict]:
    """neo4j_status가 "error"인 것만, 최신순. 재시작 후에도(파일 기반이므로)
    "지난 세션에서 Neo4j 반영에 실패한 채 남은 작업이 있다"를 확인하는 용도 -
    static/index.html 상단바 배지와 /api/neo4j-sync-status가 이걸 쓴다."""
    return [e for e in reversed(_load()) if e.get("neo4j_status") == "error"]
