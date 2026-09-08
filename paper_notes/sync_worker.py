"""Neo4j 쓰기 하나를 실제로 실행하는 백그라운드 워커. app.py의 _sync_node()/
_sync_paper()/_delete_node_from_graph() 같은 함수들이 이 모듈의 submit()으로
작업을 큐에 넣기만 하고 곧바로 리턴한다 - 실제 Neo4j 왕복(수백ms~수초 걸릴 수
있는 원격 Aura 호출)은 이 모듈의 데몬 스레드 하나가 순서대로 처리한다.

왜 FastAPI BackgroundTasks가 아니라 별도 스레드+큐인가: BackgroundTasks는
엔드포인트 함수가 `background_tasks: BackgroundTasks`를 파라미터로 받아야만
쓸 수 있는데, node_store를 바꾸는 엔드포인트가 20곳 넘게 있고 전부 이미
_sync_node() 등 공용 래퍼 하나를 호출하는 구조다(docs/neo4j/synchronization.md
참고). 그 래퍼 안에서 "큐에 넣고 반환"만 하면 호출하는 쪽(20곳 전부) 코드를
전혀 안 건드리고도 전부 비동기가 된다 - 이 모듈이 그 큐+워커다.

작업을 하나씩 순차 처리하는 이유: 같은 노드를 짧은 시간에 여러 번 고치면(예:
alias 추가 직후 카테고리 추가) Neo4j MERGE가 겹쳐 실행될 때의 경쟁 상태를
걱정할 필요가 없어진다 - 개인 도구 규모에서 처리량이 문제가 될 일은 없고,
순서 보장이 더 중요하다.
"""
from __future__ import annotations

import queue
import threading

from paper_notes import sync_queue
from paper_notes.graph_db import Neo4jNotConfigured

_queue: "queue.Queue[tuple[str, object, tuple, dict]]" = queue.Queue()
_started = False
_start_lock = threading.Lock()


def _worker_loop() -> None:
    while True:
        op_id, fn, args, kwargs = _queue.get()
        try:
            sync_queue.mark_syncing(op_id)
            fn(*args, **kwargs)
            sync_queue.mark_done(op_id)
        except Neo4jNotConfigured as exc:
            sync_queue.mark_skipped(op_id, str(exc))
        except Exception as exc:  # noqa: BLE001 - 워커 스레드가 죽으면 이후 작업이 전부 멈추므로 반드시 여기서 흡수
            sync_queue.mark_error(op_id, str(exc))
        finally:
            _queue.task_done()


def _ensure_started() -> None:
    global _started
    if _started:
        return
    with _start_lock:
        if _started:
            return
        threading.Thread(target=_worker_loop, name="neo4j-sync-worker", daemon=True).start()
        _started = True


def submit(op_id: str, fn, *args, **kwargs) -> None:
    """op_id(sync_queue.start()가 반환한 값)로 추적되는 작업 fn(*args, **kwargs)을
    큐에 넣는다. 호출 즉시 반환 - 실제 실행은 백그라운드 스레드가 나중에 한다."""
    _ensure_started()
    _queue.put((op_id, fn, args, kwargs))
