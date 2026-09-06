"""AutoNote의 Brain(node_store + Neo4j)을 Claude가 툴 호출로 직접 다룰 수 있게
하는 MCP 서버. 여기엔 비즈니스 로직을 새로 두지 않는다 - 전부 이미 떠 있는
FastAPI(app.py) REST API를 얇게 감싸서 호출할 뿐이다. 그래야 웹 UI로 만든
변경이든 이 MCP 서버(Claude)로 만든 변경이든 항상 같은 경로(node_store의
중복검사/불변식 -> Neo4j 동기화)를 거치게 되어, 두 클라이언트가 서로 다른
규칙을 적용해 데이터가 어긋나는 일이 없다.

실행 전에 app.py(uvicorn)가 먼저 떠 있어야 한다:
    uvicorn app:app

Claude Desktop/Claude Code의 MCP 설정에 이 스크립트를 등록해서 쓴다(stdio
트랜스포트로 실행됨):
    python mcp_server.py
"""
from __future__ import annotations

import json
import os

import httpx
from mcp.server.mcpserver import MCPServer

API_BASE = os.environ.get("AUTONOTE_API_BASE", "http://127.0.0.1:8000")

mcp = MCPServer(
    "autonote-brain",
    instructions=(
        "AutoNote는 사용자가 읽은 논문들로 만들어진 개인 지식 그래프(Brain)다. "
        "concept(개념)/entity(용어·모델명 등)/paper(논문) 노드와 그 사이의 관계로 "
        "구성된다. 사용자 질문에 답할 때는 항상 search_graph로 먼저 관련 노드를 "
        "찾고, 필요하면 get_node로 자세한 내용을 읽은 뒤 그 내용에 근거해 답하라 - "
        "모르면 모른다고 하고, Brain에 없는 내용을 지어내지 마라. 대화 중 사용자가 "
        "설명한 새로운 개념/관계는 create_node/link_nodes로 Brain에 바로 반영해도 "
        "된다(중복 검사는 서버가 한다 - 비슷한 노드가 있으면 409 에러(existing 필드에 "
        "기존 노드 정보 포함)로 알려주니, 그 노드로 합칠지 사용자에게 확인한 뒤 정말 "
        "새로 만들어야 하면 force=true로 다시 호출하라). 사용자는 Brain을 여러 개 "
        "가질 수 있다(예: Robot Brain, RL Brain) - 먼저 list_brains로 어떤 Brain이 "
        "있는지 확인하고, 사용자가 특정 Brain을 지목하거나 대화 맥락상 범위가 "
        "분명하면 search_graph를 호출할 때 그 Brain의 id를 brain 인자로 넘겨서 "
        "그 Brain에 속한 논문/개념으로만 검색을 좁혀라. 어느 Brain인지 불분명하면 "
        "brain을 생략해 전체에서 검색한다."
    ),
)


def _client(timeout: float = 30.0) -> httpx.Client:
    """AutoNote API(app.py)로 보내는 모든 요청에 X-AutoNote-Source: mcp 헤더를 붙인다 -
    app.py의 /api/graph-search가 이 헤더로 실제 Claude 대화 중 호출(실사용)과
    그 외 웹 호출(시각화 툴 테스트 포함)을 구분해 traversal_log.jsonl에 남긴다."""
    return httpx.Client(
        base_url=API_BASE, timeout=timeout, headers={"X-AutoNote-Source": "mcp"}
    )


def _unwrap(resp: httpx.Response) -> dict:
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise RuntimeError(f"{resp.status_code}: {detail}")
    return resp.json()


@mcp.tool()
def list_brains() -> list[dict]:
    """사용자가 만든 모든 Brain 목록을 {id, name, paper_slugs, created_at}
    형태로 반환한다. search_graph를 특정 Brain으로 좁혀 호출하기 전에 먼저
    호출해 어떤 Brain이 있는지, id가 무엇인지 확인하는 용도."""
    with _client() as c:
        return _unwrap(c.get("/api/brains"))["brains"]


@mcp.tool()
def search_graph(query: str, top_k: int = 10, brain: str | None = None) -> dict:
    """Brain에서 질의와 관련된 concept/entity 노드를 하이브리드 검색(벡터 유사도 +
    풀텍스트)으로 찾고, 각 노드의 1촌 이웃까지 같이 반환한다. 사용자 질문에
    답하기 전에 가장 먼저 호출해야 하는 툴이다. brain에 Brain id를 넘기면 그
    Brain에 속한 논문/개념/용어로만 결과를 좁힌다(list_brains로 id를 먼저
    확인하라) - 생략하면 전체 Brain에서 검색한다.

    각 이웃(neighbors[])에는 relation/direction 필드가 있다 - 논문 출처로만
    연결된 이웃(예: Paper)은 둘 다 null이지만, PART_OF/USES/EXTENDS/IS_A/
    IMPROVES_ON/OUTPERFORMS/SOLVES/EVALUATED_ON/LIMITED_BY/COMPARED_TO/
    CONTRADICTS/RELATED 중 하나로 연결된 이웃은 relation에 그 타입이,
    direction에 이 노드가 관계의 주체(outgoing)인지 대상(incoming)인지가
    채워진다. 답변할 때 이 정보를 "왜 관련 있는지"의 근거로 적극 활용하라 -
    예: relation=EXTENDS, direction=outgoing이면 "A가 B를 확장한다"는 뜻이다.

    단, COMPARED_TO/CONTRADICTS/RELATED는 원래 방향이 없는 대칭 관계라서,
    direction 값(outgoing/incoming)은 저장 시점에 어느 쪽이 정규화 소유자가
    됐는지를 반영할 뿐 실제 의미는 아니다 - 이 세 타입은 direction을 무시하고
    "A와 B가 비교된다/상충한다/관련 있다"처럼 방향 없이 서술하라.

    각 시드에는 seed_source가 붙는다 - "embedding"이면 벡터/풀텍스트 유사도로
    뽑힌 것이고, "alias"면 질문 문장에 그 노드의 이름/별칭이 그대로 등장해서
    유사도 순위와 무관하게 강제로 포함된 것이다(가장 신뢰도 높은 신호이니
    답변에서 우선 다뤄라). 응답에는 results 옆에 reasoning도 함께 온다 -
    alias_matches(정확히 매칭된 이름), intent(질문 표현으로 추정한 관계
    타입), paths(질문에 이름으로 지목된 노드가 2개 이상이면 그 노드들 사이
    가능한 모든 쌍의 최단 경로 목록 - 각 항목이 found=true면 nodes/relations에
    실제 경로가 담기고, found=false면 그 쌍은 연결이 없다는 뜻이다. 노드가
    3개 이상 지목됐으면 쌍이 여러 개 온다), summary(이 모든 걸 사람이 읽을
    문장으로 정리한 것)를 담는다. "A와 B의 관계가 뭐야"처럼 질문에 이름이
    여러 개 나오면, 각자의 이웃을 나열하기 전에 reasoning.paths부터 확인해서
    그 이름들 사이에 직접적인 연결이 있는지 먼저 보는 게 좋다 - 이름이 3개
    이상이면 paths에 여러 쌍이 들어있으니 그 이름들과 관련된 쌍을 찾아서 보면
    된다."""
    params = {"q": query, "top_k": top_k}
    if brain:
        params["brain_id"] = brain
    with _client() as c:
        return _unwrap(c.get("/api/graph-search", params=params))


@mcp.tool()
def list_nodes(node_type: str) -> list[dict]:
    """전체 노드 중 node_type("concept"/"entity"/"note")에 해당하는 것만 slug+label로
    가볍게 나열한다. 설명/메모 등 자세한 내용이 필요하면 get_node로 하나씩 더
    읽어라 - 이 툴은 "지금 Brain에 뭐가 있는지" 감을 잡는 용도다."""
    if node_type not in ("concept", "entity", "note"):
        raise ValueError("node_type은 concept, entity, note 중 하나여야 합니다.")
    with _client() as c:
        graph = _unwrap(c.get("/api/graph"))
    return [
        {"slug": n.get("node_slug") or n["id"], "label": n["label"]}
        for n in graph["nodes"]
        if n["type"] == node_type
    ]


@mcp.tool()
def get_node(node_type: str, slug: str) -> dict:
    """concept/entity/note(논문) 노드 하나의 전체 내용(설명, 별칭, 카테고리, 등장
    논문 목록, 본문/개인 메모)을 읽는다."""
    with _client() as c:
        return _unwrap(c.get(f"/api/nodes/{node_type}/{slug}"))


@mcp.tool()
def create_node(
    node_type: str,
    label: str,
    category: str | None = None,
    paper_slug: str | None = None,
    concept_slug: str | None = None,
    force: bool = False,
) -> dict:
    """새 concept/entity 노드를 만든다.

    paper_slug를 주면 그 논문에 바로 연결된다(entity면 concept_slug도 같이 줘서
    그 concept 밑에 묶을 수 있음). 둘 다 안 주면 아직 어느 논문과도 연결 안 된
    orphan 노드가 된다(나중에 link_nodes로 연결).

    category는 concept 타입일 때만 의미가 있다(problem/proposed_method/
    architecture/algorithm/theory/optimization/training_strategy/
    evaluation_setup/finding/input_representation/limitation/other 중 하나).

    이름이 완전히 같은 노드가 이미 있으면 항상 막힌다. 비슷한(퍼지 매칭) 노드가
    있으면 409 에러(similar_exists, existing 필드에 기존 노드 정보 포함)로 막힌다 -
    정말 새로 만들 게 맞다고 판단되면 force=true로 다시 호출하라."""
    if node_type not in ("concept", "entity"):
        raise ValueError("node_type은 concept 또는 entity여야 합니다.")
    with _client() as c:
        if paper_slug and node_type == "concept":
            resp = c.post(
                f"/api/papers/{paper_slug}/concepts",
                json={"label": label, "category": category, "force": force},
            )
        elif paper_slug and node_type == "entity":
            resp = c.post(
                f"/api/papers/{paper_slug}/entities",
                json={"label": label, "concept_slug": concept_slug, "force": force},
            )
        else:
            resp = c.post(
                "/api/nodes",
                json={"type": node_type, "label": label, "category": category, "force": force},
            )
        return _unwrap(resp)


@mcp.tool()
def delete_node(node_type: str, slug: str, cascade_entities: bool = False) -> dict:
    """concept/entity 노드를 통째로 지운다(파일 삭제, 되돌릴 수 없음).

    concept을 지울 때 cascade_entities=true면 그 밑에 있던 entity들도 함께
    지운다 - 기본값(false)은 entity를 남기고 "그 논문에 직접 연결"된 상태로
    되돌린다."""
    with _client() as c:
        return _unwrap(
            c.delete(f"/api/nodes/{node_type}/{slug}", params={"cascade_entities": cascade_entities})
        )


@mcp.tool()
def link_nodes(
    node_type: str, slug: str, paper_slug: str | None = None, concept_slug: str | None = None
) -> dict:
    """이미 있는 concept/entity 노드를 논문이나 concept에 연결한다(그래프에서
    드래그로 연결하는 것과 같은 동작).

    entity를 concept 밑에 묶으려면 concept_slug를 준다. paper_slug 없이
    concept_slug만 주는 건 entity에서만 가능하다(논문 없이 concept에만
    연결 - "orphan concept에 orphan entity를 붙이는" 경우)."""
    with _client() as c:
        return _unwrap(
            c.post(
                f"/api/nodes/{node_type}/{slug}/link",
                json={"paper_slug": paper_slug, "concept_slug": concept_slug},
            )
        )

@mcp.tool()
def add_note(node_type: str, slug: str, note_markdown: str) -> dict:
    """concept/entity 노드의 개인 메모(user notes) 영역 맨 끝에 note_markdown을
    덧붙인다 - 기존 메모는 그대로 두고 이어붙이기만 한다(통째로 교체하지
    않는다). 자동 생성 영역(등장 논문 목록, 파이프라인이 만든 설명)은 애초에
    건드리지 않는다. 아직 메모가 하나도 없는 노드에도 그대로 쓰면 된다."""
    if node_type not in ("concept", "entity"):
        raise ValueError("node_type은 concept 또는 entity여야 합니다.")
    with _client() as c:
        current = _unwrap(c.get(f"/api/nodes/{node_type}/{slug}"))
        existing = (current.get("user_markdown") or "").rstrip()
        combined = f"{existing}\n\n{note_markdown}" if existing else note_markdown
        return _unwrap(
            c.put(
                f"/api/nodes/{node_type}/{slug}/notes",
                json={"user_notes_markdown": combined},
            )
        )

@mcp.tool()
def unlink_paper(node_type: str, slug: str, paper_slug: str) -> dict:
    """concept/entity 노드에서 특정 논문과의 연결만 끊는다. 노드 자체는 지워지지
    않고, 다른 논문과의 연결이 남아있으면 그대로 유지된다."""
    with _client() as c:
        return _unwrap(c.delete(f"/api/nodes/{node_type}/{slug}/sources/{paper_slug}"))


@mcp.tool()
def unlink_entity_concept(entity_slug: str, concept_slug: str) -> dict:
    """entity와 concept 사이의 연결을 끊는다 - 그 concept 문맥에서 이 entity를
    완전히 뺀다는 뜻이다(논문 직접 연결로는 되돌아가지 않음)."""
    with _client() as c:
        return _unwrap(c.delete(f"/api/nodes/entity/{entity_slug}/concept/{concept_slug}"))


@mcp.tool()
def ingest_paper(pdf_path: str) -> dict:
    """로컬 PDF 파일 하나를 논문 처리 파이프라인에 넣어 요약 노트를 만들고, 그
    안의 concept/entity를 Brain에 추가한다(Claude API 호출 비용 발생, 시간이
    꽤 걸릴 수 있음). 완료되면 논문 제목/한줄요약/새로 생긴 노드 목록을 반환한다."""
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {pdf_path}")

    with open(pdf_path, "rb") as f, _client(timeout=600.0) as c:
        resp = c.post(
            "/api/process",
            files={"file": (os.path.basename(pdf_path), f, "application/pdf")},
        )
    if resp.status_code >= 400:
        raise RuntimeError(f"{resp.status_code}: {resp.text}")

    last_event: dict | None = None
    for line in resp.text.strip().splitlines():
        line = line.strip()
        if line:
            last_event = json.loads(line)
    if last_event is None:
        raise RuntimeError("파이프라인 응답을 받지 못했습니다.")
    if last_event.get("stage") == "error":
        raise RuntimeError(last_event.get("message", "처리 중 오류가 발생했습니다."))
    return last_event



if __name__ == "__main__":
    mcp.run()
