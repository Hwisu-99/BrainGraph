"""node_store(.md 파일)를 source of truth로 두고, 그걸 그대로 미러링한 Neo4j
그래프를 GraphRAG 검색 전용 인덱스로 쓴다. Neo4j에는 아무것도 직접 쓰지 않는다 -
항상 node_store.py를 먼저 갱신한 뒤, 그 결과를 이 모듈이 다시 Neo4j에 반영하는
방향으로만 흐른다(Supabase가 vault 백업 미러인 것과 같은 구조). 그래서
node_store.py에 이미 있는 중복 검사·불변식(entity의 paperless/paper-backed
전환 규칙 등)은 하나도 다시 안 짜도 된다 - Neo4j는 그 결과가 도착하는 곳일 뿐,
그 결과를 스스로 판단하지 않는다.

검색은 세 축을 합친 하이브리드다:
1. 그래프 트래버설(Cypher) - concept/entity/paper 사이의 실제 관계를 따라간다
2. 벡터 검색 - description/note를 임베딩(paper_notes/embeddings.py, 로컬 모델)해서
   의미가 비슷한 노드를 찾는다
3. 풀텍스트 검색 - 정확한 단어/구절이 일치하는 노드를 찾는다(벡터 검색이
   놓치기 쉬운 고유명사·약어에 강함)
Neo4j 5.11+가 이 셋을 전부 네이티브로 지원해서, 별도 벡터DB/검색엔진 없이
Neo4j 하나로 구성한다.
"""
from __future__ import annotations

import math
import os
import threading

from paper_notes.embeddings import embed_passage, embed_query, embedding_dimension
from paper_notes.brains import get_paper_brain_id
from paper_notes.node_store import NODE_STORE_ROOT, get_relations, get_user_section, list_nodes
from paper_notes.relation_types import describe_relation_type, load_relation_types

_driver = None
_driver_lock = threading.Lock()

_LABEL_BY_TYPE = {"concept": "Concept", "entity": "Entity", "paper": "Paper"}


class Neo4jNotConfigured(RuntimeError):
    """NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD가 .env에 없을 때. 그래프 검색 관련
    기능(MCP 서버)만 못 쓰게 하고, 앱의 나머지 기능(그래프 뷰, 노드 편집 등)은
    이 모듈과 무관하게 그대로 동작해야 하므로 import 시점이 아니라 실제 호출
    시점에 던진다."""


def get_driver():
    global _driver
    if _driver is None:
        with _driver_lock:
            if _driver is None:
                uri = os.getenv("NEO4J_URI")
                user = os.getenv("NEO4J_USER")
                password = os.getenv("NEO4J_PASSWORD")
                if not uri or not user or not password:
                    raise Neo4jNotConfigured(
                        "NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD가 .env에 설정되어 있지 않습니다."
                    )
                import neo4j

                _driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
    return _driver


def close_driver() -> None:
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


def ensure_schema() -> None:
    """제약조건 + 벡터 인덱스 + 풀텍스트 인덱스를 만든다. 전부 IF NOT EXISTS라
    몇 번을 다시 불러도 안전하다(멱등) - 앱 시작 시나 벌크 동기화 전에 호출."""
    dim = embedding_dimension()
    driver = get_driver()
    with driver.session() as session:
        session.run("CREATE CONSTRAINT paper_slug IF NOT EXISTS FOR (n:Paper) REQUIRE n.slug IS UNIQUE")
        session.run("CREATE CONSTRAINT concept_slug IF NOT EXISTS FOR (n:Concept) REQUIRE n.slug IS UNIQUE")
        session.run("CREATE CONSTRAINT entity_slug IF NOT EXISTS FOR (n:Entity) REQUIRE n.slug IS UNIQUE")

        for label in ("Concept", "Entity"):
            session.run(
                f"""
                CREATE VECTOR INDEX {label.lower()}_embedding IF NOT EXISTS
                FOR (n:{label}) ON (n.embedding)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: $dim,
                    `vector.similarity_function`: 'cosine'
                }}}}
                """,
                dim=dim,
            )

        # 필드 목록이 바뀔 때마다 IF NOT EXISTS만으로는 기존 인덱스가 새 정의로
        # 안 바뀌므로, 매번 지우고 다시 만든다 - 인덱스일 뿐이라 재생성 비용이
        # 크지 않고(Neo4j가 백그라운드로 다시 채움), 항상 지금 코드의 필드 목록과
        # 실제 인덱스가 어긋나지 않게 보장하는 쪽을 택했다.
        session.run("DROP INDEX node_fulltext IF EXISTS")
        session.run(
            """
            CREATE FULLTEXT INDEX node_fulltext IF NOT EXISTS
            FOR (n:Concept|Entity) ON EACH [n.display_label, n.description, n.note, n.user_notes]
            """
        )


def _node_text(frontmatter: dict) -> str:
    """임베딩에 넣을 텍스트 - 이름/별칭/설명/메모를 합친다(설명만 넣으면 짧은
    노드는 신호가 너무 적다)."""
    parts = [
        frontmatter.get("display_label", ""),
        ", ".join(frontmatter.get("aliases") or []),
        frontmatter.get("description", ""),
        frontmatter.get("note", ""),
    ]
    return "\n".join(p for p in parts if p)


def _node_brain_ids(frontmatter: dict) -> list[str]:
    """이 노드가 걸려 있는 논문들이 지금 어느 Brain(들)에 속하는지 - concept/entity
    자신은 Brain을 저장하지 않고(여러 Brain에 걸쳐 공유될 수 있으므로), 부를
    때마다 sources[]의 각 논문이 지금 어느 Brain인지 다시 물어서 계산한다
    (brains.get_paper_brain_id). search()가 이 brain_ids로 특정 Brain 범위의
    결과만 걸러낸다. sync_node()와 retag_node_brain() 둘 다 이 계산이
    필요해서 공용 함수로 뺐다."""
    return sorted(
        {
            bid
            for source in (frontmatter.get("sources") or [])
            if (bid := get_paper_brain_id(NODE_STORE_ROOT, source.get("slug")))
        }
    )


def sync_node(node_type: str, slug: str) -> None:
    """concept/entity 노드 하나를 node_store 상태 그대로 Neo4j에 반영한다.
    노드 속성(임베딩 포함)을 갱신하고, 이 노드로 들어오는 관계는 지금의 sources
    목록 기준으로 통째로 다시 만든다(하나씩 add/remove를 따라가지 않고 항상
    "현재 상태로 맞추기"만 하면 되므로, node_store.py의 각 함수가 sources를
    어떻게 바꿨는지 몰라도 항상 정확하다).

    관계 방향은 항상 "참조하는 쪽 -> 참조되는 쪽"이다: (Paper)-[:LINKED_TO]->
    (Concept|Entity), (Concept)-[:LINKED_TO]->(Entity). 즉 concept/entity
    노드 자신은 이 스키마에서 항상 관계의 **도착점**이고, sources[]가 그
    도착점으로 들어오는 관계 전부를 결정한다(concept->entity 에지조차,
    concept 자신의 sources가 아니라 그 entity의 sources[].concept_slug가
    결정한다 - 그래서 concept을 리싱크할 때 이 에지를 새로 만들 일은 없고,
    지우지도 않아야 한다). 그래서 정리 대상은 **나가는(outgoing)** 관계가
    아니라 **들어오는(incoming)** 관계다 - 반대로 지우면 sources를 다
    비워도(예: 논문 연결을 전부 끊어도) 예전 (Paper)-[:LINKED_TO]->(이 노드)
    관계가 하나도 안 지워져 Neo4j에 죽은 에지가 영구히 남는다(실제로 발견된
    버그 - orphan이 된 concept이 Neo4j에서는 계속 예전 논문과 연결된
    것처럼 보였다)."""
    label = _LABEL_BY_TYPE[node_type]
    nodes = list_nodes(NODE_STORE_ROOT, node_type)
    frontmatter = next((n for n in nodes if n["slug"] == slug), None)
    if frontmatter is None:
        delete_node_from_graph(node_type, slug)
        return

    embedding = embed_passage(_node_text(frontmatter))

    # Neo4j 속성은 출처별로 셋으로 나눈다 - description/note는 frontmatter의 값을
    # 그대로(둘 다 논문 요약 파이프라인이 채운 AI 생성 텍스트, resolve_or_create_node
    # 참고 - 서로 합치지 않고 각자 자기 이름의 속성에만 들어간다), user_notes는
    # node_store의 user-notes 섹션(사용자가 직접 쓴 원문, 또는 대화 중 add_note로
    # 덧붙여진 내용)을 별도 속성으로 둔다. 세 속성 모두 섞이지 않아야 검색 결과를
    # 읽는 쪽(Claude 등)이 어디서 나온 텍스트인지 헷갈리지 않는다.
    user_notes = get_user_section(NODE_STORE_ROOT, node_type, slug)
    brain_ids = _node_brain_ids(frontmatter)

    driver = get_driver()
    with driver.session() as session:
        session.run(
            f"""
            MERGE (n:{label} {{slug: $slug}})
            SET n.display_label = $display_label,
                n.aliases = $aliases,
                n.description = $description,
                n.note = $note,
                n.categories = $categories,
                n.embedding = $embedding,
                n.user_notes = $user_notes,
                n.brain_ids = $brain_ids
            """,
            slug=slug,
            display_label=frontmatter.get("display_label", slug),
            aliases=frontmatter.get("aliases") or [],
            description=frontmatter.get("description", ""),
            note=frontmatter.get("note", ""),
            categories=frontmatter.get("categories") or [],
            embedding=embedding,
            user_notes=user_notes,
            brain_ids=brain_ids,
        )

        # 이 노드로 들어오는 관계를 전부 지우고 현재 sources로 다시 만든다 -
        # 증분 동기화보다 훨씬 단순하고, 어떤 순서로 호출되든 항상 올바르다.
        session.run(f"MATCH ()-[r:LINKED_TO]->(n:{label} {{slug: $slug}}) DELETE r", slug=slug)

        # concept_slug가 가리키는 concept이 이미 지워졌을 수 있다(예: 사용자가
        # concept만 지우고 entity는 남겨둔 경우) - graph_builder.py도 이때
        # "논문에 직접 연결"로 취급하므로(죽은 참조로 그래프에서 사라지지 않게),
        # 여기서도 같은 기준(지금 실제로 존재하는 concept slug 집합)으로 판단해야
        # 그래프 뷰와 Neo4j 미러가 어긋나지 않는다.
        valid_concept_slugs = (
            {n["slug"] for n in list_nodes(NODE_STORE_ROOT, "concept")} if node_type == "entity" else set()
        )

        for source in frontmatter.get("sources") or []:
            paper_slug = source.get("slug")
            concept_slug = source.get("concept_slug") if node_type == "entity" else None
            if concept_slug and concept_slug not in valid_concept_slugs:
                concept_slug = None
            if concept_slug:
                session.run(
                    """
                    MATCH (c:Concept {slug: $concept_slug})
                    MATCH (e:Entity {slug: $entity_slug})
                    MERGE (c)-[:LINKED_TO]->(e)
                    """,
                    concept_slug=concept_slug,
                    entity_slug=slug,
                )
            elif paper_slug:
                session.run(
                    f"""
                    MATCH (p:Paper {{slug: $paper_slug}})
                    MATCH (n:{label} {{slug: $slug}})
                    MERGE (p)-[:LINKED_TO]->(n)
                    """,
                    paper_slug=paper_slug,
                    slug=slug,
                )

        # 이 노드에서 "나가는" semantic 관계(LINKED_TO가 아닌 전부)를 통째로
        # 지우고 지금 frontmatter의 relations[]로 다시 만든다 - 위 LINKED_TO
        # 재생성과 대칭이지만 방향이 정반대다: LINKED_TO는 "들어오는" 쪽이
        # sources[] 기준으로 재생성되는 반면, semantic 관계는 항상 "나가는" 쪽
        # frontmatter가 source of truth다(docs/description/relation_types.md
        # 참고 - 두 노드가 서로 상대 frontmatter에 같은 관계를 중복 기록하면
        # 이 재생성 로직이 꼬여 유령 에지가 생긴다). 이 그래프에 존재하는 관계
        # 타입은 LINKED_TO와 화이트리스트의 semantic 타입뿐이라는 불변식이
        # 있어서 "LINKED_TO가 아니면 곧 semantic"으로 안전하게 판단할 수
        # 있다.
        session.run(
            f"MATCH (n:{label} {{slug: $slug}})-[r]->() WHERE type(r) <> 'LINKED_TO' DELETE r",
            slug=slug,
        )

        relation_types = load_relation_types(NODE_STORE_ROOT)
        for rel in get_relations(NODE_STORE_ROOT, node_type, slug):
            rel_type = rel.get("type")
            target_label = _LABEL_BY_TYPE.get(rel.get("target_type"))
            # 화이트리스트에 없는 타입은 절대 Cypher 문자열에 보간하지 않는다
            # (Cypher는 관계 타입을 파라미터로 못 받아 f-string 보간이
            # 불가피한데, 검증 없이 보간하면 인젝션 위험이 생긴다).
            if rel_type not in relation_types or target_label is None:
                print(f"  [경고] 알 수 없는 관계 타입/대상 건너뜀: {rel}")
                continue
            session.run(
                f"""
                MATCH (a:{label} {{slug: $from_slug}})
                MATCH (b:{target_label} {{slug: $to_slug}})
                MERGE (a)-[r:{rel_type}]->(b)
                SET r.sources = $sources, r.rationale = $rationale
                """,
                from_slug=slug,
                to_slug=rel.get("target_slug"),
                sources=rel.get("sources") or [],
                rationale=rel.get("rationale", ""),
            )


def retag_node_brain(node_type: str, slug: str) -> None:
    """concept/entity 노드의 brain_ids만 다시 계산해 SET한다 - sync_node()
    전체를 안 거치는 가벼운 경로. Folder/Brain 소속만 바뀌었을 뿐 노드의 실제
    내용(설명/메모/별칭)이나 sources[] 자체는 안 바뀐 경우(app.py의
    _resync_paper_brain 등)에만 써야 한다 - sync_node()가 매번 하는 임베딩
    재계산(로컬 모델 forward pass)과 들어오는 관계 전체 삭제/재생성을 둘 다
    생략하므로, 논문 하나에 노드가 여러 개 걸려 있을 때 체감 속도 차이가 크다.
    sources[] 자체가 바뀌는 변경(연결 추가/해제, 새 논문에서 뽑힌 노드 등)은
    여전히 sync_node()를 써야 관계도 같이 맞는다.

    retag_paper_brain()과 같은 이유로 MERGE가 아니라 MATCH를 쓴다 - Neo4j에
    이 노드가 아직 없다면(Neo4j 미설정 등) 속성 몇 개짜리 빈 노드를 새로
    만들면 안 되므로, 있을 때만 갱신한다. 이미 삭제된 노드도 마찬가지로
    조용히 아무 일도 안 한다(호출부가 어차피 node_store에 실제로 존재하는
    노드만 찾아서 넘겨준다 - node_store.find_node_slugs_by_paper 참고)."""
    label = _LABEL_BY_TYPE[node_type]
    nodes = list_nodes(NODE_STORE_ROOT, node_type)
    frontmatter = next((n for n in nodes if n["slug"] == slug), None)
    if frontmatter is None:
        return

    brain_ids = _node_brain_ids(frontmatter)
    driver = get_driver()
    with driver.session() as session:
        session.run(
            f"MATCH (n:{label} {{slug: $slug}}) SET n.brain_ids = $brain_ids",
            slug=slug,
            brain_ids=brain_ids,
        )


def sync_paper(slug: str, title: str, tags: list[str] | None = None, brain_id: str | None = None) -> None:
    """brain_id를 안 넘기면(기존 호출부 대부분) get_paper_brain_id()로 지금
    이 논문이 속한 Brain을 직접 물어서 채운다 - 호출부가 매번 brain_id를 미리
    조회해서 넘겨줄 필요 없이, 이 함수 하나만 부르면 항상 최신 소속으로
    맞춰진다. 이미 알고 있는 값이 있으면(예: 방금 Brain을 옮긴 직후라 재조회가
    낭비인 경우) 그대로 넘겨서 재조회를 생략할 수 있다."""
    if brain_id is None:
        brain_id = get_paper_brain_id(NODE_STORE_ROOT, slug)
    driver = get_driver()
    with driver.session() as session:
        session.run(
            "MERGE (p:Paper {slug: $slug}) SET p.title = $title, p.tags = $tags, p.brain_id = $brain_id",
            slug=slug,
            title=title,
            tags=tags or [],
            brain_id=brain_id,
        )


def retag_paper_brain(slug: str, brain_id: str | None = None) -> None:
    """Paper 노드의 brain_id만 다시 태그한다(title/tags는 안 건드림) - Folder나
    Brain 소속만 바뀌었을 뿐 논문 내용 자체는 안 바뀐 경우, sync_paper()처럼
    title을 다시 조회해올 필요 없이 이 태그 하나만 갱신하면 된다. brain_id를
    안 주면(기본값) get_paper_brain_id()로 지금 소속을 다시 계산해서 채운다.
    MERGE가 아니라 MATCH를 쓴다 - Neo4j에 이 Paper 노드가 아직 없다면(Neo4j
    미설정 등) title 없이 빈 노드를 새로 만들면 안 되므로, 있을 때만 갱신한다."""
    if brain_id is None:
        brain_id = get_paper_brain_id(NODE_STORE_ROOT, slug)
    driver = get_driver()
    with driver.session() as session:
        session.run("MATCH (p:Paper {slug: $slug}) SET p.brain_id = $brain_id", slug=slug, brain_id=brain_id)


def delete_node_from_graph(node_type: str, slug: str) -> None:
    label = _LABEL_BY_TYPE[node_type]
    driver = get_driver()
    with driver.session() as session:
        session.run(f"MATCH (n:{label} {{slug: $slug}}) DETACH DELETE n", slug=slug)


def full_resync(vault_path: str) -> dict:
    """지금 vault + node_store 전체 상태로 Neo4j를 처음부터 다시 만든다(1회성
    벌크 동기화, 또는 드리프트가 의심될 때 재실행하는 안전한 재구축). 기존
    데이터를 전부 지우고 다시 쓰므로 매번 최종 상태와 정확히 일치한다."""
    from pathlib import Path

    ensure_schema()
    driver = get_driver()
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

    paper_count = 0
    autonote_dir = Path(vault_path) / "AutoNote"
    if autonote_dir.is_dir():
        import yaml

        for folder in sorted(autonote_dir.iterdir()):
            if not folder.is_dir():
                continue
            slug = folder.name
            md_path = folder / f"{slug}.md"
            if not md_path.is_file():
                continue
            text = md_path.read_text(encoding="utf-8")
            frontmatter = {}
            if text.startswith("---"):
                end = text.find("\n---", 3)
                if end != -1:
                    frontmatter = yaml.safe_load(text[3:end]) or {}
            tags = frontmatter.get("tags") or []
            if isinstance(tags, str):
                tags = [tags]
            sync_paper(slug, frontmatter.get("title") or slug, tags)
            paper_count += 1

    concept_count = 0
    for n in list_nodes(NODE_STORE_ROOT, "concept"):
        sync_node("concept", n["slug"])
        concept_count += 1

    entity_count = 0
    for n in list_nodes(NODE_STORE_ROOT, "entity"):
        sync_node("entity", n["slug"])
        entity_count += 1

    return {"papers": paper_count, "concepts": concept_count, "entities": entity_count}


_RRF_K = 60  # 정보검색에서 흔히 쓰는 상수(Cormack et al.) - 순위 1~2위 근처의 비중을
             # 과하게 키우지 않으면서도 상위권을 충분히 우대하는 완만한 감쇠를 준다.


def _reciprocal_rank_fusion(*ranked_lists: list[dict]) -> list[dict]:
    """벡터 검색(코사인 유사도, 0~1)과 풀텍스트 검색(Lucene 점수, 상한 없음)은
    점수 스케일이 완전히 달라서 그냥 숫자로 비교하면 풀텍스트 쪽이 항상
    이겨버린다(관련도와 무관하게 스케일이 커서). 대신 각 랭커 안에서의 "순위"만
    보고 합치는 RRF(Reciprocal Rank Fusion)를 쓴다 - 두 검색 방식이 서로 다른
    걸 잘 찾아내는 상황(벡터는 의미 유사, 풀텍스트는 정확한 단어/약어)에서
    표준적으로 쓰이는 병합 방식이다."""
    fused: dict[tuple[str, str], dict] = {}
    rrf_scores: dict[tuple[str, str], float] = {}
    for ranked in ranked_lists:
        for rank, hit in enumerate(ranked, start=1):
            key = (hit["type"], hit["slug"])
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (_RRF_K + rank)
            fused.setdefault(key, hit)
    for key, hit in fused.items():
        hit["score"] = rrf_scores[key]
    return sorted(fused.values(), key=lambda h: h["score"], reverse=True)


# ---- 쿼리 인지형 이웃 확장 (search(mode="routed")) ----
# docs/mcp/search_flow.md의 "개선 설계안" - LINKED_TO(provenance)는 그대로 두고,
# semantic 12종만 쿼리와 관련도 높은 타입으로 골라서 확장한다. mode="all"
# (기본값, 기존 동작)은 이 아래 함수들을 전혀 안 타므로 MCP search_graph나
# 기존 /api/graph-search 호출자는 아무 영향이 없다 - 새 파라미터를 명시적으로
# 줘야만(mode="routed") 이 경로를 탄다.

_type_embed_cache: dict[str, list[float]] = {}
_type_embed_cache_lock = threading.Lock()


def _type_embedding(rel_type: str) -> list[float]:
    """관계 타입 하나의 임베딩(이름 + 한글 설명) - 프로세스 전역에 캐싱한다.
    12개뿐이라 계산 비용 자체는 무시할 만하지만, 검색마다 다시 계산할 이유가
    없다."""
    if rel_type not in _type_embed_cache:
        with _type_embed_cache_lock:
            if rel_type not in _type_embed_cache:
                _type_embed_cache[rel_type] = embed_passage(f"{rel_type}: {describe_relation_type(rel_type)}")
    return _type_embed_cache[rel_type]


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _route_relation_types(query_vec: list[float], whitelist: list[str], top_n: int = 3) -> dict[str, float]:
    """쿼리 임베딩과 각 관계 타입 설명 임베딩의 코사인 유사도로 top_n개(+
    점수)를 고른다 - 진짜 임베딩 기반(test/search_flow_visualizer.html의 예전
    클라이언트 사이드 텍스트 겹침 데모를 대체하는 실제 구현). 전부 낮은
    점수여도 top_n개는 반환한다 - 아예 안 걸리는 것보단 그나마 가까운
    것들이라도 이웃을 보여주는 게 낫다는 판단."""
    scored = sorted(
        ((t, _cosine(query_vec, _type_embedding(t))) for t in whitelist),
        key=lambda pair: pair[1], reverse=True,
    )
    return dict(scored[:top_n])


def _ranked_semantic_neighbors(
    session, node_type: str, slug: str, brain_id: str | None,
    routed_types: dict[str, float], query_vec: list[float],
) -> list[dict]:
    """한 노드의 semantic 이웃 중 routed_types(화이트리스트 검증된 타입만 -
    Cypher는 관계 타입을 파라미터로 못 받아 문자열 보간이 필요하므로,
    호출부에서 이미 알려진 타입인지 검증된 것만 여기 들어온다)에 해당하는
    것만 뽑아서, (관계 타입 관련도 점수) x (이웃 자신의 임베딩과 쿼리의 코사인
    유사도) 점수로 내림차순 정렬해 돌려준다. routed_types가 비어 있으면
    빈 리스트."""
    if not routed_types:
        return []
    type_clause = "|".join(sorted(routed_types.keys()))
    rows = session.run(
        f"""
        MATCH (n:{node_type} {{slug: $slug}})-[r:{type_clause}]-(neighbor)
        WHERE $brain_id IS NULL OR $brain_id IN neighbor.brain_ids
        RETURN neighbor.slug AS slug, labels(neighbor)[0] AS type, neighbor.display_label AS label,
               neighbor.embedding AS embedding, type(r) AS relation,
               CASE WHEN startNode(r) = n THEN 'outgoing' ELSE 'incoming' END AS direction
        """,
        slug=slug,
        brain_id=brain_id,
    ).data()
    scored = []
    for row in rows:
        if not row["slug"]:
            continue
        type_score = routed_types.get(row["relation"], 0.0)
        similarity = _cosine(query_vec, row["embedding"] or [])
        scored.append({
            "slug": row["slug"], "type": row["type"], "label": row["label"],
            "relation": row["relation"], "direction": row["direction"],
            "score": round(type_score * similarity, 4),
        })
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored


def _expand_routed_neighbors(
    session, seed: dict, brain_id: str | None, routed_types: dict[str, float],
    neighbor_cap: int, hop2_top_n: int, query_vec: list[float],
) -> dict:
    """한 시드의 provenance(항상 전부) + semantic(routed_types로 거르고 랭킹
    후 상한 neighbor_cap) 이웃을 만든다. semantic 상위 hop2_top_n개는 그
    이웃 자신을 기준으로 한 번 더(2촌) 같은 방식으로 확장해서 "hop2" 필드에
    실어준다(2촌은 같은 routed_types를 재사용 - 원래 쿼리 의도에서 벗어나지
    않게 하기 위함, 새로 자동 라우팅하지 않는다)."""
    provenance_rows = session.run(
        f"""
        MATCH (n:{seed['type']} {{slug: $slug}})
        OPTIONAL MATCH (n)-[r:LINKED_TO]-(neighbor)
        WHERE neighbor IS NULL
           OR $brain_id IS NULL
           OR ('Paper' IN labels(neighbor) AND neighbor.brain_id = $brain_id)
           OR (NOT 'Paper' IN labels(neighbor) AND $brain_id IN neighbor.brain_ids)
        RETURN neighbor.slug AS slug, labels(neighbor)[0] AS type,
               CASE WHEN 'Paper' IN labels(neighbor) THEN neighbor.title ELSE neighbor.display_label END AS label
        """,
        slug=seed["slug"],
        brain_id=brain_id,
    ).data()
    provenance = [
        {"slug": n["slug"], "type": n["type"], "label": n["label"], "relation": None, "direction": None}
        for n in provenance_rows if n["slug"]
    ]

    semantic = _ranked_semantic_neighbors(session, seed["type"], seed["slug"], brain_id, routed_types, query_vec)
    capped = semantic[:max(1, neighbor_cap)]

    for item in capped[:max(0, hop2_top_n)]:
        item["hop2"] = _ranked_semantic_neighbors(
            session, item["type"], item["slug"], brain_id, routed_types, query_vec
        )[:max(1, neighbor_cap)]

    # neighbor_cap으로 잘리기 전 원래 개수 - "N개 -> M개로 축소" 같은 안내를
    # 만들 때 필요해서 같이 돌려준다(자르고 나면 원래 개수를 알 방법이 없다).
    return {"provenance": provenance, "semantic": capped, "semantic_total_before_cap": len(semantic)}


def search(
    query: str,
    top_k: int = 10,
    brain_id: str | None = None,
    mode: str = "all",
    relation_types: list[str] | None = None,
    neighbor_cap: int = 5,
    hop2_top_n: int = 0,
) -> list[dict]:
    """하이브리드 그래프 검색 - 벡터 유사도 + 풀텍스트 매치로 시드 노드를 찾고,
    각 시드에서 1촌 이웃까지 같이 반환한다(그래프 확장). GraphRAG의 핵심
    함수 - MCP의 search_graph 툴이 이걸 그대로 노출한다.

    brain_id를 주면 그 Brain 범위로만 결과를 좁힌다 - Concept/Entity는
    sync_node()가 미리 계산해 둔 brain_ids(그 노드가 걸린 논문들이 지금 속한
    Brain 집합) 안에 brain_id가 있어야 하고, Paper는 sync_paper()가 태그한
    brain_id가 정확히 일치해야 한다. 시드 단계(벡터/풀텍스트)뿐 아니라 이웃
    확장 단계에서도 같은 기준으로 걸러서, 특정 Brain으로 검색했을 때 다른
    Brain 소속 논문/개념이 이웃으로 섞여 나오지 않게 한다("그 Brain에 보이는
    논문에서만 나온 것처럼" 필터링 - Brain을 concept/entity 파일 자체가 아니라
    조회 시점에 계산하는 설계라, 다른 Brain에서 같은 개념을 검색하면 그 Brain
    범위로 다시 필터링된 채로 똑같이 보일 수 있다). brain_id=None(기본값)이면
    지금까지와 동일하게 전체 범위에서 검색한다.

    mode="all"(기본값)은 지금까지와 완전히 동일하게 동작한다(LINKED_TO +
    semantic 12종을 구분 없이 전부 1촌까지, 응답 형태도 "neighbors"[] 그대로) -
    아래 relation_types/neighbor_cap/hop2_top_n은 이때 전부 무시된다. 기존
    호출자(MCP search_graph, /api/graph-search의 기본 호출)는 이 함수
    시그니처가 바뀌었다는 것조차 몰라도 된다.

    mode="routed"면 docs/mcp/search_flow.md의 "개선 설계안"대로 동작한다:
    LINKED_TO(provenance)는 그대로 전부 반환하고, semantic 이웃만
    relation_types로 거른다. relation_types를 안 주면 쿼리 임베딩과 관계
    타입 설명 임베딩의 코사인 유사도로 자동 라우팅(top 3)한다. 남은 semantic
    이웃은 (타입 관련도) x (이웃 임베딩과 쿼리의 코사인 유사도)로 랭킹해서
    시드당 상위 neighbor_cap개만 남기고, 그중 상위 hop2_top_n개는 한 번 더
    (2촌) 같은 방식으로 확장한다("hop2" 필드). 이 모드의 응답은 시드마다
    "neighbors" 대신 "provenance_neighbors"/"semantic_neighbors"/
    "routed_types"를 담는다."""
    driver = get_driver()
    query_vec = embed_query(query)

    routed_types: dict[str, float] = {}
    if mode == "routed":
        known_types = list(load_relation_types(NODE_STORE_ROOT).keys())
        if relation_types:
            # 화이트리스트 검증: 여기서 걸러진 것만 밑에서 Cypher에 보간된다.
            routed_types = {t: 1.0 for t in relation_types if t in known_types}
        else:
            routed_types = _route_relation_types(query_vec, known_types, top_n=3)

    with driver.session() as session:
        # db.index.vector.queryNodes는 최신 Neo4j에서 새 SEARCH 문법으로 대체
        # 예정이라는 지원중단 경고를 내지만, 아직 정상 동작한다(경고일 뿐 오류
        # 아님) - 문법이 안정화되면 그때 옮긴다.
        vector_hits = session.run(
            """
            CALL db.index.vector.queryNodes('concept_embedding', $k, $vec) YIELD node, score
            WHERE $brain_id IS NULL OR $brain_id IN node.brain_ids
            RETURN node.slug AS slug, labels(node)[0] AS type, node.display_label AS label, score
            ORDER BY score DESC
            UNION
            CALL db.index.vector.queryNodes('entity_embedding', $k, $vec) YIELD node, score
            WHERE $brain_id IS NULL OR $brain_id IN node.brain_ids
            RETURN node.slug AS slug, labels(node)[0] AS type, node.display_label AS label, score
            ORDER BY score DESC
            """,
            k=top_k,
            vec=query_vec,
            brain_id=brain_id,
        ).data()
        vector_hits.sort(key=lambda h: h["score"], reverse=True)

        fulltext_hits = session.run(
            """
            CALL db.index.fulltext.queryNodes('node_fulltext', $q) YIELD node, score
            WHERE $brain_id IS NULL OR $brain_id IN node.brain_ids
            RETURN node.slug AS slug, labels(node)[0] AS type, node.display_label AS label, score
            ORDER BY score DESC
            LIMIT $k
            """,
            q=query,
            k=top_k,
            brain_id=brain_id,
        ).data()

        top_seeds = _reciprocal_rank_fusion(vector_hits, fulltext_hits)[:top_k]

        results = []
        for seed in top_seeds:
            if mode == "routed":
                expanded = _expand_routed_neighbors(
                    session, seed, brain_id, routed_types, neighbor_cap, hop2_top_n, query_vec
                )
                results.append({
                    "slug": seed["slug"],
                    "type": seed["type"],
                    "label": seed["label"],
                    "score": seed["score"],
                    "provenance_neighbors": expanded["provenance"],
                    "semantic_neighbors": expanded["semantic"],
                    "semantic_total_before_cap": expanded["semantic_total_before_cap"],
                    "routed_types": routed_types,
                })
                continue

            # -[r]- (타입 미지정, 무방향)로 바꿔서 LINKED_TO(출처)와 semantic
            # 관계(PART_OF/EXTENDS/... - docs/description/relation_types.md)를
            # 한 번에 같이 훑는다. Cypher는 -[r:TYPE1|TYPE2|...]- 처럼 명시적
            # 타입 나열 없이도 "아무 관계나"를 표현할 수 있어서, 화이트리스트를
            # 여기서 알 필요가 없다(sync_node()가 이미 저장 시점에 검증했다).
            # relation/direction은 LINKED_TO일 땐 의미가 없으므로 null로 두고,
            # semantic 관계일 때만 채운다 - startNode(r) = n이면 이 시드가 관계의
            # 주체(outgoing, "n이 TYPE을 neighbor에게 한다"), 아니면 대상
            # (incoming, "neighbor가 TYPE을 n에게 한다")이라는 뜻이다.
            neighbors = session.run(
                f"""
                MATCH (n:{seed['type']} {{slug: $slug}})
                OPTIONAL MATCH (n)-[r]-(neighbor)
                WHERE neighbor IS NULL
                   OR $brain_id IS NULL
                   OR ('Paper' IN labels(neighbor) AND neighbor.brain_id = $brain_id)
                   OR (NOT 'Paper' IN labels(neighbor) AND $brain_id IN neighbor.brain_ids)
                RETURN neighbor.slug AS slug, labels(neighbor)[0] AS type,
                       CASE WHEN 'Paper' IN labels(neighbor) THEN neighbor.title ELSE neighbor.display_label END AS label,
                       CASE WHEN r IS NULL OR type(r) = 'LINKED_TO' THEN NULL ELSE type(r) END AS relation,
                       CASE WHEN r IS NULL OR type(r) = 'LINKED_TO' THEN NULL
                            WHEN startNode(r) = n THEN 'outgoing' ELSE 'incoming' END AS direction
                """,
                slug=seed["slug"],
                brain_id=brain_id,
            ).data()
            results.append(
                {
                    "slug": seed["slug"],
                    "type": seed["type"],
                    "label": seed["label"],
                    "score": seed["score"],
                    "neighbors": [n for n in neighbors if n["slug"]],
                }
            )
        return results
