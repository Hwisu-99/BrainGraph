"""1회성 마이그레이션: 논문 frontmatter의 concepts/entities를 node_store의
유일한 소스로 완전히 옮긴다.

배경: 그래프를 그릴 때(paper_notes/graph_builder.py) 이제 node_store만 보고
에지를 만든다. 두 가지를 확인/보정해야 한다:

1. 일부 논문(node_store 도입 이전에 처리된 것들)은 frontmatter에 concepts/
   entities가 있는데도 node_store에 해당 노드 파일이 아예 없다 - 이런 개념/
   용어는 이 마이그레이션 없이는 그래프에서 완전히 사라진다. 이 스크립트가
   찾아서 resolve_or_create_node()로 정식 생성한다(진짜 논문 처리 때와 동일한
   경로).
2. node_store에 이미 있는 entity라도, 그 논문 소스 항목에 concept_slug가
   없을 수 있다(entity가 어느 concept 밑에 묶이는지는 예전엔 논문 frontmatter
   에만 있었음) - set_source_concept_slug()로 채워 넣는다.

코드를 배포해도 논문 .md 파일 자체의 frontmatter(concepts/entities)는 그대로
남아있으므로(이제 아무도 안 읽을 뿐, 삭제되지 않음), 이 스크립트는 배포 후
아무 때나 돌릴 수 있고 여러 번 실행해도 안전하다(이미 반영된 항목은 다시
건드리지 않음).

사용법:
    python migrate_concept_slugs.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv

from paper_notes.dedup import normalize_label
from paper_notes.node_store import (
    NODE_STORE_ROOT,
    find_node_fuzzy,
    list_nodes,
    node_index,
    resolve_or_create_node,
    set_source_concept_slug,
)

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def _parse_frontmatter(text: str) -> dict:
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}
    try:
        return yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        return {}


def main() -> None:
    load_dotenv()
    vault_path = os.environ["OBSIDIAN_VAULT_PATH"]
    autonote_dir = Path(vault_path) / "AutoNote"

    created_concepts = 0
    created_entities = 0
    updated_concept_slugs = 0

    for folder in sorted(autonote_dir.iterdir()):
        if not folder.is_dir():
            continue
        paper_slug = folder.name
        md_path = folder / f"{paper_slug}.md"
        if not md_path.is_file():
            continue
        frontmatter = _parse_frontmatter(md_path.read_text(encoding="utf-8"))
        paper_title = frontmatter.get("title") or paper_slug

        fm_concepts = [
            {"label": c, "aliases": []} if isinstance(c, str) else {"label": c["label"], "aliases": c.get("aliases") or []}
            for c in (frontmatter.get("concepts") or [])
        ]
        fm_entities = frontmatter.get("entities") or []
        if not fm_concepts and not fm_entities:
            continue

        # concept부터 처리해 이 논문 안에서 쓰는 concept 라벨 -> slug 매핑을 만든다
        # (entity의 concept 필드를 풀 때 씀). list_nodes()/node_index()는
        # _LIST_NODES_CACHE가 mtime으로 무효화를 감지하므로, 방금 새로 만든 노드도
        # 바로 다음 조회부터 반영된다.
        concept_slug_by_key: dict[str, str] = {}
        for c in fm_concepts:
            concept_nodes = list_nodes(NODE_STORE_ROOT, "concept")
            concept_idx = node_index(NODE_STORE_ROOT, "concept")
            match = find_node_fuzzy(concept_nodes, c["label"], c["aliases"], concept_idx)
            if match:
                slug = match["slug"]
            else:
                slug = resolve_or_create_node(NODE_STORE_ROOT, "concept", c["label"], c["aliases"], paper_slug, paper_title)
                created_concepts += 1
                print(f"[{paper_slug}] concept 새로 생성: '{c['label']}' -> {slug}")
            for key in [c["label"], *c["aliases"]]:
                concept_slug_by_key[normalize_label(key)] = slug

        for e in fm_entities:
            label = e.get("label", "")
            aliases = e.get("aliases") or []
            concept_label = e.get("concept")
            concept_slug = concept_slug_by_key.get(normalize_label(concept_label)) if concept_label else None
            if concept_label and not concept_slug:
                # 이 논문 자신의 concepts 목록에 없던 concept 라벨(드묾) - 전체
                # node_store에서 한 번 더 찾아본다.
                m = find_node_fuzzy(list_nodes(NODE_STORE_ROOT, "concept"), concept_label, None, node_index(NODE_STORE_ROOT, "concept"))
                concept_slug = m["slug"] if m else None

            entity_nodes = list_nodes(NODE_STORE_ROOT, "entity")
            entity_idx = node_index(NODE_STORE_ROOT, "entity")
            match = find_node_fuzzy(entity_nodes, label, aliases, entity_idx)
            if match:
                if concept_slug and set_source_concept_slug(NODE_STORE_ROOT, match["slug"], paper_slug, concept_slug):
                    updated_concept_slugs += 1
                    print(f"[{paper_slug}] {match['slug']} -> concept_slug={concept_slug}")
            else:
                slug = resolve_or_create_node(
                    NODE_STORE_ROOT, "entity", label, aliases, paper_slug, paper_title, concept_slug=concept_slug
                )
                created_entities += 1
                print(f"[{paper_slug}] entity 새로 생성: '{label}' -> {slug}" + (f" (concept_slug={concept_slug})" if concept_slug else ""))

    print(f"\n새로 생성된 concept: {created_concepts}개")
    print(f"새로 생성된 entity: {created_entities}개")
    print(f"concept_slug 백필된 기존 entity source: {updated_concept_slugs}개")


if __name__ == "__main__":
    main()
