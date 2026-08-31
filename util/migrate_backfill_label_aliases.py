"""1회성 마이그레이션: _update_node()가 label 자체를 alias로 넣지 않던 버그
때문에, 이미 어긋난 concept/entity 노드들의 alias를 backfill한다.

배경: 논문이 label="X"(과 alias 몇 개)로 노드를 처리할 때, 그 label 자체가
아니라 alias가 기존 노드와 겹쳐서 매칭되는 경우가 있다(예: "Semiseparable
Matrix"가 "Semiseparable Matrices" 노드와 "SSS representation"이라는 공유
alias로 매칭됨). 예전 _update_node()는 이때 alias 리스트만 노드에 합치고
label 자체("Semiseparable Matrix")는 넣지 않아서, 그 논문 자신의 본문
위키링크(write_note()가 그 논문이 쓴 label 그대로 [[label]]로 씀)가 나중에
그 노드를 다시 못 찾는 문제가 있었다(paper_notes/node_store.py의
_update_node() 수정으로 앞으로는 안 생김 - 이 스크립트는 이미 벌어진 것만
정리).

이미 처리된 논문 중 아직 예전 frontmatter 형식(concepts:/entities: 필드)이
남아있는 것만 대상으로 한다 - 그 형식에만 "이 label로 어느 concept/entity를
가리켰는지"가 원본 그대로 남아있어서, 지금 그 label로 다시 find_node_fuzzy를
돌리면 당시와 똑같이 어느 노드에 매칭됐는지 재현할 수 있다(그 매칭에 쓰인
alias가 지금도 그대로 노드에 남아있으므로). 리팩터 이후 처리된 논문은
frontmatter에 이 정보가 없어 이 스크립트 대상이 아니다 - 애초에 새 코드로
처리됐으니 이 버그 자체가 안 생겼어야 한다(코드 수정이 먼저 적용된 뒤 처리된
논문이라면).

사용법:
    python migrate_backfill_label_aliases.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv

from paper_notes.node_store import (
    NODE_STORE_ROOT,
    DuplicateNodeError,
    add_alias,
    find_node_fuzzy,
    list_nodes,
    node_index,
    normalize_label,
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


def _backfill_one(node_type: str, label: str, aliases: list[str]) -> str | None:
    """label(+aliases)로 기존 노드를 다시 찾아, 그 노드의 alias 목록에 label
    자체가 없으면 채워 넣는다. 실제로 채워 넣었으면 그 노드의 slug, 이미
    있었거나 매칭 자체가 안 되면 None을 반환한다."""
    nodes = list_nodes(NODE_STORE_ROOT, node_type)
    idx = node_index(NODE_STORE_ROOT, node_type)
    match = find_node_fuzzy(nodes, label, aliases, idx)
    if not match:
        return None
    if normalize_label(label) == normalize_label(match["display_label"]):
        return None
    if any(normalize_label(a) == normalize_label(label) for a in (match.get("aliases") or [])):
        return None  # 이미 alias로 있음(이번 세션 코드 수정 이후 처리됐거나 이미 backfill됨)
    try:
        add_alias(NODE_STORE_ROOT, node_type, match["slug"], label)
        return match["slug"]
    except DuplicateNodeError:
        # 이 label을 이미 "다른" 노드가 alias/표기로 쓰고 있으면 안전하게 건너뛴다 -
        # 잘못 덮어써서 두 개념이 헷갈리게 되는 것보다 나음.
        return None


def main() -> None:
    load_dotenv()
    vault_path = os.environ["OBSIDIAN_VAULT_PATH"]
    autonote_dir = Path(vault_path) / "AutoNote"

    backfilled = 0
    skipped_conflict = 0

    for folder in sorted(autonote_dir.iterdir()):
        if not folder.is_dir():
            continue
        paper_slug = folder.name
        md_path = folder / f"{paper_slug}.md"
        if not md_path.is_file():
            continue
        frontmatter = _parse_frontmatter(md_path.read_text(encoding="utf-8"))

        for c in frontmatter.get("concepts") or []:
            if isinstance(c, str):
                continue  # 라벨만 있는 아주 예전 형식 - alias 정보가 없어 대상 아님
            label = c.get("label")
            if not label:
                continue
            result = _backfill_one("concept", label, c.get("aliases") or [])
            if result:
                backfilled += 1
                print(f"[{paper_slug}] concept alias 추가: '{label}' -> {result}")

        for e in frontmatter.get("entities") or []:
            label = e.get("label")
            if not label:
                continue
            result = _backfill_one("entity", label, e.get("aliases") or [])
            if result:
                backfilled += 1
                print(f"[{paper_slug}] entity alias 추가: '{label}' -> {result}")

    print(f"\n총 {backfilled}개 alias 추가.")


if __name__ == "__main__":
    main()
