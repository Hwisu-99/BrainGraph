"""이미 처리된 논문들의 frontmatter concepts/entities를 node_store의 물리적
노드 파일(_concepts/, _entities/)로 채워넣는 1회성 마이그레이션 스크립트.

새로 처리되는 논문은 app.py가 자동으로 노드 파일을 만들지만, 이 기능이 생기기
전에 이미 처리된 논문들은 frontmatter에만 concepts/entities가 문자열로 남아있고
노드 파일이 없다. 논문을 실제 처리된 시간 순서대로 넣어야 "최초 생성 노드가
대표(slug/display_label 고정)"라는 규칙이 의미 있게 적용되므로, 여기서는 논문
.md 파일의 수정시각(mtime)을 처리 순서의 근사치로 쓴다.

사용법:
    python migrate_existing_papers.py                 # 전체 논문 마이그레이션
    python migrate_existing_papers.py --slug "<폴더명>"  # 논문 한 편만(테스트용)
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv

from paper_notes.node_store import NODE_STORE_ROOT, resolve_or_create_node

load_dotenv()

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def _read_frontmatter(path: Path) -> dict:
    match = _FRONTMATTER_RE.match(path.read_text(encoding="utf-8"))
    if not match:
        return {}
    try:
        return yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        return {}


def migrate_paper(vault_path: str, folder: Path) -> None:
    slug = folder.name
    md_path = folder / f"{slug}.md"
    if not md_path.is_file():
        print(f"[스킵] {slug} (.md 없음, 논문 폴더 아님)")
        return

    frontmatter = _read_frontmatter(md_path)
    title = frontmatter.get("title") or slug
    concepts = frontmatter.get("concepts") or []
    entities = frontmatter.get("entities") or []

    print(f"[{slug}] concept {len(concepts)}개, entity {len(entities)}개")
    for c in concepts:
        if isinstance(c, str):
            label, aliases = c, []
        else:
            label, aliases = c["label"], c.get("aliases") or []
        node_slug = resolve_or_create_node(NODE_STORE_ROOT, "concept", label, aliases, slug, title)
        print(f"  concept: {label!r:45} -> {node_slug}")

    for e in entities:
        node_slug = resolve_or_create_node(
            NODE_STORE_ROOT, "entity", e["label"], e.get("aliases") or [], slug, title
        )
        print(f"  entity:  {e['label']!r:45} -> {node_slug}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", help="특정 논문 폴더명만 마이그레이션(테스트용)")
    args = parser.parse_args()

    vault_path = os.environ.get("OBSIDIAN_VAULT_PATH")
    if not vault_path or not Path(vault_path).is_dir():
        print("OBSIDIAN_VAULT_PATH가 설정되어 있지 않거나 존재하지 않습니다.")
        return

    autonote_dir = Path(vault_path) / "AutoNote"
    folders = [f for f in autonote_dir.iterdir() if f.is_dir()]

    if args.slug:
        folders = [f for f in folders if f.name == args.slug]
        if not folders:
            print(f"'{args.slug}' 폴더를 찾을 수 없습니다.")
            return
    else:
        folders.sort(key=lambda f: (f / f"{f.name}.md").stat().st_mtime if (f / f"{f.name}.md").is_file() else 0)

    for folder in folders:
        migrate_paper(vault_path, folder)


if __name__ == "__main__":
    main()
