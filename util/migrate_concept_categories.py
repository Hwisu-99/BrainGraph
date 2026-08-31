"""1회성 마이그레이션: concept 노드의 category(단수, str|None) 필드를
categories(리스트)로 옮긴다.

배경: concept 하나가 여러 카테고리에 걸칠 수 있고, 사용자가 LLM이 매긴
카테고리를 add_category()/remove_category()로 직접 보정할 수 있어야 해서
category 필드를 리스트로 바꿨다(paper_notes/node_store.py). 이미 만들어진
concept 파일들은 아직 예전 단수 필드로 저장돼 있어 이 스크립트로 한 번
옮겨줘야 한다. 예전 값은 새 12개 카테고리 체계로 재해석하지 않고 그대로
보존한다 - 자세한 이유는 node_store.migrate_category_field() 참고.

사용법:
    python migrate_concept_categories.py
"""

from __future__ import annotations

from paper_notes.node_store import NODE_STORE_ROOT, migrate_category_field


def main() -> None:
    migrated = migrate_category_field(NODE_STORE_ROOT)
    for slug in migrated:
        print(f"이관됨: {slug}")
    print(f"\n총 {len(migrated)}개 이관.")


if __name__ == "__main__":
    main()
