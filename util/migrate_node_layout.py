"""1회성 마이그레이션: 기존 concept/entity 노드 파일들의 본문(자동 생성 영역)을
새 레이아웃(제목 -> 다른 표기 -> 카테고리(concept만) -> 등장 논문 -> ---
-> AI 설명 -> --- -> 사용자 메모)으로 다시 그린다.

배경: node_store._write_node_file()의 본문 순서를 바꿨다(예전엔 AI 설명이
먼저, "## 등장 논문" 헤딩이 나중이었음). 새로 만들어지는 파일은 이미 새
레이아웃으로 나오지만, 이미 만들어진 파일들은 그대로라 refresh_auto_section()
으로 한 번 다시 써줘야 한다. frontmatter와 사용자가 남긴 메모는 그대로
보존된다 - 오직 자동 생성 영역의 텍스트 배치만 바뀐다.

사용법:
    python migrate_node_layout.py
"""

from __future__ import annotations

from paper_notes.node_store import NODE_STORE_ROOT, list_nodes, refresh_auto_section


def main() -> None:
    refreshed = 0
    for node_type in ("concept", "entity"):
        for node in list_nodes(NODE_STORE_ROOT, node_type):
            if refresh_auto_section(NODE_STORE_ROOT, node_type, node["slug"]):
                refreshed += 1
                print(f"갱신됨: {node_type}/{node['slug']}")

    print(f"\n총 {refreshed}개 파일 레이아웃 갱신.")


if __name__ == "__main__":
    main()
