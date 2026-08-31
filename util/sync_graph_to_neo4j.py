"""1회성/필요시 재실행용: 지금 vault + node_store 전체 상태를 Neo4j에 통째로
동기화한다(paper_notes/graph_db.py의 full_resync 참고 - 기존 Neo4j 데이터를
지우고 현재 상태로 다시 만들므로, 드리프트가 의심될 때 안전하게 재실행할 수
있다).

NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD가 .env에 설정되어 있어야 한다(Neo4j Aura
콘솔에서 인스턴스를 만들면 나오는 값).

사용법:
    python sync_graph_to_neo4j.py
"""
from __future__ import annotations

import os

from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    vault_path = os.environ["OBSIDIAN_VAULT_PATH"]

    from paper_notes.graph_db import full_resync

    print("Neo4j 스키마 확인 + 전체 재동기화 시작...")
    result = full_resync(vault_path)
    print(f"완료: 논문 {result['papers']}개, 개념 {result['concepts']}개, 엔티티 {result['entities']}개")


if __name__ == "__main__":
    main()
