"""concept/entity 사이 semantic 관계 타입의 화이트리스트
(docs/description/relation_types.md 참고).

이 그래프엔 두 종류의 에지만 존재한다: 출처(provenance)를 나타내는
`LINKED_TO`(paper_notes/graph_db.py가 그대로 관리)와, concept/entity
사이의 실제 지식 관계를 나타내는 semantic 에지(여기서 관리하는 타입들).
Paper는 semantic 에지의 endpoint가 될 수 없다 - 도메인 지식 노드가
아니라 컨테이너/출처 노드이기 때문이다.

Cypher는 관계 타입을 파라미터로 못 받는다(`MERGE (a)-[r:$type]->(b)` 같은
문법 자체가 없음) - 그래서 관계 타입 문자열을 Cypher 쿼리에 f-string으로
보간해야 하는데, 검증 없이 보간하면 인젝션 위험이 있다. 이 모듈이 반환하는
화이트리스트를 보간 직전에 항상 멤버십 체크해야 한다(graph_db.py의 sync_node/
app.py의 run_pipeline이 이렇게 쓴다) - "config에 있으니 안전하다"가 아니라
"쓰기 직전마다 확인한다"가 실제 안전을 만든다는 점이 중요하다.

`config/relation_types.json`이 있으면 그걸 쓴다(도메인마다 어휘가 달라질 수
있으므로 사용자가 로컬에서 편집 가능하게 - config/의 다른 파일들과 같은
이유로 이 파일도 .gitignore 대상이다). 없거나 읽기 실패하면 아래 기본
12개로 폴백한다.
"""
from __future__ import annotations

import json
from pathlib import Path

# symmetric=True인 타입(COMPARED_TO/CONTRADICTS/RELATED)은 저장 시점에
# slug 알파벳순으로 from/to를 정규화한다 - 논문마다 어느 쪽을 먼저 언급하든
# 항상 같은 방향의 에지 하나로 합쳐지게 하기 위함
# (docs/description/relation_types.md의 "대칭 타입 정규화" 참고).
DEFAULT_RELATION_TYPES: dict[str, dict] = {
    "PART_OF": {"symmetric": False},
    "IS_A": {"symmetric": False},
    "USES": {"symmetric": False},
    "EXTENDS": {"symmetric": False},
    "IMPROVES_ON": {"symmetric": False},
    "OUTPERFORMS": {"symmetric": False},
    "SOLVES": {"symmetric": False},
    "EVALUATED_ON": {"symmetric": False},
    "LIMITED_BY": {"symmetric": False},
    "COMPARED_TO": {"symmetric": True},
    "CONTRADICTS": {"symmetric": True},
    "RELATED": {"symmetric": True},
}


# 12개 타입의 한글 설명 - docs/graph/relation_type.md의 표와 test/search_flow_visualizer.html의
# REL_TYPES를 그대로 옮긴 것(세 곳이 같은 문구를 쓰도록 유지). graph_db.py의
# search(mode="routed")가 이 설명을 임베딩해서 "쿼리와 관련도 높은 관계 타입"을
# 실제 코사인 유사도로 고르는 데 쓴다(docs/mcp/search_flow.md의 "개선 설계안" 참고).
# config/relation_types.json으로 타입을 커스터마이즈해도 설명까지 바꿀 방법은
# 아직 없다 - 알 수 없는 타입은 describe_relation_type()이 타입 이름 자체를
# 돌려준다(임베딩은 여전히 되지만 설명 없이 이름만으로).
DEFAULT_RELATION_TYPE_DESCRIPTIONS: dict[str, str] = {
    "PART_OF": "A가 B의 구성요소다",
    "IS_A": "A가 B의 구체적 사례/하위 유형이다",
    "USES": "A가 B를 도구/구성요소로 사용한다",
    "EXTENDS": "A가 B를 확장·발전시킨 후속 연구다",
    "IMPROVES_ON": "A가 B보다 개선된 성능/방식으로 제시된다",
    "OUTPERFORMS": "A가 B보다 실험적으로 우세하다",
    "COMPARED_TO": "A와 B가 서로 비교 대상이다 (대칭)",
    "CONTRADICTS": "A의 주장/결과가 B와 상충한다 (대칭)",
    "SOLVES": "A(방법)가 B(문제)를 해결한다",
    "EVALUATED_ON": "A가 B(데이터셋/벤치마크)로 검증됐다",
    "LIMITED_BY": "A가 B라는 한계를 가진다",
    "RELATED": "위 어디에도 안 맞는 약한 연결 (폴백)",
}


def describe_relation_type(relation_type: str) -> str:
    """관계 타입의 한글 설명. graph_db.py의 관계 타입 임베딩 캐싱용 - 알 수
    없는 타입이면(커스텀 config에만 있고 설명이 없는 경우) 타입 이름 자체를
    돌려준다."""
    return DEFAULT_RELATION_TYPE_DESCRIPTIONS.get(relation_type, relation_type)


def _config_path(store_root: str) -> Path:
    return Path(store_root) / "config" / "relation_types.json"


def load_relation_types(store_root: str) -> dict[str, dict]:
    """{TYPE: {"symmetric": bool}} 화이트리스트를 반환한다. 매번 파일을 다시
    읽는다 - 개인 vault 규모에서 파일 하나 읽는 비용은 무시할 만하고, 캐싱
    무효화를 신경 쓰는 것보다 항상 최신 설정을 반영하는 쪽이 더 안전하다."""
    path = _config_path(store_root)
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            types = data.get("types")
            if isinstance(types, dict) and types:
                return types
        except (json.JSONDecodeError, OSError):
            pass
    return DEFAULT_RELATION_TYPES


def is_symmetric(relation_types: dict[str, dict], relation_type: str) -> bool:
    return bool(relation_types.get(relation_type, {}).get("symmetric", False))
