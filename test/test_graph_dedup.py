"""paper_notes.dedup / graph_builder의 concept·entity 중복 제거가 정상인지 확인하는
스탠드얼론 스크립트. 외부 서비스(Supabase, Anthropic API) 호출 없이 순수 함수만 검증한다.

사용법:
    python test_graph_dedup.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import mock

from paper_notes.dedup import dedupe_labels, jaccard_estimate, labels_match, minhash_signature, normalize_label
from paper_notes.dedup import _hash_permutations, _shingles
from paper_notes.graph_builder import build_graph
from paper_notes.node_store import create_node_manual, resolve_or_create_node

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" - {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def test_normalize_label() -> None:
    check(
        "normalize_label: 대소문자/구두점/공백 차이 흡수",
        normalize_label("Self-Attention") == normalize_label("self  attention"),
        f"{normalize_label('Self-Attention')!r} vs {normalize_label('self  attention')!r}",
    )
    check(
        "normalize_label: 완전히 다른 단어는 다른 키",
        normalize_label("Self-Attention") != normalize_label("Cross Attention"),
    )


def test_minhash_similarity_ordering() -> None:
    perms = _hash_permutations(64)
    sig_a = minhash_signature(_shingles(normalize_label("self attention mechanism")), perms)
    sig_b = minhash_signature(_shingles(normalize_label("self attention")), perms)
    sig_c = minhash_signature(_shingles(normalize_label("byte pair encoding")), perms)

    sim_related = jaccard_estimate(sig_a, sig_b)
    sim_unrelated = jaccard_estimate(sig_a, sig_c)
    check(
        "MinHash: 관련 라벨의 유사도가 무관한 라벨보다 높다",
        sim_related > sim_unrelated,
        f"related={sim_related:.2f} unrelated={sim_unrelated:.2f}",
    )


def test_dedupe_labels_merges_near_duplicates() -> None:
    labels = {"Self-Attention", "self attention", "Self Attention Mechanism", "Byte Pair Encoding"}
    canon = dedupe_labels(labels)

    check(
        "dedupe_labels: 대소문자만 다른 라벨은 항상 병합",
        canon["Self-Attention"] == canon["self attention"],
        str(canon),
    )
    check(
        "dedupe_labels: 무관한 라벨은 병합되지 않음",
        canon["Byte Pair Encoding"] != canon["Self-Attention"],
        str(canon),
    )
    check(
        "dedupe_labels: 대표 라벨 선택이 결정적(재현 가능)",
        dedupe_labels(labels) == canon,
    )


def test_dedupe_labels_known_limitation_containment() -> None:
    """알려진 한계: "X" vs "X Mechanism"처럼 한쪽이 다른 쪽을 포함하는 관계는
    길이가 다른 두 문자열의 3-gram 자카드 유사도가 구조적으로 낮게 나와(둘 다
    포함해도 분모인 합집합이 커짐) 기본 threshold(0.82)에서는 병합되지 않는다.
    실측 자카드가 약 0.55에 불과해 threshold를 낮추면 무관한 라벨의 오탐 병합
    위험이 커지므로, 이 케이스는 의도적으로 병합하지 않는 보수적 선택이다.
    이 테스트는 그 한계를 문서화한다 - 통과 실패 시 threshold/유사도 함수가
    바뀌어 동작이 달라졌다는 신호다."""
    canon = dedupe_labels({"Self-Attention", "Self Attention Mechanism"})
    check(
        "dedupe_labels: 포함 관계(단어 추가)는 기본 threshold에서 병합되지 않음 (알려진 한계)",
        canon["Self-Attention"] != canon["Self Attention Mechanism"],
        str(canon),
    )


def test_dedupe_labels_merges_via_shared_alias() -> None:
    """포함 관계라 문자열 유사도로는 병합되지 않는 라벨도(위 테스트 참고),
    두 라벨이 같은 alias를 공유하면 병합돼야 한다."""
    labels = {"Self-Attention", "Attention Mechanism"}
    aliases = {
        "Self-Attention": ["Self-Attention Mechanism"],
        "Attention Mechanism": ["Self-Attention Mechanism"],
    }
    canon = dedupe_labels(labels, aliases=aliases)
    check(
        "dedupe_labels: 같은 alias를 공유하는 라벨은 병합됨",
        canon["Self-Attention"] == canon["Attention Mechanism"],
        str(canon),
    )


def test_dedupe_labels_alias_matches_other_labels_own_text() -> None:
    """alias가 vault에 이미 존재하는 다른 라벨의 표기 그 자체인 경우도 병합돼야 한다."""
    labels = {"Self-Attention", "Self-Attention Mechanism"}
    aliases = {"Self-Attention": ["Self-Attention Mechanism"]}
    canon = dedupe_labels(labels, aliases=aliases)
    check(
        "dedupe_labels: alias가 다른 라벨의 실제 표기와 일치하면 병합됨",
        canon["Self-Attention"] == canon["Self-Attention Mechanism"],
        str(canon),
    )


def test_dedupe_labels_no_alias_does_not_merge_unrelated() -> None:
    """alias가 없는 라벨끼리는 기존 문자열 유사도 로직만 적용돼야 한다(회귀 방지)."""
    labels = {"Self-Attention", "Attention Mechanism"}
    canon = dedupe_labels(labels)
    check(
        "dedupe_labels: alias 없이는 상위/하위 범주 개념이 병합되지 않음",
        canon["Self-Attention"] != canon["Attention Mechanism"],
        str(canon),
    )


def test_dedupe_labels_blocks_numeric_mismatch() -> None:
    labels = {"ResNet-50", "ResNet-101", "GPT-2", "GPT-3"}
    canon = dedupe_labels(labels)
    check(
        "dedupe_labels: 숫자가 다른 버전 라벨은 병합 안 됨",
        canon["ResNet-50"] != canon["ResNet-101"] and canon["GPT-2"] != canon["GPT-3"],
        str(canon),
    )


def test_dedupe_labels_empty_and_single() -> None:
    check("dedupe_labels: 빈 입력 -> 빈 매핑", dedupe_labels(set()) == {})
    single = dedupe_labels({"Attention"})
    check("dedupe_labels: 단일 라벨은 자기 자신에 매핑", single == {"Attention": "Attention"})


def test_labels_match_covers_dedupe_labels_criteria() -> None:
    """labels_match()는 node_store.py가 graph_builder.py의 dedupe_labels()와
    같은 기준으로 단건 비교를 할 수 있게 뽑아낸 함수다. dedupe_labels()가 쓰는
    세 기준(정규화 완전일치/MinHash 유사도/숫자 불일치 차단)이 단건 비교에서도
    그대로 적용되는지 확인한다. 실제로 겪은 버그: 그래프용 dedup이 대표로 고른
    "Selective State Space Model"(단수)과 node_store 파일의 display_label
    "Selective State Space Models"(복수)가 문자열은 다르지만 이 기준으로는
    같은 개념으로 판단돼야 한다."""
    check(
        "labels_match: 정규화 완전일치(대소문자 차이)",
        labels_match("Self-Attention", "self attention"),
    )
    check(
        "labels_match: 단수/복수 차이는 MinHash 유사도로 같은 개념 판정",
        labels_match("Selective State Space Model", "Selective State Space Models"),
    )
    check(
        "labels_match: 무관한 라벨은 매칭 안 됨",
        not labels_match("Selective State Space Model", "Byte Pair Encoding"),
    )
    check(
        "labels_match: 숫자만 다른 버전은 매칭 안 됨(dedupe_labels의 숫자 가드와 동일)",
        not labels_match("ResNet-50", "ResNet-101"),
    )


def _write_paper(vault: Path, slug: str) -> None:
    """graph_builder가 읽는 논문 노트는 이제 title/tags만 있으면 된다 -
    concepts/entities는 node_store(각 노드 파일의 sources)가 유일한 소스다."""
    folder = vault / "AutoNote" / slug
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{slug}.md").write_text(
        f"---\ntitle: {slug}\ntags: []\n---\n\n# {slug}\n", encoding="utf-8"
    )


def test_build_graph_reads_concept_edges_from_node_store() -> None:
    """graph_builder는 논문 frontmatter가 아니라 concept 파일 자신의 sources를
    읽어 논문 <-> concept 에지를 만든다. 같은 concept이 두 논문에서 처리되면
    (resolve_or_create_node가 이미 한 파일로 합쳐뒀으므로) 그래프에는 노드
    하나, 에지 두 개로 나타나야 한다."""
    with tempfile.TemporaryDirectory() as tmp:
        vault = Path(tmp) / "vault"
        store = Path(tmp) / "store"
        _write_paper(vault, "paper-a")
        _write_paper(vault, "paper-b")
        resolve_or_create_node(str(store), "concept", "Self-Attention", [], "paper-a", "Paper A")
        resolve_or_create_node(str(store), "concept", "self attention", [], "paper-b", "Paper B")

        with mock.patch("paper_notes.graph_builder.NODE_STORE_ROOT", str(store)):
            graph = build_graph(str(vault))

        concept_nodes = [n for n in graph["nodes"] if n["type"] == "concept"]
        check(
            "build_graph: node_store에서 이미 합쳐진 concept은 노드 하나로만 나타남",
            len(concept_nodes) == 1,
            f"concept nodes: {concept_nodes}",
        )
        targets = {e["target"] for e in graph["edges"] if e["source"] in {"paper-a", "paper-b"}}
        check(
            "build_graph: paper-a/paper-b 둘 다 같은 concept 노드를 가리킴",
            targets == {concept_nodes[0]["id"]},
            f"{targets}",
        )


def test_build_graph_entity_concept_slug_makes_concept_entity_edge() -> None:
    """entity의 sources 항목에 concept_slug가 있으면 논문이 아니라 concept
    노드에서 entity로 에지가 생겨야 한다(concept_slug가 없는 entity는 지금도
    논문에 직접 연결됨 - 아래 test_build_graph_reads_concept_edges_from_node_store가
    그 경로를 이미 concept으로 검증하므로 여기선 대비만 확인)."""
    with tempfile.TemporaryDirectory() as tmp:
        vault = Path(tmp) / "vault"
        store = Path(tmp) / "store"
        _write_paper(vault, "paper-a")
        concept_slug = resolve_or_create_node(str(store), "concept", "Attention", [], "paper-a", "Paper A")
        resolve_or_create_node(
            str(store), "entity", "Scaled Dot-Product Attention", [], "paper-a", "Paper A",
            concept_slug=concept_slug,
        )

        with mock.patch("paper_notes.graph_builder.NODE_STORE_ROOT", str(store)):
            graph = build_graph(str(vault))

        concept_id = next(n["id"] for n in graph["nodes"] if n["type"] == "concept")
        entity_id = next(n["id"] for n in graph["nodes"] if n["type"] == "entity")
        edge_pairs = {(e["source"], e["target"]) for e in graph["edges"]}
        check(
            "build_graph: concept_slug가 있으면 concept -> entity 에지가 생김",
            (concept_id, entity_id) in edge_pairs,
            f"{edge_pairs}",
        )
        check(
            "build_graph: concept_slug가 있으면 paper -> entity 직접 에지는 생기지 않음",
            ("paper-a", entity_id) not in edge_pairs,
            f"{edge_pairs}",
        )


def test_build_graph_orphan_node_has_no_edges() -> None:
    """sources가 비어있는(아직 어떤 논문과도 연결 안 된) concept/entity도
    항상 노드로는 나타나야 하고, 에지는 하나도 없어야 한다."""
    with tempfile.TemporaryDirectory() as tmp:
        vault = Path(tmp) / "vault"
        store = Path(tmp) / "store"
        (vault / "AutoNote").mkdir(parents=True)
        create_node_manual(str(store), "concept", "Orphan Concept", None, None)

        with mock.patch("paper_notes.graph_builder.NODE_STORE_ROOT", str(store)):
            graph = build_graph(str(vault))

        concept_nodes = [n for n in graph["nodes"] if n["type"] == "concept"]
        check("build_graph: orphan concept도 노드로 나타남", len(concept_nodes) == 1, f"{concept_nodes}")
        check(
            "build_graph: orphan concept은 에지가 없음",
            not any(concept_nodes[0]["id"] in (e["source"], e["target"]) for e in graph["edges"]),
        )


def main() -> None:
    test_normalize_label()
    test_minhash_similarity_ordering()
    test_dedupe_labels_merges_near_duplicates()
    test_dedupe_labels_known_limitation_containment()
    test_dedupe_labels_merges_via_shared_alias()
    test_dedupe_labels_alias_matches_other_labels_own_text()
    test_dedupe_labels_no_alias_does_not_merge_unrelated()
    test_dedupe_labels_blocks_numeric_mismatch()
    test_dedupe_labels_empty_and_single()
    test_labels_match_covers_dedupe_labels_criteria()
    test_build_graph_reads_concept_edges_from_node_store()
    test_build_graph_entity_concept_slug_makes_concept_entity_edge()
    test_build_graph_orphan_node_has_no_edges()

    print()
    if FAILURES:
        print(f"{len(FAILURES)}개 실패: {FAILURES}")
        raise SystemExit(1)
    print("전체 통과.")


if __name__ == "__main__":
    main()
