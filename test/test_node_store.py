"""paper_notes.node_store(concept/entity 노드 파일 생성·갱신·병합 후보 감지)가
정상인지 확인하는 스탠드얼론 스크립트. 외부 서비스 호출 없이 순수 파일 I/O만 검증한다.

사용법:
    python test_node_store.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from paper_notes.node_store import (
    build_node_index,
    execute_merge,
    find_node_fuzzy,
    find_node_slug_fuzzy,
    get_user_section,
    list_merge_candidates,
    list_nodes,
    node_index,
    reject_merge_candidate,
    resolve_or_create_node,
    save_attachment,
    update_user_section,
)

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" - {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def test_creates_new_node_on_first_mention() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        slug = resolve_or_create_node(tmp, "concept", "Self-Attention", [], "paper-a", "Paper A", category="process")
        path = Path(tmp) / "_concepts" / "self-attention.md"

        check("새 concept 첫 등장 시 노드 파일 생성", path.is_file())
        check("반환된 slug가 파일명과 일치", slug == "self-attention")

        text = path.read_text(encoding="utf-8")
        check("frontmatter에 display_label 포함", 'display_label: Self-Attention' in text)
        check("frontmatter에 categories 리스트로 포함", "categories:\n- process" in text)
        check("본문에 출처 논문 링크 포함", "[[paper-a|Paper A]]" in text)


def test_exact_label_match_updates_same_node() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        slug1 = resolve_or_create_node(tmp, "concept", "Self-Attention", [], "paper-a", "Paper A")
        slug2 = resolve_or_create_node(tmp, "concept", "self attention", [], "paper-b", "Paper B")

        concepts_dir = Path(tmp) / "_concepts"
        check("대소문자/공백만 다른 라벨은 같은 노드로 판정", slug1 == slug2)
        check("새 파일이 추가로 생기지 않음", len(list(concepts_dir.glob("*.md"))) == 1)

        text = (concepts_dir / f"{slug1}.md").read_text(encoding="utf-8")
        check("두 출처 논문이 모두 기록됨", "paper-a" in text and "paper-b" in text)


def test_alias_bridges_two_existing_nodes_as_candidate_not_merge() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        slug_a = resolve_or_create_node(tmp, "concept", "MoE", [], "paper-a", "Paper A")
        slug_b = resolve_or_create_node(tmp, "concept", "Mixture-of-Experts Layer", [], "paper-b", "Paper B")

        check("서로 다른 표기는 alias 없이 별개 노드로 생성됨", slug_a != slug_b)

        primary_slug = resolve_or_create_node(
            tmp, "concept", "MoE", ["Mixture-of-Experts Layer"], "paper-c", "Paper C"
        )
        check("bridge 발생 시 더 먼저 생성된 노드가 대표로 반환됨", primary_slug == slug_a)

        concepts_dir = Path(tmp) / "_concepts"
        check(
            "두 노드 파일이 즉시 병합되지 않고 그대로 남아있음(승인 전까지 보류)",
            len(list(concepts_dir.glob("*.md"))) == 2,
        )

        candidates_path = Path(tmp) / "config" / "_merge_candidates.json"
        check("병합 후보가 _merge_candidates.json에 기록됨", candidates_path.is_file())
        if candidates_path.is_file():
            candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
            check(
                "기록된 후보가 올바른 두 slug를 가리킴",
                len(candidates) == 1
                and {candidates[0]["slug_a"], candidates[0]["slug_b"]} == {slug_a, slug_b},
                str(candidates),
            )


def test_user_notes_section_preserved_across_updates() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        slug = resolve_or_create_node(tmp, "concept", "Self-Attention", [], "paper-a", "Paper A")
        path = Path(tmp) / "_concepts" / f"{slug}.md"

        text = path.read_text(encoding="utf-8")
        marker = "user-notes"
        idx = text.index(marker)
        end_of_marker_line = text.index("\n", idx) + 1
        text_with_note = text[:end_of_marker_line] + "이건 내가 직접 쓴 메모다.\n"
        path.write_text(text_with_note, encoding="utf-8")

        resolve_or_create_node(tmp, "concept", "Self-Attention", [], "paper-b", "Paper B")

        updated_text = path.read_text(encoding="utf-8")
        check("갱신 후에도 사용자가 작성한 메모가 그대로 남아있음", "이건 내가 직접 쓴 메모다." in updated_text)
        check("갱신 후 새 출처도 함께 반영됨", "paper-b" in updated_text)


def test_display_label_fixed_at_first_creation() -> None:
    """알려진 설계 결정: 나중 논문이 제공하는 표기가 항상 더 낫다는 보장이 없으므로
    (예: OCR 품질 차이), display_label은 slug처럼 최초 생성 시로 고정하고 이후
    절대 덮어쓰지 않는다."""
    with tempfile.TemporaryDirectory() as tmp:
        resolve_or_create_node(tmp, "concept", "Self-Attention", [], "paper-a", "Paper A")
        resolve_or_create_node(tmp, "concept", "self attention", [], "paper-b", "Paper B")

        path = Path(tmp) / "_concepts" / "self-attention.md"
        text = path.read_text(encoding="utf-8")
        check(
            "나중 논문이 다른 표기를 줘도 display_label은 최초 생성 시 표기 그대로",
            "display_label: Self-Attention" in text,
            text,
        )


def _bridge_moe_candidate(tmp: str) -> tuple[str, str]:
    slug_a = resolve_or_create_node(tmp, "concept", "MoE", [], "paper-a", "Paper A")
    slug_b = resolve_or_create_node(tmp, "concept", "Mixture-of-Experts Layer", [], "paper-b", "Paper B")
    resolve_or_create_node(tmp, "concept", "MoE", ["Mixture-of-Experts Layer"], "paper-c", "Paper C")
    return slug_a, slug_b


def test_execute_merge_keeps_earlier_node_and_stubs_the_other() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        slug_a, slug_b = _bridge_moe_candidate(tmp)
        survivor = execute_merge(tmp, "concept", slug_a, slug_b)

        check("먼저 생성된 노드(MoE)가 생존자", survivor == slug_a)

        concepts_dir = Path(tmp) / "_concepts"
        survivor_text = (concepts_dir / f"{slug_a}.md").read_text(encoding="utf-8")
        stub_text = (concepts_dir / f"{slug_b}.md").read_text(encoding="utf-8")

        check("생존 노드 aliases에 패자 라벨이 흡수됨", "Mixture-of-Experts Layer" in survivor_text)
        check("생존 노드 sources에 두 논문 다 포함", "paper-a" in survivor_text and "paper-b" in survivor_text)
        check("패자 파일은 삭제되지 않고 redirect 스텁으로 남음", "redirect_to: " + slug_a in stub_text)


def test_execute_merge_preserves_both_user_notes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        slug_a, slug_b = _bridge_moe_candidate(tmp)
        concepts_dir = Path(tmp) / "_concepts"

        for slug, note in [(slug_a, "A에 쓴 메모"), (slug_b, "B에 쓴 메모")]:
            path = concepts_dir / f"{slug}.md"
            text = path.read_text(encoding="utf-8")
            idx = text.index("\n", text.index("user-notes")) + 1
            path.write_text(text[:idx] + note + "\n", encoding="utf-8")

        execute_merge(tmp, "concept", slug_a, slug_b)
        survivor_text = (concepts_dir / f"{slug_a}.md").read_text(encoding="utf-8")
        check("병합 후에도 두 노드의 메모가 모두 보존됨", "A에 쓴 메모" in survivor_text and "B에 쓴 메모" in survivor_text)


def test_execute_merge_marks_candidate_as_merged() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        slug_a, slug_b = _bridge_moe_candidate(tmp)
        execute_merge(tmp, "concept", slug_a, slug_b)

        check("병합 후 pending 후보 목록에서 사라짐", list_merge_candidates(tmp, status="pending") == [])
        merged = list_merge_candidates(tmp, status="merged")
        check("병합 이력이 merged 상태로 남음", len(merged) == 1, str(merged))


def test_reject_merge_candidate_does_not_resurface() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        slug_a, slug_b = _bridge_moe_candidate(tmp)
        reject_merge_candidate(tmp, "concept", slug_a, slug_b)
        check("거부 후 pending 목록에서 사라짐", list_merge_candidates(tmp, status="pending") == [])

        # 같은 alias를 제공하는 논문이 또 들어와도 거부된 후보가 재등록되지 않아야 함
        resolve_or_create_node(tmp, "concept", "MoE", ["Mixture-of-Experts Layer"], "paper-d", "Paper D")
        check(
            "같은 alias가 다시 감지돼도 거부된 후보는 재등록되지 않음",
            list_merge_candidates(tmp, status="pending") == [],
        )


def test_redirect_stub_excluded_from_future_matching() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        slug_a, slug_b = _bridge_moe_candidate(tmp)
        execute_merge(tmp, "concept", slug_a, slug_b)

        # 병합 후 patper-e가 패자였던 라벨을 다시 언급해도 새 노드가 생기지 않고
        # 생존 노드로 바로 연결돼야 한다(흡수된 alias 덕분).
        resolved = resolve_or_create_node(tmp, "concept", "Mixture-of-Experts Layer", [], "paper-e", "Paper E")
        check("병합된 라벨을 다시 언급해도 생존 노드로 연결됨", resolved == slug_a)

        concepts_dir = Path(tmp) / "_concepts"
        check("redirect 스텁으로 인해 새 노드가 잘못 생성되지 않음", len(list(concepts_dir.glob("*.md"))) == 2)


def test_find_node_slug_fuzzy_matches_label_and_aliases() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        slug = resolve_or_create_node(
            tmp, "concept", "Self-Attention", ["Self-Attention Mechanism"], "paper-a", "Paper A"
        )
        resolve_or_create_node(tmp, "concept", "Byte Pair Encoding", [], "paper-b", "Paper B")

        nodes = list_nodes(tmp, "concept")
        check("정규화된 display_label로 매칭됨", find_node_slug_fuzzy(nodes, "self attention") == slug)
        check("정규화된 alias로도 매칭됨", find_node_slug_fuzzy(nodes, "self attention mechanism") == slug)
        check("무관한 라벨은 매칭 안 됨", find_node_slug_fuzzy(nodes, "byte pair encoding") != slug)


def test_resolve_or_create_node_matches_via_fuzzy_similarity_without_alias() -> None:
    """실제로 겪은 버그: alias 없이도 그래프용 dedupe_labels()가 이미 같은
    개념으로 묶는 단수/복수 같은 표기 차이는, node_store도 alias 없이 MinHash
    퍼지 매칭만으로 같은 노드로 인식해야 한다(exact match/alias로는 못 잡음)."""
    with tempfile.TemporaryDirectory() as tmp:
        slug1 = resolve_or_create_node(tmp, "concept", "Selective State Space Models", [], "paper-a", "Paper A")
        slug2 = resolve_or_create_node(tmp, "concept", "Selective State Space Model", [], "paper-b", "Paper B")

        concepts_dir = Path(tmp) / "_concepts"
        check("alias 없는 단수/복수 표기 차이도 같은 노드로 판정됨", slug1 == slug2)
        check("새 파일이 추가로 생기지 않음", len(list(concepts_dir.glob("*.md"))) == 1)

        nodes = list_nodes(tmp, "concept")
        check(
            "find_node_slug_fuzzy도 같은 기준으로 매칭됨",
            find_node_slug_fuzzy(nodes, "Selective State Space Model") == slug1,
        )


def test_build_node_index_maps_normalized_names_to_node() -> None:
    nodes = [
        {"slug": "self-attention", "display_label": "Self-Attention", "aliases": ["SA"]},
        {"slug": "moe", "display_label": "MoE", "aliases": ["Mixture-of-Experts"]},
    ]
    index = build_node_index(nodes)
    check("정규화된 display_label이 키로 등록됨", index.get("self attention") is nodes[0])
    check("정규화된 alias도 키로 등록됨", index.get("sa") is nodes[0])
    check("다른 노드의 alias도 각자 등록됨", index.get("mixture of experts") is nodes[1])
    check("등록 안 된 표기는 인덱스에 없음", index.get("attention") is None)


def test_find_node_fuzzy_with_index_matches_exact_without_scanning_all_nodes() -> None:
    """index를 넘기면 완전일치/alias는 O(1) 사전 조회로 먼저 확인해야 한다.
    무관한 노드를 잔뜩 섞어놔도 결과가 똑같이 정확해야 함을 확인한다."""
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(20):
            resolve_or_create_node(tmp, "concept", f"Filler Concept {i}", [], f"paper-{i}", f"Paper {i}")
        target_slug = resolve_or_create_node(
            tmp, "concept", "Self-Attention", ["SA"], "paper-target", "Paper Target"
        )

        nodes = list_nodes(tmp, "concept")
        idx = node_index(tmp, "concept")
        check("index 기반 조회로 display_label 매칭", find_node_fuzzy(nodes, "Self-Attention", index=idx)["slug"] == target_slug)
        check("index 기반 조회로 alias 매칭", find_node_fuzzy(nodes, "SA", index=idx)["slug"] == target_slug)
        check("index에 없는 라벨은 매칭 안 됨(관련 없는 표기)", find_node_fuzzy(nodes, "Completely Unrelated", index=idx) is None)


def test_find_node_fuzzy_with_index_still_falls_back_to_fuzzy_matching() -> None:
    """index는 정확일치/alias만 담고 있어서, 표기가 살짝 다른(단수/복수 등) 새
    라벨은 index에 없다 - 이 경우에도 기존 선형 탐색+MinHash 퍼지 매칭으로
    정상적으로 넘어가야 한다(index가 있어도 결과가 index 없을 때와 같아야 함)."""
    with tempfile.TemporaryDirectory() as tmp:
        slug = resolve_or_create_node(tmp, "concept", "Selective State Space Models", [], "paper-a", "Paper A")

        nodes = list_nodes(tmp, "concept")
        idx = node_index(tmp, "concept")
        check(
            "index에 없는 단수형도 fallback 퍼지 매칭으로 찾아짐",
            find_node_fuzzy(nodes, "Selective State Space Model", index=idx)["slug"] == slug,
        )


def test_node_index_shares_cache_and_invalidates_with_list_nodes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        resolve_or_create_node(tmp, "concept", "Self-Attention", [], "paper-a", "Paper A")
        idx1 = node_index(tmp, "concept")
        check("첫 조회 후 인덱스에 1개 등록", "self attention" in idx1)

        resolve_or_create_node(tmp, "concept", "Byte Pair Encoding", [], "paper-b", "Paper B")
        idx2 = node_index(tmp, "concept")
        check("새 노드 추가 후 인덱스가 갱신됨(list_nodes 캐시와 같은 무효화 규칙 공유)", "byte pair encoding" in idx2)


def test_list_nodes_cache_invalidates_on_new_node() -> None:
    """list_nodes()는 프로세스 메모리에 결과를 캐싱한다(디렉터리 서명이 같으면
    재사용). 캐싱 후 새 노드가 추가되면 다음 호출에서 그 변경이 반영돼야 한다 -
    안 그러면 새로 처리된 논문의 concept/entity가 그래프에서 계속 안 보이는
    회귀가 생긴다."""
    with tempfile.TemporaryDirectory() as tmp:
        resolve_or_create_node(tmp, "concept", "Self-Attention", [], "paper-a", "Paper A")
        first = list_nodes(tmp, "concept")
        check("첫 호출에서 노드 1개", len(first) == 1, str(first))

        resolve_or_create_node(tmp, "concept", "Byte Pair Encoding", [], "paper-b", "Paper B")
        second = list_nodes(tmp, "concept")
        check("새 노드 추가 후 캐시가 무효화되어 2개로 반영됨", len(second) == 2, str(second))


def test_list_nodes_cache_invalidates_on_content_update() -> None:
    """파일이 새로 생기지 않고 기존 파일 내용만 갱신돼도(같은 개념을 다른 논문이
    또 언급하는 경우) 캐시가 그 변경을 반영해야 한다."""
    with tempfile.TemporaryDirectory() as tmp:
        resolve_or_create_node(tmp, "concept", "Self-Attention", [], "paper-a", "Paper A")
        list_nodes(tmp, "concept")  # 캐시 warm-up

        resolve_or_create_node(tmp, "concept", "self attention", [], "paper-b", "Paper B")
        updated = list_nodes(tmp, "concept")
        check(
            "같은 파일의 sources 갱신도 캐시에 반영됨",
            len(updated) == 1 and len(updated[0]["sources"]) == 2,
            str(updated),
        )


def test_update_user_section_replaces_only_user_part() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        slug = resolve_or_create_node(tmp, "concept", "Self-Attention", [], "paper-a", "Paper A")

        update_user_section(tmp, "concept", slug, "내가 쓴 메모")
        check("저장한 메모가 get_user_section으로 그대로 조회됨", get_user_section(tmp, "concept", slug) == "내가 쓴 메모")

        path = Path(tmp) / "_concepts" / f"{slug}.md"
        text = path.read_text(encoding="utf-8")
        check("자동 생성 영역(등장 논문)은 그대로 남아있음", "paper-a" in text and "**등장 논문**" in text)

        update_user_section(tmp, "concept", slug, "메모를 덮어씀")
        check("두 번째 저장이 이전 메모를 완전히 대체함", get_user_section(tmp, "concept", slug) == "메모를 덮어씀")


def test_update_user_section_rejects_missing_or_merged_node() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        raised_missing = False
        try:
            update_user_section(tmp, "concept", "no-such-slug", "메모")
        except FileNotFoundError:
            raised_missing = True
        check("존재하지 않는 노드는 FileNotFoundError", raised_missing)

        slug_a, slug_b = _bridge_moe_candidate(tmp)
        execute_merge(tmp, "concept", slug_a, slug_b)
        raised_merged = False
        try:
            update_user_section(tmp, "concept", slug_b, "메모")
        except ValueError:
            raised_merged = True
        check("병합돼 사라진(redirect) 노드에는 메모를 못 씀", raised_merged)


def test_save_attachment_creates_file_and_returns_relative_path() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        slug = resolve_or_create_node(tmp, "entity", "Softmax", [], "paper-a", "Paper A")

        rel_path = save_attachment(tmp, "entity", slug, "screenshot.png", b"fake-png-bytes")
        check(
            "반환된 경로가 attachments/entities/<slug>/ 아래를 가리킴",
            rel_path.startswith(f"attachments/entities/{slug}/") and rel_path.endswith(".png"),
            rel_path,
        )

        saved_file = Path(tmp) / rel_path
        check("실제 파일이 저장됨", saved_file.is_file())
        check("저장된 내용이 업로드한 바이트와 동일함", saved_file.read_bytes() == b"fake-png-bytes")


def test_save_attachment_rejects_bad_extension_and_missing_node() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        slug = resolve_or_create_node(tmp, "concept", "Self-Attention", [], "paper-a", "Paper A")

        raised_ext = False
        try:
            save_attachment(tmp, "concept", slug, "malware.exe", b"...")
        except ValueError:
            raised_ext = True
        check("허용되지 않은 확장자는 거부됨", raised_ext)

        raised_missing = False
        try:
            save_attachment(tmp, "concept", "no-such-slug", "a.png", b"...")
        except FileNotFoundError:
            raised_missing = True
        check("존재하지 않는 노드에는 첨부 못 함", raised_missing)


def test_entity_node_has_no_category_field() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        slug = resolve_or_create_node(tmp, "entity", "Softmax", [], "paper-a", "Paper A")
        path = Path(tmp) / "_entities" / f"{slug}.md"
        text = path.read_text(encoding="utf-8")
        check("entity 노드는 concepts 폴더가 아닌 entities 폴더에 생성됨", path.is_file())
        check("entity 노드 frontmatter에는 category 필드가 없음", "category:" not in text)


def main() -> None:
    test_creates_new_node_on_first_mention()
    test_exact_label_match_updates_same_node()
    test_alias_bridges_two_existing_nodes_as_candidate_not_merge()
    test_user_notes_section_preserved_across_updates()
    test_display_label_fixed_at_first_creation()
    test_execute_merge_keeps_earlier_node_and_stubs_the_other()
    test_execute_merge_preserves_both_user_notes()
    test_execute_merge_marks_candidate_as_merged()
    test_reject_merge_candidate_does_not_resurface()
    test_redirect_stub_excluded_from_future_matching()
    test_find_node_slug_fuzzy_matches_label_and_aliases()
    test_resolve_or_create_node_matches_via_fuzzy_similarity_without_alias()
    test_build_node_index_maps_normalized_names_to_node()
    test_find_node_fuzzy_with_index_matches_exact_without_scanning_all_nodes()
    test_find_node_fuzzy_with_index_still_falls_back_to_fuzzy_matching()
    test_node_index_shares_cache_and_invalidates_with_list_nodes()
    test_list_nodes_cache_invalidates_on_new_node()
    test_list_nodes_cache_invalidates_on_content_update()
    test_update_user_section_replaces_only_user_part()
    test_update_user_section_rejects_missing_or_merged_node()
    test_save_attachment_creates_file_and_returns_relative_path()
    test_save_attachment_rejects_bad_extension_and_missing_node()
    test_entity_node_has_no_category_field()

    print()
    if FAILURES:
        print(f"{len(FAILURES)}개 실패: {FAILURES}")
        raise SystemExit(1)
    print("전체 통과.")


if __name__ == "__main__":
    main()
