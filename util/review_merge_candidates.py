"""_merge_candidates.json에 쌓인 concept/entity 병합 후보를 터미널에서 검토한다.

새 논문이 처리되다가 alias로 인해 서로 다른 두 기존 노드 파일이 이어지면
(paper_notes/node_store.py), 잘못 합쳐지면 되돌리기 어려우니 즉시 병합하지 않고
여기서 사람이 검토한 뒤 승인해야 실제로 합쳐진다.

사용법:
    python review_merge_candidates.py
"""

from __future__ import annotations

from paper_notes.node_store import (
    NODE_STORE_ROOT,
    execute_merge,
    get_display_label,
    list_merge_candidates,
    reject_merge_candidate,
)


def main() -> None:
    candidates = list_merge_candidates(NODE_STORE_ROOT, status="pending")
    if not candidates:
        print("검토할 병합 후보가 없습니다.")
        return

    print(f"검토할 병합 후보 {len(candidates)}개\n")
    for i, c in enumerate(candidates, 1):
        label_a = get_display_label(NODE_STORE_ROOT, c["type"], c["slug_a"])
        label_b = get_display_label(NODE_STORE_ROOT, c["type"], c["slug_b"])

        print(f"[병합 후보 {i}/{len(candidates)}] {c['type']}")
        print(f"  A: {label_a} ({c['slug_a']})")
        print(f"  B: {label_b} ({c['slug_b']})")
        print(f"  연결 근거: \"{c['detected_in_paper']}\" 논문이 alias \"{c['via_alias']}\"로 둘을 이음")

        answer = input("병합할까요? [y]es / [n]o(다시 안 물어봄) / [s]kip(나중에 다시): ").strip().lower()
        if answer == "y":
            survivor = execute_merge(NODE_STORE_ROOT, c["type"], c["slug_a"], c["slug_b"])
            print(f"  -> 병합 완료. 생존 노드: {survivor}\n")
        elif answer == "n":
            reject_merge_candidate(NODE_STORE_ROOT, c["type"], c["slug_a"], c["slug_b"])
            print("  -> 거부됨. 다시 묻지 않습니다.\n")
        else:
            print("  -> 스킵.\n")


if __name__ == "__main__":
    main()
