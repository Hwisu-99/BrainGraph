"""AutoNote/ vault 폴더에 이미 만들어져 있는 논문 노트(.md)들을 한 번에
Supabase Storage로 백필 업로드하는 스크립트.

사용법:
    python backfill_supabase.py
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from paper_notes.supabase_writer import upload_note

load_dotenv()


def main() -> None:
    vault_path = os.getenv("OBSIDIAN_VAULT_PATH")
    if not vault_path:
        raise RuntimeError("OBSIDIAN_VAULT_PATH가 .env에 설정되어 있지 않습니다.")

    autonote_dir = Path(vault_path) / "AutoNote"
    if not autonote_dir.is_dir():
        raise RuntimeError(f"{autonote_dir} 폴더가 없습니다.")

    uploaded = 0
    skipped = 0

    for folder in sorted(autonote_dir.iterdir()):
        if not folder.is_dir():
            continue
        title_slug = folder.name
        note_path = folder / f"{title_slug}.md"
        if not note_path.is_file():
            print(f"건너뜀 (md 없음): {folder.name}")
            skipped += 1
            continue

        try:
            storage_path = upload_note(str(note_path), title_slug)
        except Exception as exc:  # noqa: BLE001 - 스크립트 목적상 원인을 그대로 출력
            print(f"실패: {folder.name} -> {exc}")
            skipped += 1
            continue

        print(f"업로드 완료: {folder.name} -> {storage_path}")
        uploaded += 1

    print(f"\n총 {uploaded}개 업로드, {skipped}개 건너뜀.")


if __name__ == "__main__":
    main()
