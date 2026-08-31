"""Supabase Storage 연동이 정상인지 확인하는 스탠드얼론 스크립트.

사용법:
    python test_supabase_upload.py

.env의 SUPABASE_URL / SUPABASE_KEY / SUPABASE_BUCKET을 읽어 더미 텍스트 파일을
버킷에 업로드하고, 성공하면 공개/서명 URL 대신 버킷 내 경로와 파일 목록을 출력한다.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from dotenv import load_dotenv

from paper_notes.supabase_writer import _get_client

load_dotenv()


def main() -> None:
    bucket_name = os.getenv("SUPABASE_BUCKET", "autonote-notes")
    storage_path = "_connection_test/ping.md"
    content = f"# AutoNote ↔ Supabase 연동 테스트\n\n{datetime.now(timezone.utc).isoformat()}\n".encode("utf-8")

    print(f"버킷: {bucket_name}")
    print(f"업로드 경로: {storage_path}")

    client = _get_client()
    client.storage.from_(bucket_name).upload(
        storage_path,
        content,
        file_options={"content-type": "text/markdown", "upsert": "true"},
    )
    print("업로드 성공.")

    files = client.storage.from_(bucket_name).list("_connection_test")
    names = [f["name"] for f in files]
    print(f"버킷 내 _connection_test/ 폴더 목록: {names}")

    if "ping.md" in names:
        print("연동 확인 완료: 업로드한 파일이 버킷에서 조회됩니다.")
    else:
        print("경고: 업로드는 성공했지만 목록 조회에서 파일이 보이지 않습니다.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - 스크립트 목적상 원인을 그대로 출력
        print(f"연동 실패: {exc}")
        raise SystemExit(1) from exc
