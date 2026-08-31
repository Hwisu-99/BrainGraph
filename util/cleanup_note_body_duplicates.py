"""1회성 정리 스크립트: 기존 논문 노트 본문에서 frontmatter(authors/tags)와
중복으로 적혀 있던 "**저자**: .../*source_meta*/#tag..." 줄을 걷어낸다.

배경: write_note()(paper_notes/obsidian_writer.py)가 예전에는 frontmatter에
이미 있는 authors/tags를 본문 상단에도 사람이 읽기 좋은 형식(볼드 텍스트 +
언더스코어 치환 해시태그)으로 한 번 더 적었다. 이제 write_note()는 이 줄을
쓰지 않도록 고쳤지만, 그건 앞으로 새로 처리되는 논문에만 적용되고 이미 만들어진
기존 노트 파일은 그대로 남아있다 - 이 스크립트가 기존 파일들도 같은 모양으로
맞춘다.

source_meta(예: "arXiv:2603.15031v1, 2026")는 frontmatter에 대응 필드가 없는
유일한 출처라 본문에 그대로 남긴다 - 저자/태그 줄만 제거한다.

사용법:
    python cleanup_note_body_duplicates.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv

# H1 제목 + 빈 줄 뒤에 오는 "**저자**: ...\n" + (선택) "*source_meta*\n" +
# (선택) "#tag ...\n" + 빈 줄 블록을 통째로 잡는다. write_note()가 예전에
# 정확히 이 순서/형식으로 썼으므로 그 형식과 어긋나는 파일(사용자가 손으로
# 고친 노트 등)은 매칭되지 않아 건드리지 않는다.
#
# vault의 기존 노트들이 CRLF/LF 두 스타일로 섞여 있어(예전에 다른 환경에서
# 만들어진 파일들도 있는 듯) 개행을 \r?\n으로 느슨하게 매칭한다 - \n으로만
# 고정하면 CRLF 파일에서 "빈 줄" 경계(\r\n\r\n)가 \n\n과 문자 그대로 일치하지
# 않아 매칭에 실패한다.
_NL = r"\r?\n"
_HEADER_RE = re.compile(
    rf"(?P<h1>^#[^\r\n]*{_NL}{_NL})"
    rf"\*\*저자\*\*:[^\r\n]*{_NL}"
    rf"(?P<meta>\*[^\r\n]*\*{_NL})?"
    rf"(?:#\S[^\r\n]*{_NL})?"
    rf"{_NL}",
    re.MULTILINE,
)


def _strip_duplicate_header(text: str) -> tuple[str, bool]:
    match = _HEADER_RE.search(text)
    if not match:
        return text, False
    # 새로 끼워 넣는 빈 줄(meta만 남았을 때)도 파일이 원래 쓰던 개행 스타일을
    # 따라야 한 파일 안에서 줄바꿈이 섞이지 않는다 - h1/meta 캡처 그룹 자체는
    # 이미 원본 바이트를 그대로 담고 있으니 그대로 두고, 새로 추가하는 구분자만
    # 맞춰준다.
    newline = "\r\n" if "\r\n" in text else "\n"
    meta = match.group("meta") or ""
    replacement = match.group("h1") + (meta + newline if meta else "")
    return text[: match.start()] + replacement + text[match.end() :], True


def main() -> None:
    load_dotenv()
    vault_path = os.environ["OBSIDIAN_VAULT_PATH"]
    autonote_dir = Path(vault_path) / "AutoNote"

    changed = 0
    skipped = 0
    for folder in sorted(autonote_dir.iterdir()):
        if not folder.is_dir():
            continue
        md_path = folder / f"{folder.name}.md"
        if not md_path.is_file():
            continue
        # newline="" - 원본 파일의 개행 문자(LF/CRLF)를 그대로 보존한다. 기본값
        # (newline=None)으로 읽고 쓰면 파이썬이 텍스트 모드 개행 변환을 하는데,
        # Windows에서는 쓸 때 모든 \n을 os.linesep(\r\n)으로 바꿔버려 - 원래 LF로
        # 저장돼 있던 기존 vault 파일 전체가 의도치 않게 CRLF로 바뀌는 부작용이
        # 있었다(실제로 겪음: 정리 대상이 아닌 줄까지 포함해 파일 전체가 diff에
        # 다르게 나옴).
        text = md_path.read_text(encoding="utf-8", newline="")
        new_text, did_change = _strip_duplicate_header(text)
        if did_change:
            md_path.write_text(new_text, encoding="utf-8", newline="")
            changed += 1
            print(f"정리됨: {folder.name}")
        else:
            skipped += 1
            print(f"변경 없음(패턴 불일치 또는 이미 정리됨): {folder.name}")

    print(f"\n총 {changed}개 정리, {skipped}개 그대로.")


if __name__ == "__main__":
    main()
