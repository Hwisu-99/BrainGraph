"""concept/entity를 위한 물리적 Obsidian 노트 파일(_concepts/, _entities/)을 관리한다.

지금까지 concept/entity는 논문 md의 frontmatter 안에만 존재하는 문자열이었고, 실제
"같은 개념" 판정(dedup)은 /api/graph를 호출할 때마다 vault 전체를 다시 스캔해서
그 순간 계산됐다(paper_notes/graph_builder.py + dedup.py). 이 모듈은 그 판정을 논문을
처리하는 시점(쓸 때)으로 옮겨, concept/entity마다 실제로 클릭해서 들어갈 수 있는
노트 파일을 만든다 - 사용자가 그 노드에 대해 직접 메모를 남길 수 있게 하고, 나중에
brain consolidation 때 노드 단위로 비교/병합할 수 있게 하기 위함이다.

매칭 기준: 정규화 완전일치, alias 겹침에 더해 dedup.py의 labels_match()(MinHash
자카드 유사도, dedupe_labels()와 같은 기준)까지 포함한다 - graph_builder.py가
쓰는 판정과 동일한 함수를 공유해야, 그래프 뷰의 "대표 라벨"과 이 모듈이 고정한
display_label이 살짝 다른 표기(예: 단수/복수)일 때도 같은 노드로 인식되고, 두
시스템이 서로 다르게 판단해 어긋나는 문제가 생기지 않는다.

병합 정책: 새 논문이 제공한 alias가 이미 존재하는 서로 다른 두 노드 파일을 잇는
경우(예: 논문 A의 "MoE", 논문 B의 "Mixture-of-Experts Layer"를 논문 C가 이어줌),
파일을 즉시 합치지 않는다. 잘못된 병합(예: 상위/하위 개념을 같은 것으로 오판)은
사용자가 이미 노드에 메모를 남긴 뒤라면 되돌리기 어렵기 때문에, _merge_candidates.json에
후보(status=pending)로만 기록해두고, 실제 병합(execute_merge)은 util/review_merge_candidates.py
CLI로 사람이 검토·승인한 뒤에만 실행한다. 병합되면 나중 생성된 쪽은 삭제되지 않고
redirect_to를 가진 스텁으로 축소되며(다른 노트가 그 slug로 건 wikilink가 깨지지
않게), 그 라벨은 생존 노드의 aliases에 합쳐져 Obsidian 자체의 alias 해석으로도
정상 연결된다.

저장 위치: 노드 파일은 Obsidian vault가 아니라 이 프로젝트 폴더(NODE_STORE_ROOT)
아래 _concepts/_entities/config에 저장한다 - vault 안에 두면 Obsidian에서 노드를
클릭해 열람/메모할 수 있다는 장점이 있지만, vault 밖에 두기로 한 것은 사용자의
명시적 선택이다(구현 시 이 트레이드오프를 안내했고, vault 밖으로 두는 쪽을 확정함).

이번 단계는 새로 처리되는 논문에만 적용한다 - 기존에 이미 처리된 논문들은 아직 이
노드 파일 구조로 옮기지 않았고, graph_builder.py가 여전히 frontmatter 기반 실시간
dedup으로 그래프를 그린다(이 모듈은 별도로 병행되는 시스템이다).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

from paper_notes.dedup import labels_match, normalize_label
from paper_notes.relation_types import load_relation_types

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
_DIR_BY_TYPE = {"concept": "_concepts", "entity": "_entities"}
_ATTACHMENTS_DIR_BY_TYPE = {"concept": "concepts", "entity": "entities"}
# graph_builder.py도 첨부 이미지를 노드 md 본문에서 찾아 그래프의 attachment
# 노드로 만들 때 같은 확장자 목록을 써야 하므로 공개 상수로 둔다.
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

# concept 노드의 카테고리 통제 어휘(controlled vocabulary). claude_client.py의
# SCHEMA가 이 목록을 그대로 가져다 써서 LLM 출력과 사용자가 add_category()로
# 직접 추가할 수 있는 값의 범위가 항상 일치하게 한다 - 자유 텍스트로 두면
# "Optimization"/"optimization"처럼 표기만 다른 카테고리가 늘어나 필터링 의미가
# 없어진다(예전 category 필드가 실제로 이 문제를 겪었다: null 42개, process
# 17개로 사실상 두 값으로만 수렴).
CONCEPT_CATEGORIES = [
    "problem", "proposed_method", "architecture", "algorithm", "theory",
    "optimization", "training_strategy", "evaluation_setup", "finding",
    "input_representation", "limitation", "other",
]

# 노드 파일이 실제로 저장되는 기본 위치: 이 프로젝트 폴더(paper_notes/의 부모).
# resolve_or_create_node 등은 store_root를 인자로 받으므로(테스트에서는 임시
# 디렉터리를 대신 넘겨 격리한다), 실제 호출부(app.py 등)에서 이 상수를 쓴다.
NODE_STORE_ROOT = str(Path(__file__).resolve().parent.parent)


class DuplicateNodeError(Exception):
    """create_node_manual()이 사용자가 입력한 이름과 비슷한(정규화 완전일치·alias·
    MinHash 유사도) 노드가 이미 있다고 판단했을 때 던진다. slug가 완전히 같아서
    막는 FileExistsError와 달리, 이건 "그래도 새로 만들기"로 사용자가 우회할 수
    있다(force=True) - 퍼지 매칭은 오탐(비슷해 보이지만 실제론 다른 개념) 가능성이
    있어 무조건 막기보다는 사용자에게 물어보는 쪽이 안전하다."""

    def __init__(self, match: dict) -> None:
        self.match = match
        super().__init__(f"'{match['display_label']}'과(와) 비슷한 노드가 이미 있습니다.")

_AUTO_MARKER = (
    "<!-- auto-generated: 이 아래는 논문을 처리할 때마다 파이프라인이 자동으로 "
    "다시 씁니다. 직접 수정하지 마세요. -->"
)
_USER_MARKER = (
    "<!-- user-notes: 이 아래는 자동 생성 영역이 아닙니다. 자유롭게 메모를 "
    "남기세요 - 파이프라인은 이 아래 내용을 절대 덮어쓰지 않습니다. -->"
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _node_dir(store_root: str, node_type: str) -> Path:
    folder = Path(store_root) / _DIR_BY_TYPE[node_type]
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _slugify(label: str) -> str:
    return normalize_label(label).replace(" ", "-")


def _identity_keys(label: str, aliases: list[str]) -> set[str]:
    return {normalize_label(name) for name in [label, *aliases] if name and name.strip()}


def _names(label: str, aliases: list[str]) -> list[str]:
    return [n for n in [label, *aliases] if n and n.strip()]


def _matches_node(label: str, aliases: list[str], node: dict) -> bool:
    """새 label/aliases가 기존 node와 같은 개념인지 판단한다. 먼저 정규화
    완전일치나 alias 겹침으로 저렴하게 확인하고, 거기서 못 잡으면(예: 단수/복수
    차이처럼 표기만 살짝 다른 경우) graph_builder.py의 dedupe_labels()와 같은
    기준(labels_match, MinHash 자카드 유사도)까지 확인한다 - 두 시스템이 같은
    개념을 서로 다르게 판단해 그래프 뷰와 노드 파일이 어긋나는 문제를 막기 위해
    같은 판정 함수를 공유한다."""
    new_keys = _identity_keys(label, aliases)
    node_keys = _identity_keys(node["display_label"], node.get("aliases") or [])
    if new_keys & node_keys:
        return True

    new_names = _names(label, aliases)
    node_names = _names(node["display_label"], node.get("aliases") or [])
    return any(labels_match(a, b) for a in new_names for b in node_names)


def _read_frontmatter(path: Path) -> dict:
    match = _FRONTMATTER_RE.match(path.read_text(encoding="utf-8"))
    if not match:
        return {}
    try:
        return yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        return {}


def _extract_user_section(path: Path) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8")
    idx = text.find(_USER_MARKER)
    if idx == -1:
        return ""
    return text[idx + len(_USER_MARKER) :].lstrip("\n")


def _existing_nodes(store_root: str, node_type: str) -> list[dict]:
    nodes = []
    for path in sorted(_node_dir(store_root, node_type).glob("*.md")):
        frontmatter = _read_frontmatter(path)
        # redirect_to가 있으면 병합되어 사라진 노드의 스텁이다 - 더 이상 독립된
        # 노드가 아니므로 새 concept/entity와의 매칭 대상에서 제외한다.
        if not frontmatter.get("slug") or frontmatter.get("redirect_to"):
            continue
        nodes.append({**frontmatter, "path": path})
    return nodes


def get_display_label(store_root: str, node_type: str, slug: str) -> str:
    path = _node_dir(store_root, node_type) / f"{slug}.md"
    return _read_frontmatter(path).get("display_label", slug)


def get_user_section(store_root: str, node_type: str, slug: str) -> str:
    """사용자가 직접 쓴 메모(원본 마크다운, 자동 생성 영역 제외)만 반환한다.
    편집 UI가 textarea를 채울 때 쓴다."""
    path = _node_dir(store_root, node_type) / f"{slug}.md"
    return _extract_user_section(path)


def get_auto_section(store_root: str, node_type: str, slug: str) -> str:
    """자동 생성 영역(description + 등장 논문 목록)만 반환한다. get_user_section()과
    짝을 이뤄, 노드 화면이 이 부분은 읽기 전용으로 보여주고 user-notes 아래만
    편집 가능하게 나눠 렌더링할 수 있게 한다."""
    path = _node_dir(store_root, node_type) / f"{slug}.md"
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8")
    start = text.find(_AUTO_MARKER)
    end = text.find(_USER_MARKER)
    if start == -1 or end == -1:
        return ""
    return text[start + len(_AUTO_MARKER) : end].strip()


def refresh_auto_section(store_root: str, node_type: str, slug: str) -> bool:
    """이 노드 파일의 자동 생성 영역(제목/다른 표기/카테고리/등장 논문/AI 설명)을
    지금의 _write_node_file() 템플릿으로 다시 그려 넣는다. frontmatter와
    사용자가 쓴 메모는 그대로 두고 본문 레이아웃만 최신화한다 - 예전 형식으로
    이미 만들어진 노드 파일을 새 레이아웃으로 맞추는 1회성 마이그레이션
    스크립트가 쓴다. 병합돼 사라진 redirect 스텁은 건드리지 않는다(더 이상
    독립된 노드가 아니므로). 실제로 다시 썼으면 True, 파일이 없거나 redirect
    스텁이면 False."""
    path = _node_dir(store_root, node_type) / f"{slug}.md"
    frontmatter = _read_frontmatter(path)
    if not frontmatter.get("slug") or frontmatter.get("redirect_to"):
        return False
    user_section = _extract_user_section(path)
    _write_node_file(path, frontmatter, user_section)
    return True


def update_user_section(store_root: str, node_type: str, slug: str, user_markdown: str) -> None:
    """사용자가 편집한 메모를 저장한다. frontmatter/자동 생성 영역(등장 논문
    목록)은 건드리지 않고 user_section만 교체한다. 병합돼 사라진 redirect
    스텁에는 쓸 수 없다(더 이상 독립된 노드가 아니므로)."""
    path = _node_dir(store_root, node_type) / f"{slug}.md"
    frontmatter = _read_frontmatter(path)
    if not frontmatter.get("slug"):
        raise FileNotFoundError(f"노드 파일을 찾을 수 없습니다: {path}")
    if frontmatter.get("redirect_to"):
        raise ValueError(f"'{slug}'는 다른 노드로 병합되어 더 이상 독립된 노드가 아닙니다.")
    _write_node_file(path, frontmatter, user_markdown)


def save_attachment(store_root: str, node_type: str, slug: str, filename: str, content: bytes) -> str:
    """이미지 첨부파일을 저장하고, md 본문에서 참조할 상대경로를 반환한다
    (예: "attachments/concepts/self-attention/173..-ab12.png"). 업로드된
    파일명은 신뢰하지 않고(경로 조작 방지) 확장자만 취해 서버가 충돌 없는
    이름을 새로 붙인다. attachments/는 node_store.py의 다른 파일들과 같은 위치
    (NODE_STORE_ROOT)에 두고, app.py가 /attachments로 정적 서빙한다."""
    path = _node_dir(store_root, node_type) / f"{slug}.md"
    frontmatter = _read_frontmatter(path)
    if not frontmatter.get("slug"):
        raise FileNotFoundError(f"노드 파일을 찾을 수 없습니다: {path}")
    if frontmatter.get("redirect_to"):
        raise ValueError(f"'{slug}'는 다른 노드로 병합되어 더 이상 독립된 노드가 아닙니다.")

    ext = Path(filename).suffix.lower()
    if ext not in IMAGE_EXTENSIONS:
        raise ValueError(f"지원하지 않는 이미지 형식입니다: {ext or '(확장자 없음)'}")

    type_dir = _ATTACHMENTS_DIR_BY_TYPE[node_type]
    folder = Path(store_root) / "attachments" / type_dir / slug
    folder.mkdir(parents=True, exist_ok=True)

    generated_name = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}{ext}"
    (folder / generated_name).write_bytes(content)

    return f"attachments/{type_dir}/{slug}/{generated_name}"


def build_node_index(nodes: list[dict]) -> dict[str, dict]:
    """{정규화된 display_label 또는 alias: 노드} 역인덱스를 만든다. 완전일치/alias
    매칭을 노드 수와 무관하게 O(1)로 만들기 위함이다 - 인덱스 없이는 라벨 하나를
    찾을 때마다 노드 목록 전체를 처음부터 훑어야 해서(find_node_fuzzy의 기존
    선형 탐색), 노드가 수만 개로 늘어나면 그래프 하나 그릴 때 라벨마다 수만 번씩
    비교가 반복돼 느려진다. 같은 정규화 키를 여러 노드가 주장하는 경우(드묾)는
    먼저 등록된 쪽을 우선한다 - 결과가 매번 달라지지 않게."""
    index: dict[str, dict] = {}
    for node in nodes:
        for name in _names(node["display_label"], node.get("aliases") or []):
            index.setdefault(normalize_label(name), node)
    return index


# store_root/node_type별로 (디렉터리 서명, 파싱된 노드 목록, 역인덱스)를 프로세스
# 메모리에 캐싱한다. 파일 내용까지는 캐시하지 않고 "이 디렉터리가 지난번과
# 똑같은지"만 저렴하게(os.scandir, 내용은 안 읽음) 확인해서 무효화 여부를 판단한다.
_LIST_NODES_CACHE: dict[tuple[str, str], tuple[tuple, list[dict], dict[str, dict]]] = {}


def _dir_signature(folder: Path) -> tuple:
    """폴더 안 .md 파일들의 (이름, 수정시각) 목록. 내용을 읽지 않고 메타데이터만
    보므로 전체 파싱보다 훨씬 저렴하다 - util/review_merge_candidates.py처럼 다른
    프로세스가 파일을 바꿔도, 다음 list_nodes() 호출에서 mtime 차이로 정확히
    감지된다."""
    try:
        return tuple(
            sorted((entry.name, entry.stat().st_mtime_ns) for entry in os.scandir(folder) if entry.name.endswith(".md"))
        )
    except FileNotFoundError:
        return ()


def _cached_nodes_and_index(store_root: str, node_type: str) -> tuple[list[dict], dict[str, dict]]:
    folder = _node_dir(store_root, node_type)
    key = (store_root, node_type)
    signature = _dir_signature(folder)

    cached = _LIST_NODES_CACHE.get(key)
    if cached is not None and cached[0] == signature:
        return cached[1], cached[2]

    result = _existing_nodes(store_root, node_type)
    index = build_node_index(result)
    _LIST_NODES_CACHE[key] = (signature, result, index)
    return result, index


def list_nodes(store_root: str, node_type: str) -> list[dict]:
    """존재하는 모든 노드 파일의 frontmatter 목록을 반환한다(병합돼 사라진
    redirect 스텁은 제외). 그래프 뷰처럼 여러 label을 한꺼번에 조회해야 할 때는
    이 목록을 한 번만 불러와 find_node_fuzzy()에 재사용해야 한다 - label 하나마다
    폴더를 다시 스캔하면 노드 수만큼 스캔이 반복돼 요청이 느려진다.

    디렉터리 내용이 지난 호출과 똑같으면(_dir_signature 비교) 프로세스 메모리에
    캐싱된 결과를 그대로 반환하고, 파일이 추가/삭제/수정됐을 때만 실제로 다시
    읽고 파싱한다."""
    return _cached_nodes_and_index(store_root, node_type)[0]


def node_index(store_root: str, node_type: str) -> dict[str, dict]:
    """list_nodes()와 같은 캐시를 공유하는 역인덱스(build_node_index 결과)를
    반환한다. 여러 라벨을 조회해야 하는 호출부는 이걸 한 번만 받아서
    find_node_fuzzy(..., index=...)에 넘겨 재사용해야 한다 - 매 라벨마다
    새로 만들면 인덱스를 쓰는 의미가 없어진다."""
    return _cached_nodes_and_index(store_root, node_type)[1]


def find_node_fuzzy(
    nodes: list[dict], label: str, aliases: list[str] | None = None, index: dict[str, dict] | None = None
) -> dict | None:
    """list_nodes()로 미리 불러온 노드 목록에서 label/aliases와 매칭되는(정확
    일치, alias, 또는 MinHash 퍼지 매칭) 노드를 찾아 그 frontmatter 전체를
    반환한다. 매칭이 없으면 None.

    index(node_index() 결과)가 주어지면 정규화 완전일치/alias는 먼저 O(1) 사전
    조회로 확인한다 - 대부분의 재등장 라벨은 여기서 바로 끝나고, 인덱스에 없는
    (완전히 새로운 표기의) 라벨만 아래의 기존 선형 탐색 + MinHash 퍼지 매칭으로
    넘어간다. 노드 수가 많아져도 흔한 경우의 비용이 늘어나지 않게 하기 위함."""
    if index is not None:
        for name in _names(label, aliases or []):
            hit = index.get(normalize_label(name))
            if hit is not None:
                return hit

    for node in nodes:
        if _matches_node(label, aliases or [], node):
            return node
    return None


def find_node_slug_fuzzy(
    nodes: list[dict], label: str, aliases: list[str] | None = None, index: dict[str, dict] | None = None
) -> str | None:
    """find_node_fuzzy()와 같지만 slug만 필요한 호출부를 위한 편의 함수."""
    node = find_node_fuzzy(nodes, label, aliases, index)
    return node["slug"] if node else None


def _write_node_file(path: Path, frontmatter: dict, user_section: str) -> None:
    frontmatter_yaml = yaml.safe_dump(frontmatter, allow_unicode=True, sort_keys=False).strip()

    # 본문(자동 생성 영역)은 제목 -> 다른 표기 -> 카테고리(concept만) -> 등장 논문
    # -> (있으면) --- 구분선 + AI 설명 순으로 고정된 순서로 쓴다. Obsidian에서
    # frontmatter Properties 패널을 열지 않고 파일만 봐도 핵심 정보가 위에서부터
    # 순서대로 읽히게 하려는 의도다 - static/graph.js의 openNodeView()는 이
    # 순서를 알고 제목/다른 표기/카테고리/등장 논문 줄을 걷어내고 실제 논문 목록
    # (클릭 가능한 링크)부터 보여준다 - 제목/다른 표기/카테고리/등장 논문 수는
    # 이미 위쪽 메타 행(.node-view-aliases 등, 통일된 색상)으로 따로 보여주므로.
    aliases = frontmatter.get("aliases") or []
    auto_lines = [f"# {frontmatter['display_label']}", "", f"**다른 표기**: {', '.join(aliases) if aliases else '없음'}"]
    if frontmatter.get("type") == "concept":
        categories = frontmatter.get("categories") or []
        auto_lines += ["", f"**카테고리**: {', '.join(categories) if categories else '없음'}"]
    # entity의 sources 항목은 논문 없이(slug/title 없이) concept_slug만 있을 수도
    # 있다(orphan concept에 orphan entity를 직접 붙인 경우) - "등장 논문" 개수는
    # 그 항목을 빼고 진짜 논문만 센다(라벨 그대로의 뜻을 지키기 위해), 목록
    # 줄에는 위키링크 대신 문구로 표시한다.
    paper_sources = [s for s in frontmatter["sources"] if s.get("slug")]
    auto_lines += ["", f"**등장 논문** {len(paper_sources)}편"]
    for src in frontmatter["sources"]:
        if src.get("slug"):
            auto_lines.append(f"- [[{src['slug']}|{src['title']}]]")
        else:
            auto_lines.append("- (논문 없음 - concept에 직접 연결됨)")

    # description/note는 논문 노트의 레퍼런스 표에 있던 설명을 그대로 복사해온 것 -
    # 최초 생성 시에만 채워지므로 없을 수도 있다(그 이전에 만들어진 노드, 혹은 값을
    # 안 넘긴 호출부). 있을 때만 구분선과 함께 덧붙인다.
    description_lines = []
    if frontmatter.get("description"):
        description_lines.append(frontmatter["description"])
    if frontmatter.get("note"):
        description_lines.append(f"*{frontmatter['note']}*")
    if description_lines:
        auto_lines += ["", "---", ""] + description_lines
    auto_section = "\n".join(auto_lines)

    content = (
        f"---\n{frontmatter_yaml}\n---\n\n"
        f"{_AUTO_MARKER}\n\n{auto_section}\n\n"
        f"---\n\n{_USER_MARKER}\n{user_section}"
    )
    path.write_text(content, encoding="utf-8")


def _candidates_path(store_root: str) -> Path:
    folder = Path(store_root) / "config"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / "_merge_candidates.json"


def _load_candidates(store_root: str) -> list[dict]:
    path = _candidates_path(store_root)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []


def _save_candidates(store_root: str, candidates: list[dict]) -> None:
    _candidates_path(store_root).write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _find_candidate(candidates: list[dict], node_type: str, slug_a: str, slug_b: str) -> dict | None:
    key = tuple(sorted([slug_a, slug_b]))
    for c in candidates:
        if c["type"] == node_type and tuple(sorted([c["slug_a"], c["slug_b"]])) == key:
            return c
    return None


def _record_merge_candidate(
    store_root: str, node_type: str, primary_slug: str, other_slug: str, via_label: str, detected_in: str
) -> None:
    candidates = _load_candidates(store_root)
    # 상태(pending/rejected/merged) 상관없이 이미 기록된 적 있으면 다시 추가하지
    # 않는다 - 그렇지 않으면 사람이 거부한 후보가 새 논문이 들어올 때마다 계속
    # 재등장하게 된다.
    if _find_candidate(candidates, node_type, primary_slug, other_slug):
        return

    candidates.append(
        {
            "type": node_type,
            "slug_a": primary_slug,
            "slug_b": other_slug,
            "via_alias": via_label,
            "detected_in_paper": detected_in,
            "detected_at": _now_iso(),
            "status": "pending",
        }
    )
    _save_candidates(store_root, candidates)


def list_merge_candidates(store_root: str, status: str = "pending") -> list[dict]:
    return [c for c in _load_candidates(store_root) if c.get("status", "pending") == status]


def reject_merge_candidate(store_root: str, node_type: str, slug_a: str, slug_b: str) -> None:
    candidates = _load_candidates(store_root)
    candidate = _find_candidate(candidates, node_type, slug_a, slug_b)
    if candidate is None:
        return
    candidate["status"] = "rejected"
    candidate["rejected_at"] = _now_iso()
    _save_candidates(store_root, candidates)


def _write_redirect_stub(path: Path, slug: str, redirect_to: str) -> None:
    frontmatter = {"slug": slug, "redirect_to": redirect_to, "merged_at": _now_iso()}
    frontmatter_yaml = yaml.safe_dump(frontmatter, allow_unicode=True, sort_keys=False).strip()
    content = f"---\n{frontmatter_yaml}\n---\n\n이 노드는 [[{redirect_to}]]로 병합되었습니다.\n"
    path.write_text(content, encoding="utf-8")


def execute_merge(store_root: str, node_type: str, slug_a: str, slug_b: str) -> str:
    """slug_a/slug_b 두 노드 파일을 하나로 합친다. 더 먼저 생성된 쪽이 생존자가
    되고, 나중 생성된 쪽은 내용이 흡수된 뒤 redirect 스텁으로 축소된다. 사람이
    검토·승인한 뒤에만 호출돼야 한다(util/review_merge_candidates.py). 생존자 slug를
    반환한다."""
    node_dir = _node_dir(store_root, node_type)
    path_a, path_b = node_dir / f"{slug_a}.md", node_dir / f"{slug_b}.md"
    fm_a, fm_b = _read_frontmatter(path_a), _read_frontmatter(path_b)

    if fm_a.get("created_at", "") <= fm_b.get("created_at", ""):
        survivor_path, survivor_fm, loser_path, loser_fm = path_a, fm_a, path_b, fm_b
    else:
        survivor_path, survivor_fm, loser_path, loser_fm = path_b, fm_b, path_a, fm_a

    merged_aliases = set(survivor_fm.get("aliases") or [])
    merged_aliases.add(loser_fm["display_label"])
    merged_aliases.update(loser_fm.get("aliases") or [])
    survivor_fm["aliases"] = sorted(merged_aliases)

    survivor_sources = survivor_fm.get("sources") or []
    existing_source_slugs = {s["slug"] for s in survivor_sources}
    for s in loser_fm.get("sources") or []:
        if s["slug"] not in existing_source_slugs:
            survivor_sources.append(s)
    survivor_fm["sources"] = survivor_sources

    merged_from = survivor_fm.get("merged_from") or []
    merged_from.append(loser_fm["slug"])
    merged_from.extend(loser_fm.get("merged_from") or [])
    survivor_fm["merged_from"] = merged_from

    survivor_user_section = _extract_user_section(survivor_path)
    loser_user_section = _extract_user_section(loser_path)
    if loser_user_section.strip():
        divider = (
            f"\n\n---\n*(아래는 {loser_fm['slug']}.md에서 병합된 메모입니다, "
            f"{_now_iso()[:10]})*\n\n"
        )
        survivor_user_section = survivor_user_section + divider + loser_user_section

    _write_node_file(survivor_path, survivor_fm, survivor_user_section)
    _write_redirect_stub(loser_path, loser_fm["slug"], survivor_fm["slug"])

    candidates = _load_candidates(store_root)
    candidate = _find_candidate(candidates, node_type, slug_a, slug_b)
    if candidate is not None:
        candidate["status"] = "merged"
        candidate["merged_at"] = _now_iso()
        _save_candidates(store_root, candidates)

    return survivor_fm["slug"]


def resolve_or_create_node(
    store_root: str,
    node_type: str,
    label: str,
    aliases: list[str],
    source_slug: str,
    source_title: str,
    category: str | None = None,
    description: str = "",
    note: str = "",
    concept_slug: str | None = None,
) -> str:
    """label/aliases에 해당하는 concept/entity 노드 파일을 찾아 갱신하거나 새로
    만들고, 최종 slug를 반환한다.

    기존 노드가 둘 이상 걸리면(새 alias가 서로 다른 두 기존 노드를 잇는 경우,
    혹은 MinHash 퍼지 매칭으로 두 노드가 동시에 걸리는 경우) 파일을 바로 합치지
    않는다 - 가장 먼저 생성된 노드를 대표(primary)로 삼아 갱신하고, 나머지는
    _merge_candidates.json에 병합 후보로만 기록한다.

    description/note는 논문 요약 파이프라인이 이미 만들어둔 텍스트(레퍼런스 표용
    설명/비고)를 그대로 복사해오는 것이라 별도 API 호출이 필요 없다. display_label과
    같은 이유로 최초 생성 시에만 채워지고 이후 갱신에서는 바뀌지 않는다(어느 논문의
    설명이 "더 낫다"를 판단할 기준이 없음).

    concept_slug는 entity 타입일 때만 의미 있다 - 이 논문에서 이 entity가 어느
    concept 밑에 묶이는지(같은 entity라도 논문마다 다를 수 있음)를, entity 자신의
    sources 항목 하나하나에 기록한다. 호출부(app.py의 run_pipeline)가 concept을
    먼저 처리해서 얻은 최종 slug를 넘겨줘야 한다 - 그래야 이 논문에서 Claude가 쓴
    원본 concept 라벨이 아니라, 실제로 확정된 노드를 가리키게 된다.
    """
    existing = _existing_nodes(store_root, node_type)
    matches = [n for n in existing if _matches_node(label, aliases, n)]

    if not matches:
        return _create_node(
            store_root, node_type, label, aliases, source_slug, source_title, category, description, note,
            concept_slug=concept_slug,
        )

    matches.sort(key=lambda n: n.get("created_at", ""))
    primary = matches[0]
    _update_node(primary["path"], label, aliases, source_slug, source_title, concept_slug=concept_slug)

    for other in matches[1:]:
        _record_merge_candidate(store_root, node_type, primary["slug"], other["slug"], label, source_slug)

    return primary["slug"]


def create_node_manual(
    store_root: str,
    node_type: str,
    label: str,
    source_slug: str | None,
    source_title: str | None,
    category: str | None = None,
    anchor_id: str | None = None,
    force: bool = False,
    concept_slug: str | None = None,
) -> str:
    """사용자가 그래프 화면에서 직접 만드는 concept/entity 노드.

    slug(label을 정규화한 것)가 기존 파일과 완전히 같으면 force와 무관하게 항상
    막는다 - 그건 "비슷한 개념" 수준이 아니라 "같은 파일"이라, 그대로 진행하면
    기존 파일의 frontmatter와 사용자가 이미 남긴 개인 메모까지 통째로 사라진다.

    force=False(기본값)면 정확 일치 다음으로, resolve_or_create_node()가 쓰는 것과
    같은 기준(정규화 완전일치·alias·MinHash 유사도, find_node_fuzzy)으로 비슷한
    기존 노드가 있는지도 확인해 있으면 DuplicateNodeError를 던진다 - 사용자가 이미
    있는 개념/용어를 모르고 다시 만드는 걸 막기 위함(직접 만들 때는 resolve_or_create_node()
    와 달리 자동 병합은 하지 않는다 - 사용자 의도와 다르게 엉뚱한 노드에 조용히
    합쳐지면 안 되므로, 호출부가 사용자에게 확인받은 뒤 force=True로 다시 부르는
    구조). force=True면 이 퍼지 매칭 확인을 건너뛰고 새로 만든다("그래도 새로
    만들기"를 사용자가 명시적으로 선택한 경우).

    source_slug/source_title을 None으로 두면 sources가 빈 orphan 노드가 만들어진다 -
    그래프 배경 우클릭으로 만드는 노드가 이 경로를 쓴다(어느 논문과 연결할지는 나중에
    드래그로 정한다). carrier가 있는 기존 호출부(진입점 1/3)는 그대로 값을 넘긴다.

    anchor_id는 orphan 생성 시 그래프에서 가장 가까웠던 다른 노드의 id(그래프 표시용,
    예: "concept:Foo" 또는 논문 slug)를 그대로 저장해둔다 - 브라우저 메모리에만 있는
    화면 좌표 캐시는 새로고침/서버 재시작으로 사라지므로, "이 노드는 원래 저 노드
    근처에 있었다"는 최소한의 힌트를 파일에 남겨서 다음 세션에도 그 근처에 다시
    나타나게 한다(graph_builder.py가 읽어서 그래프 응답에 실어준다).

    concept_slug는 entity를 진입점 3(concept 뷰의 "+ 엔티티 추가")로 만들 때만
    쓴다 - carrier 논문에서 이 entity가 바로 그 concept 밑에 묶인다는 걸 기록한다."""
    slug = _slugify(label)
    path = _node_dir(store_root, node_type) / f"{slug}.md"
    if path.is_file():
        raise FileExistsError(f"'{label}'과(와) 이름이 같은 노드가 이미 있습니다.")

    if not force:
        nodes = list_nodes(store_root, node_type)
        idx = node_index(store_root, node_type)
        match = find_node_fuzzy(nodes, label, [], idx)
        if match:
            raise DuplicateNodeError(match)

    return _create_node(
        store_root, node_type, label, [], source_slug, source_title, category,
        anchor_id=anchor_id, concept_slug=concept_slug,
    )


def delete_node(store_root: str, node_type: str, slug: str) -> dict:
    """concept/entity 노드 파일과 그 첨부 이미지 폴더를 삭제하고, 삭제 전
    frontmatter를 그대로 반환한다. 호출부(app.py)가 이 frontmatter의
    display_label/aliases/sources로 - 이 노드를 참조하던 논문들의 frontmatter에서도
    참조를 걷어내야 하므로(안 그러면 그래프에 실제 파일 없는 "죽은" 라벨만 남음) -
    반환값이 필요하다."""
    path = _node_dir(store_root, node_type) / f"{slug}.md"
    if not path.is_file():
        raise FileNotFoundError(f"노드 파일을 찾을 수 없습니다: {slug}")
    frontmatter = _read_frontmatter(path)
    path.unlink()

    type_dir = _ATTACHMENTS_DIR_BY_TYPE.get(node_type)
    if type_dir:
        attachments_dir = Path(store_root) / "attachments" / type_dir / slug
        if attachments_dir.is_dir():
            shutil.rmtree(attachments_dir)

    return frontmatter


def remove_source(store_root: str, source_slug: str) -> list[tuple[str, str, bool]]:
    """source_slug(논문 slug)를 모든 concept/entity 노드 파일의 sources[]에서 제거한다.

    두 곳에서 쓰인다: (1) 논문 삭제 시 - vault/Supabase에서 노트를 지워도 지금까진
    node_store 참조가 죽은 채로 영구히 남았다. (2) 같은 논문을 다시 처리해 덮어쓸 때 -
    옛 처리 결과의 참조를 먼저 걷어내야, 이번엔 안 나온 concept/entity가 옛 흔적으로
    계속 남지 않는다.

    제거 후 sources가 비면(다른 논문도 이 노드를 참조하지 않으면) 파일 자체를 삭제한다 -
    아무 논문도 안 쓰는 개념/용어를 굳이 남겨둘 이유가 없다. sources가 남아있으면
    _write_node_file()로 다시 써서 "## 등장 논문" 자동 섹션도 같이 갱신하고, 사용자가
    남긴 개인 메모(user_section)는 그대로 보존한다.

    영향받은 (node_type, slug, 파일이 삭제됐는지) 목록을 반환한다 - 호출부(app.py)가
    이 논문 하나 때문에 시스템 전체에서 몇 개나 어떤 노드가 바뀌었는지 미리 알 방법이
    없어서, Neo4j 미러 동기화(삭제 or 갱신)를 어느 노드에 해야 할지 알려주는 용도다."""
    affected: list[tuple[str, str, bool]] = []
    for node_type, dir_name in _DIR_BY_TYPE.items():
        folder = Path(store_root) / dir_name
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.md")):
            frontmatter = _read_frontmatter(path)
            sources = frontmatter.get("sources") or []
            if not any(s.get("slug") == source_slug for s in sources):
                continue
            frontmatter["sources"] = [s for s in sources if s.get("slug") != source_slug]
            slug = frontmatter["slug"]
            if frontmatter["sources"]:
                user_section = _extract_user_section(path)
                _write_node_file(path, frontmatter, user_section)
                affected.append((node_type, slug, False))
            else:
                path.unlink()
                affected.append((node_type, slug, True))
    return affected


def find_node_slugs_by_paper(store_root: str, paper_slug: str) -> list[tuple[str, str]]:
    """paper_slug를 sources[]에 담고 있는 모든 concept/entity를 (node_type, slug)로
    나열한다. remove_source()와 같은 순회 방식이지만 읽기 전용이다 - 이 논문이
    다른 Brain으로 옮겨졌을 때, brain_ids를 다시 계산해서 Neo4j에 반영해야 할
    concept/entity가 어느 것들인지 app.py가 알아내는 용도(graph_db.sync_node를
    다시 불러야 할 대상 목록)."""
    found: list[tuple[str, str]] = []
    for node_type, dir_name in _DIR_BY_TYPE.items():
        folder = Path(store_root) / dir_name
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.md")):
            frontmatter = _read_frontmatter(path)
            sources = frontmatter.get("sources") or []
            if any(s.get("slug") == paper_slug for s in sources):
                found.append((node_type, frontmatter["slug"]))
    return found


def find_entities_by_concept(store_root: str, concept_slug: str) -> list[dict]:
    """이 concept 밑에 걸린 entity 노드들을 찾는다 - entity 각각의 sources 항목
    중 concept_slug가 일치하는 게 하나라도 있으면 포함시킨다(entity는 논문마다
    다른 concept 밑에 묶일 수 있으므로, sources 항목 단위로 확인해야 한다).
    concept 삭제 시 "딸린 entity도 같이 지울지" 물어보는 화면(app.py)이 이 함수로
    무엇이 딸려있는지 미리 확인한다. 예전에는 논문 frontmatter를 훑어 라벨을
    퍼지 매칭해야 했지만, 이제 entity 자신의 sources에 concept_slug가 그대로
    적혀 있어 직접 비교만 하면 된다."""
    return [
        n for n in list_nodes(store_root, "entity")
        if any(s.get("concept_slug") == concept_slug for s in (n.get("sources") or []))
    ]


def set_source_concept_slug(store_root: str, entity_slug: str, paper_slug: str, concept_slug: str) -> bool:
    """이미 있는 entity 노드 파일의 sources 중 특정 논문 항목 하나에 concept_slug를
    채워 넣는다. migrate_concept_slugs.py(1회성 마이그레이션 - 예전에는 논문
    frontmatter에만 있던 entity-concept 그룹핑을 entity 자신의 sources로 옮김)가
    쓴다. 실제로 값이 바뀌었으면 True, 항목을 못 찾았거나 이미 같은 값이면
    False를 반환한다."""
    path = _node_dir(store_root, "entity") / f"{entity_slug}.md"
    if not path.is_file():
        return False
    frontmatter = _read_frontmatter(path)
    sources = frontmatter.get("sources") or []
    changed = False
    for s in sources:
        if s.get("slug") == paper_slug and s.get("concept_slug") != concept_slug:
            s["concept_slug"] = concept_slug
            changed = True
    if not changed:
        return False
    frontmatter["sources"] = sources
    user_section = _extract_user_section(path)
    _write_node_file(path, frontmatter, user_section)
    return True


def migrate_category_field(store_root: str) -> list[str]:
    """concept 노드의 예전 category(단수, str|None) 필드를 categories(리스트)로
    옮긴다. migrate_concept_categories.py(1회성 마이그레이션)가 쓴다.

    예전 값을 새 CONCEPT_CATEGORIES 어휘로 재해석하지 않고 그대로 리스트에 담아
    보존한다 - 예전 5개 카테고리(input/process/result/limitation/other)와 지금
    12개 카테고리는 이름 체계가 달라서(예: "process" 하나가 지금은 architecture/
    algorithm/theory/proposed_method/training_strategy 여러 개로 나뉨) 자동으로
    정확히 재분류할 근거가 없다 - 그 논문을 다시 봐야 판단할 수 있는 일이라
    사용자가 add_category()/remove_category()로 나중에 직접 정리하도록 남겨둔다.
    이미 categories 필드로 이관된 노드는 건드리지 않아(idempotent), 여러 번
    실행해도 안전하다. 이관된 concept slug 목록을 반환한다."""
    migrated: list[str] = []
    for path in sorted(_node_dir(store_root, "concept").glob("*.md")):
        frontmatter = _read_frontmatter(path)
        if not frontmatter.get("slug") or "categories" in frontmatter:
            continue
        category = frontmatter.pop("category", None)
        frontmatter["categories"] = [category] if category else []
        user_section = _extract_user_section(path)
        _write_node_file(path, frontmatter, user_section)
        migrated.append(frontmatter["slug"])
    return migrated


def _create_node(
    store_root: str,
    node_type: str,
    label: str,
    aliases: list[str],
    source_slug: str | None,
    source_title: str | None,
    category: str | None,
    description: str = "",
    note: str = "",
    anchor_id: str | None = None,
    concept_slug: str | None = None,
) -> str:
    slug = _slugify(label)
    path = _node_dir(store_root, node_type) / f"{slug}.md"
    frontmatter = {
        "slug": slug,
        "type": node_type,
        "display_label": label,
        "aliases": aliases,
        "description": description,
        "note": note,
        "sources": (
            [{"slug": source_slug, "title": source_title, "concept_slug": concept_slug}] if source_slug else []
        ),
        "anchor_id": anchor_id,
        "created_at": _now_iso(),
    }
    if node_type == "concept":
        # LLM(또는 수동 생성 시 사용자)이 고른 초기 카테고리 하나로 시작하되,
        # 필드 자체는 리스트다 - 한 concept이 여러 관점(예: proposed_method이면서
        # architecture)에 걸칠 수 있고, 사용자가 add_category()/remove_category()로
        # 나중에 자유롭게 보정할 수 있어야 하기 때문이다.
        frontmatter["categories"] = [category] if category else []
    _write_node_file(path, frontmatter, user_section="")
    return slug


def link_node_to_paper(
    store_root: str, node_type: str, node_slug: str, paper_slug: str | None, paper_title: str | None,
    concept_slug: str | None = None,
) -> None:
    """이미 존재하는(orphan이든 아니든) concept/entity 노드 파일의 sources[]에 논문을
    연결한다. _update_node()가 하는 일 중 sources 갱신 부분만 떼어낸 것과 같다 -
    라벨/alias 재매칭은 하지 않는다(호출부가 이미 정확히 어느 노드인지 slug로 알고
    있으므로 다시 퍼지 매칭할 이유가 없다). 그래프에서 노드를 다른 노드로 드래그해
    연결하는 제스처가 이 함수를 쓴다.

    concept_slug는 entity를 concept으로 드래그해 연결하는 경우에만 쓴다 - 이
    논문에서는 그 concept 밑에 묶인다는 걸 sources 항목에 기록한다.

    paper_slug/paper_title은 None일 수 있다 - orphan concept(아직 어떤 논문과도
    연결 안 됨)에 orphan entity를 드래그해 붙이는 경우, 근거로 삼을 논문 자체가
    없기 때문이다. 이때는 sources에 slug/title 없이 concept_slug만 있는 항목이
    생긴다("논문과 무관하게 이 concept에 속한다"는 뜻) - graph_builder.py는
    concept_slug만 유효하면 논문 유무와 무관하게 concept->entity 에지를 만들므로
    이 항목만으로도 그래프에 정상적으로 나타난다. 같은 entity를 서로 다른(둘 다
    paper_slug 없는) concept에 여러 번 연결해도 각각 별개의 항목으로 쌓인다
    (concept_slug가 다르면 다른 관계이므로) - 한 entity가 논문 없는 concept
    여러 개에 동시에 속할 수 있다. 같은 concept에 다시 연결하는 것만 기존
    항목을 그대로 재사용한다(중복 방지).

    호출부(app.py)가 node_type이 entity이고 concept_slug가 있을 때만
    paper_slug=None을 허용한다 - concept 자신은 논문 없이 다른 무언가에
    "연결"될 방법이 없다(concept_slug 같은 자기 필드가 없음).

    이 논문이 sources에 이미 있어도(entity가 이 논문에 concept 없이 직접
    연결돼 있던 경우 등) 그 항목의 concept_slug를 이번 호출값으로 덮어쓴다 -
    "이미 있으니 그냥 둔다"로 처리하면, LLM이 같은 논문에서 뽑은 entity/concept을
    사용자가 나중에 드래그로 이어주려 해도(entity->note, concept->note를
    entity->concept->note로 바꾸는, 실제로 자주 나올 케이스) 아무 반응이 없는
    것처럼 보이는 채로 조용히 무시되는 버그가 있었다. paper_slug가 실제
    논문이면(None이 아니면) 이렇게 "그 논문 슬롯 하나"만 보고 판단하는 게 맞다
    - 같은 논문 안에서 이 entity는 어차피 concept 하나에만 묶일 수 있으므로.
    paper_slug가 None일 때는 이 기준을 그대로 쓰면 안 된다(위 문단 참고 - 슬롯이
    여러 개 있어야 함) - 그래서 두 경우를 아래에서 다르게 찾는다."""
    path = _node_dir(store_root, node_type) / f"{node_slug}.md"
    if not path.is_file():
        raise FileNotFoundError(f"노드 파일을 찾을 수 없습니다: {node_slug}")
    frontmatter = _read_frontmatter(path)
    sources = frontmatter.get("sources") or []
    if paper_slug is None:
        existing = next(
            (s for s in sources if s.get("slug") is None and s.get("concept_slug") == concept_slug), None
        )
    else:
        existing = next((s for s in sources if s.get("slug") == paper_slug), None)
    if existing is None:
        sources.append({"slug": paper_slug, "title": paper_title, "concept_slug": concept_slug})
    else:
        existing["concept_slug"] = concept_slug
    frontmatter["sources"] = sources
    # 실제로 논문에 연결됐으니 이제 orphan이 아니다 - 그래프가 더 이상 anchor_id를
    # 안 쓰긴 하지만(sources가 있으면 무시함), frontmatter에 죽은 값으로 계속
    # 남겨두지 않기 위해 지운다.
    frontmatter["anchor_id"] = None
    user_section = _extract_user_section(path)
    _write_node_file(path, frontmatter, user_section)

    # concept이 (논문 없이 붙어 있던 orphan 상태에서) 방금 진짜 논문에 연결됐으면,
    # 그 concept에 "논문 없이" 붙어 있던 entity들도 같은 논문을 근거로 갖게
    # 해준다 - 안 그러면 entity->concept 관계는 있는데 그 관계가 어느 논문에서
    # 나온 건지는 영영 알 수 없는 채로 남는다.
    if node_type == "concept" and paper_slug is not None:
        _propagate_paper_to_paperless_entities(store_root, node_slug, paper_slug, paper_title)


def _propagate_paper_to_paperless_entities(
    store_root: str, concept_slug: str, paper_slug: str, paper_title: str
) -> None:
    """concept이 논문에 연결될 때, 그 concept에 concept_slug로만(논문 없이)
    묶여 있던 entity들의 해당 source 항목에 이번 논문을 채워 넣는다 -
    link_node_to_paper()가 concept 타입 호출 끝에서 쓴다. entity 쪽이 이미
    이 논문을 다른 source로 갖고 있으면 건드리지 않는다(그 항목은 별개의
    실제 등장 기록)."""
    candidates = [
        n["slug"] for n in list_nodes(store_root, "entity")
        if any(s.get("concept_slug") == concept_slug and not s.get("slug") for s in (n.get("sources") or []))
    ]
    for entity_slug in candidates:
        path = _node_dir(store_root, "entity") / f"{entity_slug}.md"
        frontmatter = _read_frontmatter(path)
        sources = frontmatter.get("sources") or []
        changed = False
        for s in sources:
            if s.get("concept_slug") == concept_slug and not s.get("slug"):
                s["slug"] = paper_slug
                s["title"] = paper_title
                changed = True
        if changed:
            frontmatter["sources"] = sources
            user_section = _extract_user_section(path)
            _write_node_file(path, frontmatter, user_section)


def remove_source_from_node(store_root: str, node_type: str, node_slug: str, paper_slug: str) -> bool:
    """사용자가 그래프에서 note↔concept 또는 note↔entity(직접 연결) 에지를
    지울 때 쓴다 - 이 노드의 sources[]에서 그 논문 항목 하나만 제거한다.

    remove_source()(논문 삭제/재처리용, 모든 노드에서 그 논문을 지우고 sources가
    비면 파일 자체를 삭제)와 달리, 이건 사용자가 명시적으로 에지 하나만 끊는
    행위라 sources가 비어도 파일을 지우지 않는다 - orphan 노드(그래프 배경
    우클릭으로 만드는 것과 동일한 상태)로 남겨서, alias/description/사용자
    메모 같은 이 노드의 다른 데이터를 그대로 보존한다. 노드를 완전히 없애고
    싶으면 별도의 삭제 기능(delete_node)을 써야 한다.

    실제로 항목을 지웠으면 True, 그 논문 항목이 애초에 없었으면 False."""
    path = _node_dir(store_root, node_type) / f"{node_slug}.md"
    frontmatter = _read_frontmatter(path)
    if not frontmatter.get("slug"):
        raise FileNotFoundError(f"노드 파일을 찾을 수 없습니다: {node_slug}")

    sources = frontmatter.get("sources") or []
    remaining = [s for s in sources if s.get("slug") != paper_slug]
    if len(remaining) == len(sources):
        return False

    frontmatter["sources"] = remaining
    user_section = _extract_user_section(path)
    _write_node_file(path, frontmatter, user_section)

    # concept이 이 논문과의 연결을 잃었으면, 그 논문에서 이 concept 밑에 묶여
    # 있던 entity들의 그 등장 기록은 더 이상 "이 논문에서 등장했다"고 말할 수
    # 없다(concept 쪽엔 이미 그 논문 기록 자체가 없으니) - 논문 쪽(slug/title)만
    # 지우고 concept_slug는 그대로 남긴다(entity->concept 관계 자체는 논문과
    # 무관하게 유지 - concept_slug만으로도 그래프에 concept->entity 에지가
    # 그려지므로, 이 항목은 paperless 연결로 전환된다). _propagate_paper_to_
    # paperless_entities()의 정반대 방향이다.
    #
    # 왜 concept_slug를 지우지 않고 논문 쪽을 지우는지: concept↔entity 관계
    # (사용자가 명시적으로 만든 분류)가 note↔concept 관계(그 논문 하나의
    # 근거)보다 더 근본적이라고 본다 - 논문 근거 하나가 사라졌다고 사용자가
    # 만든 분류까지 같이 사라지면 안 된다.
    if node_type == "concept":
        _unlink_paper_from_concept_entities(store_root, node_slug, paper_slug)

    return True


def _unlink_paper_from_concept_entities(store_root: str, concept_slug: str, paper_slug: str) -> None:
    """concept이 특정 논문과의 연결을 잃었을 때, 그 논문에서 그 concept 밑에
    묶여 있던 entity들의 해당 source 항목을 정리한다 - remove_source_from_node()가
    concept 타입 호출 끝에서 쓴다.

    entity가 같은 concept_slug를 가리키는 **다른** source 항목을 이미 갖고
    있으면(concept이 다른 논문과는 여전히 연결돼 있어 그 논문 쪽 항목이 이미
    entity->concept 관계를 대표하고 있는 경우, 또는 이미 별도의 paperless
    항목이 있는 경우) 이번 항목은 그냥 통째로 지운다 - 안 그러면 같은
    concept_slug를 가리키는 항목이 중복으로 남는다(예: 논문B 항목은 그대로
    concept_slug=C인데, 논문A 항목도 paperless로 concept_slug=C를 또
    남기는 식).

    다른 항목이 전혀 없으면(이 항목이 이 entity와 이 concept을 잇는 유일한
    연결) 논문 쪽(slug/title)만 지우고 concept_slug는 남겨 paperless 연결로
    전환한다 - concept 자신이 orphan이 되더라도(더 이상 어떤 논문과도 연결
    안 됨) entity->concept 관계 자체는 유지하기 위해서다."""
    candidates = [
        n["slug"] for n in list_nodes(store_root, "entity")
        if any(
            s.get("concept_slug") == concept_slug and s.get("slug") == paper_slug
            for s in (n.get("sources") or [])
        )
    ]
    for entity_slug in candidates:
        path = _node_dir(store_root, "entity") / f"{entity_slug}.md"
        frontmatter = _read_frontmatter(path)
        sources = frontmatter.get("sources") or []
        has_other_link_to_concept = any(
            s.get("concept_slug") == concept_slug and s.get("slug") != paper_slug for s in sources
        )
        changed = False
        new_sources = []
        for s in sources:
            if s.get("concept_slug") == concept_slug and s.get("slug") == paper_slug:
                changed = True
                if has_other_link_to_concept:
                    continue  # 이미 다른 항목이 이 concept 관계를 대표하니 중복 안 만들고 버림
                s["slug"] = None
                s["title"] = None
            new_sources.append(s)
        if changed:
            frontmatter["sources"] = new_sources
            user_section = _extract_user_section(path)
            _write_node_file(path, frontmatter, user_section)


def unlink_concept_from_entity(store_root: str, entity_slug: str, concept_slug: str) -> bool:
    """사용자가 그래프에서 concept↔entity 에지를 지울 때 쓴다.

    entity 하나가 같은 concept 밑에 여러 논문에서 각각 묶일 수 있어서(두 논문을
    독립적으로 처리했는데 우연히 같은 concept으로 분류된 경우 - 흔히 있는 정상
    상황), sources[] 안에 concept_slug가 같은 항목이 여러 개 있을 수 있다.
    화면에는 concept→entity 에지가 한 줄로만 보이므로(d3가 중복 에지를
    구분하지 않음), 그 항목들 중 하나만 처리하면 나머지 때문에 에지가 그래프에
    그대로 남아 "지웠는데 안 지워진" 것처럼 보인다 - 그래서 concept_slug가
    일치하는 항목을 전부 찾아 한꺼번에 처리한다.

    각 항목은 논문 유무와 무관하게 통째로 삭제한다(entity→note 직접 연결로
    되돌리지 않음) - concept↔entity 관계를 끊는다는 건 이 entity가 "이 concept의
    맥락"에서 완전히 빠진다는 뜻이라, 그 맥락에서만 딸려온 논문 연결(특히
    "concept이 걸린 논문 전부에 연결"하는 기능으로 생긴 항목)까지 같이
    사라져야 두 관계가 깔끔하게 분리된다(예: 논문A-conceptB / entityE-conceptD-
    논문C 처럼). entity가 그 논문에 정말 직접 등장한다는 별도 근거가 있으면,
    그건 애초에 concept_slug 없는 별개의 source 항목으로 있어야 하는 사실이다.

    sources가 다 비어도 파일은 지우지 않는다(remove_source_from_node와 같은
    이유 - orphan으로 남겨 다른 데이터 보존). 실제로 뭔가 바뀌었으면 True."""
    path = _node_dir(store_root, "entity") / f"{entity_slug}.md"
    frontmatter = _read_frontmatter(path)
    if not frontmatter.get("slug"):
        raise FileNotFoundError(f"노드 파일을 찾을 수 없습니다: {entity_slug}")

    sources = frontmatter.get("sources") or []
    new_sources = [s for s in sources if s.get("concept_slug") != concept_slug]
    changed = len(new_sources) != len(sources)

    if not changed:
        return False

    frontmatter["sources"] = new_sources
    user_section = _extract_user_section(path)
    _write_node_file(path, frontmatter, user_section)
    return True


def add_alias(store_root: str, node_type: str, slug: str, alias: str) -> list[str]:
    """사용자가 노드 화면에서 직접 별칭을 추가한다. LLM이 판단한 별칭이 항상
    정확하다는 보장은 없으므로(놓친 표기가 있거나, 반대로 실제로는 다른 개념인데
    같다고 오판했을 수도 있음), 사용자가 직접 보완/수정할 수 있게 한다. 별칭을
    추가해두면 다음부터 그 표기로 들어오는 라벨도 find_node_fuzzy()의 O(1) 인덱스
    조회로 바로 이 노드에 연결된다.

    이미 다른 노드가 같은(정규화 기준) display_label/별칭을 쓰고 있으면 막는다 -
    그대로 추가하면 인덱스에서 이 표기의 "주인"이 둘이 되어(먼저 등록된 쪽이
    이기지만 어느 쪽이 먼저인지 사용자는 알 수 없음), Multi-head Latent
    Attention/MLA 사이에서 실제로 겪었던 것과 같은 혼란이 다시 생긴다. 두 노드가
    정말 같은 개념이면 별칭 추가가 아니라 병합(execute_merge)을 써야 한다.
    갱신된 aliases 목록을 반환한다."""
    alias = alias.strip()
    if not alias:
        raise ValueError("별칭을 입력하세요.")

    path = _node_dir(store_root, node_type) / f"{slug}.md"
    frontmatter = _read_frontmatter(path)
    if not frontmatter.get("slug"):
        raise FileNotFoundError(f"노드 파일을 찾을 수 없습니다: {slug}")

    key = normalize_label(alias)
    if normalize_label(frontmatter["display_label"]) == key:
        raise ValueError("이미 이 노드의 표시 이름과 같습니다.")

    owner = node_index(store_root, node_type).get(key)
    if owner is not None and owner["slug"] != slug:
        raise DuplicateNodeError(owner)

    aliases = frontmatter.get("aliases") or []
    if not any(normalize_label(a) == key for a in aliases):
        aliases.append(alias)
    frontmatter["aliases"] = aliases
    user_section = _extract_user_section(path)
    _write_node_file(path, frontmatter, user_section)
    return aliases


def rename_display_label(store_root: str, node_type: str, slug: str, new_label: str) -> dict:
    """사용자가 노드의 표시 이름(display_label)을 직접 바꾼다 - LLM이 맨 처음
    고른 표기가 항상 가장 적절한 건 아니다(예: "Mamba-2 Architecture"보다
    "Mamba-2"가 이 개념을 더 잘 설명하는 경우). slug(파일명, entity의
    concept_slug 같은 실제 참조 키)는 그대로 둔다 - 그래프 노드 ID는
    display_label로부터 매 렌더링마다 새로 계산되는 값이라 여기서 안 바꿔도
    다음 렌더부터 자동으로 새 이름을 쓰고, slug를 바꾸면 파일명·다른 노드의
    concept_slug 참조·첨부파일 경로·병합 이력(redirect_to)까지 전부 같이
    고쳐야 하는 훨씬 크고 위험한 작업이 된다 - 이름과 식별자를 분리해두면
    이런 연쇄 수정 없이 안전하게 이름만 바꿀 수 있다.

    예전 display_label은 alias로 자동 편입된다 - 다른 논문 본문에 이미 박혀있는
    위키링크([[예전 이름]])가 그대로 계속 풀리게 하기 위해서다. new_label이
    이미 이 노드 자신의 alias였다면(가장 흔한 경우 - "표시 이름으로 승격") 그
    자기 자신을 가리키는 중복 alias는 빼낸다. new_label을 이미 "다른" 노드가
    쓰고 있으면(정규화 기준) 막는다 - add_alias()와 같은 이유."""
    new_label = new_label.strip()
    if not new_label:
        raise ValueError("이름을 입력하세요.")

    path = _node_dir(store_root, node_type) / f"{slug}.md"
    frontmatter = _read_frontmatter(path)
    if not frontmatter.get("slug"):
        raise FileNotFoundError(f"노드 파일을 찾을 수 없습니다: {slug}")

    old_label = frontmatter["display_label"]
    key = normalize_label(new_label)
    if key == normalize_label(old_label):
        raise ValueError("지금 표시 이름과 같습니다.")

    owner = node_index(store_root, node_type).get(key)
    if owner is not None and owner["slug"] != slug:
        raise DuplicateNodeError(owner)

    aliases = [a for a in (frontmatter.get("aliases") or []) if normalize_label(a) != key]
    if not any(normalize_label(a) == normalize_label(old_label) for a in aliases):
        aliases.append(old_label)

    frontmatter["display_label"] = new_label
    frontmatter["aliases"] = aliases
    user_section = _extract_user_section(path)
    _write_node_file(path, frontmatter, user_section)
    return {"display_label": new_label, "aliases": aliases}


def remove_alias(store_root: str, node_type: str, slug: str, alias: str) -> list[str]:
    """사용자가 노드 화면에서 직접 별칭을 지운다 - LLM이 잘못 판단해서 붙인 별칭
    (실제로는 다른 개념인데 같다고 오판한 경우 등)을 사용자가 바로잡을 수 있게
    한다. 갱신된 aliases 목록을 반환한다."""
    path = _node_dir(store_root, node_type) / f"{slug}.md"
    frontmatter = _read_frontmatter(path)
    if not frontmatter.get("slug"):
        raise FileNotFoundError(f"노드 파일을 찾을 수 없습니다: {slug}")

    key = normalize_label(alias)
    aliases = [a for a in (frontmatter.get("aliases") or []) if normalize_label(a) != key]
    frontmatter["aliases"] = aliases
    user_section = _extract_user_section(path)
    _write_node_file(path, frontmatter, user_section)
    return aliases


def add_category(store_root: str, slug: str, category: str) -> list[str]:
    """사용자가 concept 화면에서 직접 카테고리를 추가한다. LLM이 매긴 카테고리가
    항상 사용자 마음에 들 리 없고(모호한 개념은 여러 관점에 동시에 걸칠 수도
    있음), concept 하나가 categories 리스트로 여러 값을 가질 수 있게 한다.
    CONCEPT_CATEGORIES에 없는 값은 막는다 - alias와 달리 category는 통제 어휘라,
    자유 텍스트를 허용하면 표기만 다른 카테고리가 계속 늘어나 필터링 의미가
    없어진다(예전 단일 category 필드가 실제로 이 문제를 겪었다). 갱신된
    categories 목록을 반환한다."""
    if category not in CONCEPT_CATEGORIES:
        raise ValueError(f"'{category}'는 올바른 카테고리가 아닙니다.")

    path = _node_dir(store_root, "concept") / f"{slug}.md"
    frontmatter = _read_frontmatter(path)
    if not frontmatter.get("slug"):
        raise FileNotFoundError(f"노드 파일을 찾을 수 없습니다: {slug}")

    categories = frontmatter.get("categories") or []
    if category not in categories:
        categories.append(category)
    frontmatter["categories"] = categories
    user_section = _extract_user_section(path)
    _write_node_file(path, frontmatter, user_section)
    return categories


def remove_category(store_root: str, slug: str, category: str) -> list[str]:
    """사용자가 concept 화면에서 직접 카테고리를 지운다 - LLM이 잘못 매긴
    카테고리를 바로잡을 수 있게 한다. 갱신된 categories 목록을 반환한다."""
    path = _node_dir(store_root, "concept") / f"{slug}.md"
    frontmatter = _read_frontmatter(path)
    if not frontmatter.get("slug"):
        raise FileNotFoundError(f"노드 파일을 찾을 수 없습니다: {slug}")

    categories = [c for c in (frontmatter.get("categories") or []) if c != category]
    frontmatter["categories"] = categories
    user_section = _extract_user_section(path)
    _write_node_file(path, frontmatter, user_section)
    return categories


def add_relation(
    store_root: str,
    node_type: str,
    slug: str,
    relation_type: str,
    target_type: str,
    target_slug: str,
    source_paper_slug: str,
    rationale: str = "",
) -> list[dict]:
    """slug 노드(from)에서 target_slug 노드(to)로 나가는 semantic 관계 하나를
    frontmatter의 relations[]에 반영한다(concept<->concept, concept<->entity,
    entity<->entity만 대상 - Paper는 여기 endpoint로 올 수 없다,
    docs/description/relation_types.md 참고). 같은 (relation_type, target_type,
    target_slug) 조합이 이미 있으면 새로 추가하지 않고 그 항목의 sources에
    source_paper_slug만 누적한다 - 여러 논문이 같은 관계를 반복 주장하면 에지
    하나에 근거가 쌓이는 것이지, 병렬 에지가 여러 개 생기는 게 아니다. 대칭
    타입(COMPARED_TO 등)의 from/to 정규화는 이 함수가 아니라 호출하는 쪽
    (app.py의 run_pipeline)이 slug 알파벳순으로 미리 정해서 넘겨야 한다 - 이
    함수는 "누가 from이고 누가 to인지"를 이미 확정된 입력으로 받는다.

    relation_type은 반드시 화이트리스트(load_relation_types)에 있는 값이어야
    한다 - graph_db.sync_node()가 이 값을 그대로 Cypher 관계 타입으로 쓰므로,
    여기서 걸러두지 않으면 잘못된 값이 그래프 동기화 단계까지 넘어간다.
    갱신된 relations 목록을 반환한다."""
    relation_types = load_relation_types(store_root)
    if relation_type not in relation_types:
        raise ValueError(f"'{relation_type}'는 등록된 관계 타입이 아닙니다.")
    if target_type not in ("concept", "entity"):
        raise ValueError(f"'{target_type}'는 올바른 대상 타입이 아닙니다(concept|entity만 가능).")

    path = _node_dir(store_root, node_type) / f"{slug}.md"
    frontmatter = _read_frontmatter(path)
    if not frontmatter.get("slug"):
        raise FileNotFoundError(f"노드 파일을 찾을 수 없습니다: {slug}")
    if frontmatter.get("redirect_to"):
        raise ValueError(f"'{slug}'는 다른 노드로 병합되어 더 이상 독립된 노드가 아닙니다.")

    relations = frontmatter.get("relations") or []
    existing = next(
        (
            r for r in relations
            if r.get("type") == relation_type
            and r.get("target_type") == target_type
            and r.get("target_slug") == target_slug
        ),
        None,
    )
    if existing is None:
        relations.append(
            {
                "type": relation_type,
                "target_type": target_type,
                "target_slug": target_slug,
                "rationale": rationale,
                "sources": [source_paper_slug] if source_paper_slug else [],
            }
        )
    else:
        if rationale and not existing.get("rationale"):
            existing["rationale"] = rationale
        sources = existing.setdefault("sources", [])
        if source_paper_slug and source_paper_slug not in sources:
            sources.append(source_paper_slug)
    frontmatter["relations"] = relations

    user_section = _extract_user_section(path)
    _write_node_file(path, frontmatter, user_section)
    return relations


def get_relations(store_root: str, node_type: str, slug: str) -> list[dict]:
    """slug 노드에서 나가는 semantic 관계 목록(frontmatter의 relations[])을
    그대로 반환한다. graph_db.sync_node()가 이 값으로 Neo4j 에지를 다시
    만든다."""
    path = _node_dir(store_root, node_type) / f"{slug}.md"
    return _read_frontmatter(path).get("relations") or []


def remove_relation(
    store_root: str,
    node_type: str,
    slug: str,
    relation_type: str,
    target_type: str,
    target_slug: str,
) -> bool:
    """사용자가 그래프 뷰에서 semantic 관계 에지 하나를 지울 때 쓴다. slug 노드
    (from)의 frontmatter relations[]에서 (relation_type, target_type,
    target_slug) 조합과 일치하는 항목을 통째로 제거한다.

    add_relation()의 단일 소유권 규칙(docs/description/relation_types.md 참고)
    덕분에 이 함수는 remove_source_from_node()/unlink_concept_from_entity()보다
    단순하다: semantic 관계는 from 노드의 frontmatter에만 존재하고 to 노드
    쪽에는 애초에 사본이 전혀 없으므로(대칭 타입도 저장 시점에 이미 한쪽으로
    정규화됨), from 쪽 파일 하나만 고치면 끝이고 "지웠는데 to 쪽에 유령
    항목이 남는" 경우 자체가 구조적으로 생길 수 없다. 호출하는 쪽(그래프
    뷰 서버)이 그래프에 그려진 에지의 from/to를 그대로 넘기기만 하면 된다 -
    대칭 타입이라도 그래프 데이터가 이미 정규화된 방향으로 들어 있어서 여기서
    다시 방향을 판단할 필요가 없다.

    relations가 비어도 파일은 지우지 않는다(remove_source_from_node/
    unlink_concept_from_entity와 같은 이유 - alias/description/사용자 메모 같은
    이 노드의 다른 데이터를 보존한다). 실제로 항목을 지웠으면 True, 그런
    항목이 애초에 없었으면 False."""
    path = _node_dir(store_root, node_type) / f"{slug}.md"
    frontmatter = _read_frontmatter(path)
    if not frontmatter.get("slug"):
        raise FileNotFoundError(f"노드 파일을 찾을 수 없습니다: {slug}")

    relations = frontmatter.get("relations") or []
    remaining = [
        r for r in relations
        if not (
            r.get("type") == relation_type
            and r.get("target_type") == target_type
            and r.get("target_slug") == target_slug
        )
    ]
    if len(remaining) == len(relations):
        return False

    frontmatter["relations"] = remaining
    user_section = _extract_user_section(path)
    _write_node_file(path, frontmatter, user_section)
    return True


def remove_relations_targeting(store_root: str, target_type: str, target_slug: str) -> list[tuple[str, str]]:
    """target_type/target_slug 노드를 semantic 관계의 "to"로 가리키는 항목을
    시스템 전체(모든 concept/entity 파일)에서 찾아 지운다.

    remove_relation()은 관계를 "만든 쪽"(from) 노드 하나만 고치면 되지만(단일
    소유 원칙), 노드 자체를 통째로 지울 때는 사정이 다르다 - delete_node()가
    지우는 건 "to"가 될 수도 있던 그 노드 파일 자신이고, 그 노드를 relations[]에서
    target_slug로 가리키던 다른 노드들(from)은 전혀 건드려지지 않은 채로 남는다.
    그대로 두면 remove_source()가 없던 시절의 문제(파일 없는 죽은 참조)와
    똑같이, 이제는 relations[]에 "가리키는 노드 파일 자체가 없는" 유령 관계가
    영구히 남는다 - remove_source()가 논문 삭제 시 모든 노드의 sources[]를
    훑어 정리하는 것과 정확히 같은 이유로, 노드 삭제 시에도 이 정리가 필요하다.

    영향받은 (node_type, slug) 목록을 반환한다 - 호출부(app.py의
    delete_node_endpoint)가 이 목록으로 Neo4j 미러도 다시 동기화한다."""
    affected: list[tuple[str, str]] = []
    for node_type, dir_name in _DIR_BY_TYPE.items():
        folder = Path(store_root) / dir_name
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.md")):
            frontmatter = _read_frontmatter(path)
            relations = frontmatter.get("relations") or []
            if not relations:
                continue
            remaining = [
                r for r in relations
                if not (r.get("target_type") == target_type and r.get("target_slug") == target_slug)
            ]
            if len(remaining) == len(relations):
                continue
            frontmatter["relations"] = remaining
            user_section = _extract_user_section(path)
            _write_node_file(path, frontmatter, user_section)
            affected.append((node_type, frontmatter["slug"]))
    return affected


def _update_node(
    path: Path, label: str, aliases: list[str], source_slug: str, source_title: str,
    concept_slug: str | None = None,
) -> None:
    frontmatter = _read_frontmatter(path)

    # display_label은 최초 생성 시 표기로 고정한다("나중 표기가 항상 더 낫다"는
    # 보장이 없다 - OCR 품질 등으로 오히려 나중 논문이 더 못생긴 표기를 줄 수도
    # 있다). slug와 마찬가지로 노드 정체성은 최초 생성 시 확정, 이후 변경 없음.
    existing_aliases = set(frontmatter.get("aliases") or [])
    existing_aliases.update(a for a in aliases if a)
    # label 자체도 alias 집합에 넣는다 - 이 논문이 준 aliases가 아니라 label
    # 하나만으로(또는 label+alias 조합으로) 기존 노드와 매칭됐을 수도 있는데,
    # 그 경우 매칭에 실제로 쓰인 신호(alias)만 합치고 label 자체는 안 넣으면,
    # 이 논문 자신의 본문 위키링크([[label]], write_note()가 그대로 씀)가
    # 나중에 이 노드를 다시 못 찾는 문제가 생긴다(실제로 겪음 - Semiseparable
    # Matrix/Matrices 사례). display_label과 정규화 기준으로 같으면 중복이니
    # 굳이 안 넣는다.
    if label and normalize_label(label) != normalize_label(frontmatter["display_label"]):
        existing_aliases.add(label)
    frontmatter["aliases"] = sorted(existing_aliases)

    sources = frontmatter.get("sources") or []
    # 같은 논문이 이미 sources에 있으면(재처리 등) 새로 추가하는 대신 그 항목의
    # concept_slug를 이번 값으로 덮어쓴다 - link_node_to_paper()와 같은 이유
    # (그 주석 참고). 재처리 시엔 remove_source()가 먼저 옛 참조를 지우니 보통
    # 이 분기를 안 타지만, 그것과 무관하게 항상 안전하게 동작해야 한다.
    existing = next((s for s in sources if s["slug"] == source_slug), None)
    if existing is None:
        sources.append({"slug": source_slug, "title": source_title, "concept_slug": concept_slug})
    else:
        existing["concept_slug"] = concept_slug
    frontmatter["sources"] = sources

    user_section = _extract_user_section(path)
    _write_node_file(path, frontmatter, user_section)
