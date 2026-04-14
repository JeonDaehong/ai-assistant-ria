"""
modules/obsidian.py — Obsidian 볼트 연동

노트 목록 조회, 검색, 읽기, 쓰기 기능을 제공한다.
볼트 경로는 config.OBSIDIAN_VAULT_PATH에서 읽는다.

실행:
    python modules/obsidian.py
"""

from pathlib import Path
from typing import Optional

from loguru import logger

from config import OBSIDIAN_VAULT_PATH


# ── 내부 유틸 ─────────────────────────────────────────────


def _vault() -> Path:
    """볼트 경로를 검증하고 반환한다."""
    if not OBSIDIAN_VAULT_PATH.exists():
        raise FileNotFoundError(
            f"Obsidian 볼트를 찾을 수 없습니다: {OBSIDIAN_VAULT_PATH}"
        )
    return OBSIDIAN_VAULT_PATH


def _safe_path(vault: Path, note_path: str) -> Path:
    """사용자 입력 경로를 볼트 내부로 제한한다 (Path Traversal 방지).

    Args:
        vault: 볼트 루트 Path
        note_path: 노트 상대 경로 또는 이름

    Returns:
        볼트 내 절대 경로

    Raises:
        PermissionError: 볼트 외부 접근 시도 시
    """
    resolved = (vault / note_path).resolve()
    if not str(resolved).startswith(str(vault.resolve())):
        raise PermissionError(f"볼트 외부 경로 접근 거부: {note_path}")
    return resolved


# ── 공개 API ──────────────────────────────────────────────


def list_notes(subfolder: str = "") -> list[str]:
    """볼트 내 모든 마크다운 노트의 상대 경로 목록을 반환한다.

    Args:
        subfolder: 검색할 하위 폴더 (기본: 볼트 루트 전체)

    Returns:
        마크다운 파일의 볼트 루트 기준 상대 경로 문자열 리스트.
        볼트가 없으면 빈 리스트.
    """
    try:
        vault = _vault()
    except FileNotFoundError as e:
        logger.warning("{e}", e=e)
        return []

    base = _safe_path(vault, subfolder) if subfolder else vault

    if not base.is_dir():
        logger.warning("list_notes: 디렉토리 없음 → {path}", path=base)
        return []

    notes = sorted(
        str(p.relative_to(vault))
        for p in base.rglob("*.md")
        if p.is_file()
    )
    logger.info("list_notes: {n}개 노트 | 폴더={f}", n=len(notes), f=subfolder or "(루트)")
    return notes


def search_notes(query: str, max_results: int = 10) -> list[dict]:
    """노트 제목과 내용에서 쿼리를 검색하고 결과를 반환한다.

    대소문자를 구분하지 않으며, 제목 일치를 내용 일치보다 우선한다.

    Args:
        query: 검색어 (단어 단위)
        max_results: 반환할 최대 결과 수

    Returns:
        [{"path": str, "title": str, "snippet": str}, ...] 리스트.
        볼트 없음 또는 결과 없음 시 빈 리스트.
    """
    if not query or not query.strip():
        logger.warning("search_notes: 빈 쿼리")
        return []

    try:
        vault = _vault()
    except FileNotFoundError as e:
        logger.warning("{e}", e=e)
        return []

    q = query.strip().lower()
    title_hits: list[dict] = []
    content_hits: list[dict] = []

    for md_file in vault.rglob("*.md"):
        if not md_file.is_file():
            continue

        title = md_file.stem
        rel_path = str(md_file.relative_to(vault))

        # 제목 검색 (우선 순위 높음)
        if q in title.lower():
            title_hits.append({
                "path": rel_path,
                "title": title,
                "snippet": f"[제목 일치] {title}",
            })
            continue

        # 내용 검색
        try:
            content = md_file.read_text(encoding="utf-8")
        except Exception as e:
            logger.debug("노트 읽기 실패 | {path} | {e}", path=rel_path, e=e)
            continue

        if q in content.lower():
            # 쿼리 주변 컨텍스트 추출
            idx = content.lower().find(q)
            start = max(0, idx - 60)
            end = min(len(content), idx + len(q) + 60)
            snippet = content[start:end].replace("\n", " ").strip()
            if start > 0:
                snippet = "..." + snippet
            if end < len(content):
                snippet = snippet + "..."

            content_hits.append({
                "path": rel_path,
                "title": title,
                "snippet": snippet,
            })

    results = (title_hits + content_hits)[:max_results]
    logger.info(
        "search_notes: 쿼리={q} | 제목={t}개 | 내용={c}개 | 반환={r}개",
        q=query,
        t=len(title_hits),
        c=len(content_hits),
        r=len(results),
    )
    return results


def get_note(note_path: str) -> Optional[str]:
    """노트 내용을 읽어 문자열로 반환한다.

    Args:
        note_path: 볼트 루트 기준 상대 경로 (예: "일기/2024-01-01.md")
                   확장자 없이 제목만 입력해도 자동 검색.

    Returns:
        노트 내용 문자열. 파일 없음 또는 읽기 실패 시 None.
    """
    try:
        vault = _vault()
    except FileNotFoundError as e:
        logger.warning("{e}", e=e)
        return None

    # .md 확장자 없으면 볼트 전체에서 제목으로 검색
    if not note_path.endswith(".md"):
        matches = list(vault.rglob(f"{note_path}.md"))
        if not matches:
            logger.warning("get_note: 노트를 찾을 수 없음 | {path}", path=note_path)
            return None
        target = matches[0]
    else:
        target = _safe_path(vault, note_path)
        if not target.exists():
            logger.warning("get_note: 파일 없음 | {path}", path=target)
            return None

    try:
        content = target.read_text(encoding="utf-8")
        logger.info("get_note: {path} | {n}자", path=note_path, n=len(content))
        return content
    except Exception as e:
        logger.error("get_note 읽기 실패: {e}", e=e)
        return None


def create_note(title: str, content: str, subfolder: str = "") -> str:
    """새 노트를 생성하거나 기존 노트를 덮어쓴다.

    Args:
        title: 노트 제목 (파일명, 확장자 제외)
        content: 노트 내용 (마크다운)
        subfolder: 저장할 하위 폴더 (없으면 볼트 루트)

    Returns:
        생성된 파일의 볼트 기준 상대 경로 문자열

    Raises:
        FileNotFoundError: 볼트 경로가 없을 때
        PermissionError: 볼트 외부 경로 접근 시도 시
    """
    vault = _vault()

    # 파일명에 사용할 수 없는 문자 제거
    safe_title = "".join(c for c in title if c not in r'\/:*?"<>|').strip()
    if not safe_title:
        raise ValueError(f"유효하지 않은 노트 제목: '{title}'")

    if subfolder:
        target_dir = _safe_path(vault, subfolder)
    else:
        target_dir = vault

    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{safe_title}.md"

    target.write_text(content, encoding="utf-8")
    rel_path = str(target.relative_to(vault))

    logger.info("create_note: {path} | {n}자", path=rel_path, n=len(content))
    return rel_path


# ── 단독 테스트 ───────────────────────────────────────────

if __name__ == "__main__":
    logger.info("=== obsidian.py 단독 테스트 시작 ===")
    logger.info("볼트 경로: {v}", v=OBSIDIAN_VAULT_PATH)

    # --- 정상 케이스 1: 노트 목록 ---
    logger.info("--- [1] list_notes() ---")
    try:
        notes = list_notes()
        logger.info("list_notes 통과 | {n}개 노트", n=len(notes))
        for note in notes[:5]:
            logger.debug("  - {note}", note=note)
    except Exception as e:
        logger.error("list_notes 실패: {e}", e=e)

    # --- 정상 케이스 2: 노트 생성 ---
    logger.info("--- [2] create_note() ---")
    try:
        test_content = "# 테스트 노트\n\nRia AI 어시스턴트 연동 테스트입니다.\n"
        created_path = create_note("Ria_테스트", test_content, subfolder="")
        logger.info("create_note 통과 | path={p}", p=created_path)
    except Exception as e:
        logger.error("create_note 실패: {e}", e=e)

    # --- 정상 케이스 3: 노트 읽기 ---
    logger.info("--- [3] get_note() ---")
    try:
        note_content = get_note("Ria_테스트")
        if note_content is not None:
            logger.info("get_note 통과 | {n}자", n=len(note_content))
            logger.debug("내용(앞 100자): {c}", c=note_content[:100])
        else:
            logger.warning("get_note: None 반환 (노트 없음)")
    except Exception as e:
        logger.error("get_note 실패: {e}", e=e)

    # --- 정상 케이스 4: 검색 ---
    logger.info("--- [4] search_notes() ---")
    try:
        results = search_notes("Ria")
        logger.info("search_notes 통과 | {n}개 결과", n=len(results))
        for r in results:
            logger.debug("  path={p} | snippet={s}", p=r["path"], s=r["snippet"][:60])
    except Exception as e:
        logger.error("search_notes 실패: {e}", e=e)

    # --- 에러 케이스: 빈 쿼리 ---
    logger.info("--- [5] search_notes 에러 케이스 (빈 쿼리) ---")
    try:
        empty = search_notes("")
        assert empty == [], f"빈 리스트 기대, 실제: {empty}"
        logger.info("에러 케이스 통과 (빈 쿼리 → 빈 리스트)")
    except Exception as e:
        logger.error("에러 케이스 실패: {e}", e=e)

    # --- 에러 케이스: 존재하지 않는 노트 ---
    logger.info("--- [6] get_note 에러 케이스 (없는 노트) ---")
    try:
        missing = get_note("절대존재하지않는노트XYZ123")
        assert missing is None, f"None 기대, 실제: {missing}"
        logger.info("에러 케이스 통과 (없는 노트 → None)")
    except Exception as e:
        logger.error("에러 케이스 실패: {e}", e=e)

    logger.info("=== obsidian.py 단독 테스트 완료 ===")
