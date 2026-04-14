"""
modules/memory.py — 대화 장기 기억 저장/검색

ChromaDB PersistentClient + jhgan/ko-sroberta-multitask 임베딩으로
대화 내용을 로컬 벡터 DB에 저장하고 의미 기반으로 검색한다.

저장 경로: BASE_DIR / data / memory  (config.MEMORY_DIR)
컬렉션명: "ria_memory"

주요 함수:
    add_message(role, content, session_id)  — 대화 한 턴 저장
    search(query, n_results, role_filter)   — 의미 유사도 검색
    get_recent(n, session_id)               — 최신 N개 조회
    clear_collection(session_id)            — 전체 또는 세션 삭제
    get_collection_info()                   — 컬렉션 통계
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from loguru import logger

from config import DEVICE, MEMORY_DIR


# ── 상수 ─────────────────────────────────────────────────────────────────────

COLLECTION_NAME: str = "ria_memory"
EMBEDDING_MODEL: str = "jhgan/ko-sroberta-multitask"

# 역할 허용 값
_VALID_ROLES: frozenset[str] = frozenset({"user", "assistant", "system"})


# ── 결과 데이터 클래스 ────────────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    """검색/조회 결과 한 건."""

    id: str
    role: str                           # "user" | "assistant" | "system"
    content: str
    timestamp: str                      # ISO 8601 문자열
    session_id: str
    distance: Optional[float] = None    # 검색 시에만 채워짐 (낮을수록 유사)


# ── 모듈 레벨 싱글턴 ─────────────────────────────────────────────────────────

_client = None       # chromadb.PersistentClient
_collection = None   # chromadb.Collection


# ── 내부 헬퍼 ────────────────────────────────────────────────────────────────

def _make_embedding_function():
    """ChromaDB용 SentenceTransformer 임베딩 함수를 생성한다.

    chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction을
    사용해 jhgan/ko-sroberta-multitask 모델을 로드한다.

    Returns:
        SentenceTransformerEmbeddingFunction 인스턴스

    Raises:
        ImportError: chromadb 또는 sentence-transformers 미설치 시
    """
    from chromadb.utils.embedding_functions import (
        SentenceTransformerEmbeddingFunction,
    )

    logger.info(
        "임베딩 모델 로드 | model={model} | device={device}",
        model=EMBEDDING_MODEL,
        device=DEVICE,
    )
    ef = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        device=DEVICE,
    )
    logger.info("임베딩 모델 로드 완료")
    return ef


def _get_collection():
    """ChromaDB 컬렉션 싱글턴을 반환한다. 없으면 초기화한다.

    Returns:
        chromadb.Collection 인스턴스

    Raises:
        ImportError: chromadb 미설치 시
        RuntimeError: 컬렉션 초기화 실패 시
    """
    global _client, _collection

    if _collection is not None:
        return _collection

    try:
        import chromadb
    except ImportError as e:
        logger.error("chromadb 미설치: {e}", e=e)
        raise

    logger.info("ChromaDB 초기화 | path={path}", path=str(MEMORY_DIR))

    try:
        _client = chromadb.PersistentClient(path=str(MEMORY_DIR))
        ef = _make_embedding_function()
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "컬렉션 준비 완료 | name={name} | count={n}",
            name=COLLECTION_NAME,
            n=_collection.count(),
        )
    except Exception as e:
        logger.error("ChromaDB 초기화 실패: {e}", e=e)
        raise RuntimeError(f"ChromaDB 초기화 실패: {e}") from e

    return _collection


def _make_id() -> str:
    """중복 없는 메모리 ID를 생성한다."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{ts}_{uuid.uuid4().hex[:8]}"


# ── 공개 함수 ─────────────────────────────────────────────────────────────────

def add_message(
    role: str,
    content: str,
    session_id: str = "default",
) -> str:
    """대화 한 턴을 벡터 DB에 저장한다.

    Args:
        role: 발화 주체 ("user" | "assistant" | "system")
        content: 저장할 텍스트
        session_id: 대화 세션 식별자 (기본 "default")

    Returns:
        저장된 메모리 항목의 ID 문자열

    Raises:
        ValueError: role이 허용되지 않은 값일 때
        ValueError: content가 빈 문자열일 때
        RuntimeError: ChromaDB 저장 실패 시
    """
    if role not in _VALID_ROLES:
        raise ValueError(f"허용되지 않은 role: '{role}' (허용: {sorted(_VALID_ROLES)})")
    if not content or not content.strip():
        raise ValueError("content가 비어 있습니다.")

    col = _get_collection()
    entry_id = _make_id()
    timestamp = datetime.now().isoformat()

    try:
        col.add(
            ids=[entry_id],
            documents=[content],
            metadatas=[{
                "role": role,
                "timestamp": timestamp,
                "session_id": session_id,
            }],
        )
    except Exception as e:
        logger.error("add_message 저장 실패: {e}", e=e)
        raise RuntimeError(f"메모리 저장 실패: {e}") from e

    logger.info(
        "메모리 저장 | id={id} | role={role} | session={sid} | len={n}",
        id=entry_id,
        role=role,
        sid=session_id,
        n=len(content),
    )
    return entry_id


def search(
    query: str,
    n_results: int = 5,
    role_filter: Optional[str] = None,
    session_id: Optional[str] = None,
) -> list[MemoryEntry]:
    """의미 유사도 기반으로 기억을 검색한다.

    Args:
        query: 검색 쿼리 텍스트
        n_results: 반환할 최대 결과 수 (기본 5)
        role_filter: "user" / "assistant" / "system" 중 하나로 역할 필터링 (None이면 전체)
        session_id: 특정 세션으로 검색 범위 제한 (None이면 전체)

    Returns:
        유사도 순 MemoryEntry 리스트 (distance 낮을수록 유사)

    Raises:
        ValueError: query가 빈 문자열일 때
    """
    if not query or not query.strip():
        raise ValueError("검색 query가 비어 있습니다.")

    col = _get_collection()
    total = col.count()

    if total == 0:
        logger.debug("search: 컬렉션이 비어 있음")
        return []

    # where 필터 조합
    where: Optional[dict] = None
    conditions: list[dict] = []
    if role_filter:
        conditions.append({"role": {"$eq": role_filter}})
    if session_id:
        conditions.append({"session_id": {"$eq": session_id}})

    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    n_results = min(n_results, total)

    try:
        result = col.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        logger.error("search 쿼리 실패: {e}", e=e)
        return []

    entries: list[MemoryEntry] = []
    ids       = result.get("ids", [[]])[0]
    docs      = result.get("documents", [[]])[0]
    metas     = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    for eid, doc, meta, dist in zip(ids, docs, metas, distances):
        entries.append(MemoryEntry(
            id=eid,
            role=meta.get("role", ""),
            content=doc,
            timestamp=meta.get("timestamp", ""),
            session_id=meta.get("session_id", ""),
            distance=round(dist, 4),
        ))

    logger.info(
        "search 완료 | query_len={n} | 결과={k}개",
        n=len(query),
        k=len(entries),
    )
    return entries


def get_recent(
    n: int = 10,
    session_id: Optional[str] = None,
) -> list[MemoryEntry]:
    """timestamp 역순으로 최신 N개 기억을 반환한다.

    Args:
        n: 반환할 최대 개수 (기본 10)
        session_id: 세션 필터 (None이면 전체)

    Returns:
        최신순 MemoryEntry 리스트
    """
    col = _get_collection()

    if col.count() == 0:
        logger.debug("get_recent: 컬렉션이 비어 있음")
        return []

    where: Optional[dict] = {"session_id": {"$eq": session_id}} if session_id else None

    try:
        result = col.get(
            where=where,
            include=["documents", "metadatas"],
        )
    except Exception as e:
        logger.error("get_recent 조회 실패: {e}", e=e)
        return []

    ids   = result.get("ids", [])
    docs  = result.get("documents", [])
    metas = result.get("metadatas", [])

    entries: list[MemoryEntry] = [
        MemoryEntry(
            id=eid,
            role=meta.get("role", ""),
            content=doc,
            timestamp=meta.get("timestamp", ""),
            session_id=meta.get("session_id", ""),
        )
        for eid, doc, meta in zip(ids, docs, metas)
    ]

    # timestamp 내림차순 정렬 후 N개
    entries.sort(key=lambda e: e.timestamp, reverse=True)
    recent = entries[:n]

    logger.info("get_recent | session={sid} | 반환={k}개", sid=session_id, k=len(recent))
    return recent


def clear_collection(session_id: Optional[str] = None) -> int:
    """컬렉션의 기억을 삭제한다.

    Args:
        session_id: 지정 시 해당 세션만 삭제, None이면 전체 삭제

    Returns:
        삭제된 항목 수
    """
    col = _get_collection()

    if session_id:
        try:
            result = col.get(
                where={"session_id": {"$eq": session_id}},
                include=[],
            )
            ids_to_delete: list[str] = result.get("ids", [])
            if ids_to_delete:
                col.delete(ids=ids_to_delete)
            count = len(ids_to_delete)
        except Exception as e:
            logger.error("clear_collection(session) 실패: {e}", e=e)
            return 0
    else:
        count = col.count()
        try:
            # 전체 ID 조회 후 삭제
            all_ids: list[str] = col.get(include=[]).get("ids", [])
            if all_ids:
                col.delete(ids=all_ids)
        except Exception as e:
            logger.error("clear_collection(all) 실패: {e}", e=e)
            return 0

    logger.info(
        "clear_collection | session={sid} | 삭제={n}개",
        sid=session_id or "ALL",
        n=count,
    )
    return count


def get_collection_info() -> dict:
    """컬렉션 통계 정보를 반환한다.

    Returns:
        {name, total_count, memory_dir, embedding_model}
    """
    col = _get_collection()
    info = {
        "name": COLLECTION_NAME,
        "total_count": col.count(),
        "memory_dir": str(MEMORY_DIR),
        "embedding_model": EMBEDDING_MODEL,
    }
    logger.debug("collection_info: {info}", info=info)
    return info


# ── 단독 테스트 ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("=== memory.py 단독 테스트 시작 ===")
    logger.info("저장 경로: {path}", path=str(MEMORY_DIR))

    TEST_SESSION = "test_session_memory_py"

    # 테스트 전 해당 세션 초기화
    try:
        cleared = clear_collection(session_id=TEST_SESSION)
        logger.info("테스트 세션 초기화 | 삭제={n}개", n=cleared)
    except Exception as e:
        logger.warning("초기화 중 오류 (무시): {e}", e=e)

    # ── [1] add_message 정상 케이스 ──────────────────────────
    logger.info("--- [1] add_message 정상 케이스 ---")
    sample_turns = [
        ("user",      "오늘 날씨가 정말 맑아서 기분이 좋아요."),
        ("assistant", "좋은 날씨네요! 산책이라도 다녀오시는 건 어떨까요?"),
        ("user",      "저는 파이썬 프로그래밍을 공부하고 있어요."),
        ("assistant", "파이썬은 배우기 좋은 언어죠. 어떤 분야를 목표로 하시나요?"),
        ("user",      "AI 개발이 목표예요. 특히 자연어 처리에 관심이 많아요."),
    ]

    saved_ids: list[str] = []
    try:
        for role, content in sample_turns:
            mid = add_message(role, content, session_id=TEST_SESSION)
            saved_ids.append(mid)
            logger.info("  저장 | role={role} | id={id}", role=role, id=mid)
        assert len(saved_ids) == len(sample_turns)
        logger.info("add_message 정상 케이스 통과 | {n}개 저장", n=len(saved_ids))
    except Exception as e:
        logger.error("add_message 정상 케이스 실패: {e}", e=e)

    # ── [2] get_collection_info ───────────────────────────────
    logger.info("--- [2] get_collection_info 테스트 ---")
    try:
        info = get_collection_info()
        assert info["total_count"] >= len(sample_turns)
        logger.info("collection_info: {info}", info=info)
    except Exception as e:
        logger.error("get_collection_info 실패: {e}", e=e)

    # ── [3] search 정상 케이스 ────────────────────────────────
    logger.info("--- [3] search 정상 케이스 ---")
    try:
        results = search("자연어 처리 공부", n_results=3, session_id=TEST_SESSION)
        assert isinstance(results, list), "리스트여야 함"
        assert len(results) > 0, "결과가 1개 이상이어야 함"
        logger.info("search 결과: {n}개", n=len(results))
        for r in results:
            logger.info(
                "  [dist={d:.4f}] {role}: {text}",
                d=r.distance or 0,
                role=r.role,
                text=r.content[:50],
            )
        # 가장 유사한 결과가 NLP 관련 문장이어야 함
        assert results[0].distance is not None
        logger.info("search 정상 케이스 통과")
    except Exception as e:
        logger.error("search 정상 케이스 실패: {e}", e=e)

    # ── [4] search role_filter 테스트 ────────────────────────
    logger.info("--- [4] search role_filter 테스트 ---")
    try:
        user_results = search(
            "날씨 산책",
            n_results=5,
            role_filter="user",
            session_id=TEST_SESSION,
        )
        assert all(r.role == "user" for r in user_results), "role_filter=user인데 다른 role 포함됨"
        logger.info("role_filter=user 필터 통과 | {n}개", n=len(user_results))

        asst_results = search(
            "공부 목표",
            n_results=5,
            role_filter="assistant",
            session_id=TEST_SESSION,
        )
        assert all(r.role == "assistant" for r in asst_results), "role_filter=assistant 필터 실패"
        logger.info("role_filter=assistant 필터 통과 | {n}개", n=len(asst_results))
    except Exception as e:
        logger.error("role_filter 테스트 실패: {e}", e=e)

    # ── [5] get_recent 테스트 ─────────────────────────────────
    logger.info("--- [5] get_recent 테스트 ---")
    try:
        recent = get_recent(n=3, session_id=TEST_SESSION)
        assert isinstance(recent, list)
        assert len(recent) <= 3
        logger.info("get_recent(n=3) 결과: {n}개", n=len(recent))
        for r in recent:
            logger.info("  [{ts}] {role}: {text}", ts=r.timestamp[11:19], role=r.role, text=r.content[:40])

        # timestamp 내림차순 정렬 확인
        if len(recent) >= 2:
            assert recent[0].timestamp >= recent[1].timestamp, "timestamp 내림차순 정렬 실패"
        logger.info("get_recent 정상 케이스 통과")
    except Exception as e:
        logger.error("get_recent 테스트 실패: {e}", e=e)

    # ── [6] 에러 케이스 ───────────────────────────────────────
    logger.info("--- [6] 에러 케이스 ---")

    # 빈 content
    try:
        add_message("user", "")
        logger.warning("빈 content: ValueError가 발생해야 하는데 통과됨")
    except ValueError as e:
        logger.info("빈 content ValueError 정상 처리: {e}", e=e)

    # 허용되지 않는 role
    try:
        add_message("bot", "테스트")
        logger.warning("잘못된 role: ValueError가 발생해야 하는데 통과됨")
    except ValueError as e:
        logger.info("잘못된 role ValueError 정상 처리: {e}", e=e)

    # 빈 query 검색
    try:
        search("")
        logger.warning("빈 query: ValueError가 발생해야 하는데 통과됨")
    except ValueError as e:
        logger.info("빈 query ValueError 정상 처리: {e}", e=e)

    # ── [7] clear_collection 세션 삭제 ───────────────────────
    logger.info("--- [7] clear_collection 세션 삭제 테스트 ---")
    try:
        before = get_collection_info()["total_count"]
        deleted = clear_collection(session_id=TEST_SESSION)
        after = get_collection_info()["total_count"]

        assert deleted == len(sample_turns), f"삭제 수 불일치: {deleted} != {len(sample_turns)}"
        assert after == before - deleted, f"삭제 후 count 불일치: {after} != {before - deleted}"
        logger.info(
            "clear_collection 세션 삭제 통과 | 삭제={d} | before={b} → after={a}",
            d=deleted, b=before, a=after,
        )
    except Exception as e:
        logger.error("clear_collection 테스트 실패: {e}", e=e)

    logger.info("=== memory.py 단독 테스트 완료 ===")
