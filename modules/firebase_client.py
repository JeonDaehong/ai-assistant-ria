"""
modules/firebase_client.py — Firebase Cloud Messaging (FCM) 푸시 알림

firebase-admin SDK로 FCM v1 API를 통해 푸시 알림을 전송한다.
서비스 계정 키 파일 경로는 .env의 FIREBASE_KEY_PATH로 관리한다.

주요 함수:
    init_app()                                    — Firebase 앱 초기화 (최초 1회)
    send_notification(token, title, body, data)   — 단일 기기 알림 전송
    send_multicast(tokens, title, body, data)     — 다중 기기 알림 전송
    is_initialized()                              — 초기화 여부 확인
"""

from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from config import FIREBASE_KEY_PATH


# ── 결과 데이터 클래스 ────────────────────────────────────────────────────────

@dataclass
class FCMResult:
    """FCM 전송 결과 한 건."""

    success: bool
    message_id: Optional[str] = None   # 성공 시 FCM 메시지 ID
    error: Optional[str] = None        # 실패 시 에러 메시지
    token: str = ""                    # 대상 기기 토큰 (앞 10자 + ****)


@dataclass
class MulticastResult:
    """send_multicast 전체 결과."""

    total: int
    success_count: int
    failure_count: int
    results: list[FCMResult] = field(default_factory=list)


# ── 모듈 레벨 상태 ────────────────────────────────────────────────────────────

_app = None   # firebase_admin.App


# ── 내부 헬퍼 ────────────────────────────────────────────────────────────────

def _mask_token(token: str) -> str:
    """FCM 토큰 앞 10자만 노출하고 나머지를 마스킹한다."""
    return token[:10] + "****" if len(token) > 10 else "****"


def _validate_key_path() -> None:
    """FIREBASE_KEY_PATH 설정 및 파일 존재 여부를 검증한다.

    Raises:
        FileNotFoundError: 경로 미설정 또는 파일 없음
    """
    if not FIREBASE_KEY_PATH or not str(FIREBASE_KEY_PATH).strip():
        raise FileNotFoundError(
            "FIREBASE_KEY_PATH가 .env에 설정되지 않았습니다."
        )
    if not FIREBASE_KEY_PATH.exists():
        raise FileNotFoundError(
            f"firebase 키 파일을 찾을 수 없습니다: {FIREBASE_KEY_PATH}"
        )


# ── 공개 함수 ─────────────────────────────────────────────────────────────────

def is_initialized() -> bool:
    """Firebase 앱이 초기화되었는지 반환한다.

    Returns:
        초기화된 경우 True
    """
    return _app is not None


def init_app(force: bool = False) -> None:
    """Firebase Admin SDK를 초기화한다. 최초 1회만 실행된다.

    Args:
        force: True면 기존 앱을 삭제하고 재초기화한다.

    Raises:
        FileNotFoundError: FIREBASE_KEY_PATH 미설정 또는 파일 없음
        ImportError: firebase-admin 미설치
        RuntimeError: 초기화 실패
    """
    global _app

    if _app is not None and not force:
        logger.debug("Firebase 앱 이미 초기화됨, 스킵")
        return

    _validate_key_path()

    try:
        import firebase_admin
        from firebase_admin import credentials
    except ImportError as e:
        logger.error("firebase-admin 미설치: {e}", e=e)
        raise

    # 기존 앱 삭제 (force 재초기화 시)
    if force and _app is not None:
        try:
            firebase_admin.delete_app(_app)
            logger.debug("기존 Firebase 앱 삭제")
        except Exception as e:
            logger.warning("기존 앱 삭제 실패 (무시): {e}", e=e)
        _app = None

    try:
        cred = credentials.Certificate(str(FIREBASE_KEY_PATH))
        _app = firebase_admin.initialize_app(cred)
        logger.info(
            "Firebase 앱 초기화 완료 | key={key}",
            key=FIREBASE_KEY_PATH.name,
        )
    except Exception as e:
        _app = None
        logger.error("Firebase 초기화 실패: {e}", e=e)
        raise RuntimeError(f"Firebase 초기화 실패: {e}") from e


def send_notification(
    token: str,
    title: str,
    body: str,
    data: Optional[dict[str, str]] = None,
    image_url: Optional[str] = None,
) -> FCMResult:
    """단일 기기에 FCM 푸시 알림을 전송한다.

    Args:
        token: 대상 기기의 FCM 등록 토큰
        title: 알림 제목
        body: 알림 본문
        data: 추가 데이터 페이로드 (문자열 키-값 쌍). None이면 미포함.
        image_url: 알림에 포함할 이미지 URL. None이면 미포함.

    Returns:
        FCMResult (success, message_id 또는 error)

    Raises:
        ValueError: token / title / body가 비어 있을 때
        RuntimeError: Firebase 앱 미초기화 시
    """
    if not token or not token.strip():
        raise ValueError("FCM token이 비어 있습니다.")
    if not title or not title.strip():
        raise ValueError("알림 title이 비어 있습니다.")
    if not body or not body.strip():
        raise ValueError("알림 body가 비어 있습니다.")

    if not is_initialized():
        raise RuntimeError("Firebase 앱이 초기화되지 않았습니다. init_app()을 먼저 호출하세요.")

    masked = _mask_token(token)
    logger.info(
        "FCM 전송 시도 | token={token} | title={title}",
        token=masked,
        title=title,
    )

    try:
        from firebase_admin import messaging

        notification = messaging.Notification(
            title=title,
            body=body,
            image=image_url,
        )
        message = messaging.Message(
            token=token,
            notification=notification,
            data=data or {},
        )
        message_id: str = messaging.send(message)

        logger.info(
            "FCM 전송 성공 | token={token} | message_id={mid}",
            token=masked,
            mid=message_id,
        )
        return FCMResult(success=True, message_id=message_id, token=masked)

    except Exception as e:
        error_msg = str(e)
        logger.error(
            "FCM 전송 실패 | token={token} | error={e}",
            token=masked,
            e=error_msg,
        )
        return FCMResult(success=False, error=error_msg, token=masked)


def send_multicast(
    tokens: list[str],
    title: str,
    body: str,
    data: Optional[dict[str, str]] = None,
    image_url: Optional[str] = None,
) -> MulticastResult:
    """여러 기기에 FCM 푸시 알림을 전송한다.

    각 토큰에 대해 send_notification()을 순차 호출한다.
    일부 실패해도 나머지 전송을 계속한다.

    Args:
        tokens: 대상 기기 FCM 등록 토큰 리스트 (빈 문자열 항목은 건너뜀)
        title: 알림 제목
        body: 알림 본문
        data: 추가 데이터 페이로드. None이면 미포함.
        image_url: 알림 이미지 URL. None이면 미포함.

    Returns:
        MulticastResult (total, success_count, failure_count, results)

    Raises:
        ValueError: tokens가 빈 리스트일 때
        RuntimeError: Firebase 앱 미초기화 시
    """
    if not tokens:
        raise ValueError("tokens 리스트가 비어 있습니다.")
    if not is_initialized():
        raise RuntimeError("Firebase 앱이 초기화되지 않았습니다. init_app()을 먼저 호출하세요.")

    logger.info("FCM 멀티캐스트 전송 시작 | 대상={n}개", n=len(tokens))

    results: list[FCMResult] = []
    for token in tokens:
        if not token or not token.strip():
            logger.warning("빈 토큰 항목 건너뜀")
            results.append(FCMResult(success=False, error="empty token", token=""))
            continue
        result = send_notification(token, title, body, data=data, image_url=image_url)
        results.append(result)

    success_count = sum(1 for r in results if r.success)
    failure_count = len(results) - success_count

    multicast = MulticastResult(
        total=len(results),
        success_count=success_count,
        failure_count=failure_count,
        results=results,
    )

    logger.info(
        "FCM 멀티캐스트 완료 | 성공={s} | 실패={f}",
        s=success_count,
        f=failure_count,
    )
    return multicast


# ── 단독 테스트 ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("=== firebase_client.py 단독 테스트 시작 ===")
    logger.info("FIREBASE_KEY_PATH={path}", path=str(FIREBASE_KEY_PATH))

    # ── [1] 키 파일 존재 여부 확인 ───────────────────────────
    logger.info("--- [1] 키 파일 존재 여부 확인 ---")
    key_exists = FIREBASE_KEY_PATH.exists() if FIREBASE_KEY_PATH and str(FIREBASE_KEY_PATH).strip() else False

    if not key_exists:
        logger.warning(
            "firebase-key.json 없음 → 초기화/전송 테스트는 에러 케이스로만 진행"
        )
    else:
        logger.info("키 파일 확인됨: {path}", path=FIREBASE_KEY_PATH)

    # ── [2] 에러 케이스: 키 파일 없이 init_app() ─────────────
    logger.info("--- [2] 에러 케이스: 키 파일 없음 또는 미설정 ---")
    if not key_exists:
        try:
            init_app()
            logger.warning("FileNotFoundError가 발생해야 하는데 통과됨")
        except FileNotFoundError as e:
            logger.info("FileNotFoundError 정상 처리: {e}", e=e)
        except Exception as e:
            logger.info("기타 예외 처리됨: {type} — {e}", type=type(e).__name__, e=e)
    else:
        logger.info("키 파일 존재 → 키 없음 에러 케이스 스킵")

    # ── [3] 에러 케이스: 미초기화 상태에서 send_notification ──
    logger.info("--- [3] 에러 케이스: 미초기화 상태에서 전송 ---")
    # is_initialized()가 False인 상태여야 함 (키 없으면 init_app 실패)
    if not is_initialized():
        try:
            send_notification("dummy_token", "테스트", "본문")
            logger.warning("RuntimeError가 발생해야 하는데 통과됨")
        except RuntimeError as e:
            logger.info("RuntimeError 정상 처리 (미초기화): {e}", e=e)
    else:
        logger.info("이미 초기화된 상태 → 미초기화 에러 케이스 스킵")

    # ── [4] 에러 케이스: 잘못된 인수 ─────────────────────────
    logger.info("--- [4] 에러 케이스: 잘못된 인수 ---")

    # send_notification에 빈 token
    try:
        send_notification("", "제목", "본문")
        logger.warning("ValueError(빈 token)가 발생해야 하는데 통과됨")
    except ValueError as e:
        logger.info("빈 token ValueError 정상 처리: {e}", e=e)

    # send_notification에 빈 title
    try:
        send_notification("some_token", "", "본문")
        logger.warning("ValueError(빈 title)가 발생해야 하는데 통과됨")
    except ValueError as e:
        logger.info("빈 title ValueError 정상 처리: {e}", e=e)

    # send_multicast에 빈 리스트
    try:
        send_multicast([], "제목", "본문")
        logger.warning("ValueError(빈 tokens)가 발생해야 하는데 통과됨")
    except ValueError as e:
        logger.info("빈 tokens ValueError 정상 처리: {e}", e=e)
    except RuntimeError as e:
        logger.info("RuntimeError(미초기화) 먼저 발생: {e}", e=e)

    # ── [5] 정상 케이스: init_app + 더미 토큰 전송 ───────────
    logger.info("--- [5] 정상 케이스: init_app + 더미 토큰 전송 (키 파일 필요) ---")
    if not key_exists:
        logger.warning("키 파일 없음 → 정상 케이스 스킵 (.env에 FIREBASE_KEY_PATH 설정 필요)")
    else:
        # init_app
        try:
            init_app()
            assert is_initialized(), "초기화 후 is_initialized()가 True여야 함"
            logger.info("init_app() 성공 | is_initialized=True")
        except Exception as e:
            logger.error("init_app() 실패: {e}", e=e)

        if is_initialized():
            # 더미 토큰으로 전송 — FCM이 유효하지 않은 토큰 오류를 반환해야 함
            dummy_token = "aaaaaaaaaa_invalid_fcm_token_for_test_only"
            result = send_notification(
                token=dummy_token,
                title="Ria 테스트",
                body="단독 테스트 알림입니다.",
                data={"source": "scheduler", "level": "test"},
            )
            assert isinstance(result, FCMResult)
            # 더미 토큰이므로 실패가 정상
            if result.success:
                logger.info("전송 성공 (예상치 못한 성공) | message_id={mid}", mid=result.message_id)
            else:
                logger.info(
                    "더미 토큰 전송 실패 (예상된 결과) | error={e}",
                    e=result.error,
                )

            # send_multicast 테스트
            mc_result = send_multicast(
                tokens=[dummy_token, "", dummy_token],
                title="멀티캐스트 테스트",
                body="다중 기기 알림 테스트",
            )
            assert mc_result.total == 3
            assert mc_result.failure_count >= 1  # 빈 토큰 + 더미 토큰 실패
            logger.info(
                "send_multicast 결과 | total={t} | 성공={s} | 실패={f}",
                t=mc_result.total,
                s=mc_result.success_count,
                f=mc_result.failure_count,
            )

            # force 재초기화
            try:
                init_app(force=True)
                assert is_initialized()
                logger.info("force 재초기화 통과")
            except Exception as e:
                logger.error("force 재초기화 실패: {e}", e=e)

    logger.info("=== firebase_client.py 단독 테스트 완료 ===")
