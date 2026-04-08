"""
modules/character.py — VTube Studio 연동 캐릭터 리액션 모듈

감정 분석 결과(EmotionResult)를 받아 VTube Studio WebSocket API를 통해
캐릭터 모션/표정을 제어한다.

인증 흐름:
  1. AuthenticationTokenRequest → VTS 팝업에서 사용자 허용 → 토큰 획득
  2. Authentication 요청으로 세션 인증
  3. 이후 세션 유지 (토큰 재사용 불필요)

토큰은 .env의 VTS_AUTH_TOKEN에 저장/로드된다.
"""
import asyncio
import json
import os
import threading
import uuid
from typing import Optional

from dotenv import load_dotenv, set_key
from loguru import logger

from config import BASE_DIR, IS_WINDOWS
from modules.emotion import EmotionResult

# .env 로드 (단독 실행 시를 위해 재호출)
_ENV_PATH = BASE_DIR / ".env"
load_dotenv(_ENV_PATH)

# ── 환경 변수 ──────────────────────────────────────────────
VTS_WS_URL: str = os.getenv("VTS_WS_URL", "ws://localhost:8001")
VTS_PLUGIN_NAME: str = os.getenv("VTS_PLUGIN_NAME", "Ria")
VTS_PLUGIN_DEVELOPER: str = os.getenv("VTS_PLUGIN_DEVELOPER", "RiaDev")

# ── 감정 → 모션 기본 매핑 ─────────────────────────────────
_DEFAULT_EMOTION_MOTION_MAP: dict[str, str] = {
    "기쁨":  "motion_happy",
    "슬픔":  "motion_sad",
    "분노":  "motion_angry",
    "불안":  "motion_nervous",
    "당황":  "motion_surprised",
    "중립":  "motion_idle",
}

# .env의 VTS_MOTION_<감정> 으로 오버라이드된 최종 매핑
def _build_emotion_motion_map() -> dict[str, str]:
    """환경 변수로 오버라이드된 감정-모션 매핑 테이블을 구성한다."""
    result: dict[str, str] = dict(_DEFAULT_EMOTION_MOTION_MAP)
    for emotion, default_motion in _DEFAULT_EMOTION_MOTION_MAP.items():
        env_key = f"VTS_MOTION_{emotion}"
        overridden = os.getenv(env_key)
        if overridden:
            result[emotion] = overridden
            logger.debug(
                "모션 오버라이드: {emotion} → {motion} (env={key})",
                emotion=emotion,
                motion=overridden,
                key=env_key,
            )
    return result

EMOTION_MOTION_MAP: dict[str, str] = _build_emotion_motion_map()
_DEFAULT_MOTION: str = "motion_idle"

# ── 모듈 레벨 싱글턴 ───────────────────────────────────────
_ws = None                   # websockets.WebSocketClientProtocol
_auth_token: Optional[str] = os.getenv("VTS_AUTH_TOKEN") or None
_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_thread: Optional[threading.Thread] = None


# ── 이벤트 루프 관리 ───────────────────────────────────────
def _get_or_create_loop() -> asyncio.AbstractEventLoop:
    """전용 백그라운드 이벤트 루프를 반환한다. 없으면 생성 후 스레드에서 실행한다."""
    global _loop, _loop_thread

    if _loop is not None and _loop.is_running():
        return _loop

    _loop = asyncio.new_event_loop()

    def _run_loop(loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    _loop_thread = threading.Thread(target=_run_loop, args=(_loop,), daemon=True)
    _loop_thread.start()
    logger.debug("VTS 전용 이벤트 루프 시작 (IS_WINDOWS={v})", v=IS_WINDOWS)
    return _loop


def _run_async(coro) -> object:
    """백그라운드 루프에서 코루틴을 동기적으로 실행하고 결과를 반환한다."""
    loop = _get_or_create_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=15)


# ── VTS 요청/응답 ─────────────────────────────────────────
async def _send_request(data: dict) -> dict:
    """VTS WebSocket에 JSON 요청을 보내고 응답을 반환한다.

    Args:
        data: 전송할 VTS API 페이로드 (messageType, data 포함)

    Returns:
        VTS로부터 받은 응답 딕셔너리

    Raises:
        RuntimeError: WebSocket 연결이 없을 때
        ConnectionError: 응답 수신 실패 시
    """
    global _ws

    if _ws is None:
        raise RuntimeError("VTS WebSocket 미연결 — connect()를 먼저 호출하세요.")

    payload = {
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": str(uuid.uuid4()),
        **data,
    }
    raw = json.dumps(payload, ensure_ascii=False)
    logger.debug(
        "VTS 요청 | type={msg_type} | id={req_id}",
        msg_type=payload.get("messageType"),
        req_id=payload["requestID"],
    )

    try:
        await _ws.send(raw)
        response_raw = await _ws.recv()
        response: dict = json.loads(response_raw)
    except Exception as e:
        logger.error("VTS 요청/응답 실패: {e}", e=e)
        raise ConnectionError(f"VTS 통신 오류: {e}") from e

    logger.debug(
        "VTS 응답 | type={msg_type}",
        msg_type=response.get("messageType"),
    )
    return response


# ── 인증 헬퍼 ─────────────────────────────────────────────
async def _request_auth_token() -> str:
    """AuthenticationTokenRequest를 전송하여 신규 토큰을 발급받는다.

    VTS 팝업에서 사용자가 허용해야 토큰이 발급된다.

    Returns:
        발급된 인증 토큰 문자열

    Raises:
        RuntimeError: 토큰 발급 실패 시
    """
    logger.info("VTS 토큰 발급 요청 — VTube Studio에서 팝업을 허용하세요.")
    response = await _send_request({
        "messageType": "AuthenticationTokenRequest",
        "data": {
            "pluginName": VTS_PLUGIN_NAME,
            "pluginDeveloper": VTS_PLUGIN_DEVELOPER,
        },
    })

    token: Optional[str] = response.get("data", {}).get("authenticationToken")
    if not token:
        raise RuntimeError(
            f"토큰 발급 실패 — VTS 응답: {response.get('data', {})}"
        )

    logger.info("VTS 토큰 발급 성공")
    return token


async def _authenticate_with_token(token: str) -> bool:
    """발급된 토큰으로 Authentication 요청을 수행한다.

    Args:
        token: 사전 발급된 VTS 인증 토큰

    Returns:
        인증 성공 여부
    """
    masked = token[:6] + "****" if len(token) > 6 else "****"
    logger.debug("VTS 인증 시도 | token={masked}", masked=masked)

    response = await _send_request({
        "messageType": "AuthenticationRequest",
        "data": {
            "pluginName": VTS_PLUGIN_NAME,
            "pluginDeveloper": VTS_PLUGIN_DEVELOPER,
            "authenticationToken": token,
        },
    })

    # APIError 응답 명시적 처리
    if "Error" in response.get("messageType", ""):
        error_id = response.get("data", {}).get("errorID")
        msg = response.get("data", {}).get("message", "")
        logger.warning(
            "VTS AuthenticationRequest APIError | errorID={eid} | msg={msg}",
            eid=error_id,
            msg=msg,
        )
        return False

    authenticated: bool = response.get("data", {}).get("authenticated", False)
    if authenticated:
        logger.info("VTS 인증 성공 | token={masked}", masked=masked)
    else:
        reason = response.get("data", {}).get("reason", "알 수 없음")
        logger.warning("VTS 인증 실패 | reason={reason}", reason=reason)
    return authenticated


def _save_auth_token(token: str) -> None:
    """토큰을 .env 파일의 VTS_AUTH_TOKEN에 저장한다.

    Args:
        token: 저장할 인증 토큰
    """
    masked = token[:6] + "****" if len(token) > 6 else "****"
    try:
        set_key(str(_ENV_PATH), "VTS_AUTH_TOKEN", token)
        logger.info("VTS 토큰 .env 저장 완료 | masked={masked}", masked=masked)
    except Exception as e:
        logger.warning("VTS 토큰 .env 저장 실패: {e}", e=e)


# ── 연결/해제 ─────────────────────────────────────────────
async def _connect_async() -> None:
    """WebSocket 연결 및 VTS 인증을 비동기로 수행한다."""
    global _ws, _auth_token

    try:
        import websockets
    except ImportError as e:
        logger.error("websockets 패키지 없음: {e}", e=e)
        raise

    logger.info("VTS WebSocket 연결 시도 | url={url}", url=VTS_WS_URL)

    try:
        _ws = await websockets.connect(VTS_WS_URL)
    except (ConnectionRefusedError, OSError) as e:
        logger.error("VTS 연결 거부 — VTube Studio가 실행 중인지 확인하세요: {e}", e=e)
        raise

    logger.info("VTS WebSocket 연결 성공")

    # 기존 토큰으로 인증 시도
    if _auth_token:
        success = await _authenticate_with_token(_auth_token)
        if success:
            return
        logger.warning("기존 토큰 인증 실패 — 새 토큰 발급을 시도합니다.")

    # 신규 토큰 발급
    _auth_token = await _request_auth_token()
    _save_auth_token(_auth_token)

    success = await _authenticate_with_token(_auth_token)
    if not success:
        raise RuntimeError("VTS 신규 토큰으로 인증 실패")


async def _disconnect_async() -> None:
    """WebSocket 연결을 비동기로 종료한다."""
    global _ws

    if _ws is None:
        logger.debug("VTS 이미 연결 해제 상태")
        return

    try:
        await _ws.close()
        logger.info("VTS WebSocket 연결 종료")
    except Exception as e:
        logger.warning("VTS 연결 종료 중 오류: {e}", e=e)
    finally:
        _ws = None


# ── 공개 함수 ─────────────────────────────────────────────
def connect() -> None:
    """VTube Studio WebSocket에 연결하고 인증을 수행한다.

    .env의 VTS_AUTH_TOKEN이 있으면 기존 토큰으로 인증한다.
    없거나 실패하면 AuthenticationTokenRequest를 통해 신규 토큰을 발급한다.

    Raises:
        ConnectionRefusedError: VTube Studio가 실행 중이지 않을 때
        RuntimeError: 인증 실패 시
    """
    _run_async(_connect_async())


def disconnect() -> None:
    """VTube Studio WebSocket 연결을 종료한다."""
    _run_async(_disconnect_async())


def is_connected() -> bool:
    """현재 VTS WebSocket 연결 상태를 반환한다.

    Returns:
        연결 중이면 True, 아니면 False
    """
    if _ws is None:
        return False
    # websockets 라이브러리의 연결 상태 확인
    try:
        return not _ws.closed
    except AttributeError:
        return _ws is not None


def trigger_motion(motion_name: str) -> bool:
    """VTS HotkeyTriggerRequest로 지정된 모션을 트리거한다.

    Args:
        motion_name: 트리거할 모션(핫키) 이름

    Returns:
        성공 시 True, 실패 시 False
    """
    async def _trigger() -> bool:
        try:
            response = await _send_request({
                "messageType": "HotkeyTriggerRequest",
                "data": {
                    "hotkeyID": motion_name,
                },
            })
            msg_type: str = response.get("messageType", "")
            if "Error" in msg_type:
                error_id = response.get("data", {}).get("errorID")
                error_msg = response.get("data", {}).get("message", "")
                logger.warning(
                    "모션 트리거 실패 | motion={motion} | errorID={eid} | msg={msg}",
                    motion=motion_name,
                    eid=error_id,
                    msg=error_msg,
                )
                return False
            logger.info("모션 트리거 성공 | motion={motion}", motion=motion_name)
            return True
        except Exception as e:
            logger.error("모션 트리거 예외 | motion={motion} | {e}", motion=motion_name, e=e)
            return False

    try:
        return bool(_run_async(_trigger()))
    except Exception as e:
        logger.error("trigger_motion 실행 실패: {e}", e=e)
        return False


def trigger_expression(expression_name: str, active: bool = True) -> bool:
    """VTS ExpressionActivationRequest로 표정을 전환한다.

    Args:
        expression_name: 표정 파일명 (예: "happy.exp3.json")
        active: True면 활성화, False면 비활성화

    Returns:
        성공 시 True, 실패 시 False
    """
    async def _trigger_expr() -> bool:
        try:
            response = await _send_request({
                "messageType": "ExpressionActivationRequest",
                "data": {
                    "expressionFile": expression_name,
                    "active": active,
                },
            })
            msg_type: str = response.get("messageType", "")
            if "Error" in msg_type:
                error_id = response.get("data", {}).get("errorID")
                error_msg = response.get("data", {}).get("message", "")
                logger.warning(
                    "표정 전환 실패 | expr={expr} | active={active} | errorID={eid} | msg={msg}",
                    expr=expression_name,
                    active=active,
                    eid=error_id,
                    msg=error_msg,
                )
                return False
            logger.info(
                "표정 전환 성공 | expr={expr} | active={active}",
                expr=expression_name,
                active=active,
            )
            return True
        except Exception as e:
            logger.error(
                "표정 전환 예외 | expr={expr} | {e}",
                expr=expression_name,
                e=e,
            )
            return False

    try:
        return bool(_run_async(_trigger_expr()))
    except Exception as e:
        logger.error("trigger_expression 실행 실패: {e}", e=e)
        return False


def react_to_emotion(result: EmotionResult) -> bool:
    """EmotionResult를 받아 감정에 대응하는 모션을 트리거한다.

    score가 0.5 미만이면 확신도가 낮다고 판단하여 기본 모션(motion_idle)을 사용한다.

    Args:
        result: emotion.py의 analyze()가 반환한 EmotionResult

    Returns:
        trigger_motion() 결과 (성공 True, 실패 False)
    """
    if result.score < 0.5:
        motion = _DEFAULT_MOTION
        logger.info(
            "감정 확신도 낮음 (score={score:.2f}) — 기본 모션 사용: {motion}",
            score=result.score,
            motion=motion,
        )
    else:
        motion = EMOTION_MOTION_MAP.get(result.label, _DEFAULT_MOTION)
        logger.info(
            "감정 반응 | label={label} | score={score:.2f} | motion={motion}",
            label=result.label,
            score=result.score,
            motion=motion,
        )

    return trigger_motion(motion)


def get_current_model() -> Optional[dict]:
    """현재 VTube Studio에 로드된 모델 정보를 반환한다.

    Returns:
        {"modelName": str, "modelID": str} 형태의 딕셔너리,
        실패 시 None
    """
    async def _get_model() -> Optional[dict]:
        try:
            response = await _send_request({
                "messageType": "CurrentModelRequest",
                "data": {},
            })
            msg_type: str = response.get("messageType", "")
            if "Error" in msg_type:
                error_id = response.get("data", {}).get("errorID")
                logger.warning(
                    "모델 정보 조회 실패 | errorID={eid}",
                    eid=error_id,
                )
                return None
            data = response.get("data", {})
            model_info = {
                "modelName": data.get("modelName", ""),
                "modelID": data.get("modelID", ""),
            }
            logger.info(
                "현재 VTS 모델 | name={name} | id={mid}",
                name=model_info["modelName"],
                mid=model_info["modelID"],
            )
            return model_info
        except Exception as e:
            logger.error("모델 정보 조회 예외: {e}", e=e)
            return None

    try:
        return _run_async(_get_model())  # type: ignore[return-value]
    except Exception as e:
        logger.error("get_current_model 실행 실패: {e}", e=e)
        return None


# ── 단독 테스트 ────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=== character.py 단독 테스트 시작 ===")

    # --- 정상 케이스 (VTS 실행 중인 경우) ---
    logger.info("--- 정상 케이스: VTS 연결 시도 ---")
    try:
        connect()
        logger.info("connect() 성공 | is_connected={v}", v=is_connected())

        # 모델 정보 조회
        model = get_current_model()
        if model:
            logger.info("현재 모델: {model}", model=model)
        else:
            logger.warning("모델 정보 없음 (VTS에 모델이 로드되지 않았을 수 있음)")

        # 기본 모션 트리거
        result_motion = trigger_motion("motion_idle")
        logger.info("trigger_motion('motion_idle') 결과: {v}", v=result_motion)

        # 감정 반응 — 기쁨, score 높음
        emotion_joy = EmotionResult(label="기쁨", score=0.9, raw={})
        result_joy = react_to_emotion(emotion_joy)
        logger.info("react_to_emotion(기쁨, 0.9) 결과: {v}", v=result_joy)

        disconnect()
        logger.info("disconnect() 완료 | is_connected={v}", v=is_connected())

    except (ConnectionRefusedError, OSError) as e:
        logger.warning("VTS 미실행 — 정상 케이스 스킵: {e}", e=e)
    except Exception as e:
        logger.error("정상 케이스 예외: {e}", e=e)

    # --- 에러 케이스 1: VTS 미실행 시 연결 오류 ---
    logger.info("--- 에러 케이스 1: VTS 미실행 상태 연결 시도 ---")
    # disconnect 후 재연결 시도하여 오류 처리 확인
    try:
        # 이미 disconnect 상태이므로 is_connected()가 False여야 함
        assert not is_connected(), "연결 해제 후 is_connected()가 False여야 함"
        logger.info("is_connected() = False 확인 완료")
    except AssertionError as e:
        logger.warning("is_connected 상태 불일치: {e}", e=e)

    # --- 에러 케이스 2: 연결 없이 trigger_motion 호출 ---
    logger.info("--- 에러 케이스 2: 미연결 상태 trigger_motion ---")
    try:
        result_no_conn = trigger_motion("motion_test")
        if not result_no_conn:
            logger.info("미연결 상태 trigger_motion → False 반환 확인")
        else:
            logger.warning("예상치 못한 True 반환")
    except Exception as e:
        logger.info("미연결 예외 정상 처리: {e}", e=e)

    # --- 에러 케이스 3: 알 수 없는 감정 레이블 → idle 폴백 ---
    logger.info("--- 에러 케이스 3: 알 수 없는 감정 → idle 폴백 ---")
    unknown_emotion = EmotionResult(label="알수없음", score=0.3, raw={})
    # score 0.3 < 0.5 이므로 motion_idle 사용해야 함
    logger.info(
        "EMOTION_MOTION_MAP.get('알수없음', default) = {v}",
        v=EMOTION_MOTION_MAP.get("알수없음", _DEFAULT_MOTION),
    )
    # react_to_emotion은 미연결 상태이므로 False 반환 예상
    try:
        result_unknown = react_to_emotion(unknown_emotion)
        logger.info(
            "react_to_emotion(알수없음, score=0.3) → {v} (idle 폴백 적용됨)",
            v=result_unknown,
        )
    except Exception as e:
        logger.info("react_to_emotion 예외 정상 처리: {e}", e=e)

    # --- 에러 케이스 4: score 높은 알수없음 레이블 → idle 폴백 ---
    logger.info("--- 에러 케이스 4: score 높지만 매핑 없는 레이블 → idle 폴백 ---")
    unknown_high = EmotionResult(label="알수없음", score=0.8, raw={})
    expected_motion = EMOTION_MOTION_MAP.get("알수없음", _DEFAULT_MOTION)
    assert expected_motion == _DEFAULT_MOTION, (
        f"매핑 없는 레이블은 기본 모션({_DEFAULT_MOTION})이어야 함: {expected_motion}"
    )
    logger.info(
        "알수없음(score=0.8) 모션 폴백 확인: {motion}",
        motion=expected_motion,
    )

    logger.info("=== character.py 단독 테스트 완료 ===")
