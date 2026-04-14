"""
main.py — Ria AI 어시스턴트 통합 파이프라인

STT → 감정 분석 → 기억 검색 → LLM(+Tools) → TTS → 캐릭터 모션 → 기억 저장

실행:
    python main.py           — 전체 파이프라인 (HyperX QuadCast 마이크 필요)
    python main.py --no-hw   — 텍스트 입력 모드 (마이크 없이 테스트)

종료: Ctrl+C
"""

import queue
import re
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from loguru import logger

from config import BASE_DIR
import modules.character as character
import modules.emotion as emotion
import modules.memory as memory
import modules.stt as stt
import modules.tts as tts
from modules.llm import is_ollama_running, query, query_stream
from modules.scheduler import get_scheduler, init_scheduler


# ── 파이프라인 상수 ───────────────────────────────────────────────────────────

HISTORY_LIMIT: int = 10         # 유지할 대화 이력 최대 턴 수 (양방향)
MEMORY_CONTEXT_N: int = 3       # 기억 검색 상위 N개

_SYSTEM_PROMPT_PATH = BASE_DIR / "data" / "prompts" / "system.txt"


def _load_system_prompt() -> str:
    """data/prompts/system.txt를 읽어 시스템 프롬프트 문자열을 반환한다.

    파일이 없거나 읽기 실패 시 하드코딩된 기본값을 반환한다.
    """
    try:
        text = _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
        logger.info("시스템 프롬프트 로드 | path={p}", p=_SYSTEM_PROMPT_PATH)
        return text
    except FileNotFoundError:
        logger.warning(
            "system.txt 없음 ({p}) → 기본 프롬프트 사용",
            p=_SYSTEM_PROMPT_PATH,
        )
    except Exception as e:
        logger.warning("system.txt 읽기 실패 → 기본 프롬프트 사용: {e}", e=e)

    return (
        "너는 리아야. 귀엽고 친근한 여동생 같은 스타일로 반말로 짧게 대화해."
    )


_SYSTEM_BASE: str = _load_system_prompt()

# "종료"로 인식하는 키워드
_EXIT_KEYWORDS: frozenset[str] = frozenset({"종료", "그만", "exit", "quit", "bye"})


# ── 런타임 상태 ───────────────────────────────────────────────────────────────

_history: list[dict] = []           # LLM 대화 이력
_hyperx_device: Optional[int] = None
_vts_connected: bool = False


# ── 초기화 헬퍼 ──────────────────────────────────────────────────────────────

def _init_stt() -> None:
    """HyperX QuadCast 장치 인덱스를 탐색한다."""
    global _hyperx_device
    _hyperx_device = stt.find_hyperx_device_index()
    if _hyperx_device is not None:
        logger.info("HyperX QuadCast 감지 | device_index={idx}", idx=_hyperx_device)
    else:
        logger.warning("HyperX QuadCast 미감지 → 시스템 기본 마이크 사용")


def _init_character() -> None:
    """VTube Studio WebSocket에 연결을 시도한다. 실패해도 계속 진행한다."""
    global _vts_connected
    try:
        character.connect()
        _vts_connected = character.is_connected()
        logger.info(
            "VTube Studio | connected={v}",
            v=_vts_connected,
        )
    except Exception as e:
        logger.warning("VTS 연결 실패 (캐릭터 모션 비활성화): {e}", e=e)
        _vts_connected = False


def _init_tts() -> None:
    """GPT-SoVITS 서버 기동 및 가중치를 로드한다."""
    try:
        tts.ensure_server()
        logger.info("TTS 서버 준비 완료")
    except Exception as e:
        logger.warning("TTS 초기화 실패 (TTS 비활성화될 수 있음): {e}", e=e)


def _init_models() -> None:
    """감정·기억 모델을 미리 로드해 첫 발화 지연과 초기화 오류를 방지한다."""
    try:
        emotion.load_model()
        logger.info("감정 모델 워밍업 완료")
    except Exception as e:
        logger.warning("감정 모델 워밍업 실패 (무시): {e}", e=e)

    try:
        memory._get_collection()
        logger.info("기억 DB 워밍업 완료")
    except Exception as e:
        logger.warning("기억 DB 워밍업 실패 (무시): {e}", e=e)


def _init_scheduler() -> None:
    """APScheduler 백그라운드 스케줄러를 시작한다."""
    init_scheduler(
        on_speak=tts.speak,
        boredom_check_interval_sec=60,
        autonomous_action_cooldown_min=10,
    )
    logger.info("스케줄러 백그라운드 시작")


def _warmup_llm() -> None:
    """Ollama 모델을 GPU에 미리 로드하여 첫 응답 콜드스타트를 제거한다."""
    try:
        t0 = time.time()
        query(".", system=".", history=None)
        logger.info("LLM 워밍업 완료 | {t:.1f}초", t=time.time() - t0)
    except Exception as e:
        logger.warning("LLM 워밍업 실패 (무시): {e}", e=e)


def _build_system_prompt(emotion_hint: str, memory_context: str) -> str:
    """기본 시스템 프롬프트에 감정 힌트와 기억 컨텍스트를 결합한다."""
    parts: list[str] = [_SYSTEM_BASE]
    if emotion_hint:
        parts.append(emotion_hint)
    if memory_context:
        parts.append(f"[관련 기억]\n{memory_context}")
    return "\n".join(parts)


# ── 파이프라인 단계 ───────────────────────────────────────────────────────────

def _step_stt(text_mode: bool) -> Optional[str]:
    """음성 입력 또는 텍스트 입력으로 사용자 발화를 반환한다."""
    if text_mode:
        try:
            user_input = input("\n[입력] >> ").strip()
        except EOFError:
            return None
        return user_input or None

    logger.info("VAD 대기 중... (말씀하세요)")
    try:
        text = stt.listen_and_transcribe(
            device_index=_hyperx_device,
        )
    except Exception as e:
        logger.error("STT 실패: {e}", e=e)
        return None

    if not text or not text.strip():
        logger.debug("STT 결과 없음 (무음 또는 잡음)")
        return None

    logger.info("STT 결과: {text}", text=text)
    return text.strip()


def _step_emotion(
    text: str,
) -> tuple[str, Optional[emotion.EmotionResult]]:
    """텍스트에서 감정을 분석하고 (프롬프트 힌트, EmotionResult)를 반환한다."""
    try:
        result = emotion.analyze(text)
        hint = emotion.to_prompt_hint(result)
        logger.info(
            "감정 분석 | label={label} | score={score:.2f}",
            label=result.label,
            score=result.score,
        )
        return hint, result
    except Exception as e:
        logger.warning("감정 분석 실패 (무시): {e}", e=e)
        return "", None


def _step_memory_search(text: str) -> str:
    """관련 기억을 검색해 컨텍스트 문자열로 반환한다."""
    try:
        results = memory.search(text, n_results=MEMORY_CONTEXT_N)
        if not results:
            return ""
        lines = [f"- [{r.role}] {r.content}" for r in results]
        return "\n".join(lines)
    except Exception as e:
        logger.warning("기억 검색 실패 (무시): {e}", e=e)
        return ""


_META_TAG_PATTERN: re.Pattern = re.compile(
    r"\[사용자 감정:[^\]]*\]"   # [사용자 감정: 기쁨(0.92) — ...] 형태
    r"|\[관련 기억\].*?(?=\n\S|\Z)",  # [관련 기억] 블록 (혹시 새어 나오는 경우 대비)
    re.DOTALL,
)


def _clean_response(text: str) -> str:
    """LLM 응답에서 시스템 프롬프트 메타 태그를 제거한다.

    시스템 프롬프트에 삽입한 [사용자 감정: ...], [관련 기억] 블록이
    LLM 응답에 그대로 포함되는 경우를 처리한다.
    LLM에 전달하는 프롬프트 자체는 변경하지 않는다.

    Args:
        text: LLM 원본 응답 문자열

    Returns:
        메타 태그가 제거된 문자열 (앞뒤 공백 정리 포함)
    """
    cleaned = _META_TAG_PATTERN.sub("", text).strip()
    if cleaned != text.strip():
        logger.debug("메타 태그 제거 | before={b!r} → after={a!r}", b=text[:60], a=cleaned[:60])
    return cleaned


_SENTENCE_DELIMITERS = re.compile(r"(?<=[.!?。！？~])\s*|(?<=\n)")
_SENTINEL = None  # 큐 종료 신호


def _tts_worker(q: queue.Queue) -> None:
    """TTS 전용 스레드: 큐에서 문장을 꺼내 순서대로 재생한다."""
    tts.begin_session()
    try:
        while True:
            sentence = q.get()
            if sentence is _SENTINEL:
                break
            try:
                tts.speak_direct(sentence)
            except Exception as e:
                logger.error("TTS 실패: {e}", e=e)
    finally:
        tts.end_session()


def _step_llm_stream_tts(user_text: str, system_prompt: str) -> Optional[str]:
    """LLM 스트리밍 응답을 문장 단위로 TTS에 즉시 전달한다.

    LLM 읽기(메인 스레드)와 TTS 재생(워커 스레드)을 분리하여
    TTS 재생 중에도 LLM 스트림이 끊기지 않는다.
    """
    recent_history = _history[-(HISTORY_LIMIT * 2):] if _history else None

    try:
        tts.ensure_server()
    except Exception as e:
        logger.error("TTS 서버 준비 실패: {e}", e=e)

    tts_queue: queue.Queue = queue.Queue()
    worker = threading.Thread(target=_tts_worker, args=(tts_queue,), daemon=True)
    worker.start()

    buf = ""
    full_response = ""

    try:
        for token in query_stream(
            prompt=user_text,
            system=system_prompt,
            history=recent_history,
        ):
            buf += token
            full_response += token

            parts = _SENTENCE_DELIMITERS.split(buf)
            if len(parts) > 1:
                sentence = parts[0].strip()
                buf = parts[-1]
                if sentence:
                    logger.debug("LLM→TTS 큐 | sentence={s}", s=sentence[:40])
                    tts_queue.put(sentence)

        leftover = buf.strip()
        if leftover:
            tts_queue.put(leftover)

    except ConnectionError as e:
        logger.error("Ollama 미실행: {e}", e=e)
        full_response = "죄송해요, 지금 LLM 서버에 연결할 수 없어요."
    except Exception as e:
        logger.error("LLM 오류: {e}", e=e)
    finally:
        tts_queue.put(_SENTINEL)
        worker.join(timeout=120)

    return full_response or None


def _step_character(emotion_result: Optional[emotion.EmotionResult]) -> None:
    """VTS가 연결된 경우 감정에 맞는 캐릭터 모션을 트리거한다."""
    if not _vts_connected or emotion_result is None:
        return
    try:
        character.react_to_emotion(emotion_result)
    except Exception as e:
        logger.warning("캐릭터 모션 실패 (무시): {e}", e=e)


def _step_memory_save(user_text: str, assistant_text: str) -> None:
    """사용자 발화와 어시스턴트 응답을 장기 기억에 저장한다."""
    try:
        memory.add_message("user", user_text)
        memory.add_message("assistant", assistant_text)
    except Exception as e:
        logger.warning("기억 저장 실패 (무시): {e}", e=e)


# ── 핵심 파이프라인 ───────────────────────────────────────────────────────────

def run_pipeline_turn(user_text: str) -> Optional[str]:
    """파이프라인 한 턴을 실행하고 어시스턴트 응답을 반환한다.

    단계: 감정 분석 → 기억 검색 → LLM → TTS → 캐릭터 모션 → 기억 저장

    Args:
        user_text: STT 또는 텍스트 입력으로 얻은 사용자 발화

    Returns:
        어시스턴트 응답 문자열. LLM 오류 시 None.
    """
    # 1+2. 감정 분석 + 기억 검색 (병렬)
    with ThreadPoolExecutor(max_workers=2) as pool:
        emotion_future = pool.submit(_step_emotion, user_text)
        memory_future = pool.submit(_step_memory_search, user_text)
        emotion_hint, emotion_result = emotion_future.result()
        memory_context = memory_future.result()

    # 3. 시스템 프롬프트 조합
    system_prompt = _build_system_prompt(emotion_hint, memory_context)

    # 4. LLM 스트리밍 + TTS 파이프라이닝 (문장 단위 즉시 재생)
    response = _step_llm_stream_tts(user_text, system_prompt)
    if not response:
        return None

    response = _clean_response(response)
    logger.info("LLM 응답 (앞 80자): {text}", text=response[:80])

    # 5. 캐릭터 모션
    _step_character(emotion_result)

    # 7. 기억 저장
    _step_memory_save(user_text, response)

    # 8. 대화 이력 갱신 + 디스크 저장
    _history.append({"role": "user",      "content": user_text})
    _history.append({"role": "assistant", "content": response})

    # 9. 스케줄러 상호작용 타이머 리셋
    sc = get_scheduler()
    if sc:
        sc.update_last_interaction()

    return response


# ── 종료 처리 ─────────────────────────────────────────────────────────────────

def _shutdown() -> None:
    """스케줄러 종료, VTS 연결 해제를 수행한다."""
    logger.info("Ria 종료 중...")

    sc = get_scheduler()
    if sc and sc._running:
        sc.stop()
        logger.info("스케줄러 종료")

    if _vts_connected:
        try:
            character.disconnect()
            logger.info("VTS 연결 해제")
        except Exception as e:
            logger.warning("VTS 해제 중 오류 (무시): {e}", e=e)

    logger.info("Ria 종료 완료")


# ── 메인 엔트리 ──────────────────────────────────────────────────────────────

def main(text_mode: bool = False) -> None:
    """Ria 메인 루프.

    Args:
        text_mode: True면 마이크 대신 콘솔 텍스트 입력을 사용한다.
    """
    logger.info("=== Ria AI 어시스턴트 시작 ===")

    # ── 사전 확인 ──────────────────────────────────────────
    if not is_ollama_running():
        logger.warning("Ollama 서버 미실행 — LLM 응답 불가. 계속하려면 Ollama를 실행하세요.")

    # ── 모듈 초기화 ────────────────────────────────────────
    if not text_mode:
        _init_stt()
    _init_tts()
    _init_models()
    _init_character()
    _init_scheduler()
    _warmup_llm()

    # ── Ctrl+C 핸들러 ──────────────────────────────────────
    def _handle_sigint(sig: int, frame: object) -> None:
        print()
        _shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_sigint)

    # ── 시작 안내 ──────────────────────────────────────────
    mode_label = (
        "텍스트 입력 모드 (--no-hw)"
        if text_mode
        else f"음성 입력 모드 | HyperX device={_hyperx_device}"
    )
    logger.info("파이프라인 시작 | {mode}", mode=mode_label)

    print(f"\n{'=' * 52}")
    print(f"  Ria 가동 중 - {mode_label}")
    print(f"  종료 키워드: {', '.join(sorted(_EXIT_KEYWORDS))}")
    print(f"  종료 단축키: Ctrl+C")
    print(f"{'=' * 52}\n")

    # ── 메인 루프 ──────────────────────────────────────────
    while True:
        user_text = _step_stt(text_mode)

        if user_text is None:
            if text_mode:
                break  # EOF → 루프 종료
            time.sleep(0.1)  # 무음 시 CPU 점유 방지
            continue

        if user_text.lower().strip() in _EXIT_KEYWORDS:
            try:
                tts.speak("안녕히 계세요!")
            except Exception:
                pass
            print("\nRia: 안녕히 계세요!\n")
            break

        print(f"사용자: {user_text}")

        response = run_pipeline_turn(user_text)
        if response:
            print(f"Ria:   {response}\n")
        else:
            logger.warning("파이프라인 응답 없음 — 다음 턴으로 진행")

    _shutdown()


if __name__ == "__main__":
    text_mode: bool = "--no-hw" in sys.argv
    main(text_mode=text_mode)
