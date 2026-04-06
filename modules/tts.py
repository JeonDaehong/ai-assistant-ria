"""
modules/tts.py — 텍스트 → 음성 (edge-tts + pygame)

edge-tts (Microsoft Edge TTS)로 한국어 음성을 합성하고,
pygame.mixer로 재생한다. 비동기 합성 + 동기 재생 인터페이스 제공.
"""
import asyncio
import os
from pathlib import Path
from typing import Optional

import edge_tts
import pygame
from loguru import logger

from config import BASE_DIR, IS_WINDOWS

# ── 기본 설정 ─────────────────────────────────────────────
EDGE_TTS_VOICE: str = os.getenv("EDGE_TTS_VOICE", "ko-KR-SunHiNeural")
_DEFAULT_OUTPUT: Path = BASE_DIR / "temp_tts.mp3"


# ── 음성 목록 ─────────────────────────────────────────────

def list_voices(language: str = "ko") -> list[dict]:
    """edge-tts에서 사용 가능한 음성 목록을 반환한다.

    Args:
        language: 필터링할 언어 코드 (기본값 "ko" → 한국어)

    Returns:
        각 항목이 {"name": str, "gender": str, "locale": str} 형태인 리스트
    """
    voices_raw: list[dict] = asyncio.run(_fetch_voices())
    filtered = [
        {
            "name": v["ShortName"],
            "gender": v["Gender"],
            "locale": v["Locale"],
        }
        for v in voices_raw
        if v["Locale"].lower().startswith(language.lower())
    ]
    logger.info(
        "음성 목록 조회 완료 | language={lang} | count={n}",
        lang=language,
        n=len(filtered),
    )
    return filtered


async def _fetch_voices() -> list[dict]:
    """edge-tts에서 전체 음성 목록을 비동기로 가져온다."""
    voices = await edge_tts.list_voices()
    return voices


# ── 합성 ──────────────────────────────────────────────────

def synthesize(
    text: str,
    output_path: Optional[Path] = None,
    voice: str = EDGE_TTS_VOICE,
) -> Path:
    """텍스트를 mp3 파일로 합성한다.

    Args:
        text: 합성할 텍스트 (빈 문자열 불가)
        output_path: 저장 경로. None이면 BASE_DIR/temp_tts.mp3 사용
        voice: edge-tts 음성 이름

    Returns:
        저장된 mp3 파일의 Path

    Raises:
        ValueError: text가 빈 문자열이거나 공백만인 경우
        RuntimeError: 합성 실패 시
    """
    if not text or not text.strip():
        raise ValueError("합성할 텍스트가 비어 있습니다.")

    dest: Path = output_path if output_path is not None else _DEFAULT_OUTPUT

    logger.debug(
        "TTS 합성 시작 | voice={voice} | text_len={n} | dest={dest}",
        voice=voice,
        n=len(text),
        dest=dest,
    )

    asyncio.run(_synthesize_async(text, dest, voice))

    logger.info("TTS 합성 완료 | dest={dest}", dest=dest)
    return dest


async def _synthesize_async(text: str, dest: Path, voice: str) -> None:
    """edge-tts 비동기 합성 코어.

    Args:
        text: 합성할 텍스트
        dest: 저장 경로
        voice: 음성 이름

    Raises:
        RuntimeError: edge-tts 오류 발생 시
    """
    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(dest))
    except Exception as e:
        raise RuntimeError(f"edge-tts 합성 오류: {e}") from e


# ── 재생 ──────────────────────────────────────────────────

def play(file_path: Path) -> None:
    """mp3 파일을 pygame.mixer로 재생한다. 재생 완료까지 블로킹.

    Args:
        file_path: 재생할 mp3 파일 경로

    Raises:
        FileNotFoundError: 파일이 존재하지 않을 경우
        RuntimeError: pygame 초기화 또는 재생 오류 시
    """
    if not file_path.exists():
        raise FileNotFoundError(f"재생 파일이 없습니다: {file_path}")

    logger.debug("오디오 재생 시작 | file={file}", file=file_path)

    try:
        pygame.mixer.init()
        pygame.mixer.music.load(str(file_path))
        pygame.mixer.music.play()

        # 재생 완료까지 블로킹 (100ms 간격으로 폴링)
        clock = pygame.time.Clock()
        while pygame.mixer.music.get_busy():
            clock.tick(10)

        logger.info("오디오 재생 완료 | file={file}", file=file_path)
    except Exception as e:
        raise RuntimeError(f"pygame 재생 오류: {e}") from e
    finally:
        _safe_mixer_quit()


def _safe_mixer_quit() -> None:
    """pygame.mixer가 초기화된 경우에만 종료한다."""
    try:
        if pygame.mixer.get_init():
            pygame.mixer.quit()
    except Exception as e:
        logger.warning("pygame.mixer 종료 중 오류 (무시): {e}", e=e)


# ── 통합 인터페이스 ───────────────────────────────────────

def speak(text: str, voice: str = EDGE_TTS_VOICE) -> None:
    """텍스트를 합성 후 재생한다. 재생 완료 후 임시 파일을 삭제한다.

    Args:
        text: 읽을 텍스트
        voice: edge-tts 음성 이름

    Raises:
        ValueError: text가 빈 문자열이거나 공백만인 경우
    """
    mp3_path = synthesize(text, voice=voice)
    try:
        play(mp3_path)
    finally:
        _cleanup_temp_file(mp3_path)


def _cleanup_temp_file(file_path: Path) -> None:
    """임시 파일을 삭제한다. 삭제 실패 시 경고만 기록하고 진행한다."""
    try:
        if file_path.exists():
            file_path.unlink()
            logger.debug("임시 파일 삭제 | file={file}", file=file_path)
    except Exception as e:
        logger.warning("임시 파일 삭제 실패 (무시): {file} | {e}", file=file_path, e=e)


def speak_async(text: str, voice: str = EDGE_TTS_VOICE) -> None:
    """비동기 기반 TTS를 동기 인터페이스로 제공한다.

    내부적으로 asyncio.run()을 사용하므로 외부에서는 동기 함수처럼 호출한다.
    합성은 비동기, 재생은 pygame 동기 블로킹으로 처리한다.

    Args:
        text: 읽을 텍스트
        voice: edge-tts 음성 이름

    Raises:
        ValueError: text가 빈 문자열이거나 공백만인 경우
    """
    if not text or not text.strip():
        raise ValueError("합성할 텍스트가 비어 있습니다.")

    logger.debug("speak_async 호출 | text_len={n}", n=len(text))

    dest = _DEFAULT_OUTPUT
    asyncio.run(_synthesize_async(text, dest, voice))
    logger.info("비동기 합성 완료 | dest={dest}", dest=dest)

    try:
        play(dest)
    finally:
        _cleanup_temp_file(dest)


# ── 단독 테스트 ───────────────────────────────────────────

if __name__ == "__main__":
    logger.info("=== tts.py 단독 테스트 시작 ===")
    logger.info("IS_WINDOWS={v}", v=IS_WINDOWS)

    # ── 정상 케이스 1: 한국어 음성 목록 조회 ──────────────
    logger.info("--- 정상 케이스 1: list_voices('ko') ---")
    try:
        voices = list_voices("ko")
        assert len(voices) > 0, "한국어 음성이 하나도 없음"
        for v in voices:
            logger.info("  음성: {name} | {gender} | {locale}", **v)
        logger.info("정상 케이스 1 통과 | 음성 수={n}", n=len(voices))
    except Exception as e:
        logger.error("정상 케이스 1 실패: {e}", e=e)

    # ── 정상 케이스 2: 첫 번째 문장 재생 ─────────────────
    logger.info("--- 정상 케이스 2: speak('안녕하세요, 저는 Ria입니다. 잘 부탁드립니다.') ---")
    try:
        speak("안녕하세요, 저는 Ria입니다. 잘 부탁드립니다.")
        logger.info("정상 케이스 2 통과")
    except Exception as e:
        logger.error("정상 케이스 2 실패: {e}", e=e)

    # ── 정상 케이스 3: 두 번째 문장 재생 ─────────────────
    logger.info("--- 정상 케이스 3: speak('오늘 날씨가 참 좋네요.') ---")
    try:
        speak("오늘 날씨가 참 좋네요.")
        logger.info("정상 케이스 3 통과")
    except Exception as e:
        logger.error("정상 케이스 3 실패: {e}", e=e)

    # ── 에러 케이스 1: 빈 문자열 ──────────────────────────
    logger.info("--- 에러 케이스 1: 빈 문자열 입력 ---")
    try:
        speak("")
        logger.warning("에러 케이스 1: 예외가 발생해야 하는데 통과됨")
    except ValueError as e:
        logger.info("에러 케이스 1 정상 처리 (ValueError): {e}", e=e)
    except Exception as e:
        logger.error("에러 케이스 1 예상 외 예외: {e}", e=e)

    # ── 에러 케이스 2: 공백만 있는 문자열 ────────────────
    logger.info("--- 에러 케이스 2: 공백만 있는 문자열 ---")
    try:
        speak("   ")
        logger.warning("에러 케이스 2: 예외가 발생해야 하는데 통과됨")
    except ValueError as e:
        logger.info("에러 케이스 2 정상 처리 (ValueError): {e}", e=e)
    except Exception as e:
        logger.error("에러 케이스 2 예상 외 예외: {e}", e=e)

    # ── 에러 케이스 3: 존재하지 않는 음성 이름 ───────────
    logger.info("--- 에러 케이스 3: 잘못된 음성 이름 ---")
    try:
        speak("테스트", voice="xx-XX-InvalidVoiceNeural")
        logger.warning("에러 케이스 3: 예외 없이 통과됨 (edge-tts 동작에 따라 허용될 수 있음)")
    except (RuntimeError, ValueError) as e:
        logger.info("에러 케이스 3 정상 처리: {e}", e=e)
    except Exception as e:
        logger.warning("에러 케이스 3 기타 예외: {type} | {e}", type=type(e).__name__, e=e)

    logger.info("=== tts.py 단독 테스트 완료 ===")
