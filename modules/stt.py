"""
stt.py — 음성 입력 및 전사 모듈 (faster-whisper + sounddevice)

공개 함수:
    list_microphones()        — 사용 가능한 마이크 목록 반환
    record_audio()            — 마이크에서 오디오 녹음
    transcribe()              — 오디오 배열 또는 파일을 텍스트로 변환
    listen_and_transcribe()   — record + transcribe 통합 단일 호출
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from loguru import logger

from config import DEVICE, STT_MODEL

# ── 모델 싱글턴 ────────────────────────────────────────────────────────────────
_whisper_model: WhisperModel | None = None

HYPERX_KEYWORDS: list[str] = ["hyperx", "quadcast", "HyperX"]


def _get_model() -> WhisperModel:
    """WhisperModel 싱글턴을 반환한다. 최초 호출 시 로드한다."""
    global _whisper_model
    if _whisper_model is None:
        compute_type = _resolve_compute_type(DEVICE)
        logger.info(
            "Whisper 모델 로드 중 | model={model} | device={device} | compute_type={ct}",
            model=STT_MODEL,
            device=DEVICE,
            ct=compute_type,
        )
        _whisper_model = WhisperModel(
            STT_MODEL,
            device=DEVICE if DEVICE != "mps" else "cpu",
            compute_type=compute_type,
        )
        logger.info("Whisper 모델 로드 완료")
    return _whisper_model


def _resolve_compute_type(device: str) -> str:
    """디바이스에 맞는 compute_type 문자열을 반환한다."""
    if device == "cuda":
        return "float16"
    if device == "mps":
        return "int8"
    return "int8"


# ── 공개 함수 ──────────────────────────────────────────────────────────────────


def list_microphones() -> list[dict]:
    """
    현재 시스템에서 사용 가능한 마이크 목록을 반환한다.

    Returns:
        각 마이크 정보를 담은 dict 리스트.
        dict 키: index (int), name (str), max_input_channels (int)
    """
    devices = sd.query_devices()
    microphones: list[dict] = []

    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            microphones.append(
                {
                    "index": idx,
                    "name": dev["name"],
                    "max_input_channels": dev["max_input_channels"],
                }
            )

    logger.debug("감지된 마이크 수: {count}", count=len(microphones))
    return microphones


def find_hyperx_device_index() -> int | None:
    """
    HyperX QuadCast 마이크의 디바이스 인덱스를 반환한다.
    찾지 못하면 None을 반환한다.

    Returns:
        HyperX 디바이스 인덱스 또는 None.
    """
    for mic in list_microphones():
        name_lower = mic["name"].lower()
        if any(kw.lower() in name_lower for kw in HYPERX_KEYWORDS):
            logger.info(
                "HyperX QuadCast 감지: index={idx} | name={name}",
                idx=mic["index"],
                name=mic["name"],
            )
            return mic["index"]
    logger.debug("HyperX QuadCast 마이크를 찾지 못함 — 기본 디바이스 사용")
    return None


def record_audio(
    duration: float,
    device_index: int | None = None,
    sample_rate: int = 16000,
) -> np.ndarray:
    """
    마이크에서 지정 시간만큼 오디오를 녹음하고 numpy 배열로 반환한다.

    Args:
        duration:      녹음 시간 (초).
        device_index:  사용할 마이크 디바이스 인덱스. None이면 기본 디바이스.
        sample_rate:   샘플링 레이트 (Hz). Whisper 권장값은 16000.

    Returns:
        shape (N,)의 float32 numpy 배열 (mono).

    Raises:
        sd.PortAudioError: 잘못된 디바이스 인덱스 또는 녹음 오류.
        ValueError:        duration이 0 이하인 경우.
    """
    if duration <= 0:
        raise ValueError(f"duration은 0보다 커야 합니다. 입력값: {duration}")

    _validate_device_index(device_index)

    logger.info(
        "녹음 시작 | duration={dur}s | device={dev} | sample_rate={sr}",
        dur=duration,
        dev=device_index,
        sr=sample_rate,
    )

    frames = int(duration * sample_rate)
    audio: np.ndarray = sd.rec(
        frames,
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=device_index,
    )
    sd.wait()

    audio = audio.flatten()
    logger.info("녹음 완료 | samples={n}", n=len(audio))
    return audio


def _validate_device_index(device_index: int | None) -> None:
    """
    디바이스 인덱스가 유효한지 검사한다.

    Args:
        device_index: 검사할 디바이스 인덱스. None이면 검사 생략.

    Raises:
        ValueError: 인덱스가 범위를 벗어나거나 입력 채널이 없는 경우.
    """
    if device_index is None:
        return

    devices = sd.query_devices()
    if device_index < 0 or device_index >= len(devices):
        raise ValueError(
            f"디바이스 인덱스 {device_index}가 범위를 벗어났습니다. "
            f"유효 범위: 0 ~ {len(devices) - 1}"
        )

    dev = devices[device_index]
    if dev["max_input_channels"] == 0:
        raise ValueError(
            f"디바이스 인덱스 {device_index} ({dev['name']})은 입력 채널이 없습니다."
        )


def transcribe(
    audio: Union[np.ndarray, str, Path],
    language: str = "ko",
) -> str:
    """
    오디오 배열 또는 파일 경로를 텍스트로 변환한다.

    Args:
        audio:    float32 numpy 배열(mono, 16kHz) 또는 오디오 파일 경로.
        language: 전사 언어 코드. 기본값 "ko" (한국어).

    Returns:
        전사된 텍스트 문자열.

    Raises:
        FileNotFoundError: 파일 경로를 지정했으나 파일이 없는 경우.
        ValueError:        audio 타입이 지원되지 않는 경우.
    """
    model = _get_model()
    audio_input = _prepare_audio_input(audio)

    logger.debug("전사 시작 | language={lang}", lang=language)
    segments, info = model.transcribe(
        audio_input,
        language=language,
        beam_size=5,
        vad_filter=True,
    )
    logger.debug(
        "언어 감지: {lang} (확률 {prob:.2f})",
        lang=info.language,
        prob=info.language_probability,
    )

    text = _join_segments(segments)
    logger.info("전사 완료 | 길이={length}자", length=len(text))
    return text


def _prepare_audio_input(
    audio: Union[np.ndarray, str, Path],
) -> Union[np.ndarray, str]:
    """
    transcribe()에 전달할 입력 형태를 준비한다.

    Args:
        audio: numpy 배열 또는 파일 경로.

    Returns:
        faster-whisper에 전달 가능한 numpy 배열 또는 파일 경로 문자열.

    Raises:
        FileNotFoundError: 파일이 존재하지 않는 경우.
        ValueError:        지원하지 않는 타입인 경우.
    """
    if isinstance(audio, np.ndarray):
        return audio.astype("float32")

    if isinstance(audio, (str, Path)):
        path = Path(audio)
        if not path.exists():
            raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {path}")
        logger.debug("파일 경로 전사: {path}", path=path)
        return str(path)

    raise ValueError(
        f"audio는 np.ndarray, str, Path 중 하나여야 합니다. 받은 타입: {type(audio)}"
    )


def _join_segments(segments) -> str:
    """
    faster-whisper 세그먼트 이터레이터에서 텍스트를 이어 붙여 반환한다.

    Args:
        segments: faster-whisper의 세그먼트 이터레이터.

    Returns:
        전체 전사 텍스트.
    """
    parts: list[str] = []
    for segment in segments:
        parts.append(segment.text.strip())
        logger.debug(
            "세그먼트 [{start:.2f}s → {end:.2f}s]: {text}",
            start=segment.start,
            end=segment.end,
            text=segment.text.strip(),
        )
    return " ".join(parts).strip()


def listen_and_transcribe(
    duration: float = 5.0,
    device_index: int | None = None,
    language: str = "ko",
) -> str:
    """
    마이크 녹음과 전사를 한 번에 수행한다.

    Args:
        duration:     녹음 시간 (초). 기본값 5.0.
        device_index: 사용할 마이크 디바이스 인덱스. None이면 기본 디바이스.
        language:     전사 언어 코드. 기본값 "ko".

    Returns:
        전사된 텍스트 문자열.
    """
    audio = record_audio(duration=duration, device_index=device_index)
    return transcribe(audio=audio, language=language)


# ── 단독 테스트 ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logger.info("=== stt.py 단독 테스트 시작 ===")

    # ── 정상 케이스 1: 마이크 목록 출력 ──────────────────────────────────────
    logger.info("--- [정상 케이스 1] 마이크 목록 ---")
    mics = list_microphones()
    for mic in mics:
        logger.info(
            "  [{idx}] {name} (채널: {ch})",
            idx=mic["index"],
            name=mic["name"],
            ch=mic["max_input_channels"],
        )

    # ── 정상 케이스 2: HyperX 자동 감지 후 3초 녹음 → 전사 ──────────────────
    logger.info("--- [정상 케이스 2] HyperX 자동 감지 → 3초 녹음 → 전사 ---")
    hyperx_idx = find_hyperx_device_index()
    if hyperx_idx is None:
        logger.warning("HyperX를 찾지 못했습니다. 시스템 기본 마이크로 녹음합니다.")

    try:
        logger.info("3초 후 녹음이 종료됩니다. 말씀해 주세요...")
        result = listen_and_transcribe(duration=3.0, device_index=hyperx_idx)
        logger.info("전사 결과: '{text}'", text=result)
        assert isinstance(result, str), "반환값이 str이어야 함"
        logger.info("정상 케이스 통과")
    except Exception as e:
        logger.error("정상 케이스 실패: {e}", e=e)
        sys.exit(1)

    # ── 에러 케이스 1: 존재하지 않는 디바이스 인덱스 ────────────────────────
    logger.info("--- [에러 케이스 1] 잘못된 디바이스 인덱스 ---")
    try:
        record_audio(duration=1.0, device_index=9999)
        logger.warning("에러가 발생해야 하는데 통과됨 — 검토 필요")
    except ValueError as e:
        logger.info("에러 케이스 1 정상 처리 (ValueError): {e}", e=e)
    except Exception as e:
        logger.info("에러 케이스 1 정상 처리 ({t}): {e}", t=type(e).__name__, e=e)

    # ── 에러 케이스 2: 존재하지 않는 파일 경로 전사 ─────────────────────────
    logger.info("--- [에러 케이스 2] 존재하지 않는 파일 경로 ---")
    try:
        transcribe("nonexistent_audio_file.wav")
        logger.warning("에러가 발생해야 하는데 통과됨 — 검토 필요")
    except FileNotFoundError as e:
        logger.info("에러 케이스 2 정상 처리 (FileNotFoundError): {e}", e=e)

    # ── 에러 케이스 3: duration이 0 이하 ─────────────────────────────────────
    logger.info("--- [에러 케이스 3] duration <= 0 ---")
    try:
        record_audio(duration=0.0)
        logger.warning("에러가 발생해야 하는데 통과됨 — 검토 필요")
    except ValueError as e:
        logger.info("에러 케이스 3 정상 처리 (ValueError): {e}", e=e)

    logger.info("=== stt.py 단독 테스트 완료 ===")
