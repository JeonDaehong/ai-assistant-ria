"""
modules/stt.py — 음성 입력 및 전사 모듈 (faster-whisper + VAD)

VAD(Voice Activity Detection) 방식으로 말소리가 시작되면 자동 녹음을 시작하고
지정 시간 이상 침묵이 지속되면 자동으로 녹음을 종료한다.

VAD 백엔드 우선순위:
    1. webrtcvad  — 경량, 별도 GPU 불필요 (pip install webrtcvad-wheels)
    2. silero-vad — 고정밀, PyTorch 필요  (pip install silero-vad)
    둘 다 없으면 ImportError 발생.

공개 함수:
    list_microphones()            — 사용 가능한 마이크 목록 반환
    find_hyperx_device_index()    — HyperX QuadCast 장치 인덱스 탐색
    record_audio()                — 고정 시간 녹음 (하위호환)
    record_with_vad()             — VAD 기반 자동 녹음
    transcribe()                  — 오디오 배열/파일 → 텍스트
    listen_and_transcribe()       — VAD 녹음 + 전사 통합 (권장)
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Union

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from loguru import logger

from config import DEVICE, STT_MODEL


# ── 상수 ─────────────────────────────────────────────────────────────────────

HYPERX_KEYWORDS: list[str] = ["hyperx", "quadcast"]

# VAD 프레임 크기 (webrtcvad는 10/20/30ms만 허용)
_FRAME_MS: int = 30
_SAMPLE_RATE: int = 16000
_FRAME_SAMPLES: int = int(_SAMPLE_RATE * _FRAME_MS / 1000)  # 480


# ── Whisper 모델 싱글턴 ───────────────────────────────────────────────────────

_whisper_model: WhisperModel | None = None


def _get_model() -> WhisperModel:
    """WhisperModel 싱글턴을 반환한다. 최초 호출 시 로드한다."""
    global _whisper_model
    if _whisper_model is None:
        compute_type = "float16" if DEVICE == "cuda" else "int8"
        logger.info(
            "Whisper 모델 로드 | model={m} | device={d} | compute={c}",
            m=STT_MODEL, d=DEVICE, c=compute_type,
        )
        _whisper_model = WhisperModel(
            STT_MODEL,
            device=DEVICE if DEVICE != "mps" else "cpu",
            compute_type=compute_type,
        )
        logger.info("Whisper 모델 로드 완료")
    return _whisper_model


# ── VAD 백엔드 ────────────────────────────────────────────────────────────────

def _make_webrtcvad(aggressiveness: int):
    """webrtcvad.Vad 인스턴스를 생성한다."""
    import webrtcvad  # pip install webrtcvad-wheels
    vad = webrtcvad.Vad(aggressiveness)
    logger.debug("VAD 백엔드: webrtcvad (aggressiveness={a})", a=aggressiveness)
    return vad


def _make_silero_vad():
    """silero-vad 모델과 유틸 함수를 로드해 반환한다."""
    import torch
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
    )
    logger.debug("VAD 백엔드: silero-vad")
    return model, utils


class _WebRTCVADBackend:
    """webrtcvad 기반 VAD 래퍼."""

    def __init__(self, aggressiveness: int = 2) -> None:
        self._vad = _make_webrtcvad(aggressiveness)

    def is_speech(self, frame: np.ndarray) -> bool:
        """float32 프레임(480 샘플)을 int16 bytes로 변환해 VAD를 수행한다."""
        pcm = (frame * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
        return self._vad.is_speech(pcm, _SAMPLE_RATE)


class _SileroVADBackend:
    """silero-vad 기반 VAD 래퍼."""

    def __init__(self) -> None:
        import torch
        self._torch = torch
        self._model, _ = _make_silero_vad()

    def is_speech(self, frame: np.ndarray) -> bool:
        """float32 프레임을 tensor로 변환해 silero-vad 신뢰도를 반환한다."""
        tensor = self._torch.from_numpy(frame).float()
        confidence: float = self._model(tensor, _SAMPLE_RATE).item()
        return confidence >= 0.5


def _get_vad_backend(aggressiveness: int = 2) -> _WebRTCVADBackend | _SileroVADBackend:
    """webrtcvad → silero-vad 순으로 사용 가능한 VAD 백엔드를 반환한다."""
    try:
        return _WebRTCVADBackend(aggressiveness)
    except ImportError:
        pass

    try:
        return _SileroVADBackend()
    except ImportError:
        pass

    raise ImportError(
        "VAD 패키지가 설치되지 않았습니다.\n"
        "  pip install webrtcvad-wheels    # 권장 (경량)\n"
        "  pip install silero-vad torch    # 고정밀"
    )


# ── 마이크 유틸 ───────────────────────────────────────────────────────────────

def list_microphones() -> list[dict]:
    """현재 시스템에서 사용 가능한 마이크 목록을 반환한다.

    Returns:
        {"index": int, "name": str, "max_input_channels": int} 리스트
    """
    devices = sd.query_devices()
    mics: list[dict] = [
        {"index": i, "name": d["name"], "max_input_channels": d["max_input_channels"]}
        for i, d in enumerate(devices)
        if d["max_input_channels"] > 0
    ]
    logger.debug("감지된 마이크: {n}개", n=len(mics))
    return mics


def find_hyperx_device_index() -> int | None:
    """HyperX QuadCast 마이크의 장치 인덱스를 반환한다.

    Returns:
        장치 인덱스 정수. 없으면 None.
    """
    for mic in list_microphones():
        if any(kw in mic["name"].lower() for kw in HYPERX_KEYWORDS):
            logger.info(
                "HyperX QuadCast 감지 | index={i} | name={n}",
                i=mic["index"], n=mic["name"],
            )
            return mic["index"]
    logger.debug("HyperX QuadCast 미감지 → 기본 마이크 사용")
    return None


def _validate_device_index(device_index: int | None) -> None:
    """장치 인덱스 유효성 검사."""
    if device_index is None:
        return
    devices = sd.query_devices()
    if not (0 <= device_index < len(devices)):
        raise ValueError(
            f"장치 인덱스 {device_index} 범위 초과 (0~{len(devices) - 1})"
        )
    if devices[device_index]["max_input_channels"] == 0:
        raise ValueError(
            f"장치 {device_index} ({devices[device_index]['name']})은 입력 채널 없음"
        )


# ── 녹음 ─────────────────────────────────────────────────────────────────────

def record_audio(
    duration: float,
    device_index: int | None = None,
    sample_rate: int = _SAMPLE_RATE,
) -> np.ndarray:
    """고정 시간 녹음. 하위호환 목적으로 유지한다.

    Args:
        duration: 녹음 시간 (초)
        device_index: 마이크 장치 인덱스. None이면 기본 장치.
        sample_rate: 샘플링 레이트 (Hz)

    Returns:
        shape (N,) float32 numpy 배열 (mono)

    Raises:
        ValueError: duration ≤ 0 또는 잘못된 장치 인덱스
    """
    if duration <= 0:
        raise ValueError(f"duration은 0보다 커야 합니다: {duration}")
    _validate_device_index(device_index)

    logger.info("고정 녹음 시작 | {d}초 | device={dev}", d=duration, dev=device_index)
    frames = int(duration * sample_rate)
    audio: np.ndarray = sd.rec(
        frames, samplerate=sample_rate, channels=1, dtype="float32", device=device_index
    )
    sd.wait()
    logger.info("고정 녹음 완료 | samples={n}", n=len(audio))
    return audio.flatten()


def record_with_vad(
    device_index: int | None = None,
    sample_rate: int = _SAMPLE_RATE,
    silence_sec: float = 1.5,
    max_sec: float = 30.0,
    aggressiveness: int = 3,
    pre_roll_frames: int = 5,
) -> np.ndarray:
    """VAD 기반 자동 녹음.

    말소리가 감지되면 녹음을 시작하고, silence_sec 이상 침묵이 지속되면
    자동으로 종료한다.

    Args:
        device_index: 마이크 장치 인덱스. None이면 기본 장치.
        sample_rate: 샘플링 레이트 (Hz, 기본 16000)
        silence_sec: 이 시간 이상 침묵이 지속되면 녹음 종료 (기본 1.0초)
        max_sec: 최대 녹음 시간 (기본 30초). 초과하면 강제 종료.
        aggressiveness: webrtcvad 민감도 0~3 (높을수록 음성 판정 엄격, 기본 2)
        pre_roll_frames: 음성 시작 직전 유지할 프레임 수 (끊김 방지, 기본 5)

    Returns:
        녹음된 float32 numpy 배열 (mono). 음성 미감지 시 빈 배열 반환.

    Raises:
        ImportError: webrtcvad, silero-vad 모두 미설치 시
        ValueError: 잘못된 장치 인덱스
    """
    _validate_device_index(device_index)

    vad = _get_vad_backend(aggressiveness)
    frame_samples = int(sample_rate * _FRAME_MS / 1000)
    silence_frames_needed = int(silence_sec * 1000 / _FRAME_MS)
    max_frames = int(max_sec * 1000 / _FRAME_MS)

    pre_roll: deque[np.ndarray] = deque(maxlen=pre_roll_frames)
    speech_buf: list[np.ndarray] = []
    silence_count: int = 0
    speaking: bool = False

    logger.info("VAD 대기 중... (말씀하세요 | 침묵 {s}초에 자동 종료)", s=silence_sec)

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            device=device_index,
            blocksize=frame_samples,
        ) as stream:
            for frame_idx in range(max_frames):
                frame, _ = stream.read(frame_samples)
                frame = frame.flatten()

                is_speech = vad.is_speech(frame)

                if not speaking:
                    # 아직 말 시작 전 — pre-roll 버퍼에 누적
                    pre_roll.append(frame)
                    if is_speech:
                        speaking = True
                        speech_buf.extend(pre_roll)  # pre-roll 포함해서 시작
                        silence_count = 0
                        logger.info("음성 감지 — 녹음 시작")
                else:
                    # 말 중 — 버퍼에 추가
                    speech_buf.append(frame)
                    if is_speech:
                        silence_count = 0
                    else:
                        silence_count += 1
                        if silence_count >= silence_frames_needed:
                            logger.info(
                                "침묵 {s}초 감지 — 녹음 종료 (총 {n}프레임)",
                                s=silence_sec,
                                n=len(speech_buf),
                            )
                            break

            else:
                if speaking:
                    logger.warning("최대 녹음 시간({s}초) 도달 — 강제 종료", s=max_sec)
                else:
                    logger.debug("VAD: 최대 대기 시간 내 음성 미감지")

    except Exception as e:
        logger.error("VAD 스트림 오류: {e}", e=e)
        raise

    if not speech_buf:
        return np.array([], dtype=np.float32)

    audio = np.concatenate(speech_buf)
    logger.info(
        "VAD 녹음 완료 | {n}샘플 | {s:.2f}초",
        n=len(audio),
        s=len(audio) / sample_rate,
    )
    return audio


# ── 전사 ─────────────────────────────────────────────────────────────────────

def transcribe(
    audio: Union[np.ndarray, str, Path],
    language: str = "ko",
) -> str:
    """오디오 배열 또는 파일 경로를 텍스트로 전사한다.

    Args:
        audio: float32 numpy 배열(mono, 16kHz) 또는 오디오 파일 경로
        language: 전사 언어 코드 (기본 "ko")

    Returns:
        전사된 텍스트 문자열

    Raises:
        FileNotFoundError: 파일 경로가 존재하지 않을 때
        ValueError: 지원하지 않는 audio 타입 또는 빈 배열
    """
    if isinstance(audio, np.ndarray):
        if audio.size == 0:
            logger.debug("transcribe: 빈 배열 → 빈 문자열 반환")
            return ""
        audio_input: Union[np.ndarray, str] = audio.astype("float32")
    elif isinstance(audio, (str, Path)):
        path = Path(audio)
        if not path.exists():
            raise FileNotFoundError(f"오디오 파일 없음: {path}")
        audio_input = str(path)
    else:
        raise ValueError(f"지원하지 않는 audio 타입: {type(audio)}")

    model = _get_model()
    logger.debug("전사 시작 | language={lang}", lang=language)

    segments, info = model.transcribe(
        audio_input,
        language=language,
        beam_size=5,
        vad_filter=False,
    )
    logger.debug(
        "언어 감지: {lang} (확률 {p:.2f})",
        lang=info.language, p=info.language_probability,
    )

    parts: list[str] = []
    for seg in segments:
        parts.append(seg.text.strip())
        logger.debug(
            "[{s:.2f}s→{e:.2f}s] {t}",
            s=seg.start, e=seg.end, t=seg.text.strip(),
        )

    text = " ".join(parts).strip()
    logger.info("전사 완료 | {n}자", n=len(text))
    return text


# ── 통합 인터페이스 ───────────────────────────────────────────────────────────

def listen_and_transcribe(
    device_index: int | None = None,
    language: str = "ko",
    silence_sec: float = 1.5,
    max_sec: float = 30.0,
    aggressiveness: int = 3,
) -> str:
    """VAD 기반 녹음과 전사를 한 번에 수행한다.

    말소리가 감지되면 자동으로 녹음을 시작하고,
    silence_sec 이상 침묵이 지속되면 자동으로 종료 후 전사한다.

    Args:
        device_index: 마이크 장치 인덱스. None이면 기본 장치.
        language: 전사 언어 코드 (기본 "ko")
        silence_sec: 녹음 종료 침묵 시간 (기본 1.0초)
        max_sec: 최대 녹음 시간 (기본 30초)
        aggressiveness: VAD 민감도 0~3 (기본 2)

    Returns:
        전사된 텍스트 문자열. 음성 미감지 시 빈 문자열.
    """
    audio = record_with_vad(
        device_index=device_index,
        silence_sec=silence_sec,
        max_sec=max_sec,
        aggressiveness=aggressiveness,
    )
    return transcribe(audio, language=language)


# ── 단독 테스트 ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("=== stt.py 단독 테스트 시작 (VAD 방식) ===")

    # ── [1] 마이크 목록 ────────────────────────────────────
    logger.info("--- [1] 마이크 목록 ---")
    mics = list_microphones()
    for m in mics:
        logger.info("  [{i}] {n} (채널: {c})", i=m["index"], n=m["name"], c=m["max_input_channels"])

    # ── [2] HyperX 자동 감지 ───────────────────────────────
    logger.info("--- [2] HyperX QuadCast 감지 ---")
    hyperx_idx = find_hyperx_device_index()
    if hyperx_idx is None:
        logger.warning("HyperX 미감지 → 기본 마이크 사용")

    # ── [3] VAD 백엔드 확인 ────────────────────────────────
    logger.info("--- [3] VAD 백엔드 초기화 ---")
    try:
        vad_backend = _get_vad_backend(aggressiveness=2)
        logger.info("VAD 백엔드: {t}", t=type(vad_backend).__name__)
    except ImportError as e:
        logger.error("VAD 패키지 없음: {e}", e=e)
        import sys; sys.exit(1)

    # ── [4] VAD 녹음 + 전사 정상 케이스 ───────────────────
    logger.info("--- [4] VAD 녹음 정상 케이스 (말씀하세요) ---")
    try:
        result = listen_and_transcribe(
            device_index=hyperx_idx,
            max_sec=15.0,
        )
        assert isinstance(result, str)
        logger.info("전사 결과: '{text}'", text=result)
        logger.info("정상 케이스 통과")
    except Exception as e:
        logger.error("정상 케이스 실패: {e}", e=e)

    # ── [5] 에러 케이스: 빈 오디오 배열 전사 ────────────────
    logger.info("--- [5] 에러 케이스: 빈 배열 전사 ---")
    try:
        empty_result = transcribe(np.array([], dtype=np.float32))
        assert empty_result == "", f"빈 배열 → 빈 문자열 기대: '{empty_result}'"
        logger.info("빈 배열 처리 통과 (빈 문자열 반환)")
    except Exception as e:
        logger.error("빈 배열 테스트 실패: {e}", e=e)

    # ── [6] 에러 케이스: 존재하지 않는 파일 경로 ─────────────
    logger.info("--- [6] 에러 케이스: 없는 파일 경로 ---")
    try:
        transcribe("nonexistent_file.wav")
        logger.warning("FileNotFoundError가 발생해야 하는데 통과됨")
    except FileNotFoundError as e:
        logger.info("FileNotFoundError 정상 처리: {e}", e=e)

    # ── [7] 에러 케이스: 잘못된 장치 인덱스 ─────────────────
    logger.info("--- [7] 에러 케이스: 잘못된 장치 인덱스 ---")
    try:
        record_with_vad(device_index=9999, max_sec=1.0)
        logger.warning("ValueError가 발생해야 하는데 통과됨")
    except ValueError as e:
        logger.info("ValueError 정상 처리: {e}", e=e)

    logger.info("=== stt.py 단독 테스트 완료 ===")
