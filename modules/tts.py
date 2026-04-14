"""
modules/tts.py — 텍스트 → 음성 (GPT-SoVITS 스트리밍 API + sounddevice)

GPT-SoVITS api_v2.py 서버에 streaming_mode=True 로 요청하고
sounddevice로 청크 단위 실시간 재생한다.

서버가 꺼져 있으면 SOVITS_DIR/runtime/python.exe 로 자동 기동한다.

실행:
    python modules/tts.py
"""
import io
import struct
import subprocess
import threading
import time
import wave
from pathlib import Path
from typing import Iterator

import numpy as np
import requests
import sounddevice as sd
from loguru import logger

from config import (
    BASE_DIR,
    SOVITS_API_URL,
    SOVITS_DIR,
    SOVITS_GPT_WEIGHTS,
    SOVITS_LANG,
    SOVITS_REF_AUDIO,
    SOVITS_REF_TEXT,
    SOVITS_WEIGHTS,
)

# ── 설정 ─────────────────────────────────────────────────
_SERVER_STARTUP_TIMEOUT: int = 90   # 서버 기동 대기 최대 초
_SERVER_POLL_INTERVAL: float = 1.5  # 폴링 간격 (초)


# ── 서버 관리 ─────────────────────────────────────────────

def _is_server_alive() -> bool:
    """GPT-SoVITS API 서버가 응답 중인지 확인한다."""
    try:
        resp = requests.get(f"{SOVITS_API_URL}/", timeout=2)
        return resp.status_code < 500
    except requests.exceptions.ConnectionError:
        return False
    except Exception:
        return False


def _start_server() -> subprocess.Popen:
    """GPT-SoVITS api_v2.py 서버를 백그라운드로 기동한다.

    Returns:
        기동된 서버 프로세스

    Raises:
        FileNotFoundError: SOVITS_DIR 가 존재하지 않을 경우
    """
    if not SOVITS_DIR.exists():
        raise FileNotFoundError(f"GPT-SoVITS 디렉터리를 찾을 수 없습니다: {SOVITS_DIR}")

    runtime_python = SOVITS_DIR / "runtime" / "python.exe"
    python_exe = str(runtime_python) if runtime_python.exists() else "python"

    log_path = SOVITS_DIR / "api_v2.log"
    inner_cmd = (
        f'$env:PYTHONIOENCODING="utf-8"; '
        f'cd "{SOVITS_DIR}"; '
        f'& "{python_exe}" api_v2.py -a 127.0.0.1 -p 9880 *>> "{log_path}" 2>&1'
    )
    cmd = ["powershell", "-NoExit", "-Command", inner_cmd]

    logger.info("GPT-SoVITS 서버 기동 | log={log}", log=log_path)

    proc = subprocess.Popen(
        cmd,
        creationflags=subprocess.CREATE_NEW_CONSOLE,
    )
    return proc


def _wait_for_server(timeout: int = _SERVER_STARTUP_TIMEOUT) -> None:
    """서버가 준비될 때까지 최대 timeout초 폴링 대기한다.

    Raises:
        TimeoutError: timeout 내에 서버가 응답하지 않을 경우
    """
    logger.info("GPT-SoVITS 모델 로딩 대기 중... (최대 {t}초)", t=timeout)
    # 서버 프로세스가 HTTP 바인딩하기 전에 폴링하지 않도록 초기 대기
    time.sleep(5)

    start = time.time()
    last_log = start
    deadline = start + timeout

    while time.time() < deadline:
        if _is_server_alive():
            elapsed = int(time.time() - start)
            logger.info("GPT-SoVITS 서버 준비 완료 | 소요={e}초", e=elapsed)
            return

        now = time.time()
        if now - last_log >= 10:
            elapsed = int(now - start)
            logger.info("GPT-SoVITS 기동 대기 중... {e}초 경과 / {t}초", e=elapsed, t=timeout)
            last_log = now

        time.sleep(_SERVER_POLL_INTERVAL)

    raise TimeoutError(f"GPT-SoVITS 서버가 {timeout}초 내에 응답하지 않습니다.")


def _load_weights() -> None:
    """GPT 및 SoVITS 가중치를 API에 로드한다.

    Raises:
        RuntimeError: 가중치 로드 API 호출 실패 시
    """
    for endpoint, weights, label in [
        ("/set_gpt_weights", SOVITS_GPT_WEIGHTS, "GPT"),
        ("/set_sovits_weights", SOVITS_WEIGHTS, "SoVITS"),
    ]:
        try:
            resp = requests.get(
                f"{SOVITS_API_URL}{endpoint}",
                params={"weights_path": weights},
                timeout=30,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"{label} 가중치 로드 실패 {resp.status_code}: {resp.text[:200]}")
            logger.info("{label} 가중치 로드 완료 | {w}", label=label, w=weights)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"{label} 가중치 로드 요청 실패: {e}") from e


_weights_loaded = False


def ensure_server() -> None:
    """서버가 꺼져 있으면 자동 기동하고, 가중치를 최초 1회만 로드한다."""
    global _weights_loaded
    if _is_server_alive():
        logger.debug("GPT-SoVITS 서버 이미 실행 중")
    else:
        logger.warning("GPT-SoVITS 서버 미기동 — 자동 시작합니다")
        _start_server()
        _wait_for_server()
        _weights_loaded = False

    if not _weights_loaded:
        _load_weights()
        _weights_loaded = True


# ── 레퍼런스 캐시 ─────────────────────────────────────────

_ref_cache: dict | None = None


def _get_refs() -> tuple[str, list[str]]:
    """레퍼런스 오디오 경로를 반환한다. SOVITS_REF_AUDIO만 사용."""
    global _ref_cache
    if _ref_cache is None:
        _ref_cache = {"main": str(SOVITS_REF_AUDIO), "aux": []}
    return _ref_cache["main"], _ref_cache["aux"]


# ── 합성 (스트리밍) ───────────────────────────────────────

def synthesize_stream(text: str) -> requests.Response:
    """GPT-SoVITS API에 스트리밍 모드로 합성을 요청하고 스트림 응답을 반환한다.

    Args:
        text: 합성할 텍스트 (빈 문자열 불가)

    Returns:
        스트리밍 응답 (iter_content로 청크 순회 가능)

    Raises:
        ValueError: text가 비어 있거나 공백만인 경우
        RuntimeError: API 호출 실패 시
    """
    if not text or not text.strip():
        raise ValueError("합성할 텍스트가 비어 있습니다.")

    ref_main, aux_refs = _get_refs()

    payload = {
        "text": text,
        "text_lang": SOVITS_LANG,
        "ref_audio_path": ref_main,
        "aux_ref_audio_paths": aux_refs,
        "prompt_text": SOVITS_REF_TEXT,
        "prompt_lang": SOVITS_LANG,
        "text_split_method": "cut0",
        "batch_size": 1,
        "media_type": "wav",
        "streaming_mode": 2,
    }

    logger.debug(
        "TTS 스트리밍 합성 요청 | text_len={n} | ref={r}",
        n=len(text),
        r=ref_main,
    )

    try:
        resp = requests.post(
            f"{SOVITS_API_URL}/tts",
            json=payload,
            timeout=120,
            stream=True,
        )
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"GPT-SoVITS API 요청 실패: {e}") from e

    if resp.status_code != 200:
        raise RuntimeError(f"API 오류 {resp.status_code}: {resp.text[:300]}")

    return resp


# ── WAV 스트림 파싱 ───────────────────────────────────────

_WAV_HEADER_SIZE = 44


def _parse_wav_header(header: bytes) -> tuple[int, int, int]:
    """44-byte WAV 헤더에서 sample_rate, channels, sample_width(bytes)를 추출한다."""
    channels = struct.unpack_from("<H", header, 22)[0]
    sample_rate = struct.unpack_from("<I", header, 24)[0]
    bits_per_sample = struct.unpack_from("<H", header, 34)[0]
    return sample_rate, channels, bits_per_sample // 8


def _iter_pcm_chunks(
    resp: requests.Response,
    chunk_size: int = 8192,
) -> Iterator[tuple[bytes, int, int, int]]:
    """스트리밍 응답에서 PCM 데이터를 청크 단위로 yield한다.

    GPT-SoVITS 스트리밍은 첫 44바이트가 WAV 헤더이고,
    이후 모든 데이터는 raw PCM이다.

    Yields:
        (pcm_data, sample_rate, channels, sample_width)
    """
    header_buf = b""
    sample_rate = channels = sample_width = 0
    header_parsed = False

    try:
        for raw_chunk in resp.iter_content(chunk_size=chunk_size):
            if not raw_chunk:
                continue

            if not header_parsed:
                header_buf += raw_chunk
                if len(header_buf) < _WAV_HEADER_SIZE:
                    continue

                sample_rate, channels, sample_width = _parse_wav_header(header_buf[:_WAV_HEADER_SIZE])
                header_parsed = True
                logger.debug(
                    "오디오 포맷 감지 | rate={r} ch={c} width={w}",
                    r=sample_rate, c=channels, w=sample_width,
                )

                leftover = header_buf[_WAV_HEADER_SIZE:]
                if leftover:
                    yield leftover, sample_rate, channels, sample_width
            else:
                yield raw_chunk, sample_rate, channels, sample_width

    except requests.exceptions.ChunkedEncodingError as e:
        logger.warning("ChunkedEncodingError: {e}", e=e)


# ── 재생 (스트리밍) ───────────────────────────────────────

class StreamingPlayer:
    """스트리밍 응답을 받아 실시간으로 재생하는 플레이어.

    사용법 (단건):
        player = StreamingPlayer()
        player.play(response)   # 블로킹 (재생 완료까지)

    사용법 (연속 재생 - 문장 사이 끊김 없음):
        player.begin_session()
        player.play(resp1)
        player.play(resp2)
        player.end_session()

        player.stop()           # 다른 스레드에서 호출하여 즉시 중단
    """

    def __init__(self) -> None:
        self._stop_event = threading.Event()
        self._stream: sd.RawOutputStream | None = None
        self._in_session = False

    def begin_session(self) -> None:
        """연속 재생 세션을 시작한다. 세션 중에는 스트림을 닫지 않는다."""
        self._in_session = True
        self._stop_event.clear()

    def end_session(self) -> None:
        """연속 재생 세션을 종료하고 스트림을 정리한다."""
        self._in_session = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def stop(self) -> None:
        """재생을 즉시 중단한다."""
        self._stop_event.set()
        self._in_session = False
        if self._stream is not None:
            self._stream.abort()
        logger.info("재생 중단 요청됨")

    def play(self, resp: requests.Response) -> None:
        """스트리밍 응답을 청크 단위로 수신하며 실시간 재생한다.

        세션 모드가 아니면 재생 후 스트림을 닫는다.
        세션 모드이면 스트림을 유지하여 다음 play() 호출 시 재사용한다.

        Args:
            resp: synthesize_stream()이 반환한 스트리밍 응답
        """
        if not self._in_session:
            self._stop_event.clear()
            self._stream = None
        total_bytes = 0
        dtype_map = {1: "int8", 2: "int16", 4: "int32"}

        try:
            for pcm, sr, ch, sw in _iter_pcm_chunks(resp):
                if self._stop_event.is_set():
                    break

                if self._stream is None:
                    dtype = dtype_map.get(sw, "int16")
                    self._stream = sd.RawOutputStream(
                        samplerate=sr,
                        channels=ch,
                        dtype=dtype,
                    )
                    self._stream.start()
                    logger.debug("sounddevice 스트림 시작 | rate={r} ch={c} dtype={d}", r=sr, c=ch, d=dtype)

                self._stream.write(pcm)
                total_bytes += len(pcm)

        except Exception as e:
            if not self._stop_event.is_set():
                raise RuntimeError(f"스트리밍 재생 오류: {e}") from e
        finally:
            if not self._in_session and self._stream is not None:
                if not self._stop_event.is_set():
                    self._stream.stop()
                self._stream.close()
                self._stream = None

        logger.info("스트리밍 재생 완료 | total_pcm={s}B", s=total_bytes)


# ── 전역 플레이어 (stop 접근용) ──────────────────────────

_player = StreamingPlayer()


def stop() -> None:
    """현재 재생 중인 음성을 즉시 중단한다."""
    _player.stop()


def begin_session() -> None:
    """연속 재생 세션을 시작한다.

    세션 중에는 sounddevice 스트림이 문장 사이에 유지되어
    끊김 없이 연속 재생된다. 반드시 end_session()으로 닫아야 한다.
    """
    _player.begin_session()


def end_session() -> None:
    """연속 재생 세션을 종료하고 오디오 스트림을 정리한다."""
    _player.end_session()


# ── 통합 인터페이스 ───────────────────────────────────────

def speak(text: str) -> None:
    """텍스트를 합성하여 재생한다.

    서버가 꺼져 있으면 자동 기동한다.
    다른 스레드에서 stop()을 호출하면 재생을 즉시 중단할 수 있다.

    Args:
        text: 읽을 텍스트

    Raises:
        ValueError: text가 비어 있거나 공백만인 경우
    """
    ensure_server()
    resp = synthesize_stream(text)
    _player.play(resp)


def speak_direct(text: str) -> None:
    """서버 체크 없이 바로 합성+재생한다.

    begin_session() 이후 반복 호출에 적합하다.
    ensure_server()를 호출자가 미리 해 둬야 한다.
    """
    resp = synthesize_stream(text)
    _player.play(resp)


# ── 단독 테스트 ───────────────────────────────────────────

if __name__ == "__main__":
    logger.info("=== tts.py 스트리밍 테스트 시작 ===")

    # ── 서버 확인/자동 기동 ───────────────────────────────
    logger.info("--- 케이스 1: ensure_server() ---")
    try:
        ensure_server()
        logger.info("케이스 1 통과")
    except Exception as e:
        logger.error("케이스 1 실패: {e}", e=e)

    # ── 스트리밍 합성 + 실시간 재생 ──────────────────────
    logger.info("--- 케이스 2: speak() 스트리밍 재생 ---")
    try:
        t0 = time.time()
        speak("안녕하세요, 저는 리아입니다. 스트리밍 모드로 실시간 재생하고 있어요.")
        elapsed = time.time() - t0
        logger.info("케이스 2 통과 | 소요={e:.1f}초", e=elapsed)
    except Exception as e:
        logger.error("케이스 2 실패: {e}", e=e)

    # ── 빈 문자열 에러 처리 ──────────────────────────────
    logger.info("--- 케이스 3: 빈 문자열 ---")
    try:
        speak("")
        logger.warning("케이스 3: 예외가 발생해야 하는데 통과됨")
    except ValueError as e:
        logger.info("케이스 3 정상 처리: {e}", e=e)

    # ── stop() 중단 테스트 ───────────────────────────────
    logger.info("--- 케이스 4: stop() 1초 후 중단 ---")
    try:
        def _stop_after(sec: float) -> None:
            time.sleep(sec)
            stop()

        timer = threading.Thread(target=_stop_after, args=(1.0,))
        timer.start()
        t0 = time.time()
        speak("이 문장은 중간에 잘릴 수 있습니다. 왜냐하면 1초 후에 stop이 호출되기 때문이에요.")
        elapsed = time.time() - t0
        timer.join()
        logger.info("케이스 4 통과 | 소요={e:.1f}초 (1초 근처면 정상 중단)", e=elapsed)
    except Exception as e:
        logger.error("케이스 4 실패: {e}", e=e)

    logger.info("=== tts.py 스트리밍 테스트 완료 ===")
