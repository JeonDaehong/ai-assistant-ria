"""
config.py — 환경별 설정 분기 (Windows 개발 / Mac Mini M4 운영)

OS 분기는 모두 이 파일에서만 처리한다.
다른 모듈에서 platform.system() 직접 호출 금지.
"""
import os
import platform
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# .env 로드 (없어도 무시)
_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path)

# ── OS 판별 ──────────────────────────────────────────────
_system = platform.system()
IS_MAC: bool = _system == "Darwin"
IS_WINDOWS: bool = _system == "Windows"

# ── 프로젝트 루트 ─────────────────────────────────────────
BASE_DIR: Path = Path(__file__).parent

# ── LLM ──────────────────────────────────────────────────
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL: str = os.getenv("LLM_MODEL", "bnksys/eeve:10.8b-korean-instruct-q5_k_m-v1")
LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "120"))

# ── 연산 디바이스 ─────────────────────────────────────────
# Mac M4 → mps, Windows RTX → cuda, 나머지 → cpu
def _detect_device() -> str:
    if IS_MAC:
        return "mps"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"

DEVICE: str = os.getenv("DEVICE", _detect_device())

# ── NAS 경로 ──────────────────────────────────────────────
if IS_MAC:
    NAS_PATH: Path = Path(os.getenv("NAS_PATH", "/Volumes/NAS"))
else:
    NAS_PATH: Path = Path(os.getenv("NAS_PATH", r"\\NAS"))

# ── Firebase ──────────────────────────────────────────────
FIREBASE_KEY_PATH: Path = Path(os.getenv("FIREBASE_KEY_PATH", ""))

# ── STT / TTS ─────────────────────────────────────────────
STT_MODEL: str = os.getenv("STT_MODEL", "base")
TTS_MODEL: str = os.getenv("TTS_MODEL", "kokoro")

# ── 로그 ──────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "DEBUG" if not IS_MAC else "INFO")
LOG_DIR: Path = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.add(
    LOG_DIR / "ria_{time:YYYY-MM-DD}.log",
    level=LOG_LEVEL,
    rotation="00:00",
    retention="14 days",
    encoding="utf-8",
)

logger.debug(
    "config 로드 완료 | OS={os} | DEVICE={device} | LLM={model}",
    os=_system,
    device=DEVICE,
    model=LLM_MODEL,
)


if __name__ == "__main__":
    logger.info("=== config.py 단독 테스트 ===")

    # 정상 케이스: 값 출력
    logger.info("IS_MAC={v}", v=IS_MAC)
    logger.info("IS_WINDOWS={v}", v=IS_WINDOWS)
    logger.info("BASE_DIR={v}", v=BASE_DIR)
    logger.info("OLLAMA_HOST={v}", v=OLLAMA_HOST)
    logger.info("LLM_MODEL={v}", v=LLM_MODEL)
    logger.info("DEVICE={v}", v=DEVICE)
    logger.info("NAS_PATH={v}", v=NAS_PATH)
    logger.info("LOG_DIR={v}", v=LOG_DIR)

    # 에러 케이스: FIREBASE_KEY_PATH 미설정 감지
    if not FIREBASE_KEY_PATH or not FIREBASE_KEY_PATH.exists():
        logger.warning(
            "FIREBASE_KEY_PATH 미설정 또는 파일 없음 → Firebase 모듈 사용 불가"
        )
    else:
        logger.info("FIREBASE_KEY_PATH={v}", v=FIREBASE_KEY_PATH)

    logger.info("=== config.py 테스트 완료 ===")
