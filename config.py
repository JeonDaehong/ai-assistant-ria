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
LLM_MODEL: str = os.getenv("LLM_MODEL", "gemma4:e2b")
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

# ── 메모리 DB ─────────────────────────────────────────────
MEMORY_DIR: Path = Path(os.getenv("MEMORY_DIR", str(BASE_DIR / "data" / "memory")))
MEMORY_DIR.mkdir(parents=True, exist_ok=True)

# ── Firebase ──────────────────────────────────────────────
FIREBASE_KEY_PATH: Path = Path(os.getenv("FIREBASE_KEY_PATH", ""))

# ── STT / TTS ─────────────────────────────────────────────
STT_MODEL: str = os.getenv("STT_MODEL", "base")
TTS_MODEL: str = os.getenv("TTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
XTTS_REF_AUDIO_DIR: Path = Path(os.getenv("XTTS_REF_AUDIO_DIR", str(BASE_DIR / "sound_model")))
XTTS_LANGUAGE: str = os.getenv("XTTS_LANGUAGE", "ko")

# ── GPT-SoVITS ────────────────────────────────────────────
_default_sovits_dir = str(BASE_DIR.parent / "GPT-SoVITS" / "GPT-SoVITS-v2pro-20250604")
SOVITS_DIR: Path = Path(os.getenv("SOVITS_DIR", _default_sovits_dir))
SOVITS_API_URL: str = os.getenv("SOVITS_API_URL", "http://127.0.0.1:9880")
SOVITS_REF_AUDIO: Path = Path(os.getenv("SOVITS_REF_AUDIO", str(BASE_DIR / "sound_model" / "ref_3sec.wav")))
SOVITS_REF_TEXT: str = os.getenv("SOVITS_REF_TEXT", "")
SOVITS_LANG: str = os.getenv("SOVITS_LANG", "ko")
SOVITS_GPT_WEIGHTS: str = os.getenv(
    "SOVITS_GPT_WEIGHTS",
    str(SOVITS_DIR / "GPT_SoVITS" / "pretrained_models"
        / "gsv-v2final-pretrained" / "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"),
)
SOVITS_WEIGHTS: str = os.getenv(
    "SOVITS_WEIGHTS",
    str(SOVITS_DIR / "GPT_SoVITS" / "pretrained_models"
        / "gsv-v2final-pretrained" / "s2G2333k.pth"),
)
    str(SOVITS_DIR / "GPT_SoVITS" / "pretrained_models" / "gsv-v2final-pretrained" / "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"),
)
SOVITS_WEIGHTS: str = os.getenv(
    "SOVITS_WEIGHTS",
    str(SOVITS_DIR / "GPT_SoVITS" / "pretrained_models" / "gsv-v2final-pretrained" / "s2G2333k.pth"),
)

# ── Obsidian ──────────────────────────────────────────────
if IS_WINDOWS:
    _default_obsidian = r"C:\Users\user\Documents\Obsidian Vault"
else:
    _default_obsidian = str(Path.home() / "Documents" / "Obsidian Vault")
OBSIDIAN_VAULT_PATH: Path = Path(os.getenv("OBSIDIAN_VAULT_PATH", _default_obsidian))

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
