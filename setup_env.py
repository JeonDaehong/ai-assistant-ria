"""
setup_env.py - Ria AI 어시스턴트 환경 셋업 스크립트

처음 프로젝트를 받은 후 한 번 실행하면:
  1. 필수 디렉터리 생성
  2. pip 패키지 설치
  3. PyTorch 설치 (CUDA 자동 감지)
  4. GPT-SoVITS 패키지 다운로드 + 압축 해제
  5. Ollama 모델 pull
  6. HuggingFace 모델 사전 다운로드 (감정, 임베딩, Whisper)
  7. .env 템플릿 생성
  8. 전체 검증

실행:
    python setup_env.py
    python setup_env.py --skip-pip      # pip 설치 건너뛰기
    python setup_env.py --skip-models   # 모델 다운로드 건너뛰기
"""
import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
SOVITS_PARENT = "GPT-SoVITS"
SOVITS_DIR_NAME = "GPT-SoVITS-v2pro-20250604"
IS_WINDOWS = platform.system() == "Windows"

# HuggingFace 모델 목록
HF_MODELS = {
    "감정 분석": "hun3359/klue-bert-base-sentiment",
    "임베딩 (기억 DB)": "jhgan/ko-sroberta-multitask",
}
WHISPER_MODEL = "base"

# GPT-SoVITS 다운로드 URL
SOVITS_HF_REPO = "lj1995/GPT-SoVITS-windows-package"
SOVITS_ARCHIVE = f"{SOVITS_DIR_NAME}.7z"
SOVITS_NVIDIA_ARCHIVE = f"{SOVITS_DIR_NAME}-nvidia50.7z"

# Ollama
OLLAMA_MODEL = "gemma4:e2b"


def _print_header(title: str) -> None:
    print(f"\n{'=' * 56}")
    print(f"  {title}")
    print(f"{'=' * 56}")


def _print_step(msg: str, ok: bool = True) -> None:
    icon = "[OK]" if ok else "[!!]"
    print(f"  {icon} {msg}")


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


# ── 1. 디렉터리 생성 ─────────────────────────────────

def setup_directories() -> None:
    _print_header("1/7  필수 디렉터리 생성")
    dirs = [
        BASE_DIR / "data" / "prompts",
        BASE_DIR / "data" / "memory",
        BASE_DIR / "logs",
        BASE_DIR / "sound_model",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        _print_step(f"{d.relative_to(BASE_DIR)}")

    prompt_file = BASE_DIR / "data" / "prompts" / "system.txt"
    if not prompt_file.exists():
        prompt_file.write_text(
            "너는 리아야. 귀엽고 친근한 여동생 같은 스타일로 반말로 짧게 대화해.",
            encoding="utf-8",
        )
        _print_step("data/prompts/system.txt (기본 프롬프트 생성)")


# ── 2. pip 패키지 ────────────────────────────────────

def setup_pip() -> None:
    _print_header("2/7  pip 패키지 설치")
    req_file = BASE_DIR / "requirements.txt"
    if not req_file.exists():
        _print_step("requirements.txt 없음 - 건너뜀", ok=False)
        return

    result = _run([sys.executable, "-m", "pip", "install", "-r", str(req_file)])
    if result.returncode == 0:
        _print_step("requirements.txt 설치 완료")
    else:
        _print_step(f"pip install 실패:\n{result.stderr[:500]}", ok=False)


# ── 3. PyTorch ───────────────────────────────────────

def setup_torch() -> None:
    _print_header("3/7  PyTorch 설치 확인")
    try:
        import torch
        cuda = torch.cuda.is_available()
        device = torch.cuda.get_device_name(0) if cuda else "CPU"
        _print_step(f"PyTorch {torch.__version__} | CUDA={cuda} | {device}")
        return
    except ImportError:
        pass

    print("  PyTorch 미설치 - 설치를 시도합니다...")
    index_url = "https://download.pytorch.org/whl/cu121"
    result = _run([
        sys.executable, "-m", "pip", "install",
        "torch", "--index-url", index_url,
    ])
    if result.returncode == 0:
        _print_step("PyTorch (CUDA 12.1) 설치 완료")
    else:
        _print_step("PyTorch 자동 설치 실패 - 수동 설치 필요:", ok=False)
        print(f"    pip install torch --index-url {index_url}")


# ── 4. GPT-SoVITS ────────────────────────────────────

def setup_sovits() -> None:
    _print_header("4/7  GPT-SoVITS 패키지")
    sovits_parent = BASE_DIR.parent / SOVITS_PARENT
    sovits_parent.mkdir(parents=True, exist_ok=True)
    sovits_dir = sovits_parent / SOVITS_DIR_NAME
    api_file = sovits_dir / "api_v2.py"

    if api_file.exists():
        _print_step(f"{SOVITS_PARENT}/{SOVITS_DIR_NAME} 이미 존재")
        pretrained = sovits_dir / "GPT_SoVITS" / "pretrained_models" / "gsv-v2final-pretrained"
        if pretrained.exists():
            files = list(pretrained.glob("*.ckpt")) + list(pretrained.glob("*.pth"))
            _print_step(f"pretrained 가중치: {len(files)}개 파일")
        else:
            _print_step("pretrained 가중치 디렉터리 없음", ok=False)
        return

    print(f"  {SOVITS_DIR_NAME} 미설치 - 다운로드를 시작합니다...")
    print(f"  위치: {sovits_parent}")

    if not _check_command("huggingface-cli"):
        print("  huggingface-cli 설치 중...")
        _run([sys.executable, "-m", "pip", "install", "huggingface_hub[cli]"])

    archive = SOVITS_NVIDIA_ARCHIVE if _ask_yn("  NVIDIA 50시리즈 GPU 버전을 받으시겠습니까?") else SOVITS_ARCHIVE

    print(f"  다운로드 중: {archive} (~8GB, 시간이 걸립니다)...")
    result = subprocess.run(
        ["huggingface-cli", "download",
         SOVITS_HF_REPO, archive,
         "--local-dir", str(sovits_parent)],
        text=True, capture_output=False,
    )

    if result.returncode != 0:
        _print_step(f"다운로드 실패:\n{result.stderr[:500]}", ok=False)
        print(f"  수동 다운로드: https://huggingface.co/{SOVITS_HF_REPO}")
        return

    archive_path = sovits_parent / archive
    if archive_path.exists():
        _print_step(f"다운로드 완료: {archive}")
        _extract_7z(archive_path, sovits_parent)
    else:
        _print_step("다운로드된 파일을 찾을 수 없음", ok=False)


def _extract_7z(archive: Path, target: Path) -> None:
    if _check_command("7z"):
        extractor = "7z"
    elif _check_command("7zz"):
        extractor = "7zz"
    else:
        _print_step("7z 미설치 - 수동으로 압축 해제 필요:", ok=False)
        print(f"    {archive}")
        return

    print(f"  압축 해제 중...")
    result = _run([extractor, "x", str(archive), f"-o{target}", "-y"])
    if result.returncode == 0:
        _print_step("압축 해제 완료")
        archive.unlink()
        _print_step("아카이브 파일 삭제")
    else:
        _print_step(f"압축 해제 실패:\n{result.stderr[:300]}", ok=False)


# ── 5. Ollama ────────────────────────────────────────

def setup_ollama() -> None:
    _print_header("5/7  Ollama 모델")
    ollama = _find_ollama()
    if not ollama:
        _print_step("Ollama 미설치 - https://ollama.ai 에서 설치하세요", ok=False)
        return

    result = _run([ollama, "list"])
    if OLLAMA_MODEL in (result.stdout or ""):
        _print_step(f"{OLLAMA_MODEL} 이미 존재")
        return

    print(f"  모델 pull 중: {OLLAMA_MODEL} (~10GB)...")
    pull = subprocess.run([ollama, "pull", OLLAMA_MODEL])
    if pull.returncode == 0:
        _print_step(f"{OLLAMA_MODEL} 다운로드 완료")
    else:
        _print_step("Ollama pull 실패", ok=False)


# ── 6. HuggingFace 모델 ─────────────────────────────

def setup_hf_models() -> None:
    _print_header("6/7  HuggingFace 모델 사전 다운로드")

    for label, model_id in HF_MODELS.items():
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(model_id)
            _print_step(f"{label}: {model_id}")
        except Exception as e:
            _print_step(f"{label}: {model_id} - 실패: {e}", ok=False)

    print(f"  Whisper ({WHISPER_MODEL}) 다운로드 중...")
    try:
        from faster_whisper import WhisperModel
        WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        _print_step(f"Whisper '{WHISPER_MODEL}' 준비 완료")
    except Exception as e:
        _print_step(f"Whisper 다운로드 실패: {e}", ok=False)


# ── 7. .env 템플릿 ───────────────────────────────────

def setup_env_file() -> None:
    _print_header("7/7  .env 파일")
    env_file = BASE_DIR / ".env"
    example_file = BASE_DIR / ".env.example"

    if env_file.exists():
        _print_step(".env 이미 존재 - 건너뜀")
        return

    if example_file.exists():
        shutil.copy2(example_file, env_file)
        _print_step(".env.example → .env 복사 완료")
    else:
        _print_step(".env.example 없음 - .env 생성하지 않음", ok=False)


# ── 검증 ─────────────────────────────────────────────

def verify() -> None:
    _print_header("검증 결과")
    checks = {
        "sound_model/ 디렉터리": (BASE_DIR / "sound_model").is_dir(),
        "sound_model/*.wav 파일": bool(list((BASE_DIR / "sound_model").glob("*.wav"))),
        "data/prompts/system.txt": (BASE_DIR / "data" / "prompts" / "system.txt").exists(),
        "GPT-SoVITS 설치": (BASE_DIR.parent / SOVITS_PARENT / SOVITS_DIR_NAME / "api_v2.py").exists(),
        "pretrained 가중치": (
            BASE_DIR.parent / SOVITS_PARENT / SOVITS_DIR_NAME
            / "GPT_SoVITS" / "pretrained_models" / "gsv-v2final-pretrained"
        ).is_dir(),
        "Ollama 설치": _find_ollama() is not None,
        ".env 파일": (BASE_DIR / ".env").exists(),
    }

    all_ok = True
    for label, ok in checks.items():
        _print_step(label, ok=ok)
        if not ok:
            all_ok = False

    # pip 패키지 확인
    packages = [
        "loguru", "dotenv", "requests", "numpy", "sounddevice",
        "faster_whisper", "torch", "chromadb", "transformers",
        "sentence_transformers", "websockets", "apscheduler",
        "duckduckgo_search",
    ]
    missing = []
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        _print_step(f"미설치 패키지: {', '.join(missing)}", ok=False)
        all_ok = False
    else:
        _print_step("필수 pip 패키지 모두 설치됨")

    print()
    if all_ok:
        print("  모든 검증 통과! python -m modules.tts 로 TTS 테스트해보세요.")
    else:
        print("  일부 항목이 미완료입니다. 위 [!!] 항목을 확인하세요.")
    print()


# ── 유틸 ─────────────────────────────────────────────

def _check_command(name: str) -> bool:
    return shutil.which(name) is not None


def _find_ollama() -> str | None:
    """Ollama 실행 파일 경로를 반환한다. PATH에 없으면 기본 설치 경로를 탐색."""
    found = shutil.which("ollama")
    if found:
        return found
    if IS_WINDOWS:
        for candidate in [
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
            Path("C:/Program Files/Ollama/ollama.exe"),
        ]:
            if candidate.exists():
                return str(candidate)
    return None


def _ask_yn(prompt: str, default: bool = False) -> bool:
    suffix = " [y/N]: " if not default else " [Y/n]: "
    try:
        answer = input(prompt + suffix).strip().lower()
    except EOFError:
        return default
    if not answer:
        return default
    return answer in ("y", "yes")


# ── 메인 ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Ria AI 어시스턴트 환경 셋업")
    parser.add_argument("--skip-pip", action="store_true", help="pip 설치 건너뛰기")
    parser.add_argument("--skip-models", action="store_true", help="모델 다운로드 건너뛰기")
    args = parser.parse_args()

    print()
    print("  Ria AI 어시스턴트 - 환경 셋업")
    print(f"  프로젝트: {BASE_DIR}")
    print(f"  Python:  {sys.version.split()[0]}")
    print(f"  OS:      {platform.system()} {platform.release()}")

    setup_directories()

    if not args.skip_pip:
        setup_pip()
        setup_torch()

    if not args.skip_models:
        setup_sovits()
        setup_ollama()
        setup_hf_models()

    setup_env_file()
    verify()


if __name__ == "__main__":
    main()
