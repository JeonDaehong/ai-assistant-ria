"""
modules/emotion.py — 텍스트 감정 분류 모듈

STT 결과를 받아 사용자의 감정 상태를 분류하고,
LLM에게 전달할 컨텍스트 힌트 문자열을 생성한다.

모델: hun3359/klue-bert-base-sentiment (기본값)
      .env의 EMOTION_MODEL 환경 변수로 오버라이드 가능
"""
import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from loguru import logger

# config.py에서 DEVICE만 가져옴 (platform.system() 직접 호출 금지)
from config import DEVICE, BASE_DIR

# .env 로드 (config.py에서 이미 로드하지만, 단독 실행 시를 위해 재호출)
load_dotenv(BASE_DIR / ".env")

# ── 모델 설정 ──────────────────────────────────────────────
DEFAULT_MODEL: str = "hun3359/klue-bert-base-sentiment"
EMOTION_MODEL: str = os.getenv("EMOTION_MODEL", DEFAULT_MODEL)

# ── 모듈 레벨 캐시 (최초 1회만 로드) ──────────────────────
_pipeline = None  # transformers pipeline 객체


# ── 데이터 클래스 ──────────────────────────────────────────
@dataclass
class EmotionResult:
    """감정 분석 결과."""

    label: str          # 예: "긍정", "부정", "중립"
    score: float        # 확신도 0.0~1.0
    raw: dict = field(default_factory=dict)  # 모든 레이블별 점수


# ── 내부 헬퍼 ──────────────────────────────────────────────
def _get_pipeline():
    """캐싱된 pipeline을 반환한다. 없으면 load_model()을 호출한다."""
    global _pipeline
    if _pipeline is None:
        load_model()
    return _pipeline


def _normalize_label(raw_label: str) -> str:
    """모델이 반환하는 레이블을 그대로 반환한다.

    hun3359/klue-bert-base-sentiment 모델은 60여 개의 세밀한 한국어 감정 레이블을
    직접 반환한다 (예: "기쁨", "슬픔", "불안", "분노" 등).
    영문/숫자 레이블을 사용하는 다른 모델에 대한 폴백 매핑도 포함한다.
    """
    fallback_mapping: dict[str, str] = {
        # 영문 계열
        "positive": "긍정",
        "negative": "부정",
        "neutral": "중립",
        # 숫자 레이블 계열 (일부 모델)
        "0": "부정",
        "1": "중립",
        "2": "긍정",
        "label_0": "부정",
        "label_1": "중립",
        "label_2": "긍정",
    }
    return fallback_mapping.get(raw_label.lower(), raw_label)


def _pipeline_to_result(pipeline_output: list[dict]) -> EmotionResult:
    """pipeline 단일 출력 리스트를 EmotionResult로 변환한다."""
    raw: dict = {_normalize_label(item["label"]): round(item["score"], 4)
                 for item in pipeline_output}
    best = max(pipeline_output, key=lambda x: x["score"])
    label = _normalize_label(best["label"])
    score = round(best["score"], 4)
    return EmotionResult(label=label, score=score, raw=raw)


# ── 공개 함수 ──────────────────────────────────────────────
def load_model() -> None:
    """감정 분류 모델을 초기화한다.

    모듈 레벨 캐시에 저장하여 최초 1회만 로드된다.
    직접 호출하거나, analyze() 첫 호출 시 자동으로 실행된다.
    """
    global _pipeline

    if _pipeline is not None:
        logger.debug("emotion 모델 이미 로드됨, 스킵")
        return

    try:
        from transformers import pipeline as hf_pipeline
    except ImportError as e:
        logger.error("transformers 패키지 없음: {e}", e=e)
        raise

    logger.info(
        "emotion 모델 로딩 시작 | model={model} | device={device}",
        model=EMOTION_MODEL,
        device=DEVICE,
    )

    # DEVICE 문자열을 transformers pipeline의 device 인자로 변환
    device_arg: int | str
    if DEVICE == "cuda":
        device_arg = 0          # transformers는 GPU 인덱스 정수를 받음
    elif DEVICE == "mps":
        device_arg = "mps"
    else:
        device_arg = -1         # CPU

    _pipeline = hf_pipeline(
        task="text-classification",
        model=EMOTION_MODEL,
        top_k=None,             # 모든 레이블 점수 반환
        device=device_arg,
    )

    logger.info("emotion 모델 로딩 완료 | model={model}", model=EMOTION_MODEL)


def analyze(text: str) -> EmotionResult:
    """단일 텍스트의 감정을 분류한다.

    Args:
        text: 분석할 한국어 텍스트 (빈 문자열 불가)

    Returns:
        EmotionResult (label, score, raw)

    Raises:
        ValueError: text가 빈 문자열일 때
    """
    if not text or not text.strip():
        raise ValueError("분석할 텍스트가 비어 있습니다.")

    pipe = _get_pipeline()

    logger.debug("감정 분석 시작 | text_len={n}", n=len(text))

    try:
        output: list[dict] = pipe(text)[0]  # 단일 문장 → 첫 번째 원소
        result = _pipeline_to_result(output)
    except Exception as e:
        logger.error("감정 분석 실패: {e}", e=e)
        raise

    logger.info(
        "감정 분석 완료 | label={label} | score={score:.4f}",
        label=result.label,
        score=result.score,
    )
    return result


def analyze_batch(texts: list[str]) -> list[EmotionResult]:
    """여러 텍스트의 감정을 배치로 분류한다.

    Args:
        texts: 분석할 한국어 텍스트 리스트 (빈 리스트 허용, 빈 문자열 불가)

    Returns:
        각 텍스트에 대응하는 EmotionResult 리스트

    Raises:
        ValueError: texts 안에 빈 문자열이 포함된 경우
    """
    if not texts:
        logger.warning("analyze_batch: 빈 리스트 입력, 빈 결과 반환")
        return []

    for i, t in enumerate(texts):
        if not t or not t.strip():
            raise ValueError(f"texts[{i}]가 빈 문자열입니다.")

    pipe = _get_pipeline()

    logger.debug("배치 감정 분석 시작 | count={n}", n=len(texts))

    try:
        batch_output: list[list[dict]] = pipe(texts)
    except Exception as e:
        logger.error("배치 감정 분석 실패: {e}", e=e)
        raise

    results = [_pipeline_to_result(output) for output in batch_output]

    logger.info("배치 감정 분석 완료 | count={n}", n=len(results))
    return results


def _classify_sentiment_group(label: str) -> str:
    """세밀한 감정 레이블을 긍정/부정/중립 그룹으로 매핑한다.

    hun3359/klue-bert-base-sentiment의 60여 개 레이블을 3개 그룹으로 분류한다.
    """
    positive_keywords = (
        "기쁨", "행복", "즐거움", "설렘", "사랑", "희망", "감사", "뿌듯함",
        "홀가분함", "편안함", "안도", "긍정", "positive",
    )
    negative_keywords = (
        "슬픔", "분노", "불안", "두려움", "공포", "절망", "우울", "혐오",
        "짜증", "후회", "억울함", "서러움", "그리움", "상처", "수치스러움",
        "죄책감", "부끄러움", "실망", "외로움", "허탈함", "허무함",
        "배신감", "냉담함", "불쾌함", "화남", "무서움", "스트레스",
        "힘듦", "지침", "고통", "괴로움", "부정", "negative",
        "ȸ������", "부끄",  # 인코딩 깨짐 방어
    )
    label_lower = label.lower()
    if any(kw in label for kw in positive_keywords):
        return "positive"
    if any(kw in label for kw in negative_keywords):
        return "negative"
    # 숫자 레이블 폴백
    if label_lower in ("0", "label_0"):
        return "negative"
    if label_lower in ("2", "label_2"):
        return "positive"
    return "neutral"


def to_prompt_hint(result: EmotionResult) -> str:
    """LLM 프롬프트에 삽입할 감정 힌트 문자열을 반환한다.

    세밀한 감정 레이블(예: "기쁨", "슬픔")을 그대로 표시하고,
    긍정/부정/중립 그룹에 따라 답변 지침을 추가한다.

    Args:
        result: analyze() 또는 analyze_batch()의 반환값

    Returns:
        예: "[사용자 감정: 슬픔(0.82) — 위로와 공감 위주로 답변하세요]"
    """
    sentiment_group = _classify_sentiment_group(result.label)
    guidance_map: dict[str, str] = {
        "positive": "긍정적이고 활기찬 톤으로 답변하세요",
        "negative": "위로와 공감 위주로 답변하세요",
        "neutral": "차분하고 객관적인 톤으로 답변하세요",
    }
    guidance = guidance_map.get(sentiment_group, "사용자 상황에 맞게 답변하세요")
    hint = f"[사용자 감정: {result.label}({result.score:.2f}) — {guidance}]"
    logger.debug("prompt_hint 생성: {hint}", hint=hint)
    return hint


# ── 단독 테스트 ────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=== emotion.py 단독 테스트 시작 ===")

    # --- 정상 케이스 ---
    logger.info("--- 정상 케이스: 단일 분석 ---")
    test_sentences = [
        "오늘 정말 기분이 좋아요! 최고의 하루였습니다.",   # 긍정 예상
        "그냥 평범한 하루였어요.",                          # 중립 예상
        "너무 힘들고 지쳐서 아무것도 하기 싫어요.",        # 부정 예상
    ]

    for sentence in test_sentences:
        try:
            emotion = analyze(sentence)
            hint = to_prompt_hint(emotion)
            logger.info(
                "입력: {text!r} | 결과: label={label}, score={score:.4f}",
                text=sentence,
                label=emotion.label,
                score=emotion.score,
            )
            logger.info("  raw={raw}", raw=emotion.raw)
            logger.info("  hint={hint}", hint=hint)
        except Exception as e:
            logger.error("단일 분석 예외: {e}", e=e)

    # --- 배치 케이스 ---
    logger.info("--- 배치 케이스: analyze_batch ---")
    batch_texts = [
        "이 음식 정말 맛있어요!",
        "별로 특별한 일이 없었어요.",
        "왜 이렇게 일이 안 풀리는지 모르겠어요.",
    ]
    try:
        batch_results = analyze_batch(batch_texts)
        for text, res in zip(batch_texts, batch_results):
            logger.info(
                "배치 결과 | {text!r} → {label}({score:.4f})",
                text=text,
                label=res.label,
                score=res.score,
            )
    except Exception as e:
        logger.error("배치 분석 예외: {e}", e=e)

    # --- 에러 케이스: 빈 문자열 ---
    logger.info("--- 에러 케이스: 빈 문자열 입력 ---")
    try:
        analyze("")
        logger.warning("에러가 발생해야 하는데 통과됨 — 확인 필요")
    except ValueError as e:
        logger.info("빈 문자열 에러 정상 처리: {e}", e=e)

    # --- 에러 케이스: 배치 내 빈 문자열 ---
    logger.info("--- 에러 케이스: 배치 내 빈 문자열 ---")
    try:
        analyze_batch(["정상 문장", "", "또 다른 문장"])
        logger.warning("에러가 발생해야 하는데 통과됨 — 확인 필요")
    except ValueError as e:
        logger.info("배치 빈 문자열 에러 정상 처리: {e}", e=e)

    # --- 에러 케이스: 빈 리스트 ---
    logger.info("--- 에러 케이스: 빈 리스트 입력 ---")
    empty_result = analyze_batch([])
    assert empty_result == [], f"빈 리스트 결과가 []여야 함: {empty_result}"
    logger.info("빈 리스트 입력 시 빈 리스트 반환 확인")

    logger.info("=== emotion.py 단독 테스트 완료 ===")
