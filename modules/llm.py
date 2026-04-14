"""
modules/llm.py — Ollama LLM API 연동

대화 이력 관리 + 단발성 쿼리를 제공한다.
스트리밍 응답과 일반 응답 모두 지원.
"""
from typing import Generator

import requests
from loguru import logger

from config import LLM_MODEL, LLM_TIMEOUT, OLLAMA_HOST


def _build_url(path: str) -> str:
    """Ollama API 엔드포인트 URL 조합."""
    return f"{OLLAMA_HOST.rstrip('/')}{path}"


def is_ollama_running() -> bool:
    """Ollama 서버 응답 여부 확인."""
    try:
        resp = requests.get(_build_url("/api/tags"), timeout=5)
        return resp.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def query(
    prompt: str,
    system: str = "당신은 Ria입니다. 친절하고 간결하게 한국어로 답변하세요.",
    model: str = LLM_MODEL,
    history: list[dict] | None = None,
) -> str:
    """Ollama에 단발성 쿼리를 보내고 완성된 응답 문자열을 반환.

    Args:
        prompt: 사용자 입력 텍스트
        system: 시스템 프롬프트
        model: 사용할 Ollama 모델명
        history: 이전 대화 이력 [{"role": "user"/"assistant", "content": "..."}]

    Returns:
        LLM 응답 텍스트

    Raises:
        ConnectionError: Ollama 서버 미실행 시
        RuntimeError: API 오류 응답 시
    """
    messages = _build_messages(system, prompt, history)

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "num_gpu": 999,   # 모든 레이어 GPU에 올림
            "num_ctx": 2048,  # 컨텍스트 축소로 속도 향상
        },
    }

    logger.debug("LLM 요청 | model={model} | prompt_len={n}", model=model, n=len(prompt))

    try:
        resp = requests.post(
            _build_url("/api/chat"),
            json=payload,
            timeout=LLM_TIMEOUT,
        )
    except requests.exceptions.ConnectionError as e:
        raise ConnectionError(
            f"Ollama 서버에 연결할 수 없습니다. ({OLLAMA_HOST})"
        ) from e

    if resp.status_code != 200:
        raise RuntimeError(f"Ollama API 오류: {resp.status_code} — {resp.text}")

    content: str = resp.json()["message"]["content"]
    logger.debug("LLM 응답 | response_len={n}", n=len(content))
    return content


def query_stream(
    prompt: str,
    system: str = "당신은 Ria입니다. 친절하고 간결하게 한국어로 답변하세요.",
    model: str = LLM_MODEL,
    history: list[dict] | None = None,
) -> Generator[str, None, None]:
    """Ollama 스트리밍 쿼리. 토큰 단위로 응답을 yield.

    Args:
        prompt: 사용자 입력 텍스트
        system: 시스템 프롬프트
        model: 사용할 Ollama 모델명
        history: 이전 대화 이력

    Yields:
        응답 토큰 문자열

    Raises:
        ConnectionError: Ollama 서버 미실행 시
    """
    import json

    messages = _build_messages(system, prompt, history)

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {
            "num_gpu": 999,
            "num_ctx": 2048,
        },
    }

    logger.debug("LLM 스트리밍 시작 | model={model}", model=model)

    try:
        with requests.post(
            _build_url("/api/chat"),
            json=payload,
            stream=True,
            timeout=LLM_TIMEOUT,
        ) as resp:
            if resp.status_code != 200:
                raise RuntimeError(f"Ollama API 오류: {resp.status_code}")
            for line in resp.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                token: str = chunk.get("message", {}).get("content", "")
                if token:
                    yield token
                if chunk.get("done"):
                    break
    except requests.exceptions.ConnectionError as e:
        raise ConnectionError(
            f"Ollama 서버에 연결할 수 없습니다. ({OLLAMA_HOST})"
        ) from e

    logger.debug("LLM 스트리밍 완료")


def _build_messages(
    system: str,
    prompt: str,
    history: list[dict] | None,
) -> list[dict]:
    """시스템 프롬프트 + 이력 + 현재 입력을 messages 배열로 조합."""
    messages: list[dict] = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})
    return messages


if __name__ == "__main__":
    logger.info("=== llm.py 단독 테스트 ===")

    # 사전 확인: Ollama 실행 여부
    if not is_ollama_running():
        logger.warning("Ollama 서버가 실행 중이지 않습니다. 연결 오류 케이스만 테스트합니다.")

        # 에러 케이스: 서버 미실행
        try:
            query("안녕하세요")
        except ConnectionError as e:
            logger.info("에러 케이스 정상 처리 (ConnectionError): {e}", e=e)

    else:
        logger.info("Ollama 서버 확인됨. 정상 케이스 테스트 시작.")

        # 정상 케이스 1: 단발 쿼리
        try:
            response = query("안녕하세요! 간단히 자기소개 해주세요.")
            assert isinstance(response, str) and len(response) > 0
            logger.info("정상 케이스 1 통과 | 응답 길이={n}", n=len(response))
            logger.debug("응답 내용: {r}", r=response[:100])
        except Exception as e:
            logger.error("정상 케이스 1 실패: {e}", e=e)

        # 정상 케이스 2: 대화 이력 포함
        try:
            hist = [
                {"role": "user", "content": "내 이름은 민준이야."},
                {"role": "assistant", "content": "안녕하세요, 민준님!"},
            ]
            response = query("내 이름이 뭐라고 했지?", history=hist)
            assert "민준" in response
            logger.info("정상 케이스 2 통과 (이력 기억 확인)")
        except AssertionError:
            logger.warning("정상 케이스 2: 이름 기억 실패 (모델 성능 문제일 수 있음)")
        except Exception as e:
            logger.error("정상 케이스 2 실패: {e}", e=e)

        # 정상 케이스 3: 스트리밍
        try:
            logger.info("스트리밍 테스트 시작...")
            tokens = list(query_stream("숫자 1부터 3까지 세어줘."))
            full = "".join(tokens)
            assert len(full) > 0
            logger.info("정상 케이스 3 통과 | 토큰 수={n} | 전체={r}", n=len(tokens), r=full[:80])
        except Exception as e:
            logger.error("정상 케이스 3 실패: {e}", e=e)

        # 에러 케이스: 빈 프롬프트
        try:
            response = query("")
            logger.warning("에러 케이스: 빈 프롬프트가 통과됨 — 응답: {r}", r=response[:50])
        except Exception as e:
            logger.info("에러 케이스 처리됨: {e}", e=e)

    logger.info("=== llm.py 단독 테스트 완료 ===")
