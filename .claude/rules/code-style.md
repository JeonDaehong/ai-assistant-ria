# 코드 스타일 규칙

## 타입 힌트
- 모든 함수의 매개변수와 반환값에 타입 힌트 필수
- `Optional`, `Union`, `List`, `Dict` 등 `typing` 모듈 적극 활용
- 반환 없는 함수는 `-> None` 명시

```python
# Good
def transcribe_audio(file_path: str, language: str = "ko") -> str:
    ...

# Bad
def transcribe_audio(file_path, language="ko"):
    ...
```

## 단일 책임 원칙
- 함수 하나는 한 가지 일만
- 함수 길이 50줄 초과 시 분리 검토
- 클래스 하나는 한 가지 도메인만 담당

## 로깅 (loguru)
- `print()` 사용 금지, 반드시 `loguru.logger` 사용
- 레벨 기준: `DEBUG`(개발), `INFO`(정상 흐름), `WARNING`(예외 상황), `ERROR`(오류)
- 민감 정보(토큰, 비밀번호)는 로그에 마스킹 처리

```python
from loguru import logger

logger.info("STT 완료: {length}자", length=len(text))
logger.error("Ollama 연결 실패: {error}", error=str(e))
```

## 단독 테스트 필수
- 모든 모듈 하단에 `if __name__ == "__main__":` 블록 포함
- 정상 케이스 + 에러 케이스 각각 테스트
- 테스트 통과 확인 후 커밋

```python
if __name__ == "__main__":
    # 정상 케이스
    result = transcribe_audio("test.wav")
    logger.info("결과: {result}", result=result)

    # 에러 케이스
    try:
        transcribe_audio("nonexistent.wav")
    except FileNotFoundError as e:
        logger.warning("예상된 오류 처리 확인: {e}", e=e)
```

## OS 분기 규칙
- 코드 내 `platform.system()` 직접 호출 금지
- 반드시 `config.py`에 정의된 변수 사용

```python
# Good (config.py에서 가져옴)
from config import NAS_PATH, IS_MAC

# Bad
import platform
if platform.system() == "Darwin":
    ...
```

## 민감 정보
- API 키, 토큰, 비밀번호 하드코딩 절대 금지
- `.env` 파일 경유 후 `python-dotenv`로 로드
- Firebase 키 파일 경로도 `.env`에서 읽기
