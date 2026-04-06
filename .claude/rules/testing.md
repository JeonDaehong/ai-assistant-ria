# 테스트 규칙

## 모듈 단독 테스트 (필수)
- 모든 모듈 작성 완료 즉시 단독 테스트 실행
- `if __name__ == "__main__":` 블록으로 직접 실행 가능해야 함
- 테스트 통과 전에는 다음 모듈 작성 금지

## 테스트 케이스 구성
단독 테스트는 반드시 아래 두 가지 포함:

### 1. 정상 케이스
- 예상 입력 → 예상 출력 확인
- 실제 사용 시나리오 기반으로 작성

### 2. 에러 케이스
- 파일 없음, 연결 실패, 잘못된 입력 등
- `try/except`로 예외 처리 확인
- 오류가 나도 프로그램이 죽지 않음을 검증

```python
if __name__ == "__main__":
    from loguru import logger

    # --- 정상 케이스 ---
    logger.info("=== 정상 케이스 테스트 ===")
    try:
        result = some_function("valid_input")
        assert result is not None, "결과가 None이면 안 됨"
        logger.info("정상 케이스 통과: {result}", result=result)
    except Exception as e:
        logger.error("정상 케이스 실패: {e}", e=e)

    # --- 에러 케이스 ---
    logger.info("=== 에러 케이스 테스트 ===")
    try:
        result = some_function("invalid_input")
        logger.warning("에러가 발생해야 하는데 통과됨")
    except ValueError as e:
        logger.info("에러 케이스 정상 처리: {e}", e=e)
```

## 통합 테스트
- 모든 모듈 단독 테스트 완료 후 1회만 실행
- `main.py`에서 전체 파이프라인 흐름 검증
- 통합 테스트는 실제 하드웨어(마이크, 스피커) 필요 시 별도 플래그로 스킵 가능

```python
# main.py 통합 테스트
if __name__ == "__main__":
    import sys
    skip_hardware = "--no-hw" in sys.argv

    if not skip_hardware:
        # 마이크 입력 포함 전체 파이프라인
        run_full_pipeline()
    else:
        # 텍스트 입력으로 LLM → TTS만 테스트
        run_text_pipeline("안녕하세요, 테스트입니다.")
```

## 테스트 실행 기록
- 각 모듈 테스트 결과를 CLAUDE.md 체크리스트에 체크
- 실패 시 수정 후 재테스트, 재체크
