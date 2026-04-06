# /project:new-module

새 모듈을 생성하고 품질 기준에 맞게 초기화합니다.

## 사용법
```
/project:new-module <module_name>
```

## 실행 단계

### 1. 모듈 파일 생성 (`modules/$1.py`)
아래 템플릿으로 생성:
```python
"""
modules/$1.py — [모듈 설명]
"""
from loguru import logger
# 필요한 import 추가


def main_function(param: str) -> str:
    """함수 설명.
    
    Args:
        param: 매개변수 설명
        
    Returns:
        반환값 설명
        
    Raises:
        ValueError: 잘못된 입력 시
    """
    logger.info("$1 실행: {param}", param=param)
    # 구현
    return result


if __name__ == "__main__":
    from loguru import logger

    logger.info("=== $1 단독 테스트 ===")

    # 정상 케이스
    try:
        result = main_function("test_input")
        logger.info("정상 케이스 통과: {result}", result=result)
    except Exception as e:
        logger.error("정상 케이스 실패: {e}", e=e)

    # 에러 케이스
    try:
        result = main_function("")
        logger.warning("에러 케이스: 빈 입력이 통과됨 (의도된 동작인지 확인)")
    except ValueError as e:
        logger.info("에러 케이스 정상 처리: {e}", e=e)
```

### 2. 단독 테스트 실행
```bash
python modules/$1.py
```
테스트 통과 확인 필수.

### 3. /project:review 자동 실행
생성된 모듈에 대해 코드 리뷰 수행.

### 4. CLAUDE.md 체크리스트 업데이트
`modules/$1.py` 항목에 체크 표시:
```
- [x] `modules/$1.py` — [설명]
```
