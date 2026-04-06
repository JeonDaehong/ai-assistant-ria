# 보안 규칙

## 민감 정보 하드코딩 금지
- 코드 내 API 키, 토큰, 비밀번호 발견 즉시 지적 후 수정 요청
- `.env` 파일로 분리, `python-dotenv`로 로드
- Git 커밋 전 민감 정보 포함 여부 반드시 확인

```python
# Bad — 즉시 차단
FIREBASE_KEY = "AIzaSy..."
OPENAI_API_KEY = "sk-..."

# Good
import os
from dotenv import load_dotenv
load_dotenv()
FIREBASE_KEY = os.getenv("FIREBASE_KEY")
```

## FastAPI 바인딩 제한
- FastAPI(uvicorn) 실행 시 반드시 `127.0.0.1` 바인딩만 허용
- `0.0.0.0` 바인딩 금지 (외부 노출 위험)

```python
# Good
uvicorn.run(app, host="127.0.0.1", port=8000)

# Bad — 외부 노출
uvicorn.run(app, host="0.0.0.0", port=8000)
```

## 로그 마스킹
- 로그에 `password=`, `token=`, `key=`, `secret=` 포함 금지
- 민감 값은 마스킹 처리 후 기록

```python
# Bad
logger.debug("Firebase 토큰: {token}", token=firebase_token)

# Good
masked = firebase_token[:6] + "****" if firebase_token else "None"
logger.debug("Firebase 토큰: {masked}", masked=masked)
```

## Firebase 키 파일
- `firebase-key.json` 코드에 경로 하드코딩 금지
- `.env`의 `FIREBASE_KEY_PATH` 변수로 경로 관리
- `.gitignore`에 `firebase-key.json` 반드시 추가
- 키 파일을 코드 저장소에 절대 커밋 금지

## 파일 경로 보안
- 사용자 입력을 파일 경로로 직접 사용 금지 (Path Traversal 방지)
- `pathlib.Path.resolve()`로 절대 경로 정규화 후 허용 경로 내 여부 확인

```python
from pathlib import Path

def safe_read(user_path: str, base_dir: Path) -> str:
    resolved = (base_dir / user_path).resolve()
    if not str(resolved).startswith(str(base_dir.resolve())):
        raise PermissionError("허용되지 않은 경로 접근")
    return resolved.read_text()
```

## 보안 감사 체크리스트
코드 리뷰 시 아래 항목 전체 확인:
- [ ] 하드코딩된 API 키 / 토큰 없음
- [ ] FastAPI `0.0.0.0` 바인딩 없음
- [ ] 로그에 민감 정보 마스킹됨
- [ ] `firebase-key.json` 코드 내 미포함
- [ ] 사용자 입력 경로 검증 로직 있음
