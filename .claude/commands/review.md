# /project:review

현재 프로젝트의 모든 코드를 rules/ 기준으로 검토합니다.

## 사용법
```
/project:review
/project:review modules/stt.py   # 특정 파일만
```

## 검토 항목

### 1. 타입 힌트 누락 체크 (code-style.md)
- 모든 함수 매개변수에 타입 힌트 있는지 확인
- 반환 타입 명시 여부 확인
- 누락 발견 시 해당 파일명:라인번호 리포트

### 2. 하드코딩 민감 정보 체크 (security.md)
검색 패턴:
- `= "AIza` — Firebase API 키
- `= "sk-` — OpenAI 키
- `password =`, `token =`, `secret =`, `api_key =` — 직접 할당
- `"firebase-key.json"` — 키 파일 경로 하드코딩
발견 시 즉시 경고 + 수정 방안 제시

### 3. 단독 테스트 코드 존재 여부 체크 (testing.md)
- 각 모듈에 `if __name__ == "__main__":` 블록 존재 확인
- 정상 케이스 + 에러 케이스 모두 포함 여부 확인
- 누락 모듈 목록 출력

### 4. security.md 규칙 위반 체크
- FastAPI `host="0.0.0.0"` 사용 여부
- 로그에 마스킹 없이 민감 값 출력 여부
- `platform.system()` 직접 호출 여부 (config.py 우회)
- `print()` 사용 여부 (loguru 대신)

## 리포트 형식
```
[REVIEW REPORT]
✅ 타입 힌트: 전체 통과
⚠️  단독 테스트 누락: modules/nas_browser.py
❌ 보안 위반: modules/firebase_client.py:12 — 하드코딩된 키 발견
```
