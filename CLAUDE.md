# RIA — AI Personal Assistant

## 프로젝트 개요
Ria는 Mac Mini(M4)와 Windows PC를 연결하는 개인 AI 어시스턴트입니다.
음성 입력 → STT → LLM 추론 → TTS → 음성 출력 파이프라인을 중심으로,
스케줄 관리, NAS 파일 브라우징, Firebase 연동 등을 제공합니다.

## 환경
- **개발 머신**: Windows 11 (D:/ai-assistant-ria/ria/)
- **운영 머신**: Mac Mini M4 (~/ria/)
- **LLM**: Ollama + eeve-korean-instruct:13b (로컬)
- **NAS 경로**: /Volumes/NAS (Mac) / \\NAS\ (Windows)
- **Python**: 3.11+

## 핵심 모듈 체크리스트
- [x] `modules/stt.py` — 음성 → 텍스트 (Whisper)
- [x] `modules/tts.py` — 텍스트 → 음성 (edge-tts + pygame)
- [x] `modules/llm.py` — Ollama API 연동
- [x] `modules/emotion.py` — 텍스트 감정 분석 (KR-BERT)
- [x] `modules/character.py` — VTube Studio WebSocket 연동
- [x] `modules/tools.py` — Tool Calling (파일 탐색, 웹 검색, 알람, Obsidian)
- [x] `modules/obsidian.py` — Obsidian 볼트 연동 (검색, 읽기, 쓰기)
- [x] `modules/scheduler.py` — 자율 행동 루프, 심심함 레벨, 시간대별 행동
- [x] `modules/memory.py` — ChromaDB 장기 기억 저장/검색 (ko-sroberta)
- [ ] `modules/nas_browser.py` — NAS 파일 탐색
- [x] `modules/firebase_client.py` — FCM 푸시 알림 (firebase-admin)
- [x] `main.py` — 파이프라인 통합 진입점
- [x] `config.py` — 환경별 설정 분기

## 코딩 규칙 (rules/ 참조)
1. 모든 함수에 타입 힌트 필수
2. 함수는 단일 책임 원칙
3. 로깅은 `loguru` 사용
4. 모든 모듈에 `if __name__ == "__main__":` 단독 테스트 포함
5. OS 분기는 `config.py`의 변수만 사용 (코드 내 `platform.system()` 직접 호출 금지)
6. 민감 정보 하드코딩 절대 금지 (`.env` 경유)

## 모듈 작성 순서
1. `config.py` — 환경 변수 및 OS 분기
2. `modules/llm.py` — LLM 핵심 연동
3. `modules/stt.py` — 음성 입력
4. `modules/tts.py` — 음성 출력
5. `modules/scheduler.py` — 스케줄
6. `modules/nas_browser.py` — NAS
7. `modules/firebase_client.py` — Firebase
8. `main.py` — 통합

## 자주 쓰는 명령어
- `/project:new-module <name>` — 새 모듈 생성
- `/project:review` — 전체 코드 리뷰
- `/project:deploy-mac` — Mac Mini 이전 체크리스트

## 주의사항
- Windows 개발 중 경로는 `pathlib.Path` 사용 (하드코딩 금지)
- Firebase 키 파일(`firebase-key.json`)은 `.gitignore`에 추가, 코드에 포함 금지
- FastAPI 사용 시 `127.0.0.1` 바인딩만 허용
