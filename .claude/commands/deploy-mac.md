# /project:deploy-mac

Mac Mini M4로 이전하기 전 체크리스트를 실행합니다.

## 사용법
```
/project:deploy-mac
```

## 체크리스트

### 1. requirements.txt 최신화 확인
현재 설치된 패키지와 `requirements.txt` 비교:
```bash
pip freeze > requirements_current.txt
diff requirements.txt requirements_current.txt
```
차이 발견 시 `requirements.txt` 업데이트 후 진행.

### 2. .env 항목 누락 확인
`config.py`에서 `os.getenv()` 호출되는 모든 키 목록 추출 후 `.env.example` 대조:
- 누락된 항목이 있으면 `.env.example`에 추가 (값은 빈칸)
- Mac Mini 환경의 `.env`에 실제 값 설정 안내

필수 환경 변수 목록:
```
FIREBASE_KEY_PATH=
OLLAMA_HOST=
NAS_PATH=
TTS_MODEL=
STT_MODEL=
```

### 3. config.py DEVICE 분기 확인
`config.py`에서 Mac/Windows 분기 로직 동작 확인:
```python
# config.py 확인 사항
IS_MAC = ...       # Mac Mini에서 True 여부
NAS_PATH = ...     # /Volumes/NAS (Mac) 경로 정확한지
DEVICE = ...       # "mps" (Mac M4) 또는 "cpu"
```

### 4. NAS 경로 확인
Mac Mini 환경에서 NAS 마운트 경로 확인:
```bash
ls /Volumes/NAS
```
마운트 안 된 경우 Finder에서 NAS 연결 후 재시도.

### 5. Ollama 모델 준비
```bash
ollama pull eeve-korean-instruct:13b
ollama list
```
모델 다운로드 확인 (약 8GB, 시간 소요).

## 이전 완료 후
- [ ] Mac Mini에서 `python main.py --no-hw` 실행해 텍스트 파이프라인 테스트
- [ ] 마이크 / 스피커 연결 확인 후 전체 파이프라인 테스트
- [ ] 로그 파일 경로 Mac 환경에 맞게 확인
