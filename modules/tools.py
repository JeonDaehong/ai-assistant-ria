"""
modules/tools.py — Ollama Tool Calling 통합 모듈

LLM이 직접 호출할 수 있는 3가지 도구를 정의하고,
Ollama /api/chat의 tools 파라미터를 활용해 tool calling 흐름을 처리한다.

Tools:
    - file_search: 디렉토리 내 파일 이름 패턴 검색
    - web_search: DuckDuckGo 웹 검색
    - set_alarm: 지정 시각에 알람 설정 (백그라운드 스레드)
"""

import json
import re
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests
from loguru import logger

from config import BASE_DIR, IS_WINDOWS, LLM_MODEL, LLM_TIMEOUT, OLLAMA_HOST


# ── Tool 정의 (Ollama tools 포맷) ─────────────────────────────────────────────

file_search_tool_def: dict = {
    "type": "function",
    "function": {
        "name": "file_search",
        "description": "지정한 디렉토리에서 파일을 이름 패턴으로 검색합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "검색할 디렉토리 경로",
                },
                "pattern": {
                    "type": "string",
                    "description": "파일 이름 패턴 (glob, 예: *.py, report*.xlsx)",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "하위 디렉토리 포함 여부",
                    "default": True,
                },
            },
            "required": ["directory", "pattern"],
        },
    },
}

web_search_tool_def: dict = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "DuckDuckGo에서 검색어로 웹을 검색하고 상위 결과를 반환합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "검색어",
                },
                "max_results": {
                    "type": "integer",
                    "description": "반환할 최대 결과 수 (기본 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}

set_alarm_tool_def: dict = {
    "type": "function",
    "function": {
        "name": "set_alarm",
        "description": "지정한 시각에 알람을 설정합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "time": {
                    "type": "string",
                    "description": "알람 시각 (HH:MM 형식, 24시간제)",
                },
                "label": {
                    "type": "string",
                    "description": "알람 레이블 (메모)",
                    "default": "알람",
                },
            },
            "required": ["time"],
        },
    },
}

obsidian_search_tool_def: dict = {
    "type": "function",
    "function": {
        "name": "obsidian_search",
        "description": "Obsidian 볼트(지식 저장소)에서 노트를 검색합니다. 사용자가 메모, 지식, 기록을 물어볼 때 사용하세요.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "검색어",
                },
                "max_results": {
                    "type": "integer",
                    "description": "반환할 최대 결과 수 (기본 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}

obsidian_read_tool_def: dict = {
    "type": "function",
    "function": {
        "name": "obsidian_read",
        "description": "Obsidian 볼트에서 특정 노트의 전체 내용을 읽습니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "note_path": {
                    "type": "string",
                    "description": "노트 경로 또는 제목 (예: '일기/2024-01-01' 또는 '회의록')",
                },
            },
            "required": ["note_path"],
        },
    },
}

TOOLS: list[dict] = [
    file_search_tool_def,
    web_search_tool_def,
    set_alarm_tool_def,
    obsidian_search_tool_def,
    obsidian_read_tool_def,
]


# ── Tool 1: 파일 탐색 ──────────────────────────────────────────────────────────

def file_search(
    directory: str,
    pattern: str,
    recursive: bool = True,
) -> list[str]:
    """지정한 디렉토리에서 파일 이름 패턴으로 파일을 검색한다.

    Args:
        directory: 검색할 디렉토리 경로 문자열
        pattern: glob 패턴 (예: *.py, report*.xlsx)
        recursive: 하위 디렉토리 포함 여부 (기본 True)

    Returns:
        매칭된 파일의 절대 경로 문자열 리스트 (최대 50개).
        디렉토리가 없거나 매칭 파일이 없으면 빈 리스트.
    """
    base = Path(directory).resolve()

    if not base.exists() or not base.is_dir():
        logger.warning("file_search: 디렉토리 없음 → {dir}", dir=str(base))
        return []

    logger.info(
        "file_search 실행 | dir={dir} | pattern={pat} | recursive={rec}",
        dir=str(base),
        pat=pattern,
        rec=recursive,
    )

    try:
        if recursive:
            matches = list(base.rglob(pattern))
        else:
            matches = list(base.glob(pattern))
    except Exception as e:
        logger.error("file_search glob 오류: {e}", e=e)
        return []

    # 파일만 필터링, 최대 50개 제한
    result = [str(p.resolve()) for p in matches if p.is_file()][:50]
    logger.info("file_search 결과: {n}개", n=len(result))
    return result


# ── Tool 2: 웹 검색 ────────────────────────────────────────────────────────────

def web_search(query: str, max_results: int = 5) -> list[dict]:
    """DuckDuckGo에서 검색어로 웹을 검색하고 상위 결과를 반환한다.

    Args:
        query: 검색어
        max_results: 반환할 최대 결과 수 (기본 5)

    Returns:
        검색 결과 딕셔너리 리스트. 각 항목은 {"title": str, "href": str, "body": str}.
        실패 시 빈 리스트.
    """
    if not query or not query.strip():
        logger.warning("web_search: 빈 쿼리 입력")
        return []

    logger.info("web_search 실행 | query={q} | max={n}", q=query, n=max_results)

    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS  # type: ignore[no-redef]

        results = list(DDGS().text(query, max_results=max_results))
        logger.info("web_search 결과: {n}개", n=len(results))
        return results
    except ImportError:
        logger.error(
            "웹 검색 패키지 미설치. pip install ddgs 또는 pip install duckduckgo-search"
        )
        return []
    except Exception as e:
        logger.error("web_search 실패: {e}", e=e)
        return []


# ── Tool 3: 알람 설정 ──────────────────────────────────────────────────────────

def _fire_alarm(label: str) -> None:
    """알람 발동 시 호출되는 내부 함수. 로그 출력 및 toast 알림 시도."""
    logger.info("알람: {label}", label=label)

    if IS_WINDOWS:
        try:
            from win10toast import ToastNotifier

            notifier = ToastNotifier()
            notifier.show_toast(
                "Ria 알람",
                label,
                duration=10,
                threaded=True,
            )
            logger.info("Windows toast 알림 전송 완료")
        except ImportError:
            logger.warning("win10toast 미설치 → toast 알림 생략. pip install win10toast")
        except Exception as e:
            logger.warning("toast 알림 실패 (무시): {e}", e=e)


def _parse_alarm_time(time_str: str) -> datetime:
    """HH:MM 문자열을 파싱해 오늘 또는 내일의 datetime을 반환한다.

    Args:
        time_str: "HH:MM" 형식의 24시간제 시각 문자열

    Returns:
        알람 발동 시각 datetime 객체 (과거면 내일로 자동 조정)

    Raises:
        ValueError: 올바르지 않은 HH:MM 형식
    """
    try:
        hour, minute = map(int, time_str.strip().split(":"))
    except (ValueError, AttributeError) as e:
        raise ValueError(f"잘못된 알람 시각 형식: '{time_str}' (HH:MM 필요)") from e

    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"시각 범위 초과: '{time_str}' (00:00 ~ 23:59)")

    now = datetime.now()
    trigger = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

    if trigger <= now:
        trigger += timedelta(days=1)
        logger.debug("알람 시각이 과거 → 내일로 조정: {t}", t=trigger.strftime("%Y-%m-%d %H:%M"))

    return trigger


def set_alarm(time: str, label: str = "알람") -> dict:
    """지정한 시각에 알람을 설정한다. 백그라운드 스레드로 실행된다.

    Args:
        time: 알람 시각 문자열 (HH:MM, 24시간제)
        label: 알람 레이블 메모 (기본 "알람")

    Returns:
        {"status": "set", "label": str, "trigger_at": str, "seconds_until": float}

    Raises:
        ValueError: 잘못된 time 포맷
    """
    trigger = _parse_alarm_time(time)
    now = datetime.now()
    seconds_until = (trigger - now).total_seconds()

    timer = threading.Timer(seconds_until, _fire_alarm, args=[label])
    timer.daemon = True
    timer.start()

    result = {
        "status": "set",
        "label": label,
        "trigger_at": trigger.strftime("%Y-%m-%d %H:%M"),
        "seconds_until": round(seconds_until, 1),
    }

    logger.info(
        "알람 설정 완료 | label={label} | trigger={t} | 남은시간={s}초",
        label=label,
        t=result["trigger_at"],
        s=result["seconds_until"],
    )
    return result


# ── Tool 4: Obsidian 검색 ─────────────────────────────────────────────────────

def obsidian_search(query: str, max_results: int = 5) -> list[dict]:
    """Obsidian 볼트에서 쿼리로 노트를 검색한다.

    Args:
        query: 검색어
        max_results: 반환할 최대 결과 수 (기본 5)

    Returns:
        [{"path": str, "title": str, "snippet": str}, ...] 리스트.
        볼트 없음 또는 결과 없음 시 빈 리스트.
    """
    from modules.obsidian import search_notes
    return search_notes(query, max_results)


def obsidian_read(note_path: str) -> dict:
    """Obsidian 볼트에서 노트 내용을 읽는다.

    Args:
        note_path: 노트 경로 또는 제목

    Returns:
        {"path": str, "content": str} 또는 {"error": str}
    """
    from modules.obsidian import get_note
    content = get_note(note_path)
    if content is None:
        return {"error": f"노트를 찾을 수 없습니다: {note_path}"}
    return {"path": note_path, "content": content}


# ── Tool 디스패처 ───────────────────────────────────────────────────────────────

_TOOL_REGISTRY: dict = {
    "file_search": file_search,
    "web_search": web_search,
    "set_alarm": set_alarm,
    "obsidian_search": obsidian_search,
    "obsidian_read": obsidian_read,
}


def dispatch_tool(name: str, arguments: dict) -> str:
    """tool 이름과 인수를 받아 실제 함수를 호출하고 JSON 문자열로 반환한다.

    Args:
        name: tool 이름 (file_search / web_search / set_alarm)
        arguments: tool에 전달할 인수 딕셔너리

    Returns:
        실행 결과를 JSON 직렬화한 문자열.
        알 수 없는 이름이면 {"error": "unknown tool: <name>"} 반환.
    """
    func = _TOOL_REGISTRY.get(name)

    if func is None:
        logger.warning("dispatch_tool: 알 수 없는 tool → {name}", name=name)
        return json.dumps({"error": f"unknown tool: {name}"}, ensure_ascii=False)

    logger.debug("dispatch_tool 실행 | name={name} | args={args}", name=name, args=arguments)

    try:
        result = func(**arguments)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.error("dispatch_tool 실행 오류 | name={name} | {e}", name=name, e=e)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ── LLM + Tool Calling 통합 ────────────────────────────────────────────────────

def _build_url(path: str) -> str:
    """Ollama API 엔드포인트 URL 조합."""
    return f"{OLLAMA_HOST.rstrip('/')}{path}"


def _post_chat(messages: list[dict], model: str) -> dict:
    """Ollama /api/chat에 POST 요청을 보내고 응답 JSON을 반환한다.

    Args:
        messages: 메시지 배열
        model: 사용할 모델명

    Returns:
        Ollama 응답 JSON 딕셔너리

    Raises:
        ConnectionError: Ollama 서버 미실행 시
        RuntimeError: API 오류 응답 시
    """
    payload: dict = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "num_gpu": 999,
            "num_ctx": 2048,
        },
    }

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

    return resp.json()


_TOOL_DETECTION_PROMPT: str = """당신은 Ria입니다. 사용자 질문에 답변하기 전, 아래 도구가 필요한지 판단하세요.

[사용 가능한 도구]
1. file_search — 디렉토리 내 파일을 이름 패턴으로 검색
   args: directory(str, 필수), pattern(str, 필수), recursive(bool, 기본 true)

2. web_search — DuckDuckGo 웹 검색
   args: query(str, 필수), max_results(int, 기본 5)

3. set_alarm — 지정 시각에 알람 설정 (HH:MM 24시간제)
   args: time(str, 필수), label(str, 기본 "알람")

4. obsidian_search — Obsidian 볼트(개인 지식 저장소) 노트 검색
   사용자가 메모, 기록, 지식, 노트를 물어볼 때 사용
   args: query(str, 필수), max_results(int, 기본 5)

5. obsidian_read — Obsidian 특정 노트 내용 전체 읽기
   args: note_path(str, 필수) — 노트 제목 또는 경로

[응답 규칙]
- 도구가 필요하면: JSON 한 줄만 출력 (다른 텍스트 없이)
  형식: {{"tool": "도구이름", "args": {{"파라미터": "값"}}}}
  예시: {{"tool": "web_search", "args": {{"query": "파이썬 asyncio", "max_results": 3}}}}
- 도구가 필요 없으면: 자연어로 바로 답변

{user_system}"""


def _extract_tool_call(text: str) -> Optional[dict]:
    """LLM 응답에서 tool call JSON을 추출한다.

    코드 블록(```json ... ```) 또는 인라인 JSON 객체를 모두 탐색한다.

    Args:
        text: LLM 응답 텍스트

    Returns:
        {"tool": str, "args": dict} 딕셔너리. 없거나 파싱 실패 시 None.
    """
    # 코드 블록 우선 탐색
    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = [code_block.group(1)] if code_block else []

    # 인라인 JSON 객체 탐색 (중첩 대괄호 고려)
    for m in re.finditer(r"\{", text):
        start = m.start()
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : i + 1])
                    break

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue

        tool_name = parsed.get("tool")
        args = parsed.get("args")

        if not isinstance(tool_name, str) or tool_name not in _TOOL_REGISTRY:
            continue
        if not isinstance(args, dict):
            continue

        logger.debug("_extract_tool_call: tool={name} | args={args}", name=tool_name, args=args)
        return {"tool": tool_name, "args": args}

    return None


def query_with_tools(
    prompt: str,
    system: str = "당신은 Ria입니다. 필요하면 도구를 사용해 정확히 답변하세요.",
    model: str = LLM_MODEL,
    history: Optional[list[dict]] = None,
) -> str:
    """프롬프트 기반 tool 판단 + Ollama 쿼리.

    LLM에게 JSON 형식으로 tool 이름과 파라미터를 반환하도록 유도하고,
    응답을 파싱해 dispatch_tool()을 직접 호출한다.
    eeve 등 tool calling API를 지원하지 않는 모델에서도 동작한다.

    Args:
        prompt: 사용자 입력 텍스트
        system: 추가 시스템 지시 (도구 탐지 프롬프트에 삽입됨)
        model: 사용할 Ollama 모델명
        history: 이전 대화 이력 [{"role": "user"/"assistant", "content": "..."}]

    Returns:
        LLM 최종 응답 텍스트 (도구 결과 반영 포함)

    Raises:
        ConnectionError: Ollama 서버 미실행 시
        RuntimeError: API 오류 응답 시
    """
    logger.debug(
        "query_with_tools 요청 | model={model} | prompt_len={n}",
        model=model,
        n=len(prompt),
    )

    # 도구 탐지용 시스템 프롬프트 조합
    detection_system = _TOOL_DETECTION_PROMPT.format(user_system=system)

    messages: list[dict] = [{"role": "system", "content": detection_system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    # 1차 요청: tool 필요 여부 판단
    first_json = _post_chat(messages, model)
    first_content: str = first_json.get("message", {}).get("content", "")

    logger.debug("1차 응답 (앞 200자): {c}", c=first_content[:200])

    tool_call = _extract_tool_call(first_content)

    # tool 불필요 → 바로 반환
    if tool_call is None:
        logger.debug("query_with_tools: tool 불필요 → 직접 반환 ({n}자)", n=len(first_content))
        return first_content

    # tool 실행
    tool_name = tool_call["tool"]
    tool_args = tool_call["args"]
    logger.info("tool 실행 | name={name} | args={args}", name=tool_name, args=tool_args)
    tool_result = dispatch_tool(tool_name, tool_args)
    logger.debug("tool 결과 (앞 300자): {r}", r=tool_result[:300])

    # 2차 요청: 도구 결과를 user 메시지로 전달해 최종 답변 요청
    messages.append({"role": "assistant", "content": first_content})
    messages.append({
        "role": "user",
        "content": (
            f"도구 실행 결과:\n{tool_result}\n\n"
            "위 결과를 바탕으로 처음 질문에 자연스럽게 답변해주세요."
        ),
    })

    logger.debug("query_with_tools: 도구 결과 포함 2차 요청")
    final_json = _post_chat(messages, model)
    final_content: str = final_json.get("message", {}).get("content", "")

    logger.info("query_with_tools 완료 | 응답_len={n}", n=len(final_content))
    return final_content


# ── 단독 테스트 ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("=== tools.py 단독 테스트 시작 ===")

    # ── file_search 테스트 ──────────────────────────────────
    logger.info("--- [1] file_search 테스트 ---")

    # 정상 케이스: BASE_DIR에서 *.py 검색
    try:
        found = file_search(str(BASE_DIR), "*.py", recursive=True)
        assert isinstance(found, list), "결과가 리스트여야 함"
        logger.info("file_search 정상 케이스 통과 | 발견={n}개", n=len(found))
        for f in found[:5]:
            logger.debug("  - {f}", f=f)
        if len(found) > 5:
            logger.debug("  ... (이하 생략)")
    except Exception as e:
        logger.error("file_search 정상 케이스 실패: {e}", e=e)

    # 에러 케이스: 존재하지 않는 디렉토리
    try:
        not_found = file_search("/nonexistent/path/xyz", "*.py")
        assert not_found == [], f"빈 리스트 기대, 실제: {not_found}"
        logger.info("file_search 에러 케이스 통과 (존재하지 않는 디렉토리 → 빈 리스트)")
    except Exception as e:
        logger.error("file_search 에러 케이스 실패: {e}", e=e)

    # ── web_search 테스트 ───────────────────────────────────
    logger.info("--- [2] web_search 테스트 ---")

    # 정상 케이스
    try:
        results = web_search("Python asyncio 사용법", max_results=3)
        assert isinstance(results, list), "결과가 리스트여야 함"
        logger.info("web_search 정상 케이스 통과 | 결과={n}개", n=len(results))
        for item in results:
            logger.debug(
                "  title={t} | href={h}",
                t=item.get("title", "")[:40],
                h=item.get("href", "")[:60],
            )
    except Exception as e:
        logger.error("web_search 정상 케이스 실패: {e}", e=e)

    # 에러 케이스: 빈 쿼리
    try:
        empty_result = web_search("")
        assert isinstance(empty_result, list), "빈 리스트 또는 리스트여야 함"
        logger.info("web_search 에러 케이스 통과 (빈 쿼리 → 빈 리스트: {r})", r=empty_result)
    except Exception as e:
        logger.info("web_search 에러 케이스 예외 처리됨: {e}", e=e)

    # ── set_alarm 테스트 ────────────────────────────────────
    logger.info("--- [3] set_alarm 테스트 ---")

    # 정상 케이스: 현재 시각 + 1분
    try:
        future = datetime.now() + timedelta(minutes=1)
        alarm_time = future.strftime("%H:%M")
        alarm_result = set_alarm(alarm_time, label="테스트 알람")
        assert alarm_result["status"] == "set", "status가 set이어야 함"
        assert alarm_result["seconds_until"] > 0, "남은 시간이 양수여야 함"
        logger.info(
            "set_alarm 정상 케이스 통과 | trigger={t} | 남은={s}초",
            t=alarm_result["trigger_at"],
            s=alarm_result["seconds_until"],
        )
    except Exception as e:
        logger.error("set_alarm 정상 케이스 실패: {e}", e=e)

    # 에러 케이스: 잘못된 시각 포맷
    try:
        set_alarm("25:99", label="잘못된 알람")
        logger.warning("set_alarm 에러 케이스: ValueError가 발생해야 하는데 통과됨")
    except ValueError as e:
        logger.info("set_alarm 에러 케이스 정상 처리 (ValueError): {e}", e=e)
    except Exception as e:
        logger.error("set_alarm 에러 케이스 예상치 못한 예외: {e}", e=e)

    # ── query_with_tools 테스트 (Ollama 실행 중인 경우만) ───
    logger.info("--- [4] query_with_tools 테스트 (Ollama 필요) ---")

    try:
        check_resp = requests.get(f"{OLLAMA_HOST.rstrip('/')}/api/tags", timeout=5)
        ollama_ok = check_resp.status_code == 200
    except Exception:
        ollama_ok = False

    if not ollama_ok:
        logger.warning("Ollama 서버 미실행 → query_with_tools 테스트 생략")
    else:
        logger.info("Ollama 서버 확인됨 → tool calling 테스트 시작")
        try:
            answer = query_with_tools(
                f"현재 {BASE_DIR} 디렉토리에서 .py 파일을 찾아줘",
                system="당신은 Ria입니다. 필요하면 도구를 사용해 정확히 답변하세요.",
            )
            assert isinstance(answer, str) and len(answer) > 0, "응답이 비어있음"
            logger.info("query_with_tools 통과 | 응답 길이={n}", n=len(answer))
            logger.debug("응답 내용(앞 200자): {r}", r=answer[:200])
        except ConnectionError as e:
            logger.warning("query_with_tools ConnectionError (예상 범위): {e}", e=e)
        except Exception as e:
            logger.error("query_with_tools 실패: {e}", e=e)

    logger.info("=== tools.py 단독 테스트 완료 ===")
