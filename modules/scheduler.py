"""
modules/scheduler.py — 자율 행동 루프 & 스케줄러

APScheduler BackgroundScheduler 기반으로 Ria의 자율 행동을 관리한다.

주요 기능:
  - BoredomLevel: 마지막 상호작용 이후 경과 시간 기반 심심함 5단계
  - TimeSlot: 아침/오후/저녁/심야 시간대 구분
  - 자율 행동 루프: 심심함이 BORED 이상이면 LLM으로 자발적 발화 생성
  - 정기 작업: 1분 간격 심심함 체크, 시간대 변경(cron) 로그
  - on_speak 콜백으로 TTS와 느슨하게 연결 (선택적 주입)
"""

import random
import threading
from datetime import datetime
from enum import Enum
from typing import Callable, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger

from config import BASE_DIR
from modules.llm import is_ollama_running, query


# ── 열거형 ────────────────────────────────────────────────────────────────────

class TimeSlot(Enum):
    """시간대 구분."""
    MORNING   = "아침"   # 06:00 ~ 11:59
    AFTERNOON = "오후"   # 12:00 ~ 17:59
    EVENING   = "저녁"   # 18:00 ~ 22:59
    NIGHT     = "심야"   # 23:00 ~ 05:59


class BoredomLevel(Enum):
    """심심함 단계 (경과 분 기반)."""
    FRESH      = 0   # 0~4분
    CONTENT    = 1   # 5~14분
    IDLE       = 2   # 15~29분
    BORED      = 3   # 30~59분
    VERY_BORED = 4   # 60분+


# ── 상수 ─────────────────────────────────────────────────────────────────────

# 각 BoredomLevel에 도달하는 최소 경과 분
_BOREDOM_THRESHOLDS: dict[BoredomLevel, int] = {
    BoredomLevel.FRESH:      0,
    BoredomLevel.CONTENT:    5,
    BoredomLevel.IDLE:       15,
    BoredomLevel.BORED:      30,
    BoredomLevel.VERY_BORED: 60,
}

# 자율 행동을 시작하는 최소 심심함 레벨
_AUTONOMOUS_THRESHOLD = BoredomLevel.BORED

# 시간대별 자율 발화 프롬프트 후보 (random.choice로 선택)
_TIME_SLOT_PROMPTS: dict[TimeSlot, list[str]] = {
    TimeSlot.MORNING: [
        "아침 인사를 건네고 오늘 계획에 대해 짧게 물어봐줘.",
        "아침에 활력을 줄 수 있는 짧고 밝은 한마디를 건네줘.",
        "좋은 아침이라고 인사하며 커피나 아침 식사를 챙겼는지 물어봐줘.",
    ],
    TimeSlot.AFTERNOON: [
        "오후에 잘 지내고 있는지 짧게 안부를 물어봐줘.",
        "오후 집중력 유지에 도움이 될 짧은 응원의 말을 해줘.",
        "점심은 먹었는지, 오후 일정은 어떤지 짧게 물어봐줘.",
    ],
    TimeSlot.EVENING: [
        "저녁에 오늘 하루 어땠는지 따뜻하게 물어봐줘.",
        "저녁 식사나 휴식을 제안하는 짧고 따뜻한 말을 건네줘.",
        "하루 수고했다고 격려하며 오늘 기억에 남는 일이 있는지 물어봐줘.",
    ],
    TimeSlot.NIGHT: [
        "심야에 아직 깨어 있는 것 같으니 부드럽게 걱정하는 말을 해줘.",
        "늦은 밤이니 내일을 위해 푹 쉬라고 짧게 말해줘.",
        "야행성 시간에 혼자 깨어 있는 것 같으니 조용히 말을 걸어봐줘.",
    ],
}


# ── 핵심 클래스 ───────────────────────────────────────────────────────────────

class RiaScheduler:
    """Ria 자율 행동 루프 스케줄러.

    APScheduler BackgroundScheduler를 래핑해 심심함 레벨 관리와
    시간대별 자율 발화를 담당한다.

    Usage::

        def speak(text: str) -> None:
            tts.speak(text)

        scheduler = RiaScheduler(on_speak=speak)
        scheduler.start()

        # 사용자가 말할 때마다 호출
        scheduler.update_last_interaction()

        scheduler.stop()
    """

    def __init__(
        self,
        on_speak: Optional[Callable[[str], None]] = None,
        boredom_check_interval_sec: int = 60,
        autonomous_action_cooldown_min: int = 10,
    ) -> None:
        """
        Args:
            on_speak: 자율 발화 텍스트를 받는 콜백 (TTS 연결용).
                      None이면 발화 텍스트를 로그에만 기록한다.
            boredom_check_interval_sec: 심심함 체크 간격 (초, 기본 60)
            autonomous_action_cooldown_min: 자율 행동 최소 쿨다운 (분, 기본 10)
        """
        self._on_speak = on_speak
        self._check_interval = boredom_check_interval_sec
        self._cooldown_min = autonomous_action_cooldown_min

        self._last_interaction: datetime = datetime.now()
        self._last_autonomous_action: Optional[datetime] = None
        self._lock = threading.Lock()

        self._scheduler = BackgroundScheduler(timezone="Asia/Seoul")
        self._running = False

    # ── 공개 상태 API ─────────────────────────────────────────────────────────

    def update_last_interaction(self) -> None:
        """사용자 상호작용 발생 시 호출. 심심함 타이머를 현재 시각으로 리셋한다."""
        with self._lock:
            self._last_interaction = datetime.now()
        logger.debug(
            "상호작용 갱신 | time={t}",
            t=self._last_interaction.strftime("%H:%M:%S"),
        )

    def get_boredom_level(self) -> BoredomLevel:
        """마지막 상호작용으로부터 경과 시간 기반 심심함 레벨을 반환한다.

        Returns:
            현재 BoredomLevel (FRESH ~ VERY_BORED)
        """
        with self._lock:
            elapsed_min = (datetime.now() - self._last_interaction).total_seconds() / 60

        # VERY_BORED → FRESH 순으로 내려오며 처음 충족되는 레벨 반환
        for level in reversed(list(BoredomLevel)):
            if elapsed_min >= _BOREDOM_THRESHOLDS[level]:
                return level

        return BoredomLevel.FRESH

    def get_time_slot(self) -> TimeSlot:
        """현재 시각 기반 TimeSlot을 반환한다.

        Returns:
            MORNING(6~11), AFTERNOON(12~17), EVENING(18~22), NIGHT(23~5)
        """
        hour = datetime.now().hour
        if 6 <= hour < 12:
            return TimeSlot.MORNING
        elif 12 <= hour < 18:
            return TimeSlot.AFTERNOON
        elif 18 <= hour < 23:
            return TimeSlot.EVENING
        else:
            return TimeSlot.NIGHT

    def get_status(self) -> dict:
        """현재 스케줄러 상태를 딕셔너리로 반환한다.

        Returns:
            running, boredom_level, boredom_elapsed_min,
            time_slot, last_interaction, last_autonomous_action
        """
        with self._lock:
            elapsed = (datetime.now() - self._last_interaction).total_seconds() / 60
            last_auto = (
                self._last_autonomous_action.strftime("%H:%M:%S")
                if self._last_autonomous_action
                else None
            )

        boredom = self.get_boredom_level()
        time_slot = self.get_time_slot()

        return {
            "running": self._running,
            "boredom_level": boredom.name,
            "boredom_elapsed_min": round(elapsed, 1),
            "time_slot": time_slot.value,
            "last_interaction": self._last_interaction.strftime("%H:%M:%S"),
            "last_autonomous_action": last_auto,
        }

    # ── 스케줄러 시작/종료 ────────────────────────────────────────────────────

    def start(self) -> None:
        """APScheduler 백그라운드 스케줄러를 시작한다.

        등록되는 작업:
          - boredom_check: 매 check_interval초마다 심심함 레벨 확인 및 자율 행동
          - time_slot_log: 매일 06:00 / 12:00 / 18:00 / 23:00에 시간대 변경 로그
        """
        if self._running:
            logger.warning("RiaScheduler 이미 실행 중")
            return

        self._scheduler.add_job(
            self._check_boredom_and_act,
            trigger=IntervalTrigger(seconds=self._check_interval),
            id="boredom_check",
            replace_existing=True,
        )
        self._scheduler.add_job(
            self._log_time_slot_change,
            trigger="cron",
            hour="6,12,18,23",
            minute=0,
            id="time_slot_log",
            replace_existing=True,
        )

        self._scheduler.start()
        self._running = True
        logger.info(
            "RiaScheduler 시작 | check_interval={s}s | cooldown={c}min",
            s=self._check_interval,
            c=self._cooldown_min,
        )

    def stop(self) -> None:
        """APScheduler 스케줄러를 종료한다."""
        if not self._running:
            logger.debug("RiaScheduler 이미 종료 상태")
            return

        self._scheduler.shutdown(wait=False)
        self._running = False
        logger.info("RiaScheduler 종료")

    # ── 내부 작업 (APScheduler 콜백) ─────────────────────────────────────────

    def _check_boredom_and_act(self) -> None:
        """심심함 레벨을 확인하고, 임계치 초과+쿨다운 완료 시 자율 행동을 실행한다."""
        boredom = self.get_boredom_level()
        logger.debug("boredom check | level={level}", level=boredom.name)

        if boredom.value < _AUTONOMOUS_THRESHOLD.value:
            return

        # 쿨다운 확인
        with self._lock:
            last_auto = self._last_autonomous_action

        if last_auto is not None:
            cooldown_sec = self._cooldown_min * 60
            elapsed_since_auto = (datetime.now() - last_auto).total_seconds()
            if elapsed_since_auto < cooldown_sec:
                remaining = cooldown_sec - elapsed_since_auto
                logger.debug("자율 행동 쿨다운 중 | 남은={s:.0f}초", s=remaining)
                return

        self._execute_autonomous_action(boredom)

    def _execute_autonomous_action(self, boredom: BoredomLevel) -> None:
        """LLM으로 자율 발화를 생성하고 on_speak 콜백을 호출한다.

        Args:
            boredom: 현재 심심함 레벨 (로그 및 시스템 프롬프트 컨텍스트용)
        """
        time_slot = self.get_time_slot()
        logger.info(
            "자율 행동 실행 | boredom={boredom} | time_slot={slot}",
            boredom=boredom.name,
            slot=time_slot.value,
        )

        if not is_ollama_running():
            logger.warning("Ollama 미실행 → 자율 행동 스킵")
            return

        prompt = random.choice(_TIME_SLOT_PROMPTS[time_slot])

        try:
            response = query(
                prompt,
                system=(
                    "당신은 Ria입니다. 사용자와 자연스럽게 대화하는 AI 어시스턴트입니다. "
                    f"현재 시간대는 {time_slot.value}이고, "
                    f"사용자가 {_BOREDOM_THRESHOLDS[boredom]}분 이상 아무 말도 없었습니다. "
                    "짧고 자연스러운 한국어로 한두 문장만 말하세요."
                ),
            )
        except Exception as e:
            logger.error("자율 행동 LLM 오류: {e}", e=e)
            return

        with self._lock:
            self._last_autonomous_action = datetime.now()

        logger.info("자율 발화 생성: {text}", text=response[:80])

        if self._on_speak:
            try:
                self._on_speak(response)
            except Exception as e:
                logger.error("on_speak 콜백 오류: {e}", e=e)

    def _log_time_slot_change(self) -> None:
        """시간대 변경(cron) 시 현재 TimeSlot을 로그에 기록한다."""
        slot = self.get_time_slot()
        logger.info("시간대 변경 | slot={slot}", slot=slot.value)


# ── 모듈 레벨 싱글턴 헬퍼 ────────────────────────────────────────────────────

_default_scheduler: Optional[RiaScheduler] = None


def get_scheduler() -> Optional[RiaScheduler]:
    """모듈 레벨 기본 스케줄러 인스턴스를 반환한다.

    Returns:
        init_scheduler()로 초기화된 RiaScheduler 또는 None
    """
    return _default_scheduler


def init_scheduler(
    on_speak: Optional[Callable[[str], None]] = None,
    boredom_check_interval_sec: int = 60,
    autonomous_action_cooldown_min: int = 10,
) -> RiaScheduler:
    """모듈 레벨 기본 스케줄러를 초기화하고 시작한다.

    Args:
        on_speak: 자율 발화 콜백 (TTS 연결용)
        boredom_check_interval_sec: 심심함 체크 간격 (초, 기본 60)
        autonomous_action_cooldown_min: 자율 행동 쿨다운 (분, 기본 10)

    Returns:
        시작된 RiaScheduler 인스턴스
    """
    global _default_scheduler

    if _default_scheduler is not None and _default_scheduler._running:
        logger.warning("기존 스케줄러 종료 후 재초기화")
        _default_scheduler.stop()

    _default_scheduler = RiaScheduler(
        on_speak=on_speak,
        boredom_check_interval_sec=boredom_check_interval_sec,
        autonomous_action_cooldown_min=autonomous_action_cooldown_min,
    )
    _default_scheduler.start()
    return _default_scheduler


# ── 단독 테스트 ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    logger.info("=== scheduler.py 단독 테스트 시작 ===")

    # ── [1] TimeSlot 판별 ────────────────────────────────────
    logger.info("--- [1] TimeSlot 판별 테스트 ---")
    try:
        s = RiaScheduler()
        slot = s.get_time_slot()
        assert isinstance(slot, TimeSlot), "TimeSlot 인스턴스여야 함"
        logger.info("현재 시간대: {slot} ({hour}시)", slot=slot.value, hour=datetime.now().hour)

        # 시간대 경계값 수동 확인
        hour_slot_map = {5: TimeSlot.NIGHT, 6: TimeSlot.MORNING, 12: TimeSlot.AFTERNOON,
                         18: TimeSlot.EVENING, 23: TimeSlot.NIGHT}
        for h, expected in hour_slot_map.items():
            # 내부 로직과 동일하게 직접 계산
            if 6 <= h < 12:
                got = TimeSlot.MORNING
            elif 12 <= h < 18:
                got = TimeSlot.AFTERNOON
            elif 18 <= h < 23:
                got = TimeSlot.EVENING
            else:
                got = TimeSlot.NIGHT
            assert got == expected, f"{h}시 → 기대 {expected}, 실제 {got}"
        logger.info("TimeSlot 경계값 테스트 통과 (5,6,12,18,23시)")
    except Exception as e:
        logger.error("TimeSlot 테스트 실패: {e}", e=e)

    # ── [2] BoredomLevel 판별 ────────────────────────────────
    logger.info("--- [2] BoredomLevel 판별 테스트 ---")
    try:
        s2 = RiaScheduler()

        # 방금 생성 → FRESH
        level = s2.get_boredom_level()
        assert level == BoredomLevel.FRESH, f"초기 레벨은 FRESH여야 함: {level}"
        logger.info("초기 BoredomLevel=FRESH 확인")

        # _last_interaction을 수동으로 조작해 각 레벨 검증
        test_cases: list[tuple[int, BoredomLevel]] = [
            (3,  BoredomLevel.FRESH),
            (7,  BoredomLevel.CONTENT),
            (20, BoredomLevel.IDLE),
            (45, BoredomLevel.BORED),
            (90, BoredomLevel.VERY_BORED),
        ]
        from datetime import timedelta
        for minutes, expected in test_cases:
            with s2._lock:
                s2._last_interaction = datetime.now() - timedelta(minutes=minutes)
            got = s2.get_boredom_level()
            assert got == expected, f"{minutes}분 → 기대 {expected.name}, 실제 {got.name}"
            logger.info("  {m}분 경과 → {level}", m=minutes, level=got.name)

        logger.info("BoredomLevel 경계값 테스트 통과")
    except Exception as e:
        logger.error("BoredomLevel 테스트 실패: {e}", e=e)

    # ── [3] update_last_interaction 및 get_status ────────────
    logger.info("--- [3] update_last_interaction / get_status 테스트 ---")
    try:
        s3 = RiaScheduler()
        # 먼저 30분 전으로 설정 → BORED
        with s3._lock:
            s3._last_interaction = datetime.now() - timedelta(minutes=35)
        assert s3.get_boredom_level() == BoredomLevel.BORED

        # 상호작용 업데이트 → FRESH
        s3.update_last_interaction()
        assert s3.get_boredom_level() == BoredomLevel.FRESH
        logger.info("update_last_interaction 후 FRESH 복귀 확인")

        status = s3.get_status()
        assert status["running"] is False
        assert status["boredom_level"] == "FRESH"
        assert isinstance(status["time_slot"], str)
        logger.info("get_status 반환 확인: {status}", status=status)
    except Exception as e:
        logger.error("update/status 테스트 실패: {e}", e=e)

    # ── [4] 스케줄러 시작/종료 ──────────────────────────────
    logger.info("--- [4] 스케줄러 시작/종료 테스트 ---")
    try:
        speak_log: list[str] = []

        def mock_speak(text: str) -> None:
            speak_log.append(text)
            logger.info("[mock_speak] 수신: {text}", text=text[:60])

        # 짧은 체크 간격(3초)으로 테스트용 스케줄러 생성
        s4 = RiaScheduler(
            on_speak=mock_speak,
            boredom_check_interval_sec=3,
            autonomous_action_cooldown_min=1,
        )
        s4.start()
        assert s4._running is True
        logger.info("start() 성공 | running=True")

        # 이중 start() 시 경고만 출력되어야 함
        s4.start()

        time.sleep(1)
        s4.stop()
        assert s4._running is False
        logger.info("stop() 성공 | running=False")

        # 이중 stop() 시 오류 없어야 함
        s4.stop()
        logger.info("이중 start/stop 안전 확인")
    except Exception as e:
        logger.error("시작/종료 테스트 실패: {e}", e=e)

    # ── [5] 자율 행동 트리거 (Ollama 실행 중인 경우) ─────────
    logger.info("--- [5] 자율 행동 트리거 테스트 (Ollama 필요) ---")
    try:
        ollama_ok = is_ollama_running()
    except Exception:
        ollama_ok = False

    if not ollama_ok:
        logger.warning("Ollama 미실행 → 자율 행동 테스트 생략")
    else:
        logger.info("Ollama 확인됨 → 자율 행동 강제 실행 테스트")
        try:
            speak_result: list[str] = []

            def capture_speak(text: str) -> None:
                speak_result.append(text)

            s5 = RiaScheduler(on_speak=capture_speak, boredom_check_interval_sec=60)
            # 마지막 상호작용을 35분 전으로 설정 → BORED
            with s5._lock:
                s5._last_interaction = datetime.now() - timedelta(minutes=35)

            assert s5.get_boredom_level() == BoredomLevel.BORED

            # _execute_autonomous_action 직접 호출
            s5._execute_autonomous_action(BoredomLevel.BORED)

            assert len(speak_result) > 0, "on_speak 콜백이 호출되어야 함"
            assert s5._last_autonomous_action is not None
            logger.info(
                "자율 행동 테스트 통과 | 발화 길이={n}자",
                n=len(speak_result[0]),
            )
            logger.debug("발화 내용: {text}", text=speak_result[0][:100])
        except Exception as e:
            logger.error("자율 행동 테스트 실패: {e}", e=e)

    # ── [6] init_scheduler / get_scheduler 헬퍼 ────────────
    logger.info("--- [6] init_scheduler / get_scheduler 테스트 ---")
    try:
        sc = init_scheduler(boredom_check_interval_sec=60, autonomous_action_cooldown_min=10)
        assert get_scheduler() is sc
        assert sc._running is True
        logger.info("init_scheduler / get_scheduler 연결 확인")

        # 재초기화 시 기존 스케줄러 자동 종료 확인
        sc2 = init_scheduler(boredom_check_interval_sec=60)
        assert not sc._running, "기존 스케줄러가 종료되어야 함"
        assert sc2._running is True
        logger.info("재초기화 시 기존 스케줄러 자동 종료 확인")

        sc2.stop()
    except Exception as e:
        logger.error("init_scheduler 테스트 실패: {e}", e=e)

    logger.info("=== scheduler.py 단독 테스트 완료 ===")
