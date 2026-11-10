# client_agent/session.py
"""Thread-safe, session-bounded lifecycle manager for telemetry collection."""

import enum
import logging
import math
import threading
import uuid
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("vigilance.agent")


class SessionState(enum.Enum):
    IDLE = "IDLE"
    STARTING = "STARTING"
    ACTIVE = "ACTIVE"
    STOPPING = "STOPPING"


class SessionManager:
    """
    Controls the telemetry worker lifecycle.

    Worker contract:
      * worker_target(stop_event, generation_id, session_info) performs collection.
      * After its initialization succeeds, it MUST call
        manager.mark_worker_ready(generation_id).
      * The worker must check stop_event before every sample and before dispatch.
      * Its finally/cleanup logic must release any resources it owns.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.state = SessionState.IDLE
        self.generation_id: Optional[str] = None
        self.active_session: Optional[Dict[str, Any]] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event: Optional[threading.Event] = None

    @staticmethod
    def _validate_inputs(
        room_id: str, student_id: str, profile_multiplier: float
    ) -> float:
        if not isinstance(room_id, str) or not room_id.strip():
            raise ValueError("room_id must be a non-empty string")
        if not isinstance(student_id, str) or not student_id.strip():
            raise ValueError("student_id must be a non-empty string")

        try:
            multiplier = float(profile_multiplier)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "profile_multiplier must be a number between 0.1 and 10.0"
            ) from exc

        if not math.isfinite(multiplier) or not 0.1 <= multiplier <= 10.0:
            raise ValueError(
                "profile_multiplier must be a finite number between 0.1 and 10.0"
            )
        return multiplier

    def _reset_locked(self, generation_id: str) -> None:
        """Reset state only if the caller still owns the current generation."""
        if self.generation_id != generation_id:
            return

        self.state = SessionState.IDLE
        self.generation_id = None
        self.active_session = None
        self._stop_event = None
        self._worker_thread = None

    def is_active(self) -> bool:
        with self._lock:
            return (
                self.state is SessionState.ACTIVE
                and self._stop_event is not None
                and not self._stop_event.is_set()
                and self._worker_thread is not None
                and self._worker_thread.is_alive()
            )

    def mark_worker_ready(self, generation_id: str) -> bool:
        """
        Called by worker_target after its startup/initialization succeeds.
        A stale worker cannot activate a newer session.
        """
        with self._lock:
            if (
                generation_id != self.generation_id
                or self.state is not SessionState.STARTING
                or self._stop_event is None
                or self._stop_event.is_set()
            ):
                return False

            self.state = SessionState.ACTIVE
            logger.info("Telemetry worker is active [%s]", generation_id)
            return True

    def start_session(
        self,
        room_id: str,
        student_id: str,
        worker_target: Callable[[threading.Event, str, Dict[str, Any]], None],
        profile_multiplier: float = 1.0,
    ) -> bool:
        multiplier = self._validate_inputs(
            room_id, student_id, profile_multiplier
        )
        if not callable(worker_target):
            raise TypeError("worker_target must be callable")

        with self._lock:
            # STARTING/ACTIVE/STOPPING all mean a previous lifecycle is owned.
            if self.state is not SessionState.IDLE:
                logger.warning(
                    "Session start rejected while state is %s", self.state.value
                )
                return False

            generation_id = str(uuid.uuid4())
            stop_event = threading.Event()
            session_info: Dict[str, Any] = {
                "session_id": generation_id,
                "room_id": room_id.strip(),
                "student_id": student_id.strip(),
                "profile_multiplier": multiplier,
            }

            self.state = SessionState.STARTING
            self.generation_id = generation_id
            self._stop_event = stop_event
            self.active_session = dict(session_info)

            def _thread_runner() -> None:
                try:
                    # Give the worker its own dictionary so it cannot mutate
                    # the manager's session state accidentally.
                    worker_target(stop_event, generation_id, dict(session_info))
                except Exception:
                    logger.exception(
                        "Unhandled telemetry worker failure [%s]", generation_id
                    )
                finally:
                    with self._lock:
                        # A stale worker must never reset a newer generation.
                        self._reset_locked(generation_id)
                    logger.info("Telemetry worker exited [%s]", generation_id)

            worker = threading.Thread(
                target=_thread_runner,
                name=f"VigilanceCollector-{generation_id[:8]}",
                daemon=True,
            )
            self._worker_thread = worker

            try:
                worker.start()
            except Exception:
                logger.exception("Failed to start telemetry worker [%s]", generation_id)
                self._reset_locked(generation_id)
                return False

            # The session remains STARTING until the worker calls
            # mark_worker_ready() after completing its own initialization.
            return True

    def stop_session(self, timeout: float = 3.0) -> bool:
        """
        Request shutdown and wait up to timeout seconds.

        Returns True if shutdown is complete (or no session existed).
        Returns False if the worker did not exit before the timeout.
        A timed-out worker remains tracked in STOPPING; a new session is
        rejected until that worker actually exits.
        """
        try:
            timeout_value = float(timeout)
        except (TypeError, ValueError) as exc:
            raise ValueError("timeout must be a finite, non-negative number") from exc

        if not math.isfinite(timeout_value) or timeout_value < 0:
            raise ValueError("timeout must be a finite, non-negative number")

        current = threading.current_thread()

        with self._lock:
            if self.state is SessionState.IDLE:
                return True

            generation_id = self.generation_id
            worker = self._worker_thread
            stop_event = self._stop_event

            self.state = SessionState.STOPPING
            if stop_event is not None:
                stop_event.set()

            # A worker cannot join itself. Its wrapper's finally block will
            # perform the final transition to IDLE.
            if worker is current:
                return True

        if worker is not None and worker.is_alive():
            worker.join(timeout=timeout_value)
            if worker.is_alive():
                logger.error(
                    "Worker %s did not exit within %.2f seconds",
                    worker.name,
                    timeout_value,
                )
                return False

        with self._lock:
            if generation_id is not None:
                self._reset_locked(generation_id)
            return self.state is SessionState.IDLE

    def get_session_info(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if self.active_session is None:
                return None
            return dict(self.active_session)


# Global singleton used by the local bridge and agent modules.
session_guard = SessionManager()

