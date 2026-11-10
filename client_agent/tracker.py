# client_agent/tracker.py
"""Session-bounded telemetry sampling and batch dispatch."""

import logging
import threading
import time
from typing import Any, Dict, List, Optional

import requests

from client_agent.config import (
    FASTAPI_TELEMETRY_URL,
    REQUEST_TIMEOUT_SECONDS,
    SAMPLE_INTERVAL_SECONDS,
    SEQUENCE_LENGTH,
)
from client_agent.platform_hooks import NetworkSampler, harvest_telemetry_slice
from client_agent.session import session_guard

logger = logging.getLogger("vigilance.tracker")

MAX_BATCH_ATTEMPTS = 3
INITIAL_RETRY_BACKOFF_SECONDS = 0.5
MAX_CONSECUTIVE_SAMPLE_ERRORS = 5
RETRYABLE_HTTP_STATUSES = {408, 425, 429}


def send_batch_to_backend(
    payload: Dict[str, Any],
    stop_event: Optional[threading.Event] = None,
) -> int:
    """Send one batch. Return its HTTP status, or 0 on request failure."""
    if stop_event is not None and stop_event.is_set():
        return 0

    try:
        with requests.post(
            FASTAPI_TELEMETRY_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        ) as response:
            return response.status_code
    except requests.exceptions.RequestException as exc:
        logger.warning("Telemetry batch transmission failed: %s", exc)
        return 0


def _dispatch_batch(
    payload: Dict[str, Any],
    stop_event: threading.Event,
) -> int:
    """Retry transient failures a bounded number of times."""
    for attempt in range(1, MAX_BATCH_ATTEMPTS + 1):
        if stop_event.is_set():
            return 0

        status_code = send_batch_to_backend(payload, stop_event)

        if 200 <= status_code < 300 or status_code == 403:
            return status_code

        retryable = (
            status_code == 0
            or status_code in RETRYABLE_HTTP_STATUSES
            or 500 <= status_code < 600
        )

        if not retryable:
            logger.error(
                "Backend rejected telemetry batch with HTTP %s; "
                "batch will not be retried",
                status_code,
            )
            return status_code

        if attempt == MAX_BATCH_ATTEMPTS:
            logger.error(
                "Telemetry batch failed after %s attempts (last status: %s)",
                MAX_BATCH_ATTEMPTS,
                status_code,
            )
            return status_code

        backoff = min(
            INITIAL_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)),
            2.0,
        )
        logger.warning(
            "Transient dispatch failure (status=%s); retrying attempt %s/%s "
            "in %.1fs",
            status_code,
            attempt + 1,
            MAX_BATCH_ATTEMPTS,
            backoff,
        )

        if stop_event.wait(backoff):
            return 0

    return 0


def telemetry_worker(
    stop_event: threading.Event,
    generation_id: str,
    session_info: Dict[str, Any],
) -> None:
    """
    Sample at the configured interval and dispatch fixed-size batches.

    The worker calls SessionManager.mark_worker_ready() only after the
    network sampler has initialized successfully.
    """
    room_id = session_info["room_id"]
    student_id = session_info["student_id"]
    profile_multiplier = session_info.get("profile_multiplier", 1.0)

    buffer: List[Dict[str, Any]] = []
    consecutive_sample_errors = 0
    batch_number = 0

    logger.info(
        "Collector initializing for student %s in room %s [generation=%s]",
        student_id,
        room_id,
        generation_id,
    )

    try:
        if stop_event.is_set():
            return

        # Establish the initial network-counter baseline.
        sampler = NetworkSampler()

        if stop_event.is_set():
            return

        # SessionManager remains STARTING until sampler initialization succeeds.
        if not session_guard.mark_worker_ready(generation_id):
            logger.info(
                "Worker startup cancelled or generation is stale [%s]",
                generation_id,
            )
            return

        logger.info("Collector active [generation=%s]", generation_id)

        # First sample occurs one full interval after the baseline.
        next_tick = time.monotonic() + SAMPLE_INTERVAL_SECONDS

        while not stop_event.is_set():
            wait_seconds = max(0.0, next_tick - time.monotonic())
            if stop_event.wait(wait_seconds):
                break

            if stop_event.is_set():
                break

            try:
                record = harvest_telemetry_slice(sampler)

                # Do not retain a sample if stop was requested during harvest.
                if stop_event.is_set():
                    break

                buffer.append(record)
                consecutive_sample_errors = 0

            except Exception:
                consecutive_sample_errors += 1
                logger.exception(
                    "Telemetry sampling failed (%s/%s) [generation=%s]",
                    consecutive_sample_errors,
                    MAX_CONSECUTIVE_SAMPLE_ERRORS,
                    generation_id,
                )

                if consecutive_sample_errors >= MAX_CONSECUTIVE_SAMPLE_ERRORS:
                    logger.error(
                        "Stopping collector after repeated sampling failures "
                        "[generation=%s]",
                        generation_id,
                    )
                    break

            if len(buffer) == SEQUENCE_LENGTH:
                if stop_event.is_set():
                    break

                batch_number += 1
                payload = {
                    "student_id": student_id,
                    "room_id": room_id,
                    "profile_multiplier": profile_multiplier,
                    "records": list(buffer),
                }

                logger.debug(
                    "Dispatching batch %s containing %s records [generation=%s]",
                    batch_number,
                    SEQUENCE_LENGTH,
                    generation_id,
                )

                status_code = _dispatch_batch(payload, stop_event)

                if status_code == 403:
                    logger.warning(
                        "Backend returned HTTP 403; ending session [generation=%s]",
                        generation_id,
                    )
                    stop_event.set()
                    break

                # A stop request during an in-flight request takes priority.
                if stop_event.is_set():
                    break

                if 200 <= status_code < 300:
                    logger.info(
                        "Telemetry batch %s accepted [generation=%s]",
                        batch_number,
                        generation_id,
                    )
                else:
                    # Explicit volatile-data policy: after bounded retries,
                    # discard rather than persisting telemetry to disk.
                    logger.error(
                        "Discarding batch %s after dispatch failure "
                        "(status=%s) [generation=%s]",
                        batch_number,
                        status_code,
                        generation_id,
                    )

                # Clear after success or an explicit final discard.
                buffer.clear()

            # Avoid catch-up bursts if harvesting or dispatch took too long.
            now = time.monotonic()
            next_tick += SAMPLE_INTERVAL_SECONDS
            if next_tick <= now:
                next_tick = now + SAMPLE_INTERVAL_SECONDS

    except Exception:
        logger.exception(
            "Telemetry worker terminated unexpectedly [generation=%s]",
            generation_id,
        )

    finally:
        # Telemetry remains volatile; discard any partial batch on shutdown.
        buffer.clear()
        logger.info(
            "Collector terminated; in-memory buffer cleared [generation=%s]",
            generation_id,
        )
