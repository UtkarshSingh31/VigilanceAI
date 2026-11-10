# client_agent/config.py
"""Central configuration for the Vigilance OS background agent."""

import math
import os
from urllib.parse import urlparse


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer; received {raw!r}") from exc

    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _positive_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number; received {raw!r}") from exc

    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite number greater than zero")
    return value


def _backend_url() -> str:
    value = os.getenv(
        "VIGILANCE_BACKEND_URL",
        "http://127.0.0.1:8000/api/v1/telemetry",
    ).strip()

    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or not parsed.hostname:
        raise ValueError(
            "VIGILANCE_BACKEND_URL must be an absolute HTTP(S) URL"
        )
    if parsed.username or parsed.password:
        raise ValueError("VIGILANCE_BACKEND_URL must not contain embedded credentials")
    return value


# Central FastAPI backend endpoint.
FASTAPI_TELEMETRY_URL: str = _backend_url()

# Local loopback bridge (browser -> native agent).
LOCAL_HOST: str = "127.0.0.1"
LOCAL_PORT: int = _positive_int("VIGILANCE_LOCAL_PORT", 49152)
if LOCAL_PORT > 65535:
    raise ValueError("VIGILANCE_LOCAL_PORT must be between 1 and 65535")

# Telemetry contract.
SEQUENCE_LENGTH: int = _positive_int("VIGILANCE_SEQUENCE_LENGTH", 30)
SAMPLE_INTERVAL_SECONDS: float = _positive_float(
    "VIGILANCE_SAMPLE_INTERVAL_SECONDS", 1.0
)
REQUEST_TIMEOUT_SECONDS: float = _positive_float(
    "VIGILANCE_REQUEST_TIMEOUT_SECONDS", 5.0
)
