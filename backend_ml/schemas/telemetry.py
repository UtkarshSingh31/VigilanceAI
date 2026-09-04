from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


class TelemetryRecord(BaseModel):
    """
    1-second raw telemetry slice captured by the edge agent.
    """
    window_title: str = Field(..., description="Active foreground window title")
    network_speed_kb: float = Field(..., ge=0.0, description="Instantaneous aggregate speed in KB/s")
    packet_frequency: float = Field(..., ge=0.0, description="Packets per second count")


class TelemetryBatchRequest(BaseModel):
    """
    The 30-step sequential payload received by the FastAPI /telemetry endpoint.
    """
    student_id: str = Field(..., examples=["STU_101"])
    room_id: str = Field(..., examples=["room_101"])
    profile_multiplier: float = Field(default=1.0, gt=0.0, description="Room sensitivity scaling factor")
    records: List[TelemetryRecord] = Field(..., min_length=30, max_length=30)

    @field_validator("records")
    def validate_exact_30_records(cls, v):
        if len(v) != 30:
            raise ValueError(f"Batch must contain exactly 30 records, got {len(v)}")
        return v
    