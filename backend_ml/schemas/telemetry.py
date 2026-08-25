from pydantic import BaseModel, Field, field_validator
from typing import List

class TelemetryRecord(BaseModel):
    window_title: str
    network_speed_kb: float
    packet_frequency: float
    # You can add process_name or created_at here if your agent sends them

class TelemetryBatch(BaseModel):
    room_id: str
    records: List[TelemetryRecord] = Field(..., min_length=30, max_length=30)
    profile_multiplier: float = 1.0

    @field_validator('records')
    def check_exact_length(cls, v):
        if len(v) != 30:
            raise ValueError(f"Batch must contain exactly 30 records, got {len(v)}")
        return v

    