from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class StudentEnrollment(BaseModel):
    """
    Contract for enrolling a student's machine into an active classroom session.
    """
    student_id: str = Field(..., min_length=3, max_length=50, examples=["STU_101"])
    room_id: str = Field(..., min_length=3, max_length=50, examples=["room_101"])
    device_fingerprint: Optional[str] = Field(
        default=None, 
        max_length=64, 
        description="Local hardware hash to ensure one active stream per student"
    )


class StudentSessionStatus(BaseModel):
    """
    View schema for real-time tracking of a student's connection state in a room.
    """
    student_id: str
    room_id: str
    is_active: bool = True
    last_batch_received: Optional[datetime] = None
    consecutive_anomalies: int = Field(default=0, ge=0)

    