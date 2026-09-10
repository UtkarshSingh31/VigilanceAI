from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class IncidentRecord(BaseModel):
    """
    Represents an isolated 30-second anomaly occurrence.
    """
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    reconstruction_loss: float = Field(..., description="Calculated MSE score from LSTM")
    effective_threshold: float = Field(..., description="Applied room threshold cutoff")
    point_anomaly_detected: bool = Field(default=False)
    flagged_window_title: Optional[str] = Field(None, description="Active window during breach")


class StudentAnomalyHistoryResponse(BaseModel):
    """
    Returned to the instructor's dashboard when inspecting a student.
    Handles the 'anomalous at start, normal now' state cleanly.
    """
    student_id: str
    room_id: str
    is_currently_anomalous: bool = Field(
        ..., 
        description="True if the LATEST 30-second batch breached threshold, False if back to normal"
    )
    total_anomaly_count: int = Field(
        default=0, 
        ge=0, 
        description="Total number of 30-second sequences flagged during the entire session"
    )
    incidents: List[IncidentRecord] = Field(
        default_factory=list, 
        description="Chronological log of past breaches"
    )

    