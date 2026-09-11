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
    point_anomaly_detected: bool = Field(
        default=False, 
        description="True if Isolation Forest flagged a 1-second spike"
    )
    sequence_anomaly_detected: bool = Field(
        default=False, 
        description="True if LSTM Autoencoder exceeded threshold"
    )
    flagged_window_title: Optional[str] = Field(
        None, 
        description="Offending window title during the breach"
    )
    flagged_process_name: Optional[str] = Field(
        None, 
        description="Process executable associated with the breach"
    )


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
    point_anomaly_count: int = Field(
        default=0, 
        ge=0, 
        description="Total batches flagged for instantaneous point spikes (IsoForest)"
    )
    total_incidents: int = Field(
        default=0, 
        ge=0, 
        description="Combined total of all flagged 30-second batches"
    )
    incidents: List[IncidentRecord] = Field(
        default_factory=list, 
        description="Chronological log of past breaches"
    )

    

