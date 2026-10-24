from typing import Dict
from pydantic import BaseModel, Field, field_validator


class RoomConfigCreate(BaseModel):
    """
    Contract for registering or provisioning a classroom's ML artifacts and baselines.
    """
    room_id: str = Field(..., min_length=2, max_length=50, examples=["room_101"])
    room_name: str = Field(..., min_length=2, max_length=100, examples=["CS101 Systems Lab"])
    clusterer_path: str = Field(..., description="Filepath to pre-fitted room HDBSCAN artifact")
    cluster_speed_map: Dict[int, float] = Field(
        ..., 
        description="Dictionary mapping integer cluster IDs to historical average speed (KB/s)"
    )
    global_train_mean: float = Field(
        ..., 
        ge=0.0, 
        description="Fallback historical baseline network speed (KB/s) for outliers or noise (-1)"
    )
    base_threshold: float = Field(
        default=1.6923, 
        gt=0.0, 
        description="Baseline reconstruction error cutoff for the room"
    )

    @field_validator("cluster_speed_map")
    def validate_speed_map_not_empty(cls, v):
        if not v:
            raise ValueError("cluster_speed_map cannot be empty; at least one cluster must be mapped.")
        return v


class RoomConfigResponse(BaseModel):
    """
    Read-only view of a room's active ML context.
    """
    room_id: str
    room_name: str
    clusterer_path: str
    global_train_mean: float
    base_threshold: float = 1.6923
    is_active: bool = True
    