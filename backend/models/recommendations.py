from typing import Any, Dict, List
from pydantic import BaseModel, Field, model_validator

class UserLocation(BaseModel):
    latitude: float = Field(..., description="User's latitude", ge=-90, le=90)
    longitude: float = Field(..., description="User's longitude", ge=-180, le=180)

class RecommendationRequest(BaseModel):
    liked_restaurants: List[str] = Field(
        default=[],
        description="List of business IDs the user likes"
    )
    disliked_restaurants: List[str] = Field(
        default=[],
        description="List of business IDs the user dislikes"
    )
    location: UserLocation = Field(
        ...,
        description="User's current location"
    )
    radius_miles: float = Field(
        default=10.0,
        description="Preferred radius for recommendations in miles",
        gt=0,
        le=100
    )
    default_radius_miles: float = Field(
        default=25.0,
        description="Fallback radius if not enough restaurants found",
        gt=0,
        le=100
    )
    min_recommendations: int = Field(
        default=5,
        description="Minimum number of recommendations to return",
        ge=1,
        le=50
    )
    location_weight: float = Field(
        default=1.0,
        description="Weight for location proximity in scoring",
        ge=0,
        le=10
    )
    top_n: int = Field(
        default=5,
        description="Number of recommendations to return",
        ge=1,
        le=50
    )
    @model_validator(mode="before")
    def check_radius_values(cls, values):
        radius = values.get('radius_miles')
        default_radius = values.get('default_radius_miles')
        
        if radius and default_radius and default_radius < radius:
            raise ValueError("default_radius_miles must be greater than or equal to radius_miles")
        return values
class WeightsRequest(BaseModel):
    content_weight: float = Field(
        default=1.0,
        description="Weight for content-based features",
        ge=0,
        le=10
    )
    rating_weight: float = Field(
        default=1.0,
        description="Weight for ratings-based similarity",
        ge=0,
        le=10
    )

class HealthResponse(BaseModel):
    status: str
    version: str
    model_info: Dict[str, Any]
    timestamp: str