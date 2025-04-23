from typing import Annotated, Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, model_validator

from backend.models.restaurant import Restaurant

class RecommendationRequest(BaseModel):
    liked_ids: List[str] = Field(..., description="List of business IDs that the user likes")
    disliked_ids: List[str] = Field(default=[], description="List of business IDs that the user dislikes")
    user_location: List[float]
    radius_miles: Optional[float] = Field(default=10.0, ge=0, description="Maximum distance in miles to search")
    top_n: int = Field(default=5, ge=1, description="Number of recommendations to return")
    desired_price: Optional[int] = Field(default=None, ge=1, le=4, description="Desired price range (1-4, where 1 is cheapest and 4 is most expensive)")

class RecommendationScore(BaseModel):
    content_score: float  # Similarity based on attribute name embeddings
    rating_score: float
    proximity_score: float
    price_score: float
    final_score: float
    distance_miles: float

class RecommendationDetail(BaseModel):
    restaurant: Restaurant
    scores: RecommendationScore

class RecommendationResponse(BaseModel):
    recommendations: List[RecommendationDetail]
    actual_radius: float
    timestamp: str

class WeightsRequest(BaseModel):
    content_weight: Optional[float] = Field(default=None, ge=0, description="Weight for attribute embedding similarity")
    rating_weight: Optional[float] = Field(default=None, ge=0, description="Weight for rating similarity")

class HealthResponse(BaseModel):
    status: str
    version: str
    model_info: dict[str, Any]
    timestamp: str