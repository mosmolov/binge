from fastapi import APIRouter, HTTPException, status
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
from backend.models.restaurant import Restaurant
from backend.routers.restaurants import get_restaurant_by_id
from backend.models.recommendations import RecommendationRequest, HealthResponse
from backend.recommendations.model.recommendation_model import RestaurantRecommender

# Define request and response models
class RecommendationRequest(BaseModel):
    liked_ids: List[str]
    disliked_ids: List[str] = []
    top_n: int = 5

class RecommendationResponse(BaseModel):
    recommended_ids: List[str]

router = APIRouter(prefix="/recommendations", tags=["recommendations"])

# Load the recommendation model
try:
    recommendation_model = RestaurantRecommender(data_path="../data/cleaned_restaurants.csv")
    recommendation_model.load_model()
except Exception as e:
    print(f"Error loading recommendation model: {e}")
    recommendation_model = None

@router.post("/", response_model=List[Restaurant])
async def get_recommendations(request: RecommendationRequest):
    """
    Get restaurant recommendations based on liked and disliked restaurants.
    """
    if recommendation_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation model is not available"
        )
    
    try:
        recommended_ids = recommendation_model.recommend_restaurants(
            request.liked_ids,
            request.disliked_ids,
            request.top_n
        )
        recommended_restaurants = []
        for restaurant_id in recommended_ids:
            restaurant = await get_restaurant_by_id(restaurant_id)
            recommended_restaurants.append(restaurant)
        return recommended_restaurants
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating recommendations: {str(e)}"
        )

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint providing system status."""
    if recommendation_model is None:
        raise HTTPException(status_code=503, detail="Recommendation system not initialized")
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_info={
            "restaurants_count": len(recommendation_model.df),
            "content_weight": recommendation_model.content_weight,
            "rating_weight": recommendation_model.rating_weight
        },
        timestamp=datetime.now().isoformat()
    )
