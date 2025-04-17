from fastapi import APIRouter, HTTPException, status, Depends, Query
from typing import List, Optional, Dict, Any, Tuple, Annotated
from pydantic import BaseModel, Field, confloat  # confloat requires 'pip install pydantic[email]'
from datetime import datetime
from beanie import PydanticObjectId

try:
    from backend.models.restaurant import Restaurant
    from backend.routers.restaurants import get_restaurant_by_id
    from backend.models.recommendations import HealthResponse
    from backend.recommendations.model.recommendation_model import RestaurantRecommender
    from backend.models.user import User
except ImportError as e:
    print(f"Error importing backend modules: {e}")
    print("Please ensure the script is run from the correct directory or PYTHONPATH is set.")
    # Define dummy classes/functions for syntactical validity
    class Restaurant(BaseModel):
        id: str
        name: str

    async def get_restaurant_by_id(id: str): 
        return Restaurant(id=id, name=f"Dummy Restaurant {id}")

    class HealthResponse(BaseModel):
        status: str
        version: str
        model_info: Dict
        timestamp: str

    import pandas as pd
    import numpy as np
    class RestaurantRecommender:
        def __init__(self, data_path="dummy.csv", embedding_model_name='all-MiniLM-L6-v2'):
            self.df = pd.DataFrame({'business_id': []})
            self.content_weight = 1.0
            self.rating_weight = 1.0
        def recommend_restaurants(self, liked_ids, disliked_ids, user_location, radius_miles, top_n):
            return [], 10.0
        def set_weights(self, content_weight, rating_weight):
            self.content_weight = content_weight
            self.rating_weight = rating_weight
        @property
        def business_ids(self): 
            return []
        @property
        def sim_matrix(self): 
            return np.array([])
        @property
        def rating_features(self): 
            return np.array([])

import numpy as np
import pandas as pd

# Define request and response models

class RecommendationRequest(BaseModel):
    liked_ids: List[str] = Field(..., description="List of business IDs that the user likes")
    disliked_ids: List[str] = Field(default=[], description="List of business IDs that the user dislikes")
    user_latitude: Annotated[float, Field(..., description="User's current latitude", ge=-90, le=90)]
    user_longitude: Annotated[float, Field(..., description="User's current longitude", ge=-180, le=180)]
    radius_miles: Optional[float] = Field(default=10.0, ge=0, description="Maximum distance in miles to search")
    top_n: int = Field(default=5, ge=1, description="Number of recommendations to return")

class RecommendationScore(BaseModel):
    content_score: float  # Similarity based on attribute name embeddings
    rating_score: float
    proximity_score: float
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

router = APIRouter(prefix="/recommendations", tags=["recommendations"])

# Global variable to hold the model instance
recommendation_model: Optional[RestaurantRecommender] = None

def get_recommendation_model():
    """Dependency to get the initialized recommendation model."""
    global recommendation_model
    if recommendation_model is None:
        print("Initializing recommendation model from scratch...")  # Debug print
        try:
            recommendation_model = RestaurantRecommender()  # Build a new model
            print("Model initialized from scratch.")  # Debug print
        except Exception as e:
            print(f"CRITICAL ERROR initializing model: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to initialize recommendation model: {str(e)}"
            )

    if recommendation_model is None:
         raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Recommendation model is not available."
         )

    return recommendation_model

@router.post("/", response_model=RecommendationResponse)
async def get_recommendations(
    request: RecommendationRequest,
    user_id: Optional[PydanticObjectId] = Query(None, description="Optional user ID to include saved preferences"),
    model: RestaurantRecommender = Depends(get_recommendation_model)
):
    """
    Get restaurant recommendations based on user preferences and location.
    
    Uses attribute name embeddings, rating similarity, and geographic proximity.
    """
    try:
        user_location = (request.user_latitude, request.user_longitude)
        # merge saved user preferences if user_id provided
        liked_ids = request.liked_ids.copy()
        disliked_ids = request.disliked_ids.copy()
        if user_id:
            user = await User.get(user_id)
            if not user:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
            # combine and dedupe
            liked_ids = list({*user.liked_business_ids, *liked_ids})
            disliked_ids = list({*user.disliked_business_ids, *disliked_ids})
        recommendations_raw, actual_radius = model.recommend_restaurants(
            liked_ids=liked_ids,
            disliked_ids=disliked_ids,
            user_location=user_location,
            radius_miles=request.radius_miles,
            top_n=request.top_n
        )
        
        detailed_recommendations = []
        for rec in recommendations_raw:
            try:
                restaurant_details = await get_restaurant_by_id(rec['business_id'])
                if not restaurant_details:
                    print(f"Warning: No details found for restaurant {rec['business_id']}")
                    continue
                
                scores = RecommendationScore(
                    content_score=rec.get('content_score', 0.0),
                    rating_score=rec.get('rating_score', 0.0),
                    proximity_score=rec.get('proximity_score', 0.0),
                    final_score=rec.get('final_score', 0.0),
                    distance_miles=rec.get('distance_miles', 0.0)
                )
                
                detailed_recommendations.append(
                    RecommendationDetail(
                        restaurant=restaurant_details,
                        scores=scores
                    )
                )
            except Exception as e:
                print(f"Error processing recommendation for {rec.get('business_id', 'N/A')}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        return RecommendationResponse(
            recommendations=detailed_recommendations,
            actual_radius=actual_radius,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        print(f"Error in get_recommendations endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating recommendations: {str(e)}"
        )

@router.patch("/weights", response_model=Dict[str, float])
async def update_weights(
    request: WeightsRequest,
    model: RestaurantRecommender = Depends(get_recommendation_model)
):
    """
    Update the weights for content (attribute embeddings) and rating similarity.
    """
    current_weights = {
        "content_weight": model.content_weight,
        "rating_weight": model.rating_weight
    }
    
    content_weight = request.content_weight if request.content_weight is not None else current_weights["content_weight"]
    rating_weight = request.rating_weight if request.rating_weight is not None else current_weights["rating_weight"]
    
    model.set_weights(
        content_weight=content_weight,
        rating_weight=rating_weight
    )
    
    return {
        "content_weight": model.content_weight,
        "rating_weight": model.rating_weight
    }

@router.get("/health", response_model=HealthResponse)
async def health_check(
    model: RestaurantRecommender = Depends(get_recommendation_model)
):
    """
    Health check endpoint providing system status.
    """
    model_info_dict = {
        "restaurants_count": len(model.df) if hasattr(model, 'df') else 0,
        "content_weight": model.content_weight,
        "rating_weight": model.rating_weight,
    }
    
    return HealthResponse(
        status="healthy",
        version="1.1.0",
        model_info=model_info_dict,
        timestamp=datetime.now().isoformat()
    )

@router.get("/explain/{restaurant_id}")
async def explain_recommendation(
    restaurant_id: str,
    reference_id: str = Query(..., description="ID of a restaurant to compare with (e.g., a liked restaurant)"),
    model: RestaurantRecommender = Depends(get_recommendation_model)
):
    """
    Explain similarity between two restaurants based on attribute embeddings and ratings.
    """
    try:
        id_to_index = {bid: i for i, bid in enumerate(model.business_ids)}
        target_idx = id_to_index.get(restaurant_id)
        reference_idx = id_to_index.get(reference_id)
                
        if target_idx is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Restaurant with ID {restaurant_id} not found in the model"
            )
            
        if reference_idx is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Reference restaurant with ID {reference_id} not found in the model"
            )
        
        if not (0 <= reference_idx < model.sim_matrix.shape[0] and 0 <= target_idx < model.sim_matrix.shape[1]):
            raise HTTPException(status_code=500, detail="Index out of bounds for similarity matrix.")
             
        attribute_embedding_similarity = float(model.sim_matrix[reference_idx, target_idx])
        
        if not (0 <= reference_idx < len(model.rating_features) and 0 <= target_idx < len(model.rating_features)):
            raise HTTPException(status_code=500, detail="Index out of bounds for rating features.")

        rating_diff = abs(float(model.rating_features[reference_idx]) - float(model.rating_features[target_idx]))
        max_rating = 5.0  # Adjust if different
        rating_similarity = 1.0 - (rating_diff / max_rating) if max_rating > 0 else 0.0 
        rating_similarity = max(0.0, rating_similarity)

        return {
            "restaurant_id": restaurant_id,
            "reference_id": reference_id,
            "similarities": {
                "attribute_embedding_similarity": attribute_embedding_similarity,
                "rating_similarity": rating_similarity,
            },
            "current_importance_weights": {
                "content_weight": model.content_weight,
                "rating_weight": model.rating_weight
            }
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error explaining recommendation: {str(e)}"
        )
