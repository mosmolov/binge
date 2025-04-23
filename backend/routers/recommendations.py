from fastapi import APIRouter, HTTPException, status, Depends, Query
from typing import List, Optional, Dict, Any, Tuple, Annotated
from pydantic import BaseModel, Field, confloat
from datetime import datetime
from beanie import PydanticObjectId
import numpy as np
import pandas as pd
from backend.models.restaurant import Restaurant
from backend.routers.restaurants import get_restaurant_by_id
from backend.models.recommendations import HealthResponse, RecommendationDetail, RecommendationRequest, RecommendationResponse, RecommendationScore, WeightsRequest
from backend.recommendations.model.recommendation_model import RestaurantRecommender
from backend.models.user import User
from fastapi.responses import JSONResponse

# Define request and response models


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

def update_model_with_new_restaurant(model: RestaurantRecommender, restaurant: Restaurant):
    """
    Update the recommendation model with a new restaurant.
    
    Args:
        model: The recommendation model instance
        restaurant: The new restaurant document to add
    """
    if not model or not restaurant:
        raise ValueError("Model and restaurant must be provided")
    
    # Create a new row for the dataframe with the restaurant data
    new_row = {
        'business_id': restaurant.business_id,
        'latitude': restaurant.latitude,
        'longitude': restaurant.longitude,
        'stars': restaurant.stars,
    }
    
    # If price is available, add it
    if restaurant.price:
        new_row['RestaurantsPriceRange2'] = restaurant.price
    
    # Add attribute columns if provided
    if restaurant.attributes:
        # Handle cuisine
        if 'Cuisine' in restaurant.attributes and restaurant.attributes['Cuisine']:
            cuisine = restaurant.attributes['Cuisine']
            new_row[f'Cuisine_{cuisine}'] = 1
            
        # Handle ambience
        if 'Ambience' in restaurant.attributes and restaurant.attributes['Ambience']:
            ambience = restaurant.attributes['Ambience']
            new_row[f'Ambience_{ambience}'] = 1
            
        # Handle GoodFor attributes (multi-select)
        if 'GoodFor' in restaurant.attributes and restaurant.attributes['GoodFor']:
            for good_for in restaurant.attributes['GoodFor']:
                new_row[f'GoodFor_{good_for}'] = 1
    
    # Create a DataFrame with the new restaurant row
    new_restaurant_df = pd.DataFrame([new_row])
    
    # Create a dictionary for all missing columns with default values of 0
    missing_columns = {}
    for col in model.df.columns:
        if col not in new_restaurant_df.columns:
            missing_columns[col] = 0
    
    # Add all missing columns at once to avoid fragmentation
    if missing_columns:
        # Create a DataFrame with all missing columns pre-filled with zeros
        missing_df = pd.DataFrame([missing_columns])
        
        # Concatenate the original new row with the missing columns
        new_restaurant_df = pd.concat([new_restaurant_df, missing_df], axis=1)
    
    # Ensure columns are in the same order as the model dataframe
    new_restaurant_df = new_restaurant_df[model.df.columns]
    
    # Append the new row to the model's dataframe
    model.df = pd.concat([model.df, new_restaurant_df], ignore_index=True)
    model.df.to_pickle("backend/recommendations/data/cleaned_restaurants.pkl")
    # Update business_ids array
    model.business_ids = model.df['business_id'].values
    
    # Generate attribute embeddings for the new restaurant
    # First, update attribute columns list
    model.attribute_columns = [col for col in model.df.columns 
                               if col not in ['business_id', 'latitude', 'longitude', 'stars']]
    
    # Regenerate restaurant embeddings
    model._generate_attribute_name_embeddings()
    model._group_attributes_by_embedding(n_groups=50, linkage='average')
    model._create_restaurant_embeddings(grouped=True)
    
    # Re-extract features with the updated dataframe
    model._extract_features()
    
    # Rebuild Annoy index with updated embeddings
    model._build_annoy_index(n_trees=50)
    
    
    print(f"Successfully updated recommendation model with new restaurant: {restaurant.name}")
    return True

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
        user_location = request.user_location
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
            RecommendationRequest(
            liked_ids=liked_ids,
            disliked_ids=disliked_ids,
            user_location=user_location,
            radius_miles=request.radius_miles,
            top_n=request.top_n,
            desired_price=request.desired_price
            ),
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
                    price_score=rec.get('price_score', 0.0),
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

@router.get("/attributes")
async def get_attributes(
    model: RestaurantRecommender = Depends(get_recommendation_model)
):
    """
    Get restaurant attributes organized by category (Cuisine, Ambience, GoodFor).
    """
    try:
        # Call the model's get_attributes method to get categorized attributes
        attr_categories = model.get_attributes()
        return attr_categories
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

# Get recommendations for a specific user
@router.get("/{user_id}")
async def get_user_recommendations(
    user_id: PydanticObjectId,
    model: RestaurantRecommender = Depends(get_recommendation_model),
    user_location: List[float] = Query(..., description="User's location as [latitude, longitude]"),
    desired_price: Optional[int] = Query(None, ge=1, le=4, description="Desired price range (1-4, where 1 is cheapest and 4 is most expensive)"),
):
    """
    Get restaurant recommendations for a specific user.
    """
    try:
        user = await User.get(user_id)
        if not user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        
        request = RecommendationRequest(
            liked_ids=user.liked_business_ids,
            disliked_ids=user.disliked_business_ids,
            user_location=user_location,
            radius_miles=25.0,  # Default radius
            top_n=25,  # Default number of recommendations
            desired_price=desired_price
        )
        
        recommendations_raw, actual_radius = model.recommend_restaurants(request)
        
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
                    price_score=rec.get('price_score', 0.0),
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
        print(f"Error in get_user_recommendations endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating recommendations: {str(e)}"
        )