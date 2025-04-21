from fastapi import APIRouter, HTTPException, Depends, status
from typing import List, Optional, Union, Dict, Any
import uuid
from beanie import PydanticObjectId

from ..database import get_db
from backend.models.restaurant import Restaurant, RestaurantCreate


router = APIRouter(
    prefix="/restaurants",
    tags=["restaurants"]
)

@router.get("/", response_model=List[Restaurant])
async def get_restaurants(db=Depends(get_db)):
    try:
        restaurants = await Restaurant.find_all().to_list(length=100)
        return restaurants
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/random-ids", response_model=None)
async def get_random_restaurant_ids(db=Depends(get_db)):
    try:
        restaurants = await Restaurant.aggregate([
            {"$sample": {"size": 10}},
        ]).to_list()
        if not restaurants:
            raise HTTPException(status_code=404, detail="No restaurants found")
        restaurant_ids = [restaurant.get("business_id") for restaurant in restaurants]
        return restaurant_ids
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{restaurant_id}", response_model=Restaurant)
async def get_restaurant_by_id(restaurant_id: Union[str, int], db=Depends(get_db)):
    try:
        # debug statement
        print(f"Fetching restaurant with ID: {restaurant_id}")
        restaurant = await Restaurant.find_one(Restaurant.business_id == restaurant_id)
        if not restaurant:
            raise HTTPException(status_code=404, detail="Restaurant not found")
        return restaurant
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/", response_model=Restaurant, status_code=status.HTTP_201_CREATED)
async def create_restaurant(restaurant: RestaurantCreate, db=Depends(get_db)):
    try:
        # Generate a unique business ID (could use UUID or other method)
        business_id = str(uuid.uuid4())
        
        # Create a new restaurant document
        new_restaurant = Restaurant(
            _id=PydanticObjectId(),
            business_id=business_id,
            name=restaurant.name,
            address=restaurant.address,
            latitude=restaurant.latitude,
            longitude=restaurant.longitude,
            stars=restaurant.stars,
            price=restaurant.price,
        )
        
        # Save to database
        await new_restaurant.insert()
        
        # Update recommendation model
        from backend.routers.recommendations import get_recommendation_model, update_model_with_new_restaurant
        
        try:
            model = get_recommendation_model()
            update_model_with_new_restaurant(model, new_restaurant)
        except Exception as e:
            print(f"Warning: Failed to update recommendation model: {e}")

        return new_restaurant
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create restaurant: {str(e)}")