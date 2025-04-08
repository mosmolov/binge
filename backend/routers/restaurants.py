from fastapi import APIRouter, HTTPException, Depends, status
from typing import List, Optional

from ..database import get_db
from backend.models.restaurant import Restaurant


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
async def get_restaurant_by_id(restaurant_id: str, db=Depends(get_db)):
    try:
        # debug statement
        print(f"Fetching restaurant with ID: {restaurant_id}")
        restaurant = await Restaurant.find_one(Restaurant.business_id == restaurant_id)
        if not restaurant:
            raise HTTPException(status_code=404, detail="Restaurant not found")
        return restaurant
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))