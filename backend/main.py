from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

# Assuming these imports point to your actual project structure
from backend.database import init_db
from backend.routers.photos import router as photos_router
from backend.routers.restaurants import router as restaurants_router
from backend.routers.images import router as images_router
from backend.routers.recommendations import router as recommendations_router, get_recommendation_model
from backend.routers.auth import router as auth_router
from backend.recommendations.model.recommendation_model import RestaurantRecommender 
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Perform database initialization if needed on startup
    print("Application startup: Initializing database...")
    try:
        await init_db() # Ensures DB schema, connections pools etc. are ready
        get_recommendation_model()
        print("Database initialization complete.")
    except Exception as e:
        print(f"CRITICAL ERROR during database initialization: {e}")


    print("Application ready.")
    yield
    # Cleanup logic can go here if needed (e.g., closing DB connections)
    print("Application shutdown.")

app = FastAPI(
    title="Binge API", 
    description="API for restaurant information and recommendations.",
    version="1.1.0", # Update version if significant changes made
    lifespan=lifespan
)
    
# Configure CORS
# Be more specific in production environments instead of allowing all origins "*"
origins = [
    "http://localhost",
    "http://localhost:3000", # Example: Allow frontend dev server
    # Add your frontend production URL here
]
app.add_middleware(
    CORSMiddleware,
    # allow_origins=origins, # Use specific origins in production
    allow_origins=["*"],  # Keep as wildcard for now if needed, but be aware of security implications
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Include the routers
app.include_router(photos_router)
app.include_router(restaurants_router)
app.include_router(images_router)
app.include_router(recommendations_router) # This router now manages its model dependency
app.include_router(auth_router)

@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint providing a welcome message.
    """
    return {"message": "Welcome to Binge API"}

if __name__ == "__main__":
    # Recommended: Disable reload in production environments
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)