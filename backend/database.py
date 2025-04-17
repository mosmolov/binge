import motor.motor_asyncio
from beanie import init_beanie
from backend.models.photo import Photo
import os
from dotenv import load_dotenv
from backend.models.restaurant import Restaurant
from backend.models.user import User
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
client = None
db = None

async def init_db():
    global client, db
    if client is None:
        client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
        db = client[MONGO_DB_NAME]
        await init_beanie(database=db, document_models=[Photo, Restaurant, User])
        print(f"Connected to MongoDB: {MONGO_DB_NAME}")

async def close_db():
    global client
    if client is not None:
        client.close()
        print("MongoDB connection closed")

# Helper function to get database instance
def get_db():
    return db
