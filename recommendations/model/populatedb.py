from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()
uri = os.getenv("MONGO_URI")

def connect_to_mongo(uri: str) -> MongoClient:
    client = MongoClient(uri, server_api=ServerApi('1'))

    try:
        client.admin.command('ping')
        print("Successfully connected to MongoDB.")
    except Exception as e:
        print(e)
    return client

def populate_database(client: MongoClient, adding_data=True) -> None:
    
    # import data into mongo
    if adding_data:
        # Import photos data
        photos_df = pd.read_json("/Users/michaelosmolovskiy/github/binge/recommendations/data/cleaned_photos.json", lines=True)
        photos_collection = client.binge.photos
        restaurants_collection = client.binge.restaurants
        if not photos_df.empty:
            photos_collection.insert_many(photos_df.to_dict("records"))
            print(f"Inserted {len(photos_df)} photo documents.")

        # import restaurants data if file exists:
        try:
            restaurants_df = pd.read_json("/Users/michaelosmolovskiy/github/binge/recommendations/data/cleaned_restaurants.json", lines=True)
            if not restaurants_df.empty:
                restaurants_collection.insert_many(restaurants_df.to_dict("records"))
                print(f"Inserted {len(restaurants_df)} restaurant documents.")
        except Exception as ex:
            print("No restaurant data imported:", ex)
    else:
        # delete all documents in the collections
        photos_collection.delete_many({})
        restaurants_collection.delete_many({})


if __name__ == "__main__":
    client = connect_to_mongo(uri)
    populate_database(client, adding_data=True)
    client.close()