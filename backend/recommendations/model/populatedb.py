from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import pandas as pd
import os
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader
import cloudinary.api
from cloudinary.utils import cloudinary_url
import concurrent.futures
import time
from tqdm import tqdm

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
        photos_df = pd.read_json("../data/cleaned_photos.json", lines=True)
        photos_collection = client.binge.photos
        restaurants_collection = client.binge.restaurants
        if not photos_df.empty:
            photos_collection.insert_many(photos_df.to_dict("records"))
            print(f"Inserted {len(photos_df)} photo documents.")

        # import restaurants data if file exists:
        try:
            restaurants_df = pd.read_json("../data/cleaned_restaurants.json", lines=True)
            if not restaurants_df.empty:
                restaurants_collection.insert_many(restaurants_df.to_dict("records"))
                print(f"Inserted {len(restaurants_df)} restaurant documents.")
        except Exception as ex:
            print("No restaurant data imported:", ex)
    else:
        # delete all documents in the collections
        # photos_collection.delete_many({})
        restaurants_collection.delete_many({})

def get_existing_files():
    """Fetch list of already uploaded files to avoid duplicates"""
    try:
        # Get all resources in the binge_photos folder
        result = cloudinary.api.resources(
            type="upload",
            prefix="binge_photos/",
            max_results=500
        )
        
        # Extract just the filenames from the resources
        existing_files = []
        for resource in result.get('resources', []):
            # Extract filename from the public_id
            if '/' in resource['public_id']:
                # Get just the filename after the folder name
                filename = resource['public_id'].split('/')[-1]
                existing_files.append(filename)
        
        # If there are more resources than the max_results, we need to paginate
        next_cursor = result.get('next_cursor')
        while next_cursor:
            result = cloudinary.api.resources(
                type="upload",
                prefix="binge_photos/",
                max_results=500,
                next_cursor=next_cursor
            )
            for resource in result.get('resources', []):
                if '/' in resource['public_id']:
                    filename = resource['public_id'].split('/')[-1]
                    existing_files.append(filename)
            next_cursor = result.get('next_cursor')
        
        print(f"Found {len(existing_files)} existing files in Cloudinary.")
        return existing_files
    except Exception as ex:
        print(f"Error fetching existing files: {ex}")
        return []

def upload_single_image(file_path, filename, existing_files):
    """Upload a single image to Cloudinary"""
    if filename in existing_files:
        print(f"Skipping {filename} - already uploaded")
        return {
            "filename": filename,
            "status": "skipped",
            "url": None
        }
    
    try:
        response = cloudinary.uploader.upload(
            file_path, 
            folder="binge_photos", 
            public_id=filename,
            resource_type="image"
        )
        optimize_url, _ = cloudinary_url(filename, fetch_format="auto", quality="auto")
        return {
            "filename": filename,
            "status": "success",
            "url": response["url"]
        }
    except Exception as ex:
        print(f"Error uploading {filename}: {ex}")
        return {
            "filename": filename,
            "status": "error",
            "error": str(ex)
        }

def upload_images(max_workers=10):
    """Upload images to Cloudinary using multithreading"""
    # Configuration       
    cloudinary.config( 
        cloud_name = "diub0blpa", 
        api_key = "212739855472564", 
        api_secret = "wASPoB1Bs743ELLWmyhZpO0BP8g",
        secure=True
    )

    photos_dir = "../data/photos/"
    
    # Get list of files to upload
    image_files = [f for f in os.listdir(photos_dir) if f.endswith(".jpg")]
    total_files = len(image_files)
    print(f"Found {total_files} images to process")
    
    # Get list of already uploaded files
    existing_files = get_existing_files()
    
    # Prepare results container
    results = {
        "success": 0,
        "skipped": 0,
        "error": 0,
        "errors": []
    }
    
    # Use ThreadPoolExecutor for concurrent uploads
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a list of future objects
        futures = {
            executor.submit(
                upload_single_image, 
                photos_dir + filename, 
                filename, 
                existing_files
            ): filename for filename in image_files
        }
        
        # Process results as they complete with a progress bar
        with tqdm(total=total_files, desc="Uploading") as progress:
            for future in concurrent.futures.as_completed(futures):
                filename = futures[future]
                try:
                    result = future.result()
                    if result["status"] == "success":
                        results["success"] += 1
                    elif result["status"] == "skipped":
                        results["skipped"] += 1
                    else:
                        results["error"] += 1
                        results["errors"].append(f"{filename}: {result.get('error')}")
                except Exception as exc:
                    print(f"{filename} generated an exception: {exc}")
                    results["error"] += 1
                    results["errors"].append(f"{filename}: {str(exc)}")
                progress.update(1)
    
    # Report results
    elapsed_time = time.time() - start_time
    print(f"\nUpload completed in {elapsed_time:.2f} seconds")
    print(f"Successful uploads: {results['success']}")
    print(f"Skipped (already uploaded): {results['skipped']}")
    print(f"Errors: {results['error']}")
    
    if results["error"] > 0:
        print("\nError details:")
        for error in results["errors"]:
            print(f"  - {error}")

if __name__ == "__main__":
    client = connect_to_mongo(uri)
    # populate_database(client, adding_data=True)
    upload_images(max_workers=10)  # You can adjust the number of workers as needed
    client.close()
