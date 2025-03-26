import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional, Union
import pickle
import os
import logging
from datetime import datetime
from pathlib import Path
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('RestaurantRecommender')

class RestaurantRecommender:
    def __init__(self, data_path: str = "../data/cleaned_restaurants.csv"):
        """
        Initialize the recommendation system with restaurant data.
        
        Args:
            data_path: Path to the cleaned restaurant data CSV file
        """
        logger.info(f"Initializing RestaurantRecommender with data from {data_path}")
        base_dir = Path(__file__).parent
        self.df = pd.read_csv(base_dir / data_path)
        self.business_ids = self.df['business_id'].values
        
        # Verify required columns exist
        required_columns = ['business_id', 'latitude', 'longitude', 'stars']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Data missing required columns: {missing_columns}")
        
        # Check for NaN values in the dataframe
        nan_count = self.df.isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values in the dataset. Handling them...")
            
            # Fill NaN values in required columns
            for col in required_columns:
                if self.df[col].isna().any():
                    if col in ['latitude', 'longitude']:
                        # For geographic coordinates, we could drop rows with missing values
                        # as they're critical for recommendations
                        self.df = self.df.dropna(subset=[col])
                        logger.info(f"Dropped rows with NaN values in {col}")
                    elif col == 'stars':
                        # For stars, we could fill with the mean value
                        mean_stars = self.df[col].mean()
                        self.df[col] = self.df[col].fillna(mean_stars)
                        logger.info(f"Filled NaN values in {col} with mean: {mean_stars}")
                    else:
                        # For other columns, fill with 0
                        self.df[col] = self.df[col].fillna(0)
                        logger.info(f"Filled NaN values in {col} with 0")
        
        # Extract and prepare feature sets
        self._extract_features()
        
        # Build initial similarity matrix
        self._build_similarity_matrix()
        
        # Default weights for combining different similarity components
        self.content_weight = 1.0
        self.rating_weight = 1.0
        
        # Cache for distance calculations
        self.distance_cache = {}
        # if model is not saved, save it
        if not os.path.exists("./model_data/recommender_model.pkl"):
            self.save_model()
        logger.info(f"Recommender initialized with {len(self.df)} restaurants")
        

    def _extract_features(self):
        """Extract and prepare feature sets for similarity calculations."""
        # Geographic features (latitude, longitude)
        self.geo_features = self.df[['latitude', 'longitude']].values
        
        # Star ratings
        self.rating_features = self.df[['stars']].values
        scaler = StandardScaler()
        self.rating_features_scaled = scaler.fit_transform(self.rating_features)
        
        # Content features (all other columns except business_id, lat, long, stars)
        content_columns = [col for col in self.df.columns 
                          if col not in ['business_id', 'latitude', 'longitude', 'stars']]
        
        # If no content columns exist, use an empty feature set
        if not content_columns:
            logger.warning("No content features found in the dataset")
            self.content_features = np.zeros((len(self.df), 1))
        else:
            # Handle NaN values in content features
            content_data = self.df[content_columns].copy()
            nan_count_before = content_data.isna().sum().sum()
            if nan_count_before > 0:
                logger.warning(f"Found {nan_count_before} NaN values in content features. Filling with 0...")
                content_data = content_data.fillna(0)
            
            self.content_features = content_data.values
            
        # Verify no NaN values in any features
        if np.isnan(self.geo_features).any() or np.isnan(self.rating_features).any() or np.isnan(self.content_features).any():
            logger.error("NaN values found in features after processing")
            raise ValueError("Input contains NaN values even after cleaning")

    def _build_similarity_matrix(self):
        """Build the content-based similarity matrix using cosine similarity."""
        logger.info("Building content similarity matrix")
        start_time = datetime.now()
        
        # Compute similarity based on content features only
        self.sim_matrix = cosine_similarity(self.content_features)
        
        # Zero out self-similarity to avoid recommending the same restaurant
        np.fill_diagonal(self.sim_matrix, 0)
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Similarity matrix built in {duration:.2f} seconds")

    def set_weights(self, content_weight: float = 1.0, rating_weight: float = 1.0):
        """
        Set weights for different components of the recommendation algorithm.
        
        Args:
            content_weight: Weight for content-based similarity (categories, attributes)
            rating_weight: Weight for rating similarity
        """
        self.content_weight = content_weight
        self.rating_weight = rating_weight
        logger.info(f"Weights updated: content={content_weight}, rating={rating_weight}")

    def haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great circle distance between two points on earth.
        
        Args:
            lat1, lon1: Latitude and longitude of first point
            lat2, lon2: Latitude and longitude of second point
            
        Returns:
            Distance in miles
        """
        # Check cache first
        cache_key = f"{lat1:.6f},{lon1:.6f},{lat2:.6f},{lon2:.6f}"
        if cache_key in self.distance_cache:
            return self.distance_cache[cache_key]
            
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        miles = 3956 * c  # Radius of earth in miles
        
        # Cache result
        self.distance_cache[cache_key] = miles
        return miles

    def batch_haversine_distance(self, user_lat: float, user_lon: float) -> np.ndarray:
        """
        Calculate distances from a user location to all restaurants at once.
        
        Args:
            user_lat: User's latitude
            user_lon: User's longitude
            
        Returns:
            Array of distances in miles to each restaurant
        """
        # Convert decimal degrees to radians
        user_lat_rad = np.radians(user_lat)
        user_lon_rad = np.radians(user_lon)
        rest_lat_rad = np.radians(self.geo_features[:, 0])
        rest_lon_rad = np.radians(self.geo_features[:, 1])
        
        # Haversine formula vectorized
        dlon = rest_lon_rad - user_lon_rad
        dlat = rest_lat_rad - user_lat_rad
        a = np.sin(dlat/2)**2 + np.cos(user_lat_rad) * np.cos(rest_lat_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        miles = 3956 * c  # Radius of earth in miles
        
        return miles

    def recommend_restaurants(
        self, 
        liked_ids: List[str], 
        disliked_ids: List[str],
        user_location: Tuple[float, float],
        radius_miles: float = 10.0,
        default_radius_miles: float = 25.0,
        min_recommendations: int = 5,
        location_weight: float = 1.0,
        top_n: int = 5
    ) -> Tuple[List[Dict], float]:
        """
        Recommend restaurants based on user preferences and location constraints.
        
        Args:
            liked_ids: Business IDs of restaurants the user likes
            disliked_ids: Business IDs of restaurants the user dislikes
            user_location: (latitude, longitude) of the user's location
            radius_miles: Maximum distance in miles to search for restaurants
            default_radius_miles: Fallback radius if not enough restaurants found
            min_recommendations: Minimum number of restaurants to find before stopping
            location_weight: Weight for proximity score (higher values favor closer restaurants)
            top_n: Number of recommendations to return
            
        Returns:
            Tuple containing:
                - List of recommended restaurants with metadata
                - The actual radius used (may be increased if not enough restaurants found)
        """
        start_time = datetime.now()
        
        if user_location is None:
            raise ValueError("User location is required for recommendations")
        
        user_lat, user_lon = user_location
        logger.info(f"Generating recommendations for user at ({user_lat:.4f}, {user_lon:.4f})")
        
        distances = self.batch_haversine_distance(user_lat, user_lon)
        

        in_radius = distances <= radius_miles
        in_radius_count = np.sum(in_radius)
        
        actual_radius = radius_miles
        

        if in_radius_count < min_recommendations:
            logger.info(f"Only {in_radius_count} restaurants within {radius_miles} miles. "
                       f"Expanding to default radius of {default_radius_miles} miles.")
            in_radius = distances <= default_radius_miles
            in_radius_count = np.sum(in_radius)
            actual_radius = default_radius_miles
        

        if in_radius_count < min_recommendations:
            logger.info(f"Only {in_radius_count} restaurants within {default_radius_miles} miles. "
                       f"Using {min_recommendations} closest restaurants instead.")
            closest_indices = np.argsort(distances)[:min_recommendations]
            in_radius = np.zeros_like(in_radius, dtype=bool)
            in_radius[closest_indices] = True
            max_distance = np.max(distances[closest_indices])
            actual_radius = max_distance
        

        restaurant_indices = np.where(in_radius)[0]
        

        liked_indices = [i for i, bid in enumerate(self.business_ids) if bid in liked_ids]
        disliked_indices = [i for i, bid in enumerate(self.business_ids) if bid in disliked_ids]
        

        if liked_indices:
            liked_sim = np.mean(self.sim_matrix[liked_indices][:, in_radius], axis=0)
        else:
            liked_sim = np.zeros(in_radius_count)
        

        if disliked_indices:
            disliked_sim = np.mean(self.sim_matrix[disliked_indices][:, in_radius], axis=0)
        else:
            disliked_sim = np.zeros(in_radius_count)
        

        content_scores = liked_sim - disliked_sim
        

        restaurant_ratings = self.rating_features_scaled[in_radius].flatten()
        

        if liked_indices:
            liked_ratings = np.mean(self.rating_features_scaled[liked_indices])
            # Higher scores for restaurants with similar ratings to liked ones
            rating_diff = abs(restaurant_ratings - liked_ratings)
            max_diff = np.max(rating_diff) if len(rating_diff) > 0 else 1.0
            rating_scores = 1 - (rating_diff / max_diff) if max_diff > 0 else np.ones_like(rating_diff)
        else:
            # If no liked restaurants, use normalized ratings directly
            rating_scores = (restaurant_ratings - np.min(restaurant_ratings)) / \
                           (np.max(restaurant_ratings) - np.min(restaurant_ratings))
        

        distances_in_radius = distances[in_radius]
        proximity_scores = 1 - (distances_in_radius / actual_radius)
        

        final_scores = (
            self.content_weight * content_scores + 
            self.rating_weight * rating_scores + 
            location_weight * proximity_scores
        )
        

        already_rated_mask = np.zeros(len(restaurant_indices), dtype=bool)
        for i, idx in enumerate(restaurant_indices):
            if self.business_ids[idx] in liked_ids or self.business_ids[idx] in disliked_ids:
                already_rated_mask[i] = True
        
        final_scores[already_rated_mask] = -np.inf
        

        top_indices = np.argsort(final_scores)[::-1][:top_n]
        

        recommendations = []
        for i in top_indices:
            if final_scores[i] == -np.inf:
                continue  # Skip already rated restaurants
                
            idx = restaurant_indices[i]
            restaurant_id = self.business_ids[idx]
            
            recommendation = {
                'business_id': restaurant_id,
                'distance_miles': float(distances_in_radius[i]),
                'content_score': float(content_scores[i]),
                'rating_score': float(rating_scores[i]),
                'proximity_score': float(proximity_scores[i]),
                'final_score': float(final_scores[i])
            }
            
            # Add restaurant details if available
            if 'name' in self.df.columns:
                recommendation['name'] = self.df.iloc[idx]['name']
            
            if 'stars' in self.df.columns:
                recommendation['stars'] = float(self.df.iloc[idx]['stars'])
                
            recommendations.append(recommendation)
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Generated {len(recommendations)} recommendations in {duration:.2f} seconds "
                   f"within {actual_radius:.1f} miles")
        
        return recommendations, actual_radius

    def save_model(self, filepath: str = "./model_data/recommender_model.pkl"):
        """
        Save the recommendation model to disk.
        
        Args:
            filepath: Path where to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare model components to save
        model_data = {
            'business_ids': self.business_ids,
            'content_features': self.content_features,
            'geo_features': self.geo_features,
            'rating_features': self.rating_features,
            'rating_features_scaled': self.rating_features_scaled,
            'sim_matrix': self.sim_matrix,
            'content_weight': self.content_weight,
            'rating_weight': self.rating_weight,
            'df': self.df  # Include the dataframe for restaurant details
        }
        
        # Save to disk
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
        logger.info(f"Model saved to {filepath}")
        return filepath
    
    @classmethod
    def load_model(cls, filepath: str = "../model_data/recommender_model.pkl"):
        """
        Load a saved recommendation model.
        
        Args:
            filepath: Path to the saved model file
            
        Returns:
            RestaurantRecommender: Initialized model
        """
        logger.info(f"Loading model from {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create a new instance without initialization
        recommender = cls.__new__(cls)
        
        # Set up logging
        recommender.logger = logging.getLogger('RestaurantRecommender')
        
        # Set model attributes from saved data
        recommender.business_ids = model_data['business_ids']
        recommender.content_features = model_data['content_features']
        recommender.geo_features = model_data['geo_features']
        recommender.rating_features = model_data['rating_features']
        recommender.rating_features_scaled = model_data['rating_features_scaled']
        recommender.sim_matrix = model_data['sim_matrix']
        recommender.content_weight = model_data.get('content_weight', 1.0)
        recommender.rating_weight = model_data.get('rating_weight', 1.0)
        recommender.df = model_data['df']
        
        # Initialize distance cache
        recommender.distance_cache = {}
        
        logger.info(f"Model loaded successfully with {len(recommender.df)} restaurants")
        return recommender