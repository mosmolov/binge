import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from typing import List, Dict, Tuple, Optional, Union
import os
import logging
from datetime import datetime
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('RestaurantRecommender')

class RestaurantRecommender:
    def __init__(self, 
                 data_path: str = "recommendations/data/cleaned_restaurants.csv", 
                 embedding_model_name: str = 'all-MiniLM-L6-v2',
                 n_components: int = 1):
        """
        Initialize the recommendation system with restaurant data.
        
        Args:
            data_path: Path to the cleaned restaurant data CSV file
            embedding_model_name: Name of the Sentence Transformer model to use
            n_components: Number of components for PCA dimensionality reduction
        """
        self.n_components = n_components
        logger.info(f"Initializing RestaurantRecommender with data from {data_path}")
        base_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
        
        # Always build a new model from scratch
        self.df = pd.read_csv(base_dir / data_path)
        self.business_ids = self.df['business_id'].values
        # Verify required columns exist
        required_columns = ['business_id', 'latitude', 'longitude', 'stars']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Data missing required columns: {missing_columns}")

        # Identify attribute columns (used for content features)
        self.attribute_columns = [col for col in self.df.columns 
                                  if col not in ['business_id', 'latitude', 'longitude', 'stars']]
        
        # Check for NaN values in the dataframe
        nan_count = self.df.isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values in the dataset. Handling them...")
            # Fill NaN values in required columns
            for col in required_columns:
                if self.df[col].isna().any():
                    if col in ['latitude', 'longitude']:
                        self.df = self.df.dropna(subset=[col])
                        logger.info(f"Dropped rows with NaN values in {col}")
                    elif col == 'stars':
                        mean_stars = self.df[col].mean()
                        self.df[col] = self.df[col].fillna(mean_stars)
                        logger.info(f"Filled NaN values in {col} with mean: {mean_stars}")

            # Fill NaN in attribute columns with 0 (or False)
            if self.attribute_columns:
                nan_in_attributes = self.df[self.attribute_columns].isna().sum().sum()
                if nan_in_attributes > 0:
                    logger.info(f"Filling {nan_in_attributes} NaN values in attribute columns with 0.")
                    self.df[self.attribute_columns] = self.df[self.attribute_columns].fillna(0)
            
            # Re-index after potential row drops
            self.df = self.df.reset_index(drop=True)
            self.business_ids = self.df['business_id'].values

        # Load the embedding model
        logger.info(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Generate the attribute embeddings and restaurant embeddings
        self._generate_attribute_name_embeddings()
        self._create_restaurant_embeddings() 

        # Extract and prepare feature sets
        self._extract_features() 
        
        # Build similarity matrix (now based on dense embeddings)
        self._build_similarity_matrix()
        
        # Set default weights for combining similarity components
        self.content_weight = 5.0
        self.rating_weight = 1.0
        
        # Initialize distance cache
        self.distance_cache = {}

        logger.info(f"Recommender initialized with {len(self.df)} restaurants")

    def _apply_dimensionality_reduction(self):
        """Apply PCA to reduce the dimensionality of restaurant embeddings."""
        logger.info(f"Applying PCA dimensionality reduction from {self.content_features.shape[1]} to {self.n_components} dimensions")
        start_time = datetime.now()
        
        if self.n_components >= self.content_features.shape[1]:
            logger.info(f"Skipping dimensionality reduction: requested components ({self.n_components}) >= current dimensions ({self.content_features.shape[1]})")
            self.pca = None
            return
        
        self.pca = PCA(n_components=self.n_components)
        self.content_features = self.pca.fit_transform(self.content_features)
        explained_variance = sum(self.pca.explained_variance_ratio_) * 100
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Dimensionality reduction completed in {duration:.2f} seconds. New shape: {self.content_features.shape}")
        logger.info(f"Explained variance: {explained_variance:.2f}%")

    def _generate_attribute_name_embeddings(self):
        """Generates embeddings for all attribute column names."""
        logger.info("Generating embeddings for attribute names...")
        start_time = datetime.now()

        # Clean attribute names (e.g., 'Cuisine_Italian' -> 'Cuisine Italian')
        cleaned_names = [name.replace('Cuisine_', '').replace('_', ' ') for name in self.attribute_columns]
        
        if not cleaned_names:
            logger.warning("No attribute columns found to generate embeddings for.")
            self.attribute_name_embeddings = {}
            return

        embeddings = self.embedding_model.encode(cleaned_names, show_progress_bar=True)
        self.attribute_name_embeddings = {
            original_name: embedding 
            for original_name, embedding in zip(self.attribute_columns, embeddings)
        }
        
        duration = (datetime.now() - start_time).total_seconds()
        emb_dim = embeddings.shape[1] if len(embeddings) > 0 else 0
        logger.info(f"Generated {len(self.attribute_name_embeddings)} attribute name embeddings (dim={emb_dim}) in {duration:.2f}s")

    def _create_restaurant_embeddings(self):
        """Creates a dense embedding for each restaurant by averaging its attribute name embeddings."""
        logger.info("Creating dense restaurant embeddings from attribute names...")
        start_time = datetime.now()
        
        num_restaurants = len(self.df)
        if not self.attribute_name_embeddings:
            logger.warning("No attribute name embeddings available. Content features will be empty.")
            embedding_dim = 1
            self.restaurant_embeddings = np.zeros((num_restaurants, embedding_dim))
        else:
            first_embedding = next(iter(self.attribute_name_embeddings.values()))
            embedding_dim = len(first_embedding)
            self.restaurant_embeddings = np.zeros((num_restaurants, embedding_dim))
            attribute_data = self.df[self.attribute_columns].values.astype(bool)

            for i in range(num_restaurants):
                present_attribute_indices = np.where(attribute_data[i])[0]
                if len(present_attribute_indices) > 0:
                    present_attribute_names = [self.attribute_columns[idx] for idx in present_attribute_indices]
                    embeddings_to_average = [
                        self.attribute_name_embeddings[name] 
                        for name in present_attribute_names 
                        if name in self.attribute_name_embeddings
                    ]
                    if embeddings_to_average:
                        self.restaurant_embeddings[i] = np.mean(embeddings_to_average, axis=0)
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Created {num_restaurants} restaurant embeddings in {duration:.2f}s")

    def _extract_features(self):
        """Extract and prepare feature sets for similarity calculations."""
        # Geographic features (latitude, longitude)
        self.geo_features = self.df[['latitude', 'longitude']].values
        
        # Star ratings
        self.rating_features = self.df[['stars']].values
        scaler = StandardScaler()
        self.rating_features_scaled = scaler.fit_transform(self.rating_features.reshape(-1, 1))
        
        # Content features from restaurant embeddings
        if hasattr(self, 'restaurant_embeddings'):
            self.content_features = self.restaurant_embeddings
            logger.info(f"Using dense restaurant embeddings (shape: {self.content_features.shape}) as content features.")
            self._apply_dimensionality_reduction()
        else:
            logger.warning("Restaurant embeddings not found. Using empty content features.")
            self.content_features = np.zeros((len(self.df), 1))
            
        # Verify no NaN values in features
        if np.isnan(self.geo_features).any() or np.isnan(self.rating_features_scaled).any() or np.isnan(self.content_features).any():
            logger.error("NaN values found in features after processing")
            raise ValueError("Input contains NaN values even after cleaning")

    def _build_similarity_matrix(self):
        """Build the content-based similarity matrix using cosine similarity."""
        logger.info("Building content similarity matrix (using dense embeddings)")
        start_time = datetime.now()
        
        self.sim_matrix = cosine_similarity(self.content_features)
        np.fill_diagonal(self.sim_matrix, 0)
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Similarity matrix built in {duration:.2f} seconds")

    def set_weights(self, content_weight: float = 1.0, rating_weight: float = 1.0):
        """
        Set weights for different components of the recommendation algorithm.
        
        Args:
            content_weight: Weight for content-based similarity
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
        cache_key = f"{lat1:.6f},{lon1:.6f},{lat2:.6f},{lon2:.6f}"
        if cache_key in self.distance_cache:
            return self.distance_cache[cache_key]
            
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        miles = 3956 * c  # Radius of earth in miles
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
        user_lat_rad = np.radians(user_lat)
        user_lon_rad = np.radians(user_lon)
        rest_lat_rad = np.radians(self.geo_features[:, 0])
        rest_lon_rad = np.radians(self.geo_features[:, 1])
        dlon = rest_lon_rad - user_lon_rad
        dlat = rest_lat_rad - user_lat_rad
        a = np.sin(dlat/2)**2 + np.cos(user_lat_rad) * np.cos(rest_lat_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        miles = 3956 * c
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
        Recommend restaurants based on user preferences and location.
        
        Args:
            liked_ids: Business IDs of restaurants the user likes
            disliked_ids: Business IDs of restaurants the user dislikes
            user_location: (latitude, longitude) of the user's location
            radius_miles: Maximum search radius in miles
            default_radius_miles: Fallback radius if not enough restaurants are found
            min_recommendations: Minimum number of restaurants to consider
            location_weight: Weight for proximity score
            top_n: Number of recommendations to return
            
        Returns:
            A tuple of:
                - List of recommended restaurants with metadata
                - The actual radius used for recommendations
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
            logger.info(f"Only {in_radius_count} restaurants within {radius_miles} miles. Expanding to default radius of {default_radius_miles} miles.")
            in_radius = distances <= default_radius_miles
            in_radius_count = np.sum(in_radius)
            actual_radius = default_radius_miles
        
        if in_radius_count < min_recommendations:
            logger.info(f"Only {in_radius_count} restaurants within {default_radius_miles} miles. Using {min_recommendations} closest restaurants instead.")
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
            liked_sim = np.zeros(np.sum(in_radius))
        
        if disliked_indices:
            disliked_sim = np.mean(self.sim_matrix[disliked_indices][:, in_radius], axis=0)
        else:
            disliked_sim = np.zeros(np.sum(in_radius))
        
        content_scores = liked_sim - disliked_sim
        restaurant_ratings = self.rating_features_scaled[in_radius].flatten()
        
        if liked_indices:
            liked_ratings = np.mean(self.rating_features_scaled[liked_indices])
            rating_diff = abs(restaurant_ratings - liked_ratings)
            max_diff = np.max(rating_diff) if len(rating_diff) > 0 else 1.0
            rating_scores = 1 - (rating_diff / max_diff) if max_diff > 0 else np.ones_like(rating_diff)
        else:
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
                continue
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
            if 'name' in self.df.columns:
                recommendation['name'] = self.df.iloc[idx]['name']
            if 'stars' in self.df.columns:
                recommendation['stars'] = float(self.df.iloc[idx]['stars'])
            recommendations.append(recommendation)
        print(recommendation)
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Generated {len(recommendations)} recommendations in {duration:.2f} seconds within {actual_radius:.1f} miles")
        return recommendations, actual_radius
