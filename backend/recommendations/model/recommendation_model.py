import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from typing import List, Dict, Tuple, Optional, Union
import os
import logging
from datetime import datetime
from pathlib import Path
from sentence_transformers import SentenceTransformer
from scipy.sparse import csr_matrix
from annoy import AnnoyIndex
from sklearn.metrics.pairwise import cosine_similarity
from backend.models.recommendations import RecommendationRequest, RecommendationResponse
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('RestaurantRecommender')

class RestaurantRecommender:
    def __init__(self, 
                 data_path: str = "recommendations/data/cleaned_restaurants.pkl", 
                 embedding_model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the recommendation system with restaurant data.
        
        Args:
            data_path: Path to the cleaned restaurant data file (CSV or pickle)
            embedding_model_name: Name of the Sentence Transformer model to use
        """
        logger.info(f"Initializing RestaurantRecommender with data from {data_path}")
        base_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
        
        # Load data based on file extension
        data_file = base_dir / data_path
        if data_file.suffix == '.csv':
            self.df = pd.read_csv(data_file)
        elif data_file.suffix in ['.pkl', '.pickle']:
            self.df = pd.read_pickle(data_file)
        else:
            raise ValueError(f"Unsupported file extension for data_path: {data_file.suffix}")
        
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
        
        # Generate the attribute embeddings and group similar attributes
        self._generate_attribute_name_embeddings()
        self._group_attributes_by_embedding(n_groups=50, linkage='average')
        # Create restaurant embeddings directly (no caching)
        self._create_restaurant_embeddings(grouped=True)

        # Extract and prepare feature sets
        self._extract_features() 
        
        # Build approximate content search index using Annoy
        self._build_annoy_index(n_trees=100)
        
        # Set default weights for combining similarity components
        self.content_weight = 2.0
        self.rating_weight = 1.0
        # Initialize weight for price similarity component
        self.price_weight = 1.0
        
        # Initialize distance cache
        self.distance_cache = {}

        logger.info(f"Recommender initialized with {len(self.df)} restaurants")

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

    def _group_attributes_by_embedding(self, n_groups: int = 10, linkage: str = 'average'):
        """
        Group attribute columns based on embedding similarity using clustering.
        Args:
            n_groups: Number of attribute groups to form
            linkage: Linkage method for Agglomerative Clustering
        Sets:
            self.grouped_attribute_columns: List of new grouped column names
            self.df_grouped_attributes: DataFrame with grouped attribute columns
        """
        if not self.attribute_name_embeddings:
            logger.warning("No attribute name embeddings available for grouping.")
            self.grouped_attribute_columns = self.attribute_columns
            self.df_grouped_attributes = self.df[self.attribute_columns].copy()
            return

        # Stack embeddings into a matrix
        emb_matrix = np.stack([self.attribute_name_embeddings[col] for col in self.attribute_columns])
        clustering = AgglomerativeClustering(n_clusters=n_groups, metric='cosine', linkage=linkage)
        labels = clustering.fit_predict(emb_matrix)

        # Map group index to attribute columns
        group_to_cols = {i: [] for i in range(n_groups)}
        for col, label in zip(self.attribute_columns, labels):
            group_to_cols[label].append(col)

        # Create new grouped columns (sum of group columns, numeric only)
        grouped_data = {}
        grouped_names = []
        for group_idx, cols in group_to_cols.items():
            group_name = f'grouped_attr_{group_idx}'
            grouped_names.append(group_name)
            # Select only numeric columns and coerce non-numeric to NaN, then fill with 0
            numeric_df = self.df[cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            grouped_data[group_name] = numeric_df.sum(axis=1)
        self.df_grouped_attributes = pd.DataFrame(grouped_data)
        self.grouped_attribute_columns = grouped_names
        logger.info(f"Grouped {len(self.attribute_columns)} attributes into {n_groups} groups.")

    def _create_restaurant_embeddings(self, grouped: bool = False):
        """Creates a dense embedding for each restaurant by averaging its attribute name embeddings."""
        logger.info("Creating dense restaurant embeddings from attribute names...")
        start_time = datetime.now()
        if grouped and hasattr(self, 'grouped_attribute_columns'):
            attribute_columns = self.grouped_attribute_columns
            attribute_data = self.df_grouped_attributes[attribute_columns].values.astype(bool)
        else:
            attribute_columns = self.attribute_columns
            attribute_data = self.df[attribute_columns].values.astype(bool)
        num_restaurants = len(self.df)
        if not self.attribute_name_embeddings:
            logger.warning("No attribute name embeddings available. Content features will be empty.")
            embedding_dim = 1
            self.restaurant_embeddings = np.zeros((num_restaurants, embedding_dim))
        else:
            first_embedding = next(iter(self.attribute_name_embeddings.values()))
            embedding_dim = len(first_embedding)
            self.restaurant_embeddings = np.zeros((num_restaurants, embedding_dim))
            for i in range(num_restaurants):
                present_attribute_indices = np.where(attribute_data[i])[0]
                if len(present_attribute_indices) > 0:
                    present_attribute_names = [attribute_columns[idx] for idx in present_attribute_indices]
                    embeddings_to_average = [
                        self.attribute_name_embeddings.get(name, first_embedding)
                        for name in present_attribute_names
                    ]
                    if embeddings_to_average:
                        self.restaurant_embeddings[i] = np.mean(embeddings_to_average, axis=0)
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Created {num_restaurants} restaurant embeddings (shape: {self.restaurant_embeddings.shape}) in {duration:.2f}s (grouped={grouped})")
        # Add check for zero embeddings
        if np.all(self.restaurant_embeddings == 0):
            logger.warning("All generated restaurant embeddings are zero. Content scores will likely be zero.")
        elif np.isnan(self.restaurant_embeddings).any():
             logger.warning("NaN values found in generated restaurant embeddings.")

    def _load_or_create_restaurant_embeddings(self, grouped: bool = False, cache_path: str = None):
        """Create restaurant embeddings without any file caching."""
        self._create_restaurant_embeddings(grouped=grouped)

    def _extract_features(self):
        """Extract and prepare feature sets for similarity calculations."""
        # Geographic features (latitude, longitude)
        self.geo_features = self.df[['latitude', 'longitude']].values
        
        # Star ratings
        self.rating_features = self.df[['stars']].values
        scaler = StandardScaler()
        self.rating_features_scaled = scaler.fit_transform(self.rating_features.reshape(-1, 1))
        # Price features (RestaurantsPriceRange2)
        if 'RestaurantsPriceRange2' in self.df.columns:
            logger.info("Price range column found, processing...")
            # Fill missing and convert to numeric
            prices = pd.to_numeric(self.df['RestaurantsPriceRange2'], errors='coerce')
            prices = prices.fillna(prices.mean()).values.reshape(-1, 1)
            price_scaler = StandardScaler()
            self.price_features_scaled = price_scaler.fit_transform(prices).flatten()
        else:
            # Fallback to zeros if price info unavailable
            self.price_features_scaled = np.zeros(len(self.df))

        # Content features from restaurant embeddings
        if hasattr(self, 'restaurant_embeddings'):
            self.content_features = self.restaurant_embeddings
            logger.info(f"Using dense restaurant embeddings (shape: {self.content_features.shape}) as content features.")
        else:
            logger.warning("Restaurant embeddings not found. Using empty content features.")
            self.content_features = np.zeros((len(self.df), 1))
            
        # Verify no NaN values in features
        if np.isnan(self.geo_features).any() or np.isnan(self.rating_features_scaled).any() or np.isnan(self.content_features).any():
            logger.error("NaN values found in features after processing")
            raise ValueError("Input contains NaN values even after cleaning")

    def _build_annoy_index(self, n_trees: int = 10):
        """
        Builds an Annoy index for approximate nearest neighbor search on restaurant embeddings.
        """
        logger.info(f"Building Annoy index with {n_trees} trees for approximate content search")
        dim = self.restaurant_embeddings.shape[1]
        self.annoy_index = AnnoyIndex(dim, metric='angular')
        for i, emb in enumerate(self.restaurant_embeddings):
            self.annoy_index.add_item(i, emb.tolist())
        self.annoy_index.build(n_trees)

    def set_weights(self, content_weight: float = 1.0, rating_weight: float = 1.0):
        """
        Set weights for different components of the recommendation algorithm.
        
        Args:
            content_weight: Weight for content-based similarity
            rating_weight: Weight for rating similarity
        """
        self.content_weight = content_weight
        self.rating_weight = rating_weight
        # Update price weight as well
        self.price_weight = getattr(self, 'price_weight', 1.0)
        logger.info(f"Weights updated: content={content_weight}, rating={rating_weight}, price={self.price_weight}")

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
        request: RecommendationRequest,
        default_radius_miles: float = 25.0,
        min_recommendations: int = 5,
        location_weight: float = 10.0,
        top_n: int = 5,
        annoy_search_k: int = 1000, # Number of candidates to retrieve from Annoy
        location_search_k: int = 1000 # Number of closest candidates to consider by location
    ) -> Tuple[List[Dict], float]:
        """
        Recommend restaurants based on user preferences and location, using Annoy for candidate generation.
        
        Args:
            liked_ids: Business IDs of restaurants the user likes
            disliked_ids: Business IDs of restaurants the user dislikes
            user_location: (latitude, longitude) of the user's location
            radius_miles: Maximum search radius in miles
            default_radius_miles: Fallback radius if not enough restaurants are found
            min_recommendations: Minimum number of restaurants to consider after filtering
            location_weight: Weight for proximity score
            top_n: Number of recommendations to return
            annoy_search_k: Number of nearest neighbors to retrieve from Annoy index based on content
            location_search_k: Number of nearest neighbors to consider based purely on location
            
        Returns:
            A tuple of:
                - List of recommended restaurants with metadata
                - The actual radius used for recommendations
        """
        start_time = datetime.now()
        if request.user_location is None:
            raise ValueError("User location is required for recommendations")
        user_lat, user_lon = request.user_location[0], request.user_location[1]
        logger.info(f"Generating recommendations for user at ({user_lat:.4f}, {user_lon:.4f}) using Annoy (k={annoy_search_k}) and Location (k={location_search_k})")
        liked_set = set(request.liked_ids)
        disliked_set = set(request.disliked_ids)
        liked_indices = [i for i, bid in enumerate(self.business_ids) if bid in liked_set]
        disliked_indices = [i for i, bid in enumerate(self.business_ids) if bid in disliked_set]
        logger.info(f"Liked indices: {liked_indices}, Disliked indices: {disliked_indices}")

        # --- Candidate Generation ---

        # 1. Get candidates based on content similarity (Annoy)
        annoy_candidate_indices = np.array([], dtype=int)
        if liked_indices:
            # Calculate average embedding of liked restaurants
            liked_embs = self.content_features[liked_indices]
            avg_liked_emb = np.mean(liked_embs, axis=0)
            
            # Find nearest neighbors using Annoy
            annoy_candidate_indices, _ = self.annoy_index.get_nns_by_vector(
                avg_liked_emb, annoy_search_k, include_distances=True
            )
            annoy_candidate_indices = np.array(annoy_candidate_indices) # Ensure numpy array
            logger.info(f"Retrieved {len(annoy_candidate_indices)} candidates from Annoy based on liked items.")
        else:
             logger.info("No liked items provided, skipping Annoy search.")

        # 2. Get candidates based on proximity
        distances_all = self.batch_haversine_distance(user_lat, user_lon)
        # Ensure location_search_k is not larger than the total number of restaurants
        num_restaurants = len(self.df)
        actual_location_search_k = min(location_search_k, num_restaurants)
        # Get indices of the closest restaurants
        location_candidate_indices = np.argsort(distances_all)[:actual_location_search_k]
        logger.info(f"Retrieved {len(location_candidate_indices)} candidates based on proximity (top {actual_location_search_k}).")

        # 3. Combine candidate sets (Annoy + Location) and remove duplicates
        candidate_indices = np.union1d(annoy_candidate_indices, location_candidate_indices)
        logger.info(f"Combined Annoy and location candidates: {len(candidate_indices)} unique indices.")

        # --- Filtering Candidates --- 
        
        # 1. Filter by Distance Radius (using the combined candidate set)
        # Get distances only for the combined candidates
        distances_candidates = distances_all[candidate_indices]
        
        in_radius_mask = distances_candidates <= request.radius_miles
        actual_radius = request.radius_miles
        
        # Expand radius if too few candidates in the combined set are within the initial radius
        if np.sum(in_radius_mask) < min_recommendations and len(candidate_indices) >= min_recommendations:
            logger.info(f"Only {np.sum(in_radius_mask)} combined candidates within {request.radius_miles} miles. Expanding to {default_radius_miles} miles.")
            in_radius_mask = distances_candidates <= default_radius_miles
            actual_radius = default_radius_miles

        # If still too few after expanding, take the closest candidates from the combined set
        if np.sum(in_radius_mask) < min_recommendations:
             logger.info(f"Only {np.sum(in_radius_mask)} combined candidates within {actual_radius} miles. Taking {min_recommendations} closest combined candidates.")
             # Sort candidates by distance and take the top min_recommendations
             closest_candidate_indices_local = np.argsort(distances_candidates)[:min(min_recommendations, len(candidate_indices))]
             in_radius_mask = np.zeros_like(distances_candidates, dtype=bool)
             # Check if closest_candidate_indices_local is not empty before trying to index
             if len(closest_candidate_indices_local) > 0:
                 in_radius_mask[closest_candidate_indices_local] = True
                 # Update actual_radius to the distance of the furthest included candidate
                 actual_radius = distances_candidates[closest_candidate_indices_local[-1]]
             else: # Handle edge case where candidate_indices was empty
                 actual_radius = 0

        # Apply the radius mask to the original candidate_indices
        filtered_indices_global = candidate_indices[in_radius_mask]
        if len(filtered_indices_global) == 0:
            logger.warning("No candidates found after distance filtering. Returning empty list.")
            return [], actual_radius

        # 2. Filter out already rated restaurants
        already_rated_mask = np.array([self.business_ids[idx] in liked_set or self.business_ids[idx] in disliked_set 
                                       for idx in filtered_indices_global])
        
        final_candidate_indices = filtered_indices_global[~already_rated_mask]
        if len(final_candidate_indices) == 0:
            logger.warning("No candidates left after removing already rated items. Returning empty list.")
            return [], actual_radius
        
        logger.info(f"Filtered down to {len(final_candidate_indices)} final candidates.")

        # --- Score Calculation for Final Candidates --- 
        
        # Get features only for the final candidates
        candidate_content_features = self.content_features[final_candidate_indices]
        candidate_rating_features_scaled = self.rating_features_scaled[final_candidate_indices].flatten()
        candidate_distances = distances_all[final_candidate_indices]

        # 1. Content Scores
        if liked_indices:
            liked_embs = self.content_features[liked_indices]
            sim_liked = cosine_similarity(candidate_content_features, liked_embs).mean(axis=1)
        else:
            sim_liked = np.zeros(len(final_candidate_indices))
            
        if disliked_indices:
            disliked_embs = self.content_features[disliked_indices]
            sim_disliked = cosine_similarity(candidate_content_features, disliked_embs).mean(axis=1)
        else:
            sim_disliked = np.zeros(len(final_candidate_indices))
            
        content_scores = sim_liked - sim_disliked
        logger.info(f"Shape of content_scores (candidates): {content_scores.shape}, Example values: {content_scores[:5]}")

        # 2. Rating Scores
        if liked_indices:
            liked_ratings = np.mean(self.rating_features_scaled[liked_indices])
            rating_diff = np.abs(candidate_rating_features_scaled - liked_ratings)
            max_diff = np.max(rating_diff) if len(rating_diff) > 0 else 1.0
            rating_scores = 1 - (rating_diff / max_diff) if max_diff > 0 else np.ones_like(rating_diff)
        else:
            min_rating = np.min(candidate_rating_features_scaled)
            max_rating = np.max(candidate_rating_features_scaled)
            # Handle division by zero if all candidate ratings are the same
            if max_rating > min_rating:
                rating_scores = (candidate_rating_features_scaled - min_rating) / (max_rating - min_rating)
            elif len(candidate_rating_features_scaled) > 0: # Check if there are candidates
                # All candidates have the same rating, assign a neutral score
                rating_scores = np.full_like(candidate_rating_features_scaled, 0.5)
            else:
                # No candidates, return empty array
                rating_scores = np.array([])
        logger.info(f"Shape of rating_scores (candidates): {rating_scores.shape}, Example values: {rating_scores[:5]}")

        # 3. Proximity Scores
        # Avoid division by zero if actual_radius is 0
        proximity_scores = 1 - (candidate_distances / actual_radius) if actual_radius > 0 else np.ones_like(candidate_distances)
        logger.info(f"Shape of proximity_scores (candidates): {proximity_scores.shape}, Example values: {proximity_scores[:5]}")

        # 4. Price Scores
        candidate_price_scaled = self.price_features_scaled[final_candidate_indices]
        if liked_indices:
            avg_liked_price = np.mean(self.price_features_scaled[liked_indices])
            price_diff = np.abs(candidate_price_scaled - avg_liked_price)
            max_price_diff = np.max(price_diff) if price_diff.size > 0 else 1.0
            price_scores = 1 - (price_diff / max_price_diff) if max_price_diff > 0 else np.ones_like(price_diff)
        else:
            # Normalize by range if no liked items
            min_price = np.min(candidate_price_scaled)
            max_price = np.max(candidate_price_scaled)
            if max_price > min_price:
                price_scores = 1 - ((candidate_price_scaled - min_price) / (max_price - min_price))
            else:
                price_scores = np.full_like(candidate_price_scaled, 0.5)
        logger.info(f"Shape of price_scores (candidates): {price_scores.shape}, Example values: {price_scores[:5]}")

        # --- Combine Scores and Rank --- 
        final_scores = (
            self.content_weight * content_scores +
            self.rating_weight * rating_scores +
            location_weight * proximity_scores +
            self.price_weight * price_scores
        )
        logger.info(f"Shape of final_scores (candidates): {final_scores.shape}, Example values: {final_scores[:5]}")

        # Get top N indices from the final candidates
        # Need to sort based on the scores calculated for the candidates
        top_indices_local = np.argsort(final_scores)[::-1][:top_n]
        
        # Map local indices back to global indices
        top_indices_global = final_candidate_indices[top_indices_local]

        # --- Format Recommendations --- 
        recommendations = []
        for i, idx in enumerate(top_indices_global):
            local_idx = top_indices_local[i] # Get the index within the candidate arrays
            restaurant_id = self.business_ids[idx]
            recommendation = {
                'business_id': restaurant_id,
                'distance_miles': float(candidate_distances[local_idx]),
                'content_score': float(content_scores[local_idx]),
                'rating_score': float(rating_scores[local_idx]),
                'proximity_score': float(proximity_scores[local_idx]),
                'price_score': float(price_scores[local_idx]),
                'final_score': float(final_scores[local_idx])
            }
            if 'name' in self.df.columns:
                recommendation['name'] = self.df.iloc[idx]['name']
            if 'stars' in self.df.columns:
                recommendation['stars'] = float(self.df.iloc[idx]['stars'])
            recommendations.append(recommendation)
            
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Generated {len(recommendations)} recommendations in {duration:.2f} seconds using Annoy within {actual_radius:.1f} miles")
        logger.info(f"Top recommendations: {recommendations[:5] if recommendations else 'None'}")
        return recommendations, actual_radius

    def evaluate(self, test_cases: List[Dict], top_n: int = 5) -> Dict[str, float]:
        """
        Evaluate the recommender on multiple test cases.
        Each test case should be a dict with keys: 'liked_ids', 'disliked_ids', 'user_location', 'ground_truth_liked_ids'.
        Returns average precision@top_n and recall@top_n over all cases.
        """
        precisions, recalls = [], []
        logger.info(f"Evaluating recommender with {len(test_cases)} test cases at k={top_n}")
        for i, case in enumerate(test_cases):
            try:
                recs, _ = self.recommend_restaurants(
                    case.get('liked_ids', []), 
                    case.get('disliked_ids', []),
                    case['user_location'], 
                    top_n=top_n
                    # Add annoy_search_k if needed for evaluation consistency
                    # annoy_search_k=500 
                )
                rec_ids = [r['business_id'] for r in recs]
                true_ids = case.get('ground_truth_liked_ids', [])
                
                p_at_k = self.precision_at_k(rec_ids, true_ids, top_n)
                r_at_k = self.recall_at_k(rec_ids, true_ids, top_n)
                precisions.append(p_at_k)
                recalls.append(r_at_k)
                logger.info(f"Test case {i+1}: Precision@{top_n}={p_at_k:.4f}, Recall@{top_n}={r_at_k:.4f}")
            except Exception as e:
                logger.error(f"Error processing test case {i+1}: {e}", exc_info=True)
                # Optionally append default values or skip the case
                # precisions.append(0.0)
                # recalls.append(0.0)

        avg_precision = float(np.mean(precisions)) if precisions else 0.0
        avg_recall = float(np.mean(recalls)) if recalls else 0.0
        logger.info(f"Evaluation complete: Avg Precision@{top_n}={avg_precision:.4f}, Avg Recall@{top_n}={avg_recall:.4f}")
        return {'precision_at_k': avg_precision, 'recall_at_k': avg_recall}
