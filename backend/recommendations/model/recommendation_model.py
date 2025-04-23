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
                                  if col not in ['business_id', 'latitude', 'longitude', 'stars', 'is_michelin']]
        self._attr_weights = self._compute_attribute_idf(self.attribute_columns)
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
        self._create_restaurant_embeddings(grouped=False)

        # Extract and prepare feature sets
        self._extract_features() 
        
        # Build approximate content search index using Annoy
        self._build_annoy_index(n_trees=50)
        
        # Set default weights for combining similarity components
        self.content_weight = 2.0
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
    def _compute_attribute_idf(self, attr_cols: List[str]) -> np.ndarray:
        """
        Returns a weight per attribute column.  Common flags ≈ 1,   very rare flags ≪ 1
        so they contribute less when we average embeddings.
        """
        # how many restaurants have the flag turned on?
        dfreq = np.maximum(1,
                        self.df[attr_cols].astype(bool).sum(axis=0).values)
        # inverse frequency (IDF)  – use sqrt to soften the curve
        raw_idf = np.sqrt(len(self.df) / dfreq)

        # Convert to weights that *down‑weight* rarity
        #   common flag (dfreq ~ N) → raw_idf ~1   → weight ~1
        #   very rare  (dfreq ~1)   → raw_idf ~√N → weight very small
        weights = raw_idf.max() / raw_idf
        return weights.astype("float32")

    def _create_restaurant_embeddings(self, grouped: bool = False):
        """Creates a dense embedding for each restaurant by averaging its attribute name embeddings."""
        logger.info("Creating dense restaurant embeddings from attribute names...")
        start_time = datetime.now()
        
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
                    weights = self._attr_weights[present_attribute_indices]          # 1‑D array
                    embeddings_to_average = [
                        self.attribute_name_embeddings.get(name, first_embedding) * w
                        for name, w in zip(present_attribute_names, weights)
                    ]
                    self.restaurant_embeddings[i] = (
                        np.sum(embeddings_to_average, axis=0) / np.sum(weights)
                    )

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
        if np.isnan(self.geo_features).any() or np.isnan(self.content_features).any():
            logger.error("NaN values found in features after processing")
            raise ValueError("Input contains NaN values even after cleaning")

    def _build_annoy_index(self, n_trees: int = 10):
        """
        Builds an Annoy index for approximate nearest neighbor search on restaurant embeddings.
        """
        logger.info(f"Building Annoy index with {n_trees} trees for approximate content search")
        dim = self.restaurant_embeddings.shape[1]
        self.annoy = AnnoyIndex(dim, 'angular')
        for i, embedding in enumerate(self.restaurant_embeddings):
            self.annoy.add_item(i, embedding)
        self.annoy.build(n_trees)
        logger.info(f"Annoy index built with {len(self.restaurant_embeddings)} items")

    def set_weights(self, content_weight: float = 1.0):
        """
        Set weights for different components of the recommendation algorithm.
        
        Args:
            content_weight: Weight for content-based similarity
        """
        self.content_weight = content_weight
        # Update price weight as well
        self.price_weight = getattr(self, 'price_weight', 1.0)
        logger.info(f"Weights updated: content={content_weight}, price={self.price_weight}")

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
        top_n: int = 100,
        annoy_search_k: int = 2000, # Number of candidates to retrieve from Annoy
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

        annoy_candidate_indices = np.array([], dtype=int)
        if liked_indices:
            # Calculate average embedding of liked restaurants
            liked_embs = self.content_features[liked_indices]
            avg_liked_emb = np.mean(liked_embs, axis=0)
            
            annoy_candidate_indices = self.annoy.get_nns_by_vector(
                avg_liked_emb, 
                annoy_search_k, 
                include_distances=False
            )
            # Filter out disliked indices from the candidates
            annoy_candidate_indices = [idx for idx in annoy_candidate_indices if idx not in disliked_indices]
            logger.info(f"Retrieved {len(annoy_candidate_indices)} candidates from Annoy indices based on liked items.")
        else:
             logger.info("No liked items provided, skipping Annoy search.")

        
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
        final_candidate_indices = candidate_indices[in_radius_mask]
        
        
        logger.info(f"Filtered down to {len(final_candidate_indices)} final candidates.")
        
        # Get features only for the final candidates
        candidate_content_features = self.content_features[final_candidate_indices]
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
            
        content_scores = sim_liked - sim_disliked * 0.05
        logger.info(f"Shape of content_scores (candidates): {content_scores.shape}, Example values: {content_scores[:5]}")
        

        # 3. Proximity Scores
        # Avoid division by zero if actual_radius is 0
        proximity_scores = 1 - (candidate_distances / actual_radius) if actual_radius > 0 else np.ones_like(candidate_distances)
        logger.info(f"Shape of proximity_scores (candidates): {proximity_scores.shape}, Example values: {proximity_scores[:5]}")

        # 4. Price Scores and Filtering
        candidate_price_scaled = self.price_features_scaled[final_candidate_indices]
        
        # Get raw prices (not scaled) for user-specified price preference
        if 'RestaurantsPriceRange2' in self.df.columns:
            candidate_prices_raw = pd.to_numeric(self.df.loc[final_candidate_indices, 'RestaurantsPriceRange2'], errors='coerce').fillna(2).values
        else:
            candidate_prices_raw = np.ones(len(final_candidate_indices)) * 2  # Default to middle price

        # If user has specified a desired price, filter to only include exact matches
        if request.desired_price is not None:
            logger.info(f"Filtering candidates by exact price match: {request.desired_price}")
            # Create a mask for exact price matches
            price_match_mask = np.isclose(candidate_prices_raw, request.desired_price)
            
            # If we have exact matches, filter all arrays to only include those matches
            if np.any(price_match_mask):
                final_candidate_indices = final_candidate_indices[price_match_mask]
                candidate_content_features = candidate_content_features[price_match_mask]
                candidate_distances = candidate_distances[price_match_mask]
                proximity_scores = proximity_scores[price_match_mask]
                candidate_price_scaled = candidate_price_scaled[price_match_mask]
                candidate_prices_raw = candidate_prices_raw[price_match_mask]
                content_scores = content_scores[price_match_mask]
                
                logger.info(f"Filtered down to {len(final_candidate_indices)} candidates with exact price match {request.desired_price}")
                # All remaining candidates have the exact price, so give them perfect price scores
                price_scores = np.ones(len(final_candidate_indices))
            else:
                logger.warning(f"No candidates found with exact price match {request.desired_price}. Falling back to price similarity scoring.")
                # Calculate difference between restaurant price and user's desired price
                price_diff = np.abs(candidate_prices_raw - request.desired_price)
                # Normalize to [0,1] range where 1 is perfect match and 0 is worst match
                max_possible_diff = 3.0  # Max difference between price level 1 and 4
                price_scores = 1.0 - (price_diff / max_possible_diff)
        elif liked_indices:
            logger.info("No Desired price, using liked restaurants' average price")
            # Fall back to using liked restaurants' average price
            avg_liked_price = np.mean(self.price_features_scaled[liked_indices])
            price_diff = np.abs(candidate_price_scaled - avg_liked_price)
            max_price_diff = np.max(price_diff) if price_diff.size > 0 else 1.0
            price_scores = 1 - (price_diff / max_price_diff) if max_price_diff > 0 else np.ones_like(price_diff)
        else:
            # Normalize by range if no liked items or desired price
            min_price = np.min(candidate_price_scaled)
            max_price = np.max(candidate_price_scaled)
            if max_price > min_price:
                price_scores = 1 - ((candidate_price_scaled - min_price) / (max_price - min_price))
            else:
                price_scores = np.full_like(candidate_price_scaled, 0.5)
        logger.info(f"Shape of price_scores (candidates): {price_scores.shape}, Example values: {price_scores[:5]}")

        

        michelin_penalty = 0.3      # tune 0-1
        is_michelin_flag = self.df['is_michelin'].values[final_candidate_indices]
        final_scores = (
            self.content_weight * content_scores
            + location_weight       * proximity_scores
            + self.price_weight     * price_scores
            - michelin_penalty      * is_michelin_flag   # subtract if Michelin
        )

        logger.info(f"Shape of final_scores (candidates): {final_scores.shape}, Example values: {final_scores[:5]}")
        
        # select first n candidates
        top_indices_local = np.argsort(final_scores)[::-1]
        top_indices_global = final_candidate_indices[top_indices_local]

        if self.df.loc[top_indices_global, "is_michelin"].all():
            non_mich = [i for i in final_candidate_indices
                        if not self.df.at[i, "is_michelin"]]
            if non_mich:
                # pick the highest-scoring non-Michelin
                best_non = max(non_mich,
                            key=lambda i: final_scores[
                                np.where(final_candidate_indices == i)[0][0]])
                # put it in the last slot
                top_indices_global[-1] = best_non

        
        # --- Format Recommendations --- 
        recommendations = []
        for i, idx in enumerate(top_indices_global):
            local_idx = top_indices_local[i] # Get the index within the candidate arrays
            restaurant_id = self.business_ids[idx]
            recommendation = {
                'business_id': restaurant_id,
                'distance_miles': float(candidate_distances[local_idx]),
                'content_score': float(content_scores[local_idx]),
                'proximity_score': float(proximity_scores[local_idx]),
                'price_score': float(price_scores[local_idx]),
                'final_score': float(final_scores[local_idx])
            }
            if 'name' in self.df.columns:
                recommendation['name'] = self.df.iloc[idx]['name']
            recommendations.append(recommendation)
            
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Generated {len(recommendations)} recommendations in {duration:.2f} seconds using Annoy within {actual_radius:.1f} miles")
        logger.info(f"Top recommendations: {recommendations[:5] if recommendations else 'None'}")
        return recommendations, actual_radius

    def get_attributes(self) -> Dict[str, List[str]]:
        """
        Get attributes for restaurant additions across all restaurants.
        
        Returns:
            Dictionary of attribute names and their counts
        """
        # Extract unique Cuisines from columns that start with Cuisine
        all_cuisines = []
        cuisine_columns = [col for col in self.df.columns if col.startswith('Cuisine_')]
        for cuisine in cuisine_columns:
            all_cuisines.append(cuisine.replace('Cuisine_', ''))
        
        # Extract ambience types from column names that start with Ambience
        all_ambiences = []
        ambience_columns = [col for col in self.df.columns if col.startswith('Ambience_')]
        for ambience in ambience_columns:
            all_ambiences.append(ambience.replace('Ambience_', ''))
        
        # Extract GoodFor attributes
        all_good_for = []
        good_for_columns = [col for col in self.df.columns if col.startswith('GoodFor')]
        for good_for in good_for_columns:
            all_good_for.append(good_for.replace('GoodFor', ''))
        
        # Return attribute options
        return {
            'Cuisines': sorted(all_cuisines),
            'Ambiences': sorted(all_ambiences),
            'GoodFor': sorted(all_good_for)
        }
