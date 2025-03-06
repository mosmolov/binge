import pickle
import numpy as np
import pandas as pd
import os

class RecommendationModel:
    def __init__(self, model_path):
        # Load the model components
        with open(model_path, 'rb') as f:
            self.model_components = pickle.load(f)
        
        self.sim_matrix = self.model_components['sim_matrix']
        self.df = self.model_components['df']
        self.business_ids = self.model_components['business_ids']
        self.scaler = self.model_components['scaler']
    
    def recommend_restaurants(self, liked_ids, disliked_ids=None, top_n=5):
        """
        Recommend restaurants based on both liked and disliked restaurants.

        liked_ids: list of business_id strings that the user likes.
        disliked_ids: list of business_id strings that the user dislikes.
        top_n: number of recommendations to return.
        """
        if disliked_ids is None:
            disliked_ids = []
            
        # Get indices for liked and disliked restaurants.
        liked_indices = self.df.index[self.df['business_id'].isin(liked_ids)].tolist()
        disliked_indices = self.df.index[self.df['business_id'].isin(disliked_ids)].tolist()
        
        # Compute average similarity scores from liked restaurants.
        if liked_indices:
            liked_sim = self.sim_matrix[liked_indices].mean(axis=0)
        else:
            liked_sim = np.zeros(self.sim_matrix.shape[0])
        
        # Compute average similarity scores from disliked restaurants.
        if disliked_indices:
            disliked_sim = self.sim_matrix[disliked_indices].mean(axis=0)
        else:
            disliked_sim = np.zeros(self.sim_matrix.shape[0])
        
        # Calculate the net score by subtracting the disliked similarity from liked similarity.
        net_scores = liked_sim - disliked_sim
        
        # Exclude already rated restaurants (both liked and disliked).
        for idx in liked_indices + disliked_indices:
            net_scores[idx] = -np.inf
        
        # Get indices of the top recommended restaurants.
        rec_indices = np.argsort(net_scores)[::-1][:top_n]
        
        # Return the recommended business IDs
        return self.df.iloc[rec_indices]['business_id'].tolist()

def load_model(model_path=None):
    """Load the recommendation model from the specified path or use the default path"""
    if model_path is None:
        # Use default path relative to this file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, '../model_data/recommendation_model.pkl')
    
    return RecommendationModel(model_path)
