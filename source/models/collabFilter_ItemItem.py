import numpy as np
import pandas as pd
from .collabFilter import CollaborativeFilter

class CollaborativeFilter_ItemItem(CollaborativeFilter):
    def __init__(self, N):
        self.N = N
        self.cosine = lambda x, y: np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))
        
    def fit(self, U):
        self.U = U
        self.profiles = (U - U.mean(axis=0).values[np.newaxis, :]).fillna(0)
        
    def predict(self, xi, sj):
        rated_items = self.U.loc[xi].dropna().index
        item_profiles = self.profiles[rated_items]
        
        similarities = item_profiles.apply(lambda x: self.cosine(self.profiles[sj], x), axis=0)
        selected_similarities = similarities.sort_values(ascending=False).head(self.N)
        similar_items = selected_similarities.index
        
        predicted_rating = (self.U.loc[xi, similar_items] * selected_similarities).sum() / selected_similarities.sum()
        return predicted_rating

    def recommend(self, xi, n):
        rated_items = self.U.loc[xi].dropna().index
        unrated_items = self.U.columns.difference(rated_items)
        
        predictions = {item: self.predict(xi, item) for item in unrated_items}
        
        recommended_items = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:n]
        return recommended_items