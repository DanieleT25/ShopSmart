import numpy as np
from .collabFilter import CollaborativeFilter

class CollaborativeFilter_UserUser(CollaborativeFilter):
    def __init__(self, N):
        self.N = N
        self.cosine = lambda x, y: np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))
        
    def fit(self, U):
        self.U = U
        self.profiles = (U - U.mean(axis=1).values[:, np.newaxis]).fillna(0)
        
    def predict(self, xi, sj):
        similarities = self.profiles.apply(lambda x: self.cosine(self.profiles.loc[xi], x), axis=1).drop(xi)
        selected_users = self.U.loc[:, sj].dropna().index
        selected_similarities = similarities.loc[selected_users].sort_values(ascending=False).head(self.N)
        similar_users = selected_similarities.index
        predicted_rating = (self.U.loc[similar_users, sj] * selected_similarities).sum() / selected_similarities.sum()

        return predicted_rating

    def recommend(self, xi, n):
        rated_items = self.U.loc[xi].dropna().index
        unrated_items = self.U.columns.difference(rated_items)
        
        predictions = {item: self.predict(xi, item) for item in unrated_items}
        
        recommended_items = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:n]
        return recommended_items