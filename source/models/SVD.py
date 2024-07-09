import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from .base_model import baseModel

class SVDModel(baseModel):
    def __init__(self, k):
        self.k = k  # Numero di componenti latenti
        
    def fit(self, U):
        self.U = U
        self.user_item_matrix = U.fillna(0).to_numpy()  # Converti il DataFrame in una matrice NumPy
        self.user_means = np.mean(self.user_item_matrix, axis=1)
        self.user_item_matrix_demeaned = self.user_item_matrix - self.user_means.reshape(-1, 1)
        
        # Applicare SVD
        self.U_matrix, self.sigma, self.Vt_matrix = svds(self.user_item_matrix_demeaned, k=self.k)
        self.sigma = np.diag(self.sigma)
        
        # Ricostruzione della matrice
        self.predicted_ratings = np.dot(np.dot(self.U_matrix, self.sigma), self.Vt_matrix) + self.user_means.reshape(-1, 1)
        self.predicted_ratings_df = pd.DataFrame(self.predicted_ratings, columns=U.columns, index=U.index)

        
    def predict(self, xi, sj):
        return self.predicted_ratings_df.loc[xi, sj]
    
    def recommend(self, xi, n):
        user_predictions = self.predicted_ratings_df.loc[xi].sort_values(ascending=False)
        rated_items = self.U.loc[xi].dropna().index
        recommendations = user_predictions.drop(rated_items).head(n)
        return list(recommendations.index)
