import numpy as np
from tqdm import tqdm

def predict(cf, test_data):
    predicted_ratings = []
    for i, ann in tqdm(test_data.iterrows(), total=len(test_data)):
        try:
            rating = cf.predict(ann['tessera'], ann['cod_prod'])
        except:
            rating = np.nan
        predicted_ratings.append(rating)
    return np.array(predicted_ratings)

def mae(y_true, y_pred):
    return (y_true - y_pred).abs().mean()