import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

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

def parallel_predict(cfUser, cfItem, test_data):
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(predict, cfUser, test_data): 'user',
            executor.submit(predict, cfItem, test_data): 'item'
        }

        predicted_ratings_user = None
        predicted_ratings_item = None

        for future in as_completed(futures):
            model_type = futures[future]
            if model_type == 'user':
                predicted_ratings_user = future.result()
            else:
                predicted_ratings_item = future.result()

        return predicted_ratings_user, predicted_ratings_item
