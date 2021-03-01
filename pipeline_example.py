import pandas as pd
from nlp_pipeline import NLPPipeline
import numpy as np
import time

reviews_df = pd.read_csv('tripadvisor_hotel_reviews.csv')
downsample = 0.5
sample_size = int(len(reviews_df) * downsample)
print(f"Sample size: {sample_size}")
reviews_df = reviews_df.sample(sample_size)
reviews = reviews_df['Review']
ratings = np.array(reviews_df['Rating']).reshape(-1)
pipeline = NLPPipeline(reviews, ratings)

# Best pipeline so far: {'tfidf': {'max_df': 0.9, 'min_df': 0.001, 'ngram_range': (1, 3), 'max_features': 10000},
#								 'features': {'n_components': 10, 'num_top_words': 10},
#								 'model': {'type': 'rfc'}}, 0.5810019518542615]
start = time.time()
tfidf_grid = dict(max_df = [0.8, 0.9], min_df=[0.001], ngram_range = [(1, 3)], max_features=[int(1.0e4), int(5.0e4)])
features_grid = dict(n_components = [10, 20], num_top_words = [10])
model_grid = dict(type=['dtc', 'rfc'])
steps_grids = dict(tfidf=tfidf_grid, features=features_grid, model=model_grid)
grid_results = pipeline.grid_search(steps_grids)

for params, score in grid_results:
	print(f"Parameters: {params}, score = {score}")

print(f"Run time: {time.time() - start}")