import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sklearn.manifold import TSNE

import json

tqdm.pandas()

orig_dataset = pd.read_csv('data/shuffled_sql_data.csv', names=['query', 'label'], header=None)
drift_dataset = pd.read_csv('data/shuffled_data_drifted_sql_data.csv', names=['query', 'label'], header=None)
drift_dataset = drift_dataset[drift_dataset['query'].str.len() >= 1]
orig_dataset = orig_dataset.iloc[:len(drift_dataset)]

model = SentenceTransformer('C:/Users/ROHAN/time-series-kafka-demo/model/')

orig_dataset['ft_encodings'] = orig_dataset['query'].progress_apply(lambda x: model.encode(x))
orig_dataset.to_csv('data/orig_embeddings.csv')

drift_dataset['ft_encodings'] = drift_dataset['query'].progress_apply(lambda x: model.encode(x))
drift_dataset.to_csv('data/drift_embeddings.csv')

# orig_dataset = pd.read_csv('data/orig_embeddings.csv')
# drift_dataset = pd.read_csv('data/drift_embeddings.csv')

embeddings1 = orig_dataset['ft_encodings']
embeddings1 = np.stack(embeddings1.values)
embeddings2 = drift_dataset['ft_encodings']
embeddings2 = np.stack(embeddings2.values)


concatenated_embeddings = np.concatenate((embeddings1, embeddings2), axis=0)

# Apply t-SNE to reduce the dimensionality
tsne = TSNE(n_components=2, random_state=42)
embeddings_tsne = tsne.fit_transform(concatenated_embeddings)

# Separate the t-SNE results for the two sets of embeddings
embeddings1_tsne = embeddings_tsne[:len(embeddings1)]
embeddings2_tsne = embeddings_tsne[len(embeddings1):]

# Visualize the t-SNE results
plt.figure(figsize=(10, 6))
plt.scatter(embeddings1_tsne[:, 0], embeddings1_tsne[:, 1], c='r', label='Original data embeddings')
plt.scatter(embeddings2_tsne[:, 0], embeddings2_tsne[:, 1], c='b', label='Data drift embeddings')
plt.title('t-SNE Visualization of Embeddings')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()
plt.grid(True)
plt.show()