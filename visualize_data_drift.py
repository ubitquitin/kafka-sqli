import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sklearn.manifold import TSNE
import seaborn as sns
from scipy.stats import entropy, kstest


import json

tqdm.pandas()


# Population Stability index for data drift detection.
# Tests the 
# From https://sarakmair.medium.com/data-drift-in-machine-learning-d5c86979f78d
def calculate_psi(expected, observed, bins=10):
    # Bin the data to get the distribution in each bin
    expected_counts, bin_edges = np.histogram(expected, bins=bins, range=(min(expected), max(expected)))
    observed_counts, _ = np.histogram(observed, bins=bin_edges, range=(min(expected), max(expected)))
    
    # Calculate the proportions of observations in each bin
    expected_proportions = expected_counts / sum(expected_counts)
    observed_proportions = observed_counts / sum(observed_counts)
    
    # Replace 0s with a small number to avoid division by zero in PSI calculation
    expected_proportions = np.where(expected_proportions == 0, 0.0001, expected_proportions)
    observed_proportions = np.where(observed_proportions == 0, 0.0001, observed_proportions)
    
    # Calculate the PSI
    psi_values = (expected_proportions - observed_proportions) * np.log(expected_proportions / observed_proportions)
    psi = sum(psi_values)
    
    return psi



orig_dataset = pd.read_csv('data/shuffled_sql_data.csv', names=['query', 'label'], header=None)
drift_dataset = pd.read_csv('data/shuffled_data_drifted_sql_data.csv', names=['query', 'label'], header=None)
drift_dataset = drift_dataset[drift_dataset['query'].str.len() >= 1]
orig_dataset = orig_dataset.iloc[:len(drift_dataset)]

model = SentenceTransformer('C:/Users/ROHAN/time-series-kafka-demo/model/')

orig_dataset['ft_encodings'] = orig_dataset['query'].progress_apply(lambda x: model.encode(x))
orig_dataset.to_csv('data/orig_embeddings.csv')

drift_dataset['ft_encodings'] = drift_dataset['query'].progress_apply(lambda x: model.encode(x))
drift_dataset.to_csv('data/drift_embeddings.csv')

orig_dataset = pd.read_csv('data/orig_embeddings.csv')
drift_dataset = pd.read_csv('data/drift_embeddings.csv')

embeddings1 = orig_dataset['ft_encodings']
embeddings1 = np.stack(embeddings1.values)
embeddings2 = drift_dataset['ft_encodings']
embeddings2 = np.stack(embeddings2.values)


concatenated_embeddings = np.concatenate((embeddings1, embeddings2), axis=0)

#Apply t-SNE to reduce the dimensionality
tsne = TSNE(n_components=2, random_state=42)
embeddings_tsne = tsne.fit_transform(concatenated_embeddings)

#Separate the t-SNE results for the two sets of embeddings
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


#Now, create 1-d representations for statistical tests
tsne = TSNE(n_components=1, random_state=42)
embeddings_tsne = tsne.fit_transform(concatenated_embeddings)

#Separate the t-SNE results for the two sets of embeddings
embeddings1_tsne = embeddings_tsne[:len(embeddings1)]
embeddings2_tsne = embeddings_tsne[len(embeddings1):]

embeddings1_tsne = np.load('embeddings_1d_data.npy')
embeddings2_tsne = np.load('embeddings_1d_drift.npy')

embeddings1_tsne =embeddings1_tsne.squeeze()
embeddings2_tsne = embeddings2_tsne.squeeze()

embeddings1_tsne = np.asarray(embeddings1_tsne).reshape(1, -1)[0]
embeddings2_tsne = np.asarray(embeddings2_tsne).reshape(1, -1)[0]

#np.save('embeddings_1d_data', embeddings1_tsne)
#np.save('embeddings_1d_drift', embeddings2_tsne)

#normalize
embeddings1_tsne = (embeddings1_tsne - min(embeddings1_tsne))/(max(embeddings1_tsne) - min(embeddings1_tsne))
embeddings2_tsne = (embeddings2_tsne - min(embeddings2_tsne))/(max(embeddings2_tsne) - min(embeddings2_tsne))

fig, ax = plt.subplots(figsize=(12, 6))
sns.kdeplot(data=embeddings1_tsne,
            color='crimson', label='original dataset', fill=True, ax=ax)
sns.kdeplot(data=embeddings2_tsne,
            color='limegreen', label='drifted dataset', fill=True, ax=ax)
plt.title('Distributions of original and drifted dataset embeddings (1d-tSNE)')
ax.legend()
plt.tight_layout()
plt.show()

# Replace 0s with a small number to avoid division by zero in KL/PSI calculation
embeddings1_tsne = np.where(embeddings1_tsne == 0, 0.0001, embeddings1_tsne)
embeddings2_tsne = np.where(embeddings2_tsne == 0, 0.0001, embeddings2_tsne)

print(embeddings1_tsne[0:10])
print(embeddings2_tsne[0:10])


# Calculate KL divergence
kl_divergence = entropy(embeddings1_tsne, embeddings2_tsne)

print(f"KL divergence between data1 and data2: {kl_divergence}")

# Calculate PSI between data1 and data2 using bins
psi = calculate_psi(embeddings1_tsne, embeddings2_tsne)

print(f"Population Stability Index between data 1 and data2: {psi}")

KS_test_stat, pval = kstest(embeddings1_tsne, embeddings2_tsne)

if pval < 0.05:
    print(f"The p-value: {pval}, is statistically significant and we can reject the null hypothesis that the two distributions are identical.")

else:
    print(f"The p-value: {pval}, is NOT statistically significant so we cannot reject the null hypothesis.")