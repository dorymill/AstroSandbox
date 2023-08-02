# Hierarchical Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Housekeeping Variables
fname = input('\nEnter training set name | ')
#tname = input('\nEnter target set name | ')

# Importing the dataset
dataset = pd.read_csv('Datasets\%s.csv' %fname)

ug = np.array(dataset['g-r'])
FeH = np.array(dataset['[Fe/H]'])

ug = ug.reshape(len(ug),1)
FeH = FeH.reshape(len(FeH),1)

X = np.concatenate((ug[1:1000],FeH[1:1000]), axis=1)

# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Stars')
plt.ylabel('Euclidean distances')
plt.show()

# Training the Hierarchical Clustering model on the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')

plt.title('Clusters of customers')
plt.xlabel('[Fe/H]]')
plt.ylabel('g-r')
plt.legend()
plt.show()