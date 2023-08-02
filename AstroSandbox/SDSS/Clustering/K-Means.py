# K-Means Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

#Housekeeping Variables
fname = input('\nEnter training set name | ')
#tname = input('\nEnter target set name | ')

# Importing the dataset
dataset = pd.read_csv('Datasets\%s.csv' %fname)

ug = np.array(dataset['g-r'])
FeH = np.array(dataset['[Fe/H]'])

ug = ug.reshape(len(ug),1)
FeH = FeH.reshape(len(FeH),1)

X = np.concatenate((ug,FeH), axis=1)

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
print('\nApplying Elbow Method. . .\n')
wcss = []
for i in range(1, 5):
    kmeans = KMeans(n_clusters = i, algorithm='full', init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 5), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

k = int(input('Enter cluster count | '))

# Training the K-Means model on the dataset
print('\nTraining K-Means model. . .')
kmeans = KMeans(n_clusters = k, algorithm='full', init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)
tstamp = datetime.date.today()


flabels = np.array(dataset.columns[1:])
flabels = np.append(flabels,['Cluster'])
ra = kmeans.labels_
ra = ra.reshape(len(ra),1)

#Put results into single matrix and output to file
print('\nSaving results. . .')
import os
outdir = 'Results\%s' %fname
if not os.path.exists(outdir):
    os.mkdir(outdir)

prediction = pd.DataFrame(np.concatenate((dataset.iloc[:,1:],ra), axis=1), columns=flabels)
prediction.to_csv('Results\%s\%s_K-M_%s.csv' %(fname,fname,tstamp),  mode='w', index=False)

