# ANN Regression (ANN-R)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

fname = input('\nEnter training set name | ')
#tname = input('\nEnter target set name | ')

# Importing and preprocessing the dataset
trainingset = pd.read_csv('Datasets\%s.csv' %fname)
#dataset = pd.read_csv('Datasets\%s.csv' %tname)
X = trainingset.iloc[:, 1:-1].values
y = trainingset.iloc[:, -1].values

y = y.reshape(len(y),1)

from sklearn.model_selection import train_test_split
print("\nSplitting Test and Training Sets. . .\n")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
print('Scaling features. . .\n')
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)

# Initializing the ANN
ann = tf.keras.models.Sequential()

# Hyper-Parameters
N_h = 1 
n_hn = 10
hidden_afn = 'relu'
outp_afn = 'linear'
batch = 32
epochs = 500

# Input layer
ann.add(tf.keras.layers.Dense(units=13, activation='relu'))

# Hidden Layers
for i in range(0,N_h-1):
    ann.add(tf.keras.layers.Dense(units=n_hn, activation=hidden_afn))

# Output Layers
ann.add(tf.keras.layers.Dense(units='1', activation=outp_afn))

# Compiling and training the ANN
#optimizer = tf.keras.optimizers.Adam(lr=0.05)
ann.compile(optimizer = 'adam', loss = 'mean_squared_error')

ann.fit(X_train, y_train, batch_size = batch, epochs = epochs)

# Predicting Test Set results
print('Prediciting Results. . .\n')
y_pred = sc_y.inverse_transform(ann.predict(sc_X.transform(X_test)))
y_pred = y_pred.reshape(len(y_pred),1)

# Determine Accuracy
from sklearn.metrics import mean_squared_error
score = mean_squared_error(y_test,y_pred)**(1/2)
print("RMS Error | %.4f" %score)

# Write Results to file
res = pd.DataFrame(columns=['Hidden Layers','Neurons/Layer','Activation','Batch Size','Epochs','Results'], data=[[N_h,n_hn,hidden_afn,batch,epochs,score]])
res.to_csv('Results\Runs.csv', mode='a', index=False, header=False)