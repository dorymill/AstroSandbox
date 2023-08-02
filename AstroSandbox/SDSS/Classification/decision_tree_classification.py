#Decision Tree Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import datetime

# Housekeeping Variables
fname = input('\nEnter training set name | ')
tname = input('\nEnter target set name | ')

# Importing the dataset
trainingset = pd.read_csv('Datasets\%s.csv' %fname)
dataset = pd.read_csv('Datasets\%s.csv' %tname)
X = trainingset.iloc[:, :-1].values
y = trainingset.iloc[:, -1].values
X_new = dataset.iloc[:, :].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
print("\nSplitting Test and Training Sets. . .\n")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.50, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
print("Scaling Features. . .\n")
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_new = sc.transform(X_new)

# Train the DecisionTree Classification Model
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
print("\nTraining the model. . .\n")
classifier.fit(X_train, y_train)

# Predicting the Test set results
print("Prediciting test results. . .\n")
y_pred = classifier.predict(X_test)

# Predict the New Observation Results
print("Predicting new observation results. . .\n")
y_new = classifier.predict(X_new)
tstamp = datetime.date.today()

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix\n\n', cm)
print("\nAccuracy score | %.6f" %(accuracy_score(y_test, y_pred)))

# Inverse transform features for plotting
X_new = sc.inverse_transform(X_new)

# Plot the results || Note the 6 and the 2 are the columns of the plotted variables
plt.scatter(X_new[y_new == 1, 6], X_new[y_new == 1, 2], s = 10, c = 'red', label = 'MS')
plt.scatter(X_new[y_new == 2, 6], X_new[y_new == 2, 2], s = 10, c = 'orange', label = 'SGB')
plt.scatter(X_new[y_new == 3, 6], X_new[y_new == 3, 2], s = 10, c = 'gold', label = 'RGB')
plt.scatter(X_new[y_new == 4, 6], X_new[y_new == 4, 2], s = 10, c = 'green', label = 'HZB')
plt.xlim(-0.5,1)
plt.ylim(13,21)
plt.gca().invert_yaxis()
plt.ylabel('r')
plt.xlabel('g-r')
plt.title("Result CMD")
plt.legend()
plt.show()

# Put results into single matrix and output to file
import os
outdir = 'Results\%s' %tname
if not os.path.exists(outdir):
    os.mkdir(outdir)

prediction = pd.DataFrame(np.hstack((X_new,y_new.reshape(len(y_new),1))), columns=trainingset.columns)
prediction.to_csv('Results\%s\%s_DT_%s.csv' %(tname,tname,tstamp),  mode='w', index=False)

#training = pd.DataFrame(np.hstack((X_new,y_pred.reshape(len(y_pred),1))), columns=trainingset.columns)
#training.to_csv('Results\TestResults_K-NN.csv', mode='w',index=False)