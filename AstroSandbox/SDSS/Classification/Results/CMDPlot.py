import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

tname = input('\nEnter result filepath and name | ')
X_new = pd.read_csv('%s.csv' %tname)

#Stellar Groups
MS = X_new.iloc[X_new[:,-1]==1, ['g-r','r']]
SGB = X_new.iloc[X_new[:,-1]==2, ['g-r','r']]
RGB = X_new.iloc[X_new[:,-1]==3, ['g-r','r']]
HZB = X_new.iloc[X_new[:,-1]==4, ['g-r','r']]


#Plot the results || Note the 6 and the 2 are the columns of the plotted variables
plt.scatter(MS[:,0], MS[:,1], s = 10, c = 'red', label = 'MS')
plt.scatter(SGB[:,0], SGB[:,1], s = 10, c = 'orange', label = 'SGB')
plt.scatter(RGB[:,0], RGB[:,1], s = 10, c = 'gold', label = 'RGB')
plt.scatter(HZB[:,0], HZB[:,1], s = 10, c = 'green', label = 'HZB')
plt.xlim(-0.5,1)
plt.ylim(13,21)
plt.gca().invert_yaxis()
plt.ylabel('r')
plt.xlabel('g-r')
plt.title("Result CMD")
plt.legend()
plt.show()