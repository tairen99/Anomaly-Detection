
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sys import argv
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

allFt = pd.read_csv('features.csv')
allFt['class']=np.int8(allFt['class']=='BIRD')

plt.figure()
# play the pairplot for the data
sns.pairplot(allFt.iloc[0:2000,[1,11,12,13,14,15]])
# then try to use the heatmap plot
plt.figure()
a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.heatmap(allFt.iloc[1:2000,1:10])

X, y = allFt.iloc[:,2:], allFt.iloc[:,1]
clf = ExtraTreesClassifier()
clf = clf.fit(X, y)
impWeight=clf.feature_importances_
index = np.arange(0,256)
newWeightDF = pd.DataFrame([index,impWeight]).transpose()

newWeightDF.head()
print('Top 30 most important features \n')
newWeightDF.sort_values(by=[1],ascending=False)[0][0:30]

plt.show(block=True)
plt.interactive(False)

