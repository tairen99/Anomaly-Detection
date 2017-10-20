
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report,confusion_matrix

allFt = pd.read_csv('features.csv')
allFt['class']=np.int8(allFt['class']=='BIRD')

X, y = allFt.iloc[:,2:], allFt.iloc[:,1]
clf = ExtraTreesClassifier()
clf = clf.fit(X, y)
impWeight=clf.feature_importances_
index = np.arange(0,256)
model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)

newWeightDF = pd.DataFrame([index,impWeight]).transpose()
print('Top 10 most important features \n')
print(newWeightDF.sort_values(by=[1],ascending=False)[0][0:10])

X_newdf = pd.DataFrame(X_new)
trainLen = np.int(np.floor(X_newdf.shape[0]*0.7))
xTrain = X_newdf.iloc[0:trainLen,:]
yTrain = y[0:trainLen]
xTest = X_newdf.iloc[trainLen:,:]
yTest = y[trainLen:]

rng = np.random.RandomState(100)


# fit the model
clf = IsolationForest(max_samples=200, random_state=rng)
clf.fit(xTrain,yTrain)
y_pred_train = clf.predict(xTest)
yPredTrain = np.int8(y_pred_train < 0)
print(classification_report(yTest, yPredTrain))

# print the precision/recall table
print(confusion_matrix(yTest, yPredTrain))
