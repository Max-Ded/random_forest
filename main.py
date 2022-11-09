import numpy as np
from random import sample
from sklearn import datasets
from random_forest import Random_Forest

from sklearn.ensemble import RandomForestClassifier

VIZ = False

test_ratio = 0.1

data = datasets.make_classification(n_samples = 1500,n_features=6,n_classes=3,n_informative=3)

X = data[0] 
Y = data[1]

train_index = sample(range(X.shape[0]),int(X.shape[0]*(1-test_ratio)))
test_index = [x for x in range(X.shape[0]) if x not in train_index]

X_train = X[train_index]
Y_train = Y[train_index]

X_test = X[test_index]
Y_test = Y[test_index]


random_forest_model = Random_Forest(n_trees=100,feature_sampling_factor=0.5,sampling_factor=0.5)
random_forest_model.fit(X_train,Y_train)


sklearn_model = RandomForestClassifier(n_estimators=100,criterion="entropy")
sklearn_model.fit(X,Y)

if VIZ : random_forest_model.viz()

print("Accuracy is :" ,random_forest_model.test_score(X_test,Y_test) ,"custom model")
print("Accuracy is :" ,sklearn_model.score(X_test,Y_test) ,"sklearn model")


