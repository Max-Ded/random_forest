import numpy as np
from tree import Tree
from math import sqrt,ceil
from random import sample
from collections import defaultdict

class Random_Forest:

    def __init__(self,n_trees,feature_sampling_factor = None , sampling_factor = 0.8):
        
        self.n_trees = n_trees
        self.trees = []  # format [Tree::obj,[feature_index]]
        self.feature_sampling_factor = feature_sampling_factor
        self.sampling_factor = sampling_factor

        self.is_fitted = False

    def fit(self,X,Y):
        """
        X,Y : numpy arrays of data (n_data x n_feature  && n_data x 1 )

        2 process : 
            bagging -> sample part of X,Y based on a factor (e.g. take 80pct of data), replace missing data to fit on n_data rows
            feature bagging -> sample features based on a factor, need to store feature_index for the predict phase later
        """

        self.feature_sampling_factor = self.feature_sampling_factor if self.feature_sampling_factor else ceil(sqrt(X.shape[0])) 
        for _ in range(self.n_trees):
            row_sampling_index = sample(range(X.shape[0]), int(X.shape[0] * self.sampling_factor))
            doubling_index = 0
            feature_sampling_index = sample(range(X.shape[1]),int(self.feature_sampling_factor * X.shape[1]))
            while len(row_sampling_index)<X.shape[0]:
                row_sampling_index.append(doubling_index)
                doubling_index +=1
            X_b = X[row_sampling_index][:,feature_sampling_index]
            Y_b = Y[row_sampling_index]
            tree = Tree()
            tree.fit(X_b,Y_b)
            self.trees.append([tree,feature_sampling_index])
        self.is_fitted = True

    def predict_one_row(self,x):
        predictions = defaultdict(int)
        for tree,feature_index in self.trees:
            class_ = tree.predict(x[feature_index])
            predictions[class_]+=1
        return sample([x for x in predictions.keys() if predictions[x] == max(predictions.values())],1)[0]

    def predict(self,X):
        """
        X : np array of n_data x n_feature

        Looping on all the trees, they each make a prediction for the input data -> [n_data,1]
        Aggregate in a matrix T, such that T[i,j] = prediction of the i-th tree for the j-th input data (X[j])
        Transposing the matrix is easier to work with

        """
        if not self.is_fitted:
            raise Exception("Model hasen't been fitted yet!")
        if X.size == 0:
            raise Exception("Data provided is empty")
        prediction_matrix = []
        prediction_results = []
        for tree,feature_index in self.trees:
            classes_prediction_one_tree = tree.predict(X[:,feature_index])
            prediction_matrix.append(classes_prediction_one_tree)

        prediction_matrix = np.array(prediction_matrix).T

        for x_class_prediction in prediction_matrix:
            dict_count_class = { class_ : x_class_prediction.tolist().count(class_) for class_ in x_class_prediction}
            prediction_results.append(sample([x for x in dict_count_class.keys() if dict_count_class[x] == max(dict_count_class.values())],1)[0])

        return prediction_results

    def test_score(self,X,Y):
        if not self.is_fitted:
            raise Exception("Model hasen't been fitted yet!")
        if X.size == 0 or Y.size == 0:
            raise Exception("Data provided is empty")
        prediction = self.predict(X)
        correct_prediction = [prediction[i]==Y[i] for i in range(Y.shape[0])]
        return correct_prediction.count(True)/Y.shape[0]

    def viz(self):

        random_tree,_ = sample(self.trees,1)[0]
        random_tree.viz()