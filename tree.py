from node import Node
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pos_management_tree_vis import hierarchy_pos
from uuid import uuid4


class Tree:
    
    def __init__(self):
        
        self.input_data = None
        self.nodes= []
        self.nodes_dict = {}
        self.nodes.reverse()

    def fit(self,X,Y):
        if X.size == 0 or Y.size == 0:
            raise Exception("Data provided is empty")
        data = np.insert(X, X.shape[1], Y, axis=1)
        self.nodes.append(Node(self,data,uuid4(),id_parent= None, depth = 0))
        self.nodes.reverse()

        self.input_data = data

    def viz(self):
        if type(self.input_data) is not np.ndarray:
            raise Exception("Tree hasen't been fitted yet!")
        G = nx.Graph()        
        all_node_from_tree = [(n.id,{"text":n.short_repr}) for n in self.nodes]
        G.add_nodes_from(all_node_from_tree)
        edges = []

        for node in self.nodes:
            if node.right_child != None:
                edges.append((node.id,node.right_child.id))
            if node.right_child != None:
                edges.append((node.id,node.left_child.id))

        G.add_edges_from(edges)
        labeldict=nx.get_node_attributes(G,'text')
        pos= hierarchy_pos(G,self.nodes[0].id)
        nx.draw(G,pos, labels=labeldict, with_labels = True, alpha =0.5, node_color="skyblue", linewidths=40)
        plt.show()


    def predict(self,X):
        prediction = []
        if type(self.input_data) is not np.ndarray:
            raise Exception("Tree hasen't been fitted yet!")
        if X.size == 0:
            raise Exception("Data provided is empty")
        for x in X:
            current_node = self.nodes[0]
            while not current_node.leaf:
                element = current_node.element
                feature_index = current_node.index_feature
                if x[feature_index] <= element:
                    current_node = current_node.left_child
                else:
                    current_node = current_node.right_child
            prediction.append(current_node.class_)
        return prediction

    def test_score(self,X,Y):
        if type(self.input_data) is not np.ndarray:
            raise Exception("Tree hasen't been fitted yet!")
        if X.size == 0 or Y.size == 0:
            raise Exception("Data provided is empty")
        prediction = self.predict(X)
        correct_prediction = [prediction[i]==Y[i] for i in range(Y.shape[0])]
        return correct_prediction.count(True)/Y.shape[0]