import numpy as np
from math import log2
from uuid import uuid4
from random import sample


MAX_DEPTH = 3


def log(x):
    return log2(x) if x != 0 else 0

def compute_entropy_class(class_list):
    # compute entropy of input type [class_id] i.e. [0,1,2,3,3,2,1,0,0,0] -> flaot
    class_count = {i : class_list.count(i) for i in set(class_list)}
    return -sum([ log(count_/len(class_list))*count_/len(class_list) for count_ in class_count.values()])



def create_split_datasets_from_X(data):
    """
    data : zip(X,Y) shape [n_data, n_feature +1]

    From data create all possible split based on conditions : "<=" and ">" and keep non-trivial results

    """
    res = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]-1):

            element = data[i][j] # Choose pivot-element to split the remaining of data
            mask_under = (data[:, j] <= element)
            mask_over = (data[:, j] > element) 
            part_under_element = data[mask_under,:]    
            part_over_element = data[mask_over,:]  
            if part_over_element.size!=0 and part_over_element.size != 0:
                res.append([part_under_element,part_over_element,element,i,j])
    return res

def choose_best_split(splits):
    """
    splits : output from fn "create_split_datasets_from_X" type : [split1_under_element,split2_over_element,element,feature_index]

    Compute information of splits, select the one with the least (maximize delta Information)

    print(np.array(splits)[:,-2:])
    print([custom_sort_on_splits(s) for s in splits])
    print(np.array(sorted_splits)[:,-2:])

    """
    sorted_splits = sorted(splits,key= custom_sort_on_splits)
    best_split = sorted_splits[0]
    return best_split


def custom_sort_on_splits(split_element):

    class_list_under = split_element[0][:,-1]
    class_list_over = split_element[1][:,-1]
    entropy_under = compute_entropy_class(class_list_under.tolist())
    entropy_over = compute_entropy_class(class_list_over.tolist())
    weighted_entropy = entropy_under * class_list_under.shape[0] + entropy_over * class_list_over.shape[0]
    weighted_entropy /= class_list_under.shape[0] +  class_list_over.shape[0]
    return weighted_entropy


class Node:
    
    def __init__(self,tree,data,id,id_parent,depth):
        """
        data : input data like zip(X,Y) shape [n_data', n_feature +1] ,   n_data'<n_data
        """
        self.tree = tree
        self.data = data
        self.id = id
        self.id_parent = id_parent
        self.depth = depth
        self.left_child = None
        self.right_child = None
        self.element = None
        self.index_feature = None
        self.leaf = False
        self.short_repr = ""
        splits = create_split_datasets_from_X(self.data)
        end_condition = data[:,-1].tolist() == [data[:,-1][0]] * data[:,-1].size #If the data passed to the nodes only represents one class

        if len(splits)!=0 and self.depth <= MAX_DEPTH and not end_condition:
            
            best_split = choose_best_split(splits)
            self.element = best_split[-3]
            self.index_feature = best_split[-1]
            self.left_child = Node(self.tree,best_split[0],uuid4(),self.id,self.depth +1)
            self.right_child = Node(self.tree,best_split[1],uuid4(),self.id,self.depth +1)
            self.short_repr = f"x_{self.index_feature} < {self.element}"

        else:

            self.leaf = True
            class_list = data[:,-1].tolist()
            class_distrib = {class_ : class_list.count(class_) for class_ in class_list}
            self.class_ = sample([x for x in class_distrib.keys() if class_distrib[x] == max(class_distrib.values())],1)[0]
            self.short_repr = f"{self.class_}"

        self.tree.nodes.append(self)
        self.tree.nodes_dict[self.id]=self

    def __repr__(self):
        if self.leaf:
            return f"Leaf ({self.id}) of class : {self.class_} // parent id : {self.id_parent}"
        else:
            return f"Decision node ({self.id}) : X_{self.index_feature}<={self.element} // parent id : {self.id_parent}"