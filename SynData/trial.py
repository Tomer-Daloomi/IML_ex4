"""
===================================================
     Introduction to Machine Learning (67577)
===================================================
Skeleton for the decision tree classifier with real-values features.
Training algorithm: CART
Author: Noga Zaslavsky
Edited: Yoav Wald, May 2018
"""
import numpy as np


class Node(object):
    """ A node in a real-valued decision tree.
        Set all the attributes properly for viewing the tree after training.
    """
    def __init__(self, leaf=True, left=None, right=None, samples=0, feature=None, theta=0.5, misclassification=0, label=None):
        """
        Parameters
        ----------
        leaf : True if the node is a leaf, False otherwise
        left : left child
        right : right child
        samples : number of training samples that got to this node
        feature : a coordinate j in [d], where d is the dimension of x (only for internal nodes)
        theta : threshold over self.feature (only for internal nodes)
        label : the label of the node, if it is a leaf
        """
        self.leaf = leaf
        self.left = left
        self.right = right
        self.samples = samples
        self.feature = feature
        self.theta = theta
        self.label = label
        self.misclassification = misclassification


class DecisionTree(object):
    """ A decision tree for binary classification.
        max_depth - the maximum depth allowed for a node in this tree.
        Training method: CART
    """

    def __init__(self, max_depth):
        self.root = None
        self.max_depth = max_depth

    def train(self, X, y):
        """
        Train this classifier over the sample (X,y)
        """
        self.root = self.CART(X, y, X.T, 0)

    def CART(self, X, y, A, depth):
        """
        Grow a decision tree with the CART method ()
        Parameters
        ----------
        X, y : sample
        A : array of d*m real features, A[j,:] row corresponds to thresholds over x_j
        depth : current depth of the tree
        Returns
        -------
        node m : an instance of the class Node (can be either a root of a subtree or a leaf)
        """
        node = Node()
        node.samples = X.shape[0]
        if depth == self.max_depth:
            node.label = 1 if (np.sum(y != np.full(y.shape, 1)) <= np.sum(y != np.full(y.shape, -1))) else (-1)
            return node
        j, s, l_1, l_2 = None, None, None, None
        minimum = np.inf
        for feature in range(X.shape[1]):
            for th in A[feature, :]:
                ind_r_1 = X[:, feature] <= th
                ind_r_2 = X[:, feature] > th
                y_1 = y[ind_r_1]
                y_2 = y[ind_r_2]
                for l_i in [-1, 1]:
                    for l_j in [-1, 1]:
                        mistake_count = np.sum(y_1 != np.full(y_1.shape, l_i)) + np.sum(y_2 != np.full(y_2.shape, l_j))
                        if mistake_count < minimum:
                            minimum = mistake_count
                            l_1 = l_i
                            l_2 = l_j
                            s = th
                            j = feature
        ind_r_1 = X[:, j] <= s
        ind_r_2 = X[:, j] > s
        R_1, R_2 = X[ind_r_1], X[ind_r_2]
        y_1, y_2 = y[ind_r_1], y[ind_r_2]
        if minimum == 0 or R_1.shape[0] == 0 or R_2.shape[0] == 0:
            if l_1 == l_2:
                node.label = l_1
                return node
            if R_1.shape[0] == 0:
                node.label = l_2
                return node
            if R_2.shape[0] == 0:
                node.label = l_1
                return node
            node.left = Node(samples=R_1.shape[0], label=l_1)
            node.right = Node(samples=R_2.shape[0], label=l_2)
        else:
            node.left = self.CART(R_1, y_1, A, depth + 1)
            node.right = self.CART(R_2, y_2, A, depth + 1)
        node.leaf = False
        node.feature = j
        node.theta = s
        return node

    def predict(self, X):
        """
        Returns
        -------
        y_hat : a prediction vector for X
        """
        y = np.empty(X.shape[0])
        node = self.root
        for m in range(X.shape[0]):
            while not node.leaf:
                node = node.left if X[m, node.feature] <= node.theta else node.right
            y[m] = node.label
            node = self.root
        return y

    def error(self, X, y):
        """
        Returns
        -------
        the error of this classifier over the sample (X,y)
        """
        return np.sum(self.predict(X) != y) / X.shape[0]