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
import io_util as iou


class Node(object):
    """
    A node in a real-valued decision tree.
    Set all the attributes properly for viewing the tree after training.
    """
    def __init__(self, leaf=True, left=None, right=None, samples=0, feature=None, theta=0.5,
                 label=None):
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


class DecisionTree(object):
    """
    A decision tree for binary classification.
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
        A = np.transpose(X)
        self.root = self.CART(X, y, A, 0)

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
        node : an instance of the class Node (can be either a root of a subtree or a leaf)
        """
        num_of_samples = np.shape(X)[0]

        curr_node = Node()
        curr_node.samples = num_of_samples

        # the exit condition for the recursive process
        if depth == self.max_depth:
            curr_node.label = iou.labeling(y)
            return curr_node

        # assuming we haven't got to a leaf - we should now decide how to split the data in the
        # current node

        label_1, label_2, s, j, min_misclassified = iou.split(X, y, A)

        # now that we've found our optimal split, we should apply it on our current node
        R1_indexes = X[:, j] <= s
        R2_indexes = X[:, j] > s
        R1_samples = X[R1_indexes]
        R2_samples = X[R2_indexes]
        y_R1 = y[R1_indexes]
        y_R2 = y[R2_indexes]

        # in case we got to a "semi leaf" without reaching the maximal depth or that the samples
        # were split perfectly:
        if np.shape(R1_samples)[0] == 0 or np.shape(R2_samples)[0] == 0 or min_misclassified == 0:

            # if the samples best split in a way that the same label was necessary for both
            # subgroups
            if label_1 == label_2:
                curr_node.label = label_2
                return curr_node
            # if the split left R1 empty
            if np.shape(R1_samples)[0] == 0:
                curr_node.label = label_2
                return curr_node
            # if the split left R2 empty
            if np.shape(R2_samples)[0] == 0:
                curr_node.label = label_1
                return curr_node

            # if they split perfectly
            curr_node.right = Node(samples=np.shape(R2_samples)[0], label=label_2)
            curr_node.left = Node(samples=np.shape(R1_samples)[0], label=label_1)

        # if the split didn't lead to a perfect classification and we haven't reached the max depth
        else:
            curr_node.right = self.CART(R2_samples, y_R2, A, depth + 1)
            curr_node.left = self.CART(R1_samples, y_R1, A, depth + 1)

        curr_node.leaf = False
        curr_node.theta = s
        curr_node.feature = j

        return curr_node

    def predict(self, X):
        """
        Returns
        -------
        y_hat : a prediction vector for X
        """

        prediction = []
        m = np.shape(X)[0]
        curr_node = self.root

        for i in range(m):

            while not curr_node.leaf:

                j = curr_node.feature
                s = curr_node.theta
                curr_node = curr_node.left if X[i, j] <= s else curr_node.right

            prediction.append(curr_node.label)
            curr_node = self.root

        return np.array(prediction)

    def error(self, X, y):
        """
        Returns
        -------
        the error of this classifier over the sample (X,y)
        """
        m = len(y)
        dt_classification = self.predict(X)
        diff = np.reshape([dt_classification != y], (m,))
        error = (1 / m) * sum(diff)

        return error
