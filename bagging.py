"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for classification with Bagging.

Author: Yoav Wald

"""
import numpy as np

class Bagging(object):

    def __init__(self, L, B):
        """
        Parameters
        ----------
        L : the class of the base learner
        T : the number of base learners to learn
        """
        self.L = L
        self.B = B
        self.h = [None]*B     # list of base learners

    def train(self, X, y):
        """
        Train this classifier over the sample (X,y)
        """

    def predict(self, X):
        """
        Returns
        -------
        y_hat : a prediction vector for X
        """


    def error(self, X, y):
        """
        Returns
        -------
        the error of this classifier over the sample (X,y)
        """
