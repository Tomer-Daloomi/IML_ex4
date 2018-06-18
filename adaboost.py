"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

Author: Noga Zaslavsky
Edited: Yoav Wald, May 2018

"""
import numpy as np
from ex4_tools import decision_boundaries, h_opt


class AdaBoost(object):

    def __init__(self, WL, T):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.WL = WL
        self.T = T
        self.h = [None]*T     # list of base learners
        self.w = np.zeros(T)  # weights

    def train(self, X, y):
        """
        Train this classifier over the sample (X,y)
        """
        m = len(y)
        D = np.array([1/m] * m)

        for i in range(self.T):

            h = self.WL(D, X, y)
            h.train(D, X, y)
            self.h[i] = h
            # create a vector of k elements who's values are 1s (if y(k) == hi(k)) and 0s (
            # otherwise)

            prediction = h.predict(X)
            diff = np.reshape([prediction != y], (m,))
            # multiplication of this vector and D's values
            epsilon = np.sum(list(map(lambda x, y: x*y, D, diff)))
            # update the i'th weight - related to the i'th classification rule h
            self.w[i] = (1 / 2) * np.log((1 / epsilon) - 1)

            # calculate the values needed for the construction of the next distribution array D_t+1
            exp = [np.exp(-1 * self.w[i] * y[j] * prediction[j]) for j in range(m)]
            denominator = np.sum(list(map(lambda x, y: x*y, D, exp)))

            # construction of the next distribution
            for j in range(m):
                D[j] = (D[j] * np.exp(-1 * self.w[i] * y[j] * prediction[j])) / denominator

    def predict(self, X):
        """
        Returns
        -------
        y_hat : a prediction vector for X
        """
        m = X.shape[0]
        t = self.T
        predictions = [h.predict(X) for h in self.h]
        y_hat = []
        for i in range(m):
            y_hat.append(np.sign(sum([self.w[j] * predictions[j][i] for j in range(t)])))

        return np.array(y_hat)

    def error(self, X, y):
        """
        Returns
        -------
        the error of this classifier over the sample (X,y)
        """
        m = len(y)
        ada_classification = self.predict(X)
        diff = np.reshape([ada_classification != y], (m,))
        error = (1 / m) * sum(diff)
        return error
