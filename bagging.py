"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for classification with Bagging.

Author: Yoav Wald

"""
import numpy as np
import io_util as iou


M = 150


class Bagging(object):

    def __init__(self, L, B, m=M):
        """
        Parameters
        ----------
        L : the class of the base learner
        B : the number of base learners to learn
        m : the number of samples to sample from the original training samples group
        """
        self.m = m
        self.L = L
        self.B = B
        self.h = [None]*B     # list of base learners

    def train(self, X, y):
        """
        Train this classifier over the sample (X,y)
        """
        # train B classifiers over m randomized points (freshly randomized for each b value) out of
        # the original training sample
        for i in range(self.B):
            m_x_samples, m_y_samples = iou.bagging_sampler(M, X, y)
            h = self.L
            h.train(m_x_samples, m_y_samples)
            self.h.append(h)
            print(self.h)

    def predict(self, X):
        """
        Returns a prediction vector for classifying all the points in X
        -------
        y_hat : a prediction vector for X
        """
        # for every point we iterate over - C contains the predictions from all B classifiers
        # that we achieved from training
        C = np.empty(self.B)
        predictions = list()

        for point in X:
            for i, h in enumerate(self.h):
                C[i] = h.predict(point)
            # then, we add the majority vote of this point's C as the classification
            predictions.append(1 if np.count_nonzero(C == 1) > np.count_nonzero(C == -1) else -1)

        predictions = np.array(predictions)

        return predictions

    def error(self, X, y):
        """
        Returns
        -------
        the error of this classifier over the sample (X,y)
        """
        m = len(y)
        bagging_classification = self.predict(X)
        diff = np.reshape([bagging_classification != y], (m,))
        error = (1 / m) * sum(diff)
        return error
