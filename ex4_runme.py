"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Running script for Ex4.

Author:
Date: May, 2018

"""

import numpy as np
import io_util as iou
import ex4_tools
import matplotlib.pyplot as plt

T = [i for i in range(5, 205, 5)]
d = [3, 6, 8, 10, 12]
B = list(range(2, 100))


def Q3(function):  # AdaBoost

    # calculate the training and validation errors, and the classifiers for different values of T
    training_errors, validation_errors, test_errors, classifiers, x_tr, y_tr = iou.helper(3)

    # plot the training and validation errors of classifiers that were trained using adaboost
    # over the T values that are in t
    if function == 'a':
        plt.plot(T, training_errors, 'r', label="training errors")
        plt.plot(T, validation_errors, 'c', label="validation errors")
        plt.xlabel("T - number of distribution adjustment iterations")
        plt.ylabel("Error")
        plt.legend(loc=3)
        plt.show()

    # plot the decisions of the learned classifiers over [-1,1]^2 for different T values
    if function == 'b':
        counter = 0
        t_values = [1, 5, 10, 50, 100, 200]

        for classifier in classifiers:
            weights = classifier.w
            ex4_tools.decision_boundaries(classifier, x_tr, y_tr, title_str="T = " + str(
                t_values[counter]), weights=weights * 30)
            counter += 1

    # find the T value that leads to the minimal validation error,
    # and calculate it's testing error
    if function == 'c':
        gap = T[1] - T[0]
        min_t_index = np.argmin(validation_errors)
        min_t = gap * min_t_index  # get the index of the minimal value,
        # then multiply it by the constant gap of the T values (assuming it really is constant)
        test_err_min_t = test_errors[min_t_index]
        print('the minimizing T is: ', min_t, "and it's test error is: ", test_err_min_t)


def Q4(function): # decision trees

    # calculate the training and validation errors, and the classifiers for different values of T
    training_errors, validation_errors, test_errors, classifiers, x_tr, y_tr = iou.helper(4)

    # plot the training and validation errors of classifiers that were trained using decision tree
    # over the depth values that are in d
    if function == 'a':
        plt.plot(d, training_errors, 'r', label="training errors")
        plt.plot(d, validation_errors, 'c', label="validation errors")
        plt.xlabel("d - maximal tree depth")
        plt.ylabel("Error")
        plt.legend(loc=1)
        plt.show()

    # plot the decisions of the learned classifiers over [-1,1]^2 for different depth values
    if function == 'b':
        counter = 0
        d_values = [3, 6, 8, 10, 12]

        for classifier in classifiers:
            ex4_tools.decision_boundaries(classifier, x_tr, y_tr, title_str="depth = " + str(
                d_values[counter]))
            counter += 1

    # find the d value that leads to the minimal validation error,
    # and calculate it's testing error
    if function == 'c':
        min_d_index = np.argmin(validation_errors)
        min_d = d[min_d_index]  # get the index of the minimal value,
        # then multiply it by the constant gap of the T values (assuming it really is constant)
        test_err_min_d = test_errors[min_d_index]
        print('the minimizing depth is: ', min_d, "and it's test error is: ", test_err_min_d)


def bonus():
    # calculate the training and validation errors, and the classifiers for different values of T
    validation_errors, test_errors = iou.helper('bonus')

    # plot the validation errors of classifiers that were trained using bagging
    # over the different b values

    plt.plot(B, validation_errors, 'c')
    plt.xlabel("B - number of 'bags' taking place in the 'bagging'")
    plt.ylabel("Validation Error")
    plt.show()

    # find the B value that leads to the minimal validation error,
    # and calculate it's testing error

    min_b_index = np.argmin(validation_errors)
    min_b = B[min_b_index]  # get the index of the minimal value,
    # then multiply it by the constant gap of the T values (assuming it really is constant)
    test_err_min_b = test_errors[min_b_index]
    print('the minimizing number of trees is: ', min_b, "and it's test error is: ", test_err_min_b)


def Q5(): # spam data

    # calculate the training and validation errors, and the classifiers for different values of T
    validation_errors_adb, validation_errors_dt = iou.helper('5')

    mean_err_adb = list()
    mean_err_dt = list()

    for i in range(6):
        mean_err_adb.append(np.mean(validation_errors_adb[i]))
        mean_err_dt.append(np.mean(validation_errors_dt[i]))

    print('the minimal val_err for adb is: ', min(mean_err_adb), 'and for dt: ', min(mean_err_dt))

    # plot the validation errors of classifiers that were trained using adaboost
    # over the different T values

    plt.plot(iou.T_VALUES, mean_err_adb, 'c')
    plt.xlabel("T - number of distribution adjustment iterations")
    plt.ylabel("Mean Validation Error")
    plt.show()

    plt.plot(iou.D_VALUES, mean_err_dt, 'r')
    plt.xlabel("d - maximal depth allowed while training")
    plt.ylabel("Mean Validation Error")
    plt.show()

    # # find the B value that leads to the minimal validation error,
    # # and calculate it's testing error
    #
    # min_b_index = np.argmin(validation_errors)
    # min_b = B[min_b_index]  # get the index of the minimal value,
    # # then multiply it by the constant gap of the T values (assuming it really is constant)
    # test_err_min_b = test_errors[min_b_index]
    # print('the minimizing number of trees is: ', min_b, "and it's test error is: ", test_err_min_b)

if __name__ == '__main__':
    Q5()
    pass
