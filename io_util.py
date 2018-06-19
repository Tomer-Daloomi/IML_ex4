import numpy as np
import ex4_tools
import adaboost as adb
import decision_tree as dt
import ex4_runme as run
import bagging as bag


def text_lines_to_vector(text_lines):
    """
    strips the lines of text from irrelevant characters and returns an array of single values
    :param text_lines:
    :return:
    """
    value_lines = text_lines.readlines()
    values_vector = [float(value.strip()) for value in value_lines]
    return np.array(values_vector)


def text_lines_to_matrix(text_lines):
    """
    converts the lines of text into a matrix
    :param text_lines:
    :return:
    """
    value_lines = text_lines.readlines()

    matrix = []

    for vector in value_lines:
        features = vector.split()
        features[0] = float(features[0].strip())
        features[1] = float(features[1].strip())
        matrix.append(features)

    matrix = np.array(matrix)

    return matrix


def helper(question):
    """
    execution of question 3,4,bonus & 5, including training the classifiers using adaboost,
    decision trees and bagging, and calculating the error for each of the T,d or b values (both
    training, validation & test)

    :return: training_errors, validation_errors, test_errors, classifiers, x_matrix_training,
    y_vector_training
    """

    with open("./SynData/X_train.txt", 'r') as training_x, open("./SynData/y_train.txt", 'r') as \
            training_y, open("./SynData/X_val.txt", 'r') as validation_x, \
            open("./SynData/y_val.txt", 'r') as validation_y, open("./SynData/y_test.txt",
                                                                   'r') as test_y, \
            open("./SynData/X_test.txt", 'r') as test_x:

        # convert the files into vectors and matrices
        y_vector_training = text_lines_to_vector(training_y)
        x_matrix_training = text_lines_to_matrix(training_x)

        y_vector_validation = text_lines_to_vector(validation_y)
        x_matrix_validation = text_lines_to_matrix(validation_x)

        y_vector_test = text_lines_to_vector(test_y)
        x_matrix_test = text_lines_to_matrix(test_x)

        # preset the ingredients for the training
        training_errors = []
        validation_errors = []
        test_errors = []

        # written form question 3
        if question == 3:

            wl = ex4_tools.DecisionStump
            classifiers = list()

            # adding the classifier for T = 1
            adb_h_1 = adb.AdaBoost(wl, 1)
            adb_h_1.train(x_matrix_training, y_vector_training)
            classifiers.append(adb_h_1)

            t_values = [5, 10, 50, 100, 200]

            # train a classification hypothesis using adaboost for different values of T
            for t in range(5, 205, 5):
                adb_h_t = adb.AdaBoost(wl, t)
                adb_h_t.train(x_matrix_training, y_vector_training)
                training_errors.append(adb_h_t.error(x_matrix_training, y_vector_training))
                validation_errors.append(adb_h_t.error(x_matrix_validation, y_vector_validation))
                test_errors.append(adb_h_t.error(x_matrix_test, y_vector_test))
                # collecting the trained classifiers for part 2 of question 3
                if t in t_values:
                    classifiers.append(adb_h_t)

            return training_errors, validation_errors, test_errors, classifiers,\
                   x_matrix_training, y_vector_training

        # written for question 4
        if question == 4:

            classifiers = list()
            d_values = [3, 6, 8, 10, 12]
            for d in d_values:
                tree_classifier = dt.DecisionTree(d)
                tree_classifier.train(x_matrix_training, y_vector_training)
                training_errors.append(tree_classifier.error(x_matrix_training, y_vector_training))
                validation_errors.append(tree_classifier.error(x_matrix_validation, y_vector_validation))
                test_errors.append(tree_classifier.error(x_matrix_test, y_vector_test))

            return training_errors, validation_errors, test_errors, classifiers, \
                   x_matrix_training, y_vector_training

        # written for question 4
        if question == 'bonus':

            classifiers = list()
            b_values = run.B
            max_depth = 10  # since we discovered at question 4 that 10 was the optimal depth for
            #  this data
            tree_classifier = dt.DecisionTree(max_depth)
            for b in b_values:
                bagging_classifier = bag.Bagging(tree_classifier, b)
                bagging_classifier.train(x_matrix_training, y_vector_training)
                validation_errors.append(bagging_classifier.error(x_matrix_validation, y_vector_validation))
                test_errors.append(bagging_classifier.error(x_matrix_test, y_vector_test))
                classifiers.append(tree_classifier)

            return validation_errors, test_errors


def labeling(y):
    """
    determines the label of the leaf when we arrive max_depth
    :param y: vector of labels in the relevant R_j
    :return: label
    """
    label = 1 if(np.sum(y == (np.full(np.shape(y), 1))) > np.sum(y != (np.full(np.shape(y),
                                                                       1)))) else -1
    return label


def misclassification(y_1, y_2, label_1, label_2):
    """
    the amount of misclassified examples for two y vectors and two
    possible labels after a suggested split

    :param y_1:
    :param y_2:
    :param label_1:
    :param label_2:
    :return: misclassified_samples
    """
    misclassified_samples = np.sum(y_1 != np.full(np.shape(y_1), label_1)) +\
                            np.sum(y_2 != np.full(np.shape(y_2), label_2))
    return misclassified_samples


def split(X, y, A):
    """
    determines the best split of a samples group.

    :param X:
    :param y:
    :return: j - the index of the chosen feature to split upon, s - the threshold by which we
    would split around this feature, label_1, label_2 - the labels into which each classified
    sample should fall.
    """

    label_1, label_2, s, j = None, None, None, None
    num_of_features = np.shape(X)[1]
    min_misclassified = np.inf

    for feature in range(num_of_features):
        for threshold in A[feature, :]:

            r1_indexes = X[:, feature] <= threshold
            r2_indexes = X[:, feature] > threshold
            y_r_1 = y[r1_indexes]
            y_r_2 = y[r2_indexes]

            for l_1 in [1, -1]:
                for l_2 in [1, -1]:

                    misclassified_samples = misclassification(y_r_1, y_r_2, l_1, l_2)

                    if misclassified_samples < min_misclassified:
                        min_misclassified = misclassified_samples
                        label_1 = l_1
                        label_2 = l_2
                        s = threshold
                        j = feature

    return label_1, label_2, s, j, min_misclassified


def bagging_sampler(m, X, y):
    """
    sample randomly m samples out of the original training sample
    :param m:
    :param X:
    :return: a new sample group of length m
    """
    num__of_training_samples = np.shape(X)[0]
    m_random_indexes = np.random.randint(1, num__of_training_samples, m)
    m_x_samples = X[m_random_indexes]
    m_y_samples = y[m_random_indexes]

    return np.array(m_x_samples), m_y_samples



