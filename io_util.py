import numpy as np
import ex4_tools
import adaboost as adb
import decision_tree as dt
import ex4_runme as run
import bagging as bag

VAULT = 1536
T_VALUES = [5, 50, 100, 200, 500, 1000]
D_VALUES = [5, 8, 10, 12, 15, 18]


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
        spam_matrix = text_lines_to_matrix(training_x)

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
            adb_h_1.train(spam_matrix, y_vector_training)
            classifiers.append(adb_h_1)

            t_values = [5, 10, 50, 100, 200]

            # train a classification hypothesis using adaboost for different values of T
            for t in range(run.T):
                adb_h_t = adb.AdaBoost(wl, t)
                adb_h_t.train(spam_matrix, y_vector_training)
                training_errors.append(adb_h_t.error(spam_matrix, y_vector_training))
                validation_errors.append(adb_h_t.error(x_matrix_validation, y_vector_validation))
                test_errors.append(adb_h_t.error(x_matrix_test, y_vector_test))
                # collecting the trained classifiers for part 2 of question 3
                if t in t_values:
                    classifiers.append(adb_h_t)

            return training_errors, validation_errors, test_errors, classifiers,\
                   spam_matrix, y_vector_training

        # written for question 4
        if question == 4:

            classifiers = list()
            D_VALUES = [3, 6, 8, 10, 12]
            for d in D_VALUES:
                tree_classifier = dt.DecisionTree(d)
                tree_classifier.train(spam_matrix, y_vector_training)
                training_errors.append(tree_classifier.error(spam_matrix, y_vector_training))
                validation_errors.append(tree_classifier.error(x_matrix_validation, y_vector_validation))
                test_errors.append(tree_classifier.error(x_matrix_test, y_vector_test))

            return training_errors, validation_errors, test_errors, classifiers, \
                   spam_matrix, y_vector_training

        # written for question 4 - bonus
        if question == 'bonus':

            b_values = run.B
            max_depth = 10  # since we discovered at question 4 that 10 was the optimal depth for
            #  this data
            tree_classifier = dt.DecisionTree(max_depth)
            for b in b_values:
                print(b)
                bagging_classifier = bag.Bagging(tree_classifier, b)
                bagging_classifier.train(spam_matrix, y_vector_training)
                validation_errors.append(bagging_classifier.error(x_matrix_validation, y_vector_validation))
                test_errors.append(bagging_classifier.error(x_matrix_test, y_vector_test))

            return validation_errors, test_errors

        # written for question 5

    if question == '5':

        with open("./SpamData/spam.data") as spam:

            # convert the file into a matrix and a vector, and create a list of partitioned parts
            # for the cross validation

            spam_matrix = text_lines_to_matrix(spam)

            x_spam_matrix, y_spam_vector, x_vault_matrix, y_vault_vector = spam_modif(spam_matrix)

            subgroup_length = np.shape(x_spam_matrix)[0] // 5

            partitioned_x, partitioned_y = cross_val_partition(x_spam_matrix, y_spam_vector,
                                                               subgroup_length)

            # preparations for the adaboost training
            wl = ex4_tools.DecisionStump
            validation_errors_adb = []

            # train a classification hypothesis using adaboost for different values of T,
            # and through each of the subsets as a validation group (cross validation)
            for j, t in enumerate(T_VALUES):
                for i in range(5):

                    segment_len = len(partitioned_x[0])

                    x, y = sub_matrix(x_spam_matrix, y_spam_vector, segment_len, i)

                    adb_h_t = adb.AdaBoost(wl, t)
                    adb_h_t.train(x, y)
                    if i == 0:
                        validation_errors_adb.append([adb_h_t.error(partitioned_x[i],
                                                                      partitioned_y[i])])
                    else:
                        validation_errors_adb[j].append(adb_h_t.error(partitioned_x[i],
                                                                      partitioned_y[i]))
                print(validation_errors_adb)

            # preparations for the decision tree training
            validation_errors_dt = []

            # train a classification hypothesis using decision trees for different values of depth,
            # and through each of the subsets as a validation group (cross validation)
            for j, d in enumerate(D_VALUES):
                for i in range(5):
                    x, y = sub_matrix(x_spam_matrix, partitioned_x, y_spam_vector, partitioned_y)

                    tree_classifier = dt.DecisionTree(d)
                    tree_classifier.train(x, y)
                    if i == 0:
                        validation_errors_dt.append([tree_classifier.error(partitioned_x[i],
                                                                           partitioned_y[i])])
                    else:
                        validation_errors_dt[j].append(tree_classifier.error(partitioned_x[i],
                                                                             partitioned_y[i]))

            return validation_errors_adb, validation_errors_dt


def dt_labeling(y):
    """
    determines the label of the leaf when we arrive max_depth
    :param y: vector of labels in the relevant R_j
    :return: label
    """
    label = 1 if(np.sum(y == (np.full(np.shape(y), 1))) > np.sum(y != (np.full(np.shape(y),
                                                                       1)))) else -1
    return label


def dt_misclassification(y_1, y_2, label_1, label_2):
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

                    misclassified_samples = dt_misclassification(y_r_1, y_r_2, l_1, l_2)

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


def spam_modif(spam_matrix):

    last_column_number = np.shape(spam_matrix)[1]
    y_spam_vector = spam_matrix[:, last_column_number - 1]
    x_spam_matrix = np.delete(spam_matrix, last_column_number - 1, 1)

    vault_random_indexes = np.random.choice(len(x_spam_matrix), VAULT, replace=False)

    x_vault_matrix = x_spam_matrix[vault_random_indexes - 1]
    x_spam_matrix = np.delete(x_spam_matrix, vault_random_indexes - 1, 0)

    y_vault_vector = y_spam_vector[vault_random_indexes - 1]
    y_spam_vector = np.delete(y_spam_vector, vault_random_indexes - 1, 0)

    return x_spam_matrix, y_spam_vector, x_vault_matrix, y_vault_vector


def sub_matrix(original_x, original_y, segment_len, i):
    """
    return the relevant training sample, in relation to the current segment that was chosen as the
    validation sample

    :param original_x:
    :param original_y:
    :param partitioned_x:
    :param partitioned_y:
    :param i:
    :return:
    """

    full_len = len(original_x)

    x_pre = original_x[0:segment_len * i]
    x_post = original_x[segment_len * i:full_len]

    y_pre = original_y[0:segment_len * i]
    y_post = original_y[segment_len * i:full_len]

    x = np.concatenate((x_pre, x_post), 0).astype(float)
    y = np.concatenate((y_pre, y_post), 0).astype(float)

    return x, y


def cross_val_partition(original_x, original_y, subgroup_length):
    """
    create a list that contains the original spam data - split into 5 size equivalent subsets

    :param original_x:
    :param original_y:
    :param subgroup_length:
    :return:
    """
    partitioned_x = np.array([original_x[i:i + subgroup_length] for i in range(0,
                                                                         subgroup_length
                                                                         * 5,
                                                                         subgroup_length)]).astype(float)
    partitioned_y = np.array([original_y[i:i + subgroup_length] for i in range(0,
                                                                         subgroup_length
                                                                         * 5,
                                                                         subgroup_length)]).astype(float)

    return partitioned_x, partitioned_y
