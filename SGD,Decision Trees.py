# %% md

# Homework 1


# %%

# Uncomment and run this code if you want to verify your `sklearn` installation.
# If this cell outputs 'array([1])', then it's installed correctly.

# from sklearn import tree
# X = [[0, 0], [1, 1]]
# y = [0, 1]
# clf = tree.DecisionTreeClassifier(criterion='entropy')
# clf = clf.fit(X, y)
# clf.predict([[2, 2]])

# %%

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
import random
from string import ascii_lowercase


# When you turn this function in to Gradescope, it is easiest to copy and paste this cell to a new python file called hw1.py
# and upload that file instead of the full Jupyter Notebook code (which will cause problems for Gradescope)
def compute_features(names):
    """
    Given a list of names of length N, return a numpy matrix of shape (N, 260)
    with the features described in problem 2b of the homework assignment.

    Parameters
    ----------
    names: A list of strings
        The names to featurize, e.g. ["albert einstein", "marie curie"]

    Returns
    -------
    numpy.array:
        A numpy array of shape (N, 260)
    """
    text_names = np.zeros((len(names), 260))
    row = 0
    for name in names:
        
        name.lower()
        firstname, lastname = name.split(' ')
        if len(firstname) >= 5:
            firstname = firstname[0:5]
            count = 0
            alphat = 26
            for num in firstname:
                num = (ord(num) - 97) + (alphat * count)
                text_names[row, num] = 1
                count +=1
        if len(firstname) < 5:
            count = 0
            alphat = 26
            for num in firstname:
                num = (ord(num) - 97) + (alphat * count)
                text_names[row, num] = 1
                count +=1
        if len(lastname) >= 5:
            lastname = lastname[0:5]
            count = 5
            alphat = 26
            for num in lastname:
                num = (ord(num) - 97) + (alphat * count)
                text_names[row, num] = 1
                count +=1
        if len(lastname) < 5:
            count = 5
            alphat = 26
            for num in lastname:
                num = (ord(num) - 97) + (alphat * count)
                text_names[row, num] = 1
                count +=1
        row += 1
    return text_names
    raise NotImplementedError



# %% md


# %%

from sklearn.linear_model import SGDClassifier


def Accuracy(first, second):
    same_element = np.equal(first, second)
    number_same = np.sum(same_element)
    return (number_same / np.shape(first)[0])


def train_and_evaluate_sgd(X_train, y_train, X_test, y_test):
    """
    Trains a SGDClassifier on the training data and computes two accuracy scores, the
    accuracy of the classifier on the training data and the accuracy of the decision
    tree on the testing data.

    Parameters
    ----------
    X_train: np.array
        The training features of shape (N_train, k)
    y_train: np.array
        The training labels of shape (N_train)
    X_test: np.array
        The testing features of shape (N_test, k)
    y_test: np.array
        The testing labels of shape (N_test)
    """

    model = SGDClassifier(loss='log', max_iter=10000)
    model = model.fit(X_train, y_train)
    y_predict_train = model.predict(X_train)
    accuracy1 = Accuracy(y_predict_train, y_train)
    y_predict_test = model.predict(X_test)
    accuracy2 = Accuracy(y_predict_test, y_test)

    return accuracy1, accuracy2


# %%

from sklearn.tree import DecisionTreeClassifier


def train_and_evaluate_decision_tree(X_train, y_train, X_test, y_test):
    """
    Trains an unbounded decision tree on the training data and computes two accuracy scores, the
    accuracy of the decision tree on the training data and the accuracy of the decision
    tree on the testing data.

    The decision tree should use the information gain criterion (set criterion='entropy')

    Parameters
    ----------
    X_train: np.array
        The training features of shape (N_train, k)
    y_train: np.array
        The training labels of shape (N_train)
    X_test: np.array
        The testing features of shape (N_test, k)
    y_test: np.array
        The testing labels of shape (N_test)

    Returns
    -------
    The training and testing accuracies represented as a tuple of size 2.
    """
    model = DecisionTreeClassifier(criterion='entropy')
    model = model.fit(X_train, y_train)
    y_predict_train = model.predict(X_train)
    accuracy1 = Accuracy(y_predict_train, y_train)
    y_predict_test = model.predict(X_test)
    accuracy2 = Accuracy(y_predict_test, y_test)
    return accuracy1, accuracy2


def train_and_evaluate_decision_stump(X_train, y_train, X_test, y_test):
    """
    Trains a decision stump of maximum depth 4 on the training data and computes two accuracy scores, the
    accuracy of the decision stump on the training data and the accuracy of the decision
    tree on the testing data.

    The decision tree should use the information gain criterion (set criterion='entropy')

    Parameters
    ----------
    X_train: np.array
        The training features of shape (N_train, k)
    y_train: np.array
        The training labels of shape (N_train)
    X_test: np.array
        The testing features of shape (N_test, k)
    y_test: np.array
        The testing labels of shape (N_test)

    Returns
    -------
    The training and testing accuracies represented as a tuple of size 2.
    """
    model = DecisionTreeClassifier(criterion='entropy', max_depth=4)
    model = model.fit(X_train, y_train)
    y_predict_train = model.predict(X_train)
    accuracy1 = Accuracy(y_predict_train, y_train)
    y_predict_test = model.predict(X_test)
    accuracy2 = Accuracy(y_predict_test, y_test)
    return accuracy1, accuracy2


# %%

def train_and_evaluate_sgd_with_stumps(X_train, y_train, X_test, y_test):
    """
    Trains a SGDClassifier with stumps on the training data and computes two accuracy scores, the
    accuracy of the classifier on the training data and the accuracy of the decision
    tree on the testing data.

    Parameters
    ----------
    X_train: np.array
        The training features of shape (N_train, k)
    y_train: np.array
        The training labels of shape (N_train)
    X_test: np.array
        The testing features of shape (N_test, k)
    y_test: np.array
        The testing labels of shape (N_test)

    Returns
    -------
    The training and testing accuracies represented as a tuple of size 2.
    """
    num = 0
    classifier = []
    index = []
    list = []
    example = np.shape(X_train)[0]
    length = np.shape(X_train)[1]
    num_features = int(length/2)
    tree_stumps = np.zeros((example, 50))
    column = 0
    list_new_text_x =[]
    tumors = np.shape(X_test)[0]
    test_stumps = []
    test_stumps = np.zeros((tumors, 50))
    while (num <= 49):
        new_test_x = []
        new_train_x = []
        index = random.sample(range(0, length), num_features)
        if (index not in list):
            new_train_x = X_train[:, index]
            new_test_x = X_test[:, index]
            list_new_text_x.append(new_test_x)
            num += 1
            list.append(sorted(index))
            model = DecisionTreeClassifier(criterion='entropy', max_depth=4)
            model = model.fit(new_train_x, y_train)
            classifier.append(model)
            output_stumps = model.predict(new_train_x)
            output_stumps_test = model.predict(new_test_x)
            row = 0
            for label in output_stumps_test:
                test_stumps[row, column] = label
                row += 1
            row = 0
            for label in output_stumps:
                tree_stumps[row, column] = label
                row += 1
            column += 1


    final_model = SGDClassifier(loss='log', max_iter=10000).fit(tree_stumps, y_train)
    y_predict_train = final_model.predict(tree_stumps)
    accuracy1 = Accuracy(y_predict_train, y_train)
    y_predict_test = final_model.predict(test_stumps)
    accuracy2 = Accuracy(y_predict_test, y_test)
    return accuracy1, accuracy2


# %%

def load_cv_split(fold):
    """
    Parameters
    ----------
    fold: int
        The integer index of the split to load, i.e. 0, 1, 2, 3, or 4

    Returns
    -------
    A tuple of 4 numpy arrays that correspond to the following items:
        X_train, y_train, X_test, y_test
    """
    X_train = np.load('madelon/cv-train-X.' + str(fold) + '.npy')
    y_train = np.load('madelon/cv-train-y.' + str(fold) + '.npy')
    X_test = np.load('madelon/cv-heldout-X.' + str(fold) + '.npy')
    y_test = np.load('madelon/cv-heldout-y.' + str(fold) + '.npy')
    return X_train, y_train, X_test, y_test


# %%

import os
import matplotlib.pyplot as plt


def plot_results(sgd_train_acc, sgd_train_std, sgd_heldout_acc, sgd_heldout_std, sgd_test_acc,
                 dt_train_acc, dt_train_std, dt_heldout_acc, dt_heldout_std, dt_test_acc,
                 dt4_train_acc, dt4_train_std, dt4_heldout_acc, dt4_heldout_std, dt4_test_acc,
                 stumps_train_acc, stumps_train_std, stumps_heldout_acc, stumps_heldout_std, stumps_test_acc):
    """
    Plots the final results from problem 2. For each of the 4 classifiers, pass
    the training accuracy, training standard deviation, held-out accuracy, held-out
    standard deviation, and testing accuracy.

    Although it should not be necessary, feel free to edit this method.
    """
    train_x_pos = [0, 4, 8, 12]
    cv_x_pos = [1, 5, 9, 13]
    test_x_pos = [2, 6, 10, 14]
    ticks = cv_x_pos

    labels = ['sgd', 'dt', 'dt4', 'stumps (4 x 50)']

    train_accs = [sgd_train_acc, dt_train_acc, dt4_train_acc, stumps_train_acc]
    train_errors = [sgd_train_std, dt_train_std, dt4_train_std, stumps_train_std]

    cv_accs = [sgd_heldout_acc, dt_heldout_acc, dt4_heldout_acc, stumps_heldout_acc]
    cv_errors = [sgd_heldout_std, dt_heldout_std, dt4_heldout_std, stumps_heldout_std]

    test_accs = [sgd_test_acc, dt_test_acc, dt4_test_acc, stumps_test_acc]

    fig, ax = plt.subplots()
    ax.bar(train_x_pos, train_accs, yerr=train_errors, align='center', alpha=0.5, ecolor='black', capsize=10,
           label='train')
    ax.bar(cv_x_pos, cv_accs, yerr=cv_errors, align='center', alpha=0.5, ecolor='black', capsize=10, label='held-out')
    ax.bar(test_x_pos, test_accs, align='center', alpha=0.5, capsize=10, label='test')
    ax.set_ylabel('Accuracy')
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_title('Models')
    ax.yaxis.grid(True)
    ax.legend()
    plt.tight_layout()


#  validation


plot_results(0.6, 0.1, 0.7, 0.1, 0.1,
             0.7, 0.2, 0.7, 0.15, 0.2,
             0.8, 0.3, 0.7, 0.2, 0.3,
             0.9, 0.4, 0.7, 0.25, 0.4)

def deviation(model_dateset):
    num =0
    total=0
    for value in model_dateset:
        num += value
    num= (num/5)
    for value in model_dateset:
        value = (value - num)**2
        total +=value
    total = total / (len(model_dateset))
    return total



# %%
def main():
    model1_train = np.zeros(5)
    model1_heldout = np.zeros(5)
    model2_train = np.zeros(5)
    model2_heldout = np.zeros(5)
    model3_train = np.zeros(5)
    model3_heldout = np.zeros(5)
    model4_train = np.zeros(5)
    model4_heldout = np.zeros(5)
    for num in range(0, 5):
        X_train, y_train, X_test, y_test = load_cv_split(num)
        accuracy1, accuracy2 = train_and_evaluate_sgd(X_train, y_train, X_test, y_test)
        model1_train[num] = accuracy1
        model1_heldout[num] = accuracy2

        accuracy1, accuracy2 = train_and_evaluate_decision_tree(X_train, y_train, X_test, y_test)
        model2_train[num] = accuracy1
        model2_heldout[num] = accuracy2

        accuracy1, accuracy2 = train_and_evaluate_decision_stump(X_train, y_train, X_test, y_test)
        model3_train[num] = accuracy1
        model3_heldout[num] = accuracy2

        accuracy1, accuracy2 = train_and_evaluate_sgd_with_stumps(X_train, y_train, X_test, y_test)
        model4_train[num] = accuracy1
        model4_heldout[num] = accuracy2

    acctrain_1 = 0
    acctrain_2 = 0
    acctrain_3 = 0
    acctrain_4 = 0
    accheld_1 = 0
    accheld_2 = 0
    accheld_3 = 0
    accheld_4 = 0
    for i in range(0,5):
        acctrain_1 += model1_train[i]
        acctrain_2 += model2_train[i]
        acctrain_3 += model3_train[i]
        acctrain_4 += model4_train[i]
        accheld_1 += model1_heldout[i]
        accheld_2 += model2_heldout[i]
        accheld_3 += model3_heldout[i]
        accheld_4 += model4_heldout[i]
    sgd_train_acc = acctrain_1 /5
    dt_train_acc = acctrain_2 /5
    dt4_train_acc = acctrain_3 /5
    stumps_train_acc = acctrain_4 /5
    sgd_heldout_acc = accheld_1 /5
    dt_heldout_acc = accheld_2 /5
    dt4_heldout_acc = accheld_3 /5
    stumps_heldout_acc = accheld_4 /5

    X_test = np.load('madelon/test-X.npy')
    y_test = np.load('madelon/test-y.npy')
    X_train = np.load('madelon/train-X.npy')
    y_train = np.load('madelon/train-y.npy')

    sgd_test_acc = train_and_evaluate_sgd(X_train, y_train, X_test, y_test)[1]
    dt_test_acc = train_and_evaluate_decision_tree(X_train, y_train, X_test, y_test)[1]
    dt4_test_acc = train_and_evaluate_decision_stump(X_train, y_train, X_test, y_test)[1]
    stumps_test_acc = train_and_evaluate_sgd_with_stumps(X_train, y_train, X_test, y_test)[1]

    sgd_train_std = np.array(model1_train)
    sgd_train_std = np.std(sgd_train_std, ddof = 1)
    sgd_heldout_std = np.array(model1_heldout)
    sgd_heldout_std = np.std(sgd_heldout_std, ddof=1)
    dt_train_std = np.array(model2_train)
    dt_train_std  = np.std(dt_train_std , ddof=1)
    dt_heldout_std = np.array(model2_heldout)
    dt_heldout_std = np.std(dt_heldout_std, ddof=1)
    dt4_train_std = np.array(model3_train)
    dt4_train_std =  np.std(dt4_train_std, ddof=1)
    dt4_heldout_std = np.array(model3_heldout)
    dt4_heldout_std = np.std(dt4_heldout_std, ddof=1)
    stumps_train_std = np.array(model4_train)
    stumps_train_std = np.std(stumps_train_std , ddof=1)
    stumps_heldout_std = np.array(model4_heldout)
    stumps_heldout_std = np.std(stumps_heldout_std , ddof=1)
    plot_results(sgd_train_acc, sgd_train_std, sgd_heldout_acc, sgd_heldout_std, sgd_test_acc,
                 dt_train_acc, dt_train_std, dt_heldout_acc, dt_heldout_std, dt_test_acc,
                 dt4_train_acc, dt4_train_std, dt4_heldout_acc, dt4_heldout_std, dt4_test_acc,
                 stumps_train_acc, stumps_train_std, stumps_heldout_acc, stumps_heldout_std, stumps_test_acc)
    plt.show()




    test_names = open("badges/test.names.txt", "r")
    test_names = test_names.readlines()
    test_names = compute_features(test_names)
    train_names = open("badges/train.names.txt", "r")
    train_names = train_names.readlines()
    train_names = compute_features(train_names)
    test_labels = np.load('badges/test.labels.npy')
    train_labels = np.load('badges/train.labels.npy')
    acc_model1_train, acc_model1_test = train_and_evaluate_sgd(train_names, train_labels, test_names, test_labels)
    acc_model2_train, acc_model2_test = train_and_evaluate_decision_tree(train_names, train_labels, test_names, test_labels)
    acc_model3_train, acc_model3_test = train_and_evaluate_decision_stump(train_names, train_labels, test_names, test_labels)
    acc_model4_train, acc_model4_test = train_and_evaluate_sgd_with_stumps(train_names, train_labels, test_names, test_labels)
    print(acc_model1_train, acc_model1_test, acc_model2_train, acc_model2_test, acc_model3_train, acc_model3_test,
          acc_model4_train, acc_model4_test)






if __name__=='__main__':
    main()

