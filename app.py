"""
Authors:    Liu, Yuntian
            Murphy, Declan
            Porebski, Elvis
            Tyrakowski, Bartosz
Date:       22-02-2016
Purpose:    Machine Learning Team Project.

Learning Models:
            1. Linear Regression.
            2. Ridge Regression.
            3. Lasso.
            4. Elastic Net.
            5. Support Vector Machine Regression.
            6. Random Forests Decision Trees.
"""

from sklearn.cross_validation import train_test_split
import numpy as np


def load_dataset():
    """
    Load Boston Housing Dataset from scikit-learn library.

    :return: boston dataset
    """
    from sklearn.datasets import load_boston
    return load_boston()


def split_train_test(features, target, test_size=0.2):
    """
    Split dataset into random Train and Test subsets.

    :param features:    Input variables(features)
    :param target:      Output variable(target value)
    :param test_size:   Percentage of dataset used for testing, default 20%.
    :return: tuple:     (X_train, X_test, y_train, y_test) where X represents features and y represents target value
    """
    return train_test_split(features,
                            target,
                            test_size=test_size,
                            random_state=65)


def describe_dataset(dataset):
    total_houses, total_features = dataset.data.shape
    min_price = np.min(dataset.target)
    max_price = np.max(dataset.target)
    mean_price = np.mean(dataset.target)
    median_price = np.median(dataset.target)
    std_dev = np.std(dataset.target)

    print('Boston Housing Dataset')
    print('Total number of houses: ', total_houses)
    print('Total number of features: ', total_features)
    print('Minimum house price: ', min_price)
    print('Maximum house price: ', max_price)
    print('Mean house price: ', mean_price)
    print('Median house price: ', median_price)
    print('Standard deviation of house price: ', std_dev)

if __name__ == '__main__':
    # load dataset
    boston = load_dataset()
    # describe loaded dataset
    describe_dataset(boston)
    # split it into training and testing subsets
    X_train, X_test, y_train, y_test = split_train_test(features=boston.data,
                                                        target=boston.target)
