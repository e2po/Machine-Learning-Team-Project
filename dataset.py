"""
Authors:    Liu, Yuntian
            Murphy, Declan
            Porebski, Elvis
            Tyrakowski, Bartosz
Date:       March, 2016
Purpose:    Machine Learning Team Project.

Generalised Machine Learning Models:
            1. Linear Regression.
            2. Ridge Regression.
            3. Lasso.
            4. Elastic Net.
            5. Support Vector Machine Regression.
            6. Random Forests Decision Trees.
            7. Extra Trees.
            8. Gradient Boosting.
"""
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split


class DataSet:
    def __init__(self, dataset):
        self.dataset = dataset
        self.features = dataset.ix[:, 0:13]
        self.target = dataset.MEDV
        self.column_names = dataset.columns.values.tolist()

    def describe(self):
        total_houses, total_features = self.dataset.shape
        min_price = np.min(self.dataset.MEDV)
        max_price = np.max(self.dataset.MEDV)
        mean_price = np.mean(self.dataset.MEDV)
        median_price = np.median(self.dataset.MEDV)
        std_dev = np.std(self.dataset.MEDV)

        print('------------------------------------------------------------------------')
        print('Boston Housing Dataset')
        print('Total number of houses: ', total_houses)
        print('Total number of features: ', total_features)
        print('Minimum house price: ', min_price)
        print('Maximum house price: ', max_price)
        print('Mean house price: ', mean_price)
        print('Median house price: ', median_price)
        print('Standard deviation of house price: ', std_dev)

    def split_train_test(self, test_size=0.20):
        """
        Split dataset into random Train and Test subsets.

        :param test_size:   Percentage of dataset used for testing, default 20%.
        :return: tuple:     (X_train, X_test, y_train, y_test) where X represents features and y represents target value
        """
        return train_test_split(self.features,
                                self.target,
                                test_size=test_size,
                                random_state=14)

    @staticmethod
    def load():
        """
        Load Boston Housing Dataset from UCI web servers

        :return: DataSet object
        """
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
        col_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                     'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
        # load csv file from UCI web servers into DataFrame
        dataset = pd.read_csv(url, header=None, delim_whitespace=True, names=col_names)

        # if there are missing values
        if pd.isnull(dataset).any().any():
            # display message
            print('Some values are missing.')
            # and drop rows with missing values
            dataset = dataset.dropna()

        return DataSet(dataset=dataset)
