"""
Authors:    Liu, Yuntian
            Murphy, Declan
            Porebski, Elvis
            Tyrakowski, Bartosz
Date:       22-02-2016
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
import os
import time
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame
from sklearn import metrics, svm
from sklearn.cross_validation import train_test_split, KFold, cross_val_score
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

sns.set_style('whitegrid')


def load_dataset():
    """
    Load Boston Housing Dataset from scikit-learn library.

    :return: boston dataset
    """
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
    col_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    dataset = pd.read_csv(url, header=None, delim_whitespace=True, names=col_names)

    if pd.isnull(dataset).any().any():
        print('Some values are missing.')
        dataset = dataset.dropna()

    feature_cols = col_names[:-1]
    features = dataset[feature_cols]
    target = dataset.MEDV

    return feature_cols, features, target


def split_train_test(features, target, test_size=0.20):
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
                            random_state=14)


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


def train_and_evaluate(model, features, target):
    model.fit(features, target)
    print("Coefficient of determination on training set:", model.score(features, target))
    cv = KFold(X_train.shape[0], 5, shuffle=True, random_state=33)
    scores = cross_val_score(model, features, target, cv=cv)
    print("Average coefficient of determination using 5-fold cross validation:", np.mean(scores))


def measure_performance(actual_target, expected_target):
    mae = metrics.mean_absolute_error(expected_target, actual_target)
    mse = metrics.mean_squared_error(expected_target, actual_target)
    r2 = metrics.r2_score(expected_target, actual_target)
    return mae, mse, r2


def save_model(model, model_name):
    if not os.path.exists('persistence'):
        os.makedirs('persistence')
    joblib.dump(model, 'persistence/' + model_name)


def load_model(model_name):
    return joblib.load('persistence/' + model_name)


def find_best_features(model, training_features, testing_features, training_target, testing_target):
    current_best_mse = 100
    current_best_features = []
    indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    start = time.time()
    for combination_length in reversed(range(13)):
        combination_length += 1
        for c in combinations(indices, combination_length):
            model.fit(training_features.ix[:, c], training_target)
            predicted = models[-1].predict(testing_features.ix[:, c])
            t_mae, current_mse, t_r2 = measure_performance(predicted, testing_target)
            if current_mse < current_best_mse:
                print('New best MSE: ', current_mse, ' using features: ', c)
                current_best_mse = current_mse
                current_best_features = c

    end = time.time()
    print('Execution time: ', end - start)
    return current_best_mse, current_best_features


if __name__ == '__main__':
    # load dataset
    boston_feature_names, boston_features, boston_target = load_dataset()

    # describe_dataset(boston)

    # split dataset into training and testing subsets
    X_train, X_test, y_train, y_test = split_train_test(features=boston_features,
                                                        target=boston_target)

    # list of Generalised ML Models
    models = [svm.SVR(kernel='rbf', C=50000, gamma=0.00001, epsilon=.0001),
              LinearRegression(),
              Ridge(alpha=0.1),
              RandomForestRegressor(n_estimators=100, max_depth=4, min_samples_leaf=1, min_samples_split=2),
              ExtraTreesRegressor(n_estimators=10, random_state=42),
              GradientBoostingRegressor(n_estimators=200)]
    models_names = ['SVM RBF', 'Linear', 'Ridge', 'Random Forest', 'Extra Trees', 'Gradient Boosting']

    # set up residual plot to visualise how good the model is
    fig, axes = plt.subplots(3, 3)  # 3x3 grid of figures
    fig.canvas.set_window_title('Residual Plots')

    for i in range(len(models)):
        print('------------------------', models_names[i], '------------------------')

        # if model was previously saved, load it
        current_model = load_model(models_names[i])
        if current_model:
            print('loading ', models_names[i], ' ...')
            models[i] = current_model
        else:
            # otherwise, train it
            print('training ', models_names[i], ' ...')
            train_and_evaluate(models[i], X_train, y_train)

        predicted_training_target = models[i].predict(X_train)
        predicted_testing_target = models[i].predict(X_test)

        save_model(models[i], models_names[i])

        training_mae, training_mse, training_r2 = measure_performance(predicted_training_target, y_train)
        testing_mae, testing_mse, testing_r2 = measure_performance(predicted_testing_target, y_test)

        print('MAE: {0:.2f}'.format(training_mae),
              'MSE: {0:.2f}'.format(training_mse),
              'R2: {0:.2f}'.format(training_r2), "Trained " + models_names[i])
        print('MAE: {0:.2f}'.format(testing_mae),
              'MSE: {0:.2f}'.format(testing_mse),
              'R2: {0:.2f}'.format(testing_r2), " Tested " + models_names[i])

        # set up residual plot for this model
        axes[i / 3][i % 3].set_aspect('equal')
        axes[i / 3][i % 3].set_title(models_names[i])

        train = axes[i/3][i % 3].scatter(predicted_training_target,
                                         predicted_training_target - y_train, c='b', alpha=0.5)
        test = axes[i/3][i % 3].scatter(predicted_testing_target,
                                        predicted_testing_target - y_test, c='r', alpha=0.5)

        axes[i / 3][i % 3].set_aspect('equal')
        axes[i / 3][i % 3].set_title(models_names[i])
        axes[i / 3][i % 3].hlines(y=0, xmin=-10, xmax=50)
        axes[i / 3][i % 3].legend((train, test), ('Training', 'Test'), loc='lower left')

    # display all features sorted by importance in descending order.
    df = DataFrame(boston_feature_names)
    df.columns = ['Features']
    df['Importance'] = models[-1].feature_importances_  # use data from 'Gradient Boosting' model
    print(df.sort_values(by='Importance', ascending=False))

    # display residual plot
    plt.tight_layout()
    plt.show()

    # find best combination of features
    best_mse, features_list = find_best_features(models[-1], X_train, X_test, y_train, y_test)
    print('Best MSE: ', best_mse, ' using features: ', features_list)
