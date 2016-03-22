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
import os

import numpy as np
from sklearn import metrics, svm
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, Lars


class Model:
    def __init__(self, estimator, estimator_name):
        self.estimator = estimator
        self.estimator_name = estimator_name

    def train_and_evaluate(self, features, target, kfold=False):
        self.estimator.fit(features, target)
        if kfold:
            print("Coefficient of determination on training set:", self.estimator.score(features, target))
            cv = KFold(features.shape[0], 5, shuffle=True, random_state=33)
            scores = cross_val_score(self.estimator, features, target, cv=cv)
            print("Average coefficient of determination using 5-fold cross validation:", np.mean(scores))

    def predict(self, features):
        return self.estimator.predict(features)

    def save(self):
        if not os.path.exists('persistence'):
            os.makedirs('persistence')
            joblib.dump(self.estimator, 'persistence/' + self.estimator_name)

    @staticmethod
    def load(estimator_name):
        if os.path.exists('persistence/' + estimator_name):
            return Model(estimator=joblib.load('persistence/' + estimator_name), estimator_name=estimator_name)


def measure_performance(actual_target, expected_target):
    mae = metrics.mean_absolute_error(expected_target, actual_target)
    mse = metrics.mean_squared_error(expected_target, actual_target)
    r2 = metrics.r2_score(expected_target, actual_target)
    return mae, mse, r2


def get_models(features, target):
    # list of Generalised ML Models
    estimator_names = ['SVM RBF', 'Linear', 'Ridge', 'Random Forest', 'Extra Trees',
                       'Lasso', 'Elastic Net', 'LARS', 'Gradient Boosting']

    models = []
    for estimator_name in estimator_names:
        # load model from a file
        model = Model.load(estimator_name)
        # if model was loaded from a file
        if model:
            # append it to a list
            models.append(model)
        # otherwise create a new model and then append it to a list
        elif estimator_name == 'SVM RBF':
            models.append(Model(estimator=svm.SVR(kernel='rbf', C=50000, gamma=0.00001, epsilon=.0001),
                                estimator_name='SVM RBF')
                          .train_and_evaluate(features, target, kfold=True))
        elif estimator_names == 'Linear':
            models.append(Model(estimator=LinearRegression(),
                                estimator_name='Linear')
                          .train_and_evaluate(features, target, kfold=True))
        elif estimator_names == 'Ridge':
            models.append(Model(estimator=Ridge(alpha=0.1, fit_intercept=True),
                                estimator_name='Ridge')
                          .train_and_evaluate(features, target, kfold=True))
        elif estimator_names == 'Random Forest':
            models.append(Model(estimator=RandomForestRegressor(n_estimators=100,
                                                                max_depth=4,
                                                                min_samples_leaf=1,
                                                                min_samples_split=2),
                                estimator_name='Random Forest')
                          .train_and_evaluate(features, target, kfold=True))
        elif estimator_names == 'Extra Trees':
            models.append(Model(estimator=ExtraTreesRegressor(n_estimators=10, random_state=42),
                                estimator_name='Extra Trees')
                          .train_and_evaluate(features, target, kfold=True))
        elif estimator_names == 'Lasso':
            models.append(Model(estimator=Lasso(alpha=0.1, fit_intercept=True),
                                estimator_name='Lasso')
                          .train_and_evaluate(features, target, kfold=True))
        elif estimator_names == 'Elastic Net':
            models.append(Model(estimator=ElasticNet(alpha=0.1, fit_intercept=True),
                                estimator_name='Elastic Net')
                          .train_and_evaluate(features, target, kfold=True))
        elif estimator_names == 'LARS':
            models.append(Model(estimator=Lars(fit_intercept=True, n_nonzero_coefs=np.inf, normalize=True),
                                estimator_name='LARS')
                          .train_and_evaluate(features, target, kfold=True))
        elif estimator_names == 'Gradient Boosting':
            models.append(Model(estimator=GradientBoostingRegressor(n_estimators=500,
                                                                    max_depth=4,
                                                                    min_samples_split=1,
                                                                    learning_rate=0.01,
                                                                    loss='ls'),
                                estimator_name='Gradient Boosting')
                          .train_and_evaluate(features, target, kfold=True))
    return models
