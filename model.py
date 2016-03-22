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
        print('loading {} ...'.format(estimator_name))
        # load model from a file
        model = Model.load(estimator_name)
        # if model was loaded from a file
        if model:
            model.estimator_name = estimator_name
            # append it to a list
            models.append(model)
        else:
            # otherwise create a new model and then append it to a list
            if estimator_name == 'SVM RBF':
                new_model = Model(estimator=svm.SVR(kernel='rbf', C=50000, gamma=0.00001, epsilon=.0001),
                                  estimator_name='SVM RBF')
                new_model.train_and_evaluate(features, target, kfold=False)
                new_model.save()
                models.append(new_model)

            elif estimator_name == 'Linear':
                new_model = Model(estimator=LinearRegression(),
                                  estimator_name='Linear')
                new_model.train_and_evaluate(features, target, kfold=False)
                new_model.save()

                models.append(new_model)

            elif estimator_name == 'Ridge':
                new_model = Model(estimator=Ridge(alpha=0.1, fit_intercept=True),
                                  estimator_name='Ridge')
                new_model.train_and_evaluate(features, target, kfold=False)
                new_model.save()
                models.append(new_model)

            elif estimator_name == 'Random Forest':
                new_model = Model(estimator=RandomForestRegressor(n_estimators=100,
                                                                  max_depth=4,
                                                                  min_samples_leaf=1,
                                                                  min_samples_split=2),
                                  estimator_name='Random Forest')
                new_model.train_and_evaluate(features, target, kfold=False)
                new_model.save()
                models.append(new_model)

            elif estimator_name == 'Extra Trees':
                new_model = Model(estimator=ExtraTreesRegressor(n_estimators=10, random_state=42),
                                  estimator_name='Extra Trees')
                new_model.train_and_evaluate(features, target, kfold=False)
                new_model.save()
                models.append(new_model)

            elif estimator_name == 'Lasso':
                new_model = Model(estimator=Lasso(alpha=0.1, fit_intercept=True),
                                  estimator_name='Lasso')
                new_model.train_and_evaluate(features, target, kfold=False)
                new_model.save()
                models.append(new_model)

            elif estimator_name == 'Elastic Net':
                new_model = Model(estimator=ElasticNet(alpha=0.1, fit_intercept=True),
                                  estimator_name='Elastic Net')
                new_model.train_and_evaluate(features, target, kfold=False)
                new_model.save()
                models.append(new_model)

            elif estimator_name == 'LARS':
                new_model = Model(estimator=Lars(fit_intercept=True, n_nonzero_coefs=np.inf, normalize=True),
                                  estimator_name='LARS')
                new_model.train_and_evaluate(features, target, kfold=False)
                new_model.save()
                models.append(new_model)

            elif estimator_name == 'Gradient Boosting':
                new_model = Model(estimator=GradientBoostingRegressor(n_estimators=500,
                                                                      max_depth=4,
                                                                      min_samples_split=1,
                                                                      learning_rate=0.01,
                                                                      loss='ls'),
                                  estimator_name='Gradient Boosting')
                new_model.train_and_evaluate(features, target, kfold=False)
                new_model.save()
                models.append(new_model)
    return models
