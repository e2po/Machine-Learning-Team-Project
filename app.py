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
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas import DataFrame
from sklearn import metrics, svm
from sklearn.cross_validation import train_test_split, KFold, cross_val_score
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

sns.set_style('whitegrid')


def load_dataset():
    """
    Load Boston Housing Dataset from scikit-learn library.

    :return: boston dataset
    """
    from sklearn.datasets import load_boston
    return load_boston()


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


if __name__ == '__main__':
    # load dataset
    boston = load_dataset()
    # describe_dataset(boston)
    # split dataset into training and testing subsets
    X_train, X_test, y_train, y_test = split_train_test(features=boston.data,
                                                        target=boston.target)
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
        train_and_evaluate(models[i], X_train, y_train)

        predicted_training_target = models[i].predict(X_train)
        predicted_testing_target = models[i].predict(X_test)

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
    df = DataFrame(boston.feature_names)
    df.columns = ['Features']
    df['Importance'] = models[-1].feature_importances_  # use data from 'Gradient Boosting' model
    print(df.sort_values(by='Importance', ascending=False))

    # display residual plot
    plt.tight_layout()
    plt.show()
