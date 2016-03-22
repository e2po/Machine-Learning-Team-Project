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
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')


def plot_relationship(feature, target, feature_name):
    plt.scatter(feature, target)
    plt.xlabel(feature_name)
    plt.ylabel("Housing Price")
    plt.title("Relationship between Feature and Price")
    plt.show()


def plot_residual(predicted_training_target, actual_training_target,
                  predicted_testing_target, actual_testing_target,
                  model_name):
    train = plt.scatter(predicted_training_target, predicted_training_target - actual_training_target, c='b', alpha=0.5)
    test = plt.scatter(predicted_testing_target, predicted_testing_target - actual_testing_target, c='r', alpha=0.5)
    plt.hlines(y=0, xmin=-10, xmax=50)
    plt.legend((train, test), ('Training', 'Test'), loc='lower left')
    plt.title('Residual Plot for {}'.format(model_name))
    plt.show()
