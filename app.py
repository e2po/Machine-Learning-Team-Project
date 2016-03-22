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
import time
from itertools import combinations

import numpy as np
from pandas import DataFrame

from dataset import DataSet
from graph import plot_relationship, plot_residual
from model import get_models, measure_performance


def show_relationships():
    # Display options to user
    print("--------------------------------------------------------------------------")
    print("Relationship between feature and price")
    print("--------------------------------------------------------------------------")
    print("0. CRIM :     per capita crime rate by town")
    print("1. ZN :       proportion of residential land zoned for lots over 25,000 sq.ft.")
    print("2. INDUS :    proportion of non-retail business acres per town")
    print("3. CHAS :     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)")
    print("4. NOX :      nitric oxides concentration (parts per 10 million)")
    print("5. RM :       average number of rooms per dwelling")
    print("6. AGE :      proportion of owner-occupied units built prior to 1940")
    print("7. DIS :      weighted distances to five Boston employment centres")
    print("8. RAD :      index of accessibility to radial highways")
    print("9. TAX :     full-value property-tax rate per $10,000")
    print("10. PTRATIO : pupil-teacher ratio by town")
    print("11. B :       1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town")
    print("12. LSTAT :   % lower status of the population")
    print('13. Back')
    print("--------------------------------------------------------------------------")

    running = True
    while running:
        user_input = input('Enter your choice: ')
        if user_input == '13':
            running = False
        elif user_input.isdigit():
            index = int(user_input)
            if 0 <= index < 13:
                plot_relationship(dataset.dataset.ix[:, 0], dataset.target, dataset.column_names[index])


def show_model_performance(model):
    print('--------------------------------------------------------------------------')
    print(model.estimator_name, 'Performance')
    print('--------------------------------------------------------------------------')

    # Predict prices with data used for training
    predicted_training_target = model.predict(training_features)
    # Predict prices with new and unseen data
    predicted_testing_target = model.predict(testing_features)

    # Measure Mean Absolute Error, Mean Squared Error and Coefficient of Determination score.
    training_mae, training_mse, training_r2 = measure_performance(predicted_training_target, training_target)
    testing_mae, testing_mse, testing_r2 = measure_performance(predicted_testing_target, testing_target)

    print('Mean Absolute Error: {0:.2f}'.format(training_mae),
          'Mean Square Error: {0:.2f}'.format(training_mse),
          'R2: {0:.2f}'.format(training_r2), "Trained " + model.estimator_name)
    print('Mean Absolute Error: {0:.2f}'.format(testing_mae),
          'Mean Square Error: {0:.2f}'.format(testing_mse),
          'R2: {0:.2f}'.format(testing_r2), " Tested " + model.estimator_name)

    # Create a data frame
    data_frame = DataFrame()
    # Add a PREDICTED price column to the table
    data_frame['PREDICTED'] = predicted_testing_target
    # Add an ACTUAL price column to the table
    data_frame['ACTUAL'] = list(testing_target)

    print('--------------------------------------------------------------------------')
    print("First 10 predictions for unseen new data : ")
    print(data_frame[:10])


def show_model_list():
    running = True
    while running:
        print('--------------------------------------------------------------------------')
        print('Generalised Models')
        print('--------------------------------------------------------------------------')
        for index, m in enumerate(models):
            print('{}. {}'.format(index, m.estimator_name))
        print('9. Back')
        print('--------------------------------------------------------------------------')

        user_input = input('Enter your choice: ')
        if user_input == '9':
            running = False
        elif user_input.isdigit():
            index = int(user_input)
            if 0 <= index < len(models):
                show_model_options(models[index])


def show_main_menu():
    running = True
    while running:
        print('--------------------------------------------------------------------------')
        print('Main Menu')
        print('--------------------------------------------------------------------------')
        print('0. Show ML Models')
        print('1. Show Best ML Model')
        print('2. Show Relationships')
        print('3. Exit')
        print('--------------------------------------------------------------------------')
        user_input = input('Enter your choice: ')
        if user_input == '0':
            show_model_list()
        elif user_input == '2':
            show_relationships()
        elif user_input == '3':
            running = False


def show_model_options(model):
    running = True
    while running:
        print('--------------------------------------------------------------------------')
        print(model.estimator_name)
        print('--------------------------------------------------------------------------')
        print('0. Show performance')
        print('1. Show residual plot')
        print('2. Train this model again')
        print('3. Brute force combination of features')
        print('4. Display feature importance')
        print('5. Predict housing price using custom features')
        print('6. Back')
        print('--------------------------------------------------------------------------')

        user_input = input('Enter your choice: ')
        if user_input == '6':
            running = False

        # Show Performance.
        elif user_input == '0' and user_input.isdigit:
            show_model_performance(model)

        # Show Residual Plot.
        elif user_input == '1':
            # Predict prices with data used for training.
            predicted_training_target = model.predict(training_features)
            # Predict prices with new and unseen data.
            predicted_testing_target = model.predict(testing_features)

            # Generate and show residual plot.
            plot_residual(predicted_training_target=predicted_training_target, actual_training_target=training_target,
                          predicted_testing_target=predicted_testing_target, actual_testing_target=testing_target,
                          model_name=model.estimator_name)

        # Train again.
        elif user_input == '2' and user_input.isdigit:
            print('Training {} ...'.format(model.estimator_name))
            model.train_and_evaluate(features=training_features, target=training_target, kfold=True)
            print('Training completed!')
            model.save()
            print('Model persisted.')

        elif user_input == '3':
            print('searching for best combination of features...')
            best_mse, best_feature_lists = find_best_features(model)
            print('Best MSE:', best_mse, 'using features:', best_feature_lists)

        elif user_input == '4':
            df = DataFrame()
            df['FEATURE'] = dataset.column_names[0:-1]
            if model.estimator_name in ['Elastic Net', 'LARS', 'Lasso', 'Ridge', 'Linear', ]:
                df['IMPORTANCE'] = model.estimator.coef_
                print(df.sort_values(by='IMPORTANCE', ascending=False))
            elif model.estimator_name in ['Gradient Boosting', 'Random Forest', 'Extra Trees']:
                df['IMPORTANCE'] = model.estimator.feature_importances_
                print(df.sort_values(by='IMPORTANCE', ascending=False))
            elif model.estimator_name in ['SVM RBF']:
                print(model.estimator.dual_coef_)

        elif user_input == '5':
            predict_custom(model)


def find_best_features(model):
    best_mse = 100
    best_features = []
    indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    start = time.time()
    for combination_length in reversed(range(13)):
        combination_length += 1
        for c in combinations(indices, combination_length):
            model.estimator.fit(training_features.ix[:, c], training_target)
            predicted = model.predict(testing_features.ix[:, c])

            t_mae, current_mse, t_r2 = measure_performance(predicted, testing_target)
            if current_mse < best_mse:
                print('New best MSE: ', current_mse, ' using features: ', c)
                best_mse = current_mse
                best_features = c

    end = time.time()
    print('Execution time: ', end - start)

    # train model again with all features
    model.estimator.fit(training_features, training_target)
    return best_mse, best_features


def predict_custom(model):
    print("0. CRIM :     per capita crime rate by town")
    print("1. ZN :       proportion of residential land zoned for lots over 25,000 sq.ft.")
    print("2. INDUS :    proportion of non-retail business acres per town")
    print("3. CHAS :     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)")
    print("4. NOX :      nitric oxides concentration (parts per 10 million)")
    print("5. RM :       average number of rooms per dwelling")
    print("6. AGE :      proportion of owner-occupied units built prior to 1940")
    print("7. DIS :      weighted distances to five Boston employment centres")
    print("8. RAD :      index of accessibility to radial highways")
    print("9. TAX :     full-value property-tax rate per $10,000")
    print("10. PTRATIO : pupil-teacher ratio by town")
    print("11. B :       1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town")
    print("12. LSTAT :   % lower status of the population")
    print('--------------------------------------------------------------------------')
    features = []
    crim = input('CRIM: ')
    zn = input('ZN: ')
    indus = input('INDUS: ')
    chas = input('CHAS: ')
    nox = input('NOX: ')
    rm = input('RM: ')
    age = input('AGE: ')
    dis = input('DIS: ')
    rad = input('RAD: ')
    tax = input('TAX: ')
    ptratio = input('PTRATIO: ')
    b = input('B: ')
    lstat = input('LSTAT: ')

    try:
        features.append(float(crim))
        features.append(float(zn))
        features.append(float(indus))
        features.append(float(chas))
        features.append(float(nox))
        features.append(float(rm))
        features.append(float(age))
        features.append(float(dis))
        features.append(float(rad))
        features.append(float(tax))
        features.append(float(ptratio))
        features.append(float(b))
        features.append(float(lstat))

        # f = [0.00632,18,2.31,0,0.538,6.575,65.2,4.09,1,296,15.3,396.9,4.98]

        print('Predicted Value:', model.predict(np.asarray(features).reshape(1, -1))[0])
        # print('Predicted Value:', model.predict(features)[0])
    except Exception as err:
        print('An error has occured.', err)


if __name__ == '__main__':
    # load dataset
    dataset = DataSet.load()

    # split dataset into training and testing subsets
    training_features, testing_features, training_target, testing_target = dataset.split_train_test()

    # list of Generalised ML Models
    models = get_models(training_features, training_target)

    show_main_menu()
