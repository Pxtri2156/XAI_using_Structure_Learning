import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV
import random
import argparse
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import statistics as st

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def california_housing(): 
    data = pd.read_csv('/workspace/tripx/MCS/xai_causality/dataset/scaled_cali_housing.csv')
    data = data.to_numpy()
    X = data[:,1:-2]
    y = data[:,-1]
    return X, y

def boston_housing(): 
    data = pd.read_csv('/workspace/tripx/MCS/xai_causality/dataset/scaled_boston_housing.csv')
    data = data.to_numpy()
    X = data[:,1:-2]
    y = data[:,-1]
    return X, y

def diabetes(): 
    data = pd.read_csv('/workspace/tripx/MCS/xai_causality/dataset/scaled_diabetes.csv')
    data = data.to_numpy()
    X = data[:,1:-2]
    y = data[:,-1]
    return X, y

def friedman1(): 
    data = pd.read_csv('/workspace/tripx/MCS/xai_causality/dataset/scaled_friedman1.csv')
    data = data.to_numpy()
    X = data[:,1:-2]
    y = data[:,-1]
    return X, y

def dataset_split(X, y): 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return  X_train, X_test, y_train, y_test 

def dataset(dataset_arg): 
    # download dataset
    if dataset_arg == 'california_housing': 
        X, y = california_housing()
    elif dataset_arg == 'boston_housing': 
        X, y = boston_housing()
    elif dataset_arg == 'friedman1': 
        X, y = friedman1()
    elif dataset_arg == 'diabetes': 
        X, y = diabetes()
    else: 
        print('Need a dataset argument!')
        return 
    #dataset_split
    X_train, X_test, y_train, y_test = dataset_split(X, y)
    return X_train, X_test, y_train, y_test

def dataset_tuning(dataset_arg): 
    # download dataset
    if dataset_arg == 'california_housing': 
        X, y = california_housing()
    elif dataset_arg == 'boston_housing': 
        X, y = boston_housing()
    elif dataset_arg == 'friedman1': 
        X, y = friedman1()
    elif dataset_arg == 'diabetes': 
        X, y = diabetes()
    else: 
        print('Need a dataset argument!')
        return 
    return X, y

def average_result(best_model, dataset_arg):
    mse = [] 
    for seed in range(0,15): 
        set_random_seed(seed)
        model = LinearRegression()
        X_train, X_test, y_train, y_test = dataset(dataset_arg)
        if type(best_model.best_estimator_) == type(LinearRegression()): 
            model = LinearRegression()
            model.set_params(**best_model.best_params_)
        elif type(best_model.best_estimator_) == type(Ridge()): 
            model = Ridge()
            model.set_params(**best_model.best_params_)
        elif type(best_model.best_estimator_) == type(Lasso()): 
            model = Lasso()
            model.set_params(**best_model.best_params_)

        elif type(best_model.best_estimator_) == type(svm.SVR()): 
            model = svm.SVR()
            model.set_params(**best_model.best_params_)

        elif type(best_model.best_estimator_) == type(ElasticNet()): 
            model = ElasticNet()
            model.set_params(**best_model.best_params_)

        elif type(best_model.best_estimator_) == type(DecisionTreeRegressor()): 
            model = DecisionTreeRegressor()
            model.set_params(**best_model.best_params_)

        elif type(best_model.best_estimator_) == type(RandomForestRegressor()): 
            model = RandomForestRegressor()
            model.set_params(**best_model.best_params_)

        else: 
            print('Model type problem!')
        model.fit(X_train, y_train)
        # Make predictions on the test set
        y_pred = model.predict(X_test)
        # Calculate root mean squared error
        mse.append(mean_squared_error(y_test, y_pred))
    #mean
    mean_mse = round(sum(mse)/len(mse),2) 
    #std
    std_mse = round(st.pstdev(mse),5) 
    
    print(f'Average: {mean_mse}+-{std_mse}')

def linear_tuning(dataset): 
    print('-----------Linear regression------------')
    X_train, y_train = dataset_tuning(dataset)
    # Create a linear regression model
    model = LinearRegression()
    
    # Define a set of hyperparameters to tune
    param_grid = {
        'fit_intercept': [True, False],
        'copy_X': [True, False],
        'n_jobs': [None, -1],  # -1 means using all available CPU cores
        'positive': [True, False],
    }
    
    # Use GridSearchCV to search for the best hyperparameters
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    print("Best Hyperparameters:", grid_search.best_params_)
    print("Best Model:", grid_search.best_estimator_)
    average_result(grid_search, dataset)

def ridge_tuning(dataset): 
    print('-----------Ridge regression------------')
    X_train, y_train = dataset_tuning(dataset)
    # Create a linear regression model
    model = Ridge()
    
    # Define a set of hyperparameters to tune
    param_grid = {
        'alpha': range(1, 100),
        'fit_intercept': [True, False],
        'copy_X': [True, False],
        'positive': [True],
    }
    
    # Use GridSearchCV to search for the best hyperparameters
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    print("Best Hyperparameters:", grid_search.best_params_)
    print("Best Model:", grid_search.best_estimator_)
    average_result(grid_search, dataset)

def lasso_tuning(dataset): 
    print('-----------Lasso regression------------')
    X_train, y_train = dataset_tuning(dataset)
    # Create a linear regression model
    model = Lasso()
    
    # Define a set of hyperparameters to tune
    param_grid = {
        'alpha': range(1, 100),
        'fit_intercept': [True, False],
        'copy_X': [True, False],
        'positive': [True],
    }
    
    # Use GridSearchCV to search for the best hyperparameters
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    print("Best Hyperparameters:", grid_search.best_params_)
    print("Best Model:", grid_search.best_estimator_)
    average_result(grid_search, dataset)

def elasticNet_tuning(dataset): 
    print('-----------ElasticNet regression------------')
    X_train, y_train = dataset_tuning(dataset)
    # Create a linear regression model
    model = ElasticNet()
    
    # Define a set of hyperparameters to tune
    param_grid = {
        'alpha': range(1, 100),
        'l1_ratio': np.arange(0.0, 1.0, 0.1),
        'fit_intercept': [True, False],
        'copy_X': [True, False],
        'positive': [True, False]
    }
    
    # Use GridSearchCV to search for the best hyperparameters
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    print("Best Hyperparameters:", grid_search.best_params_)
    print("Best Model:", grid_search.best_estimator_)
    average_result(grid_search, dataset)


def svm_tuning(dataset): 
    print('-----------SVM regression------------')
    X_train, y_train = dataset_tuning(dataset)
    # Create a linear regression model
    model = svm.SVR()
    
    # Define a set of hyperparameters to tune
    param_grid = {
        "C": range(1,11),
        "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
        "gamma": ['scale', 'auto'], 
        'epsilon': np.arange(0.0,1.0,0.1)
    }
    
    # Use GridSearchCV to search for the best hyperparameters
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    print("Best Hyperparameters:", grid_search.best_params_)
    print("Best Model:", grid_search.best_estimator_)
    average_result(grid_search, dataset)

def decisionTree_tuning(dataset): 
    print('-----------Decision Tree Regression------------')
    X_train, y_train = dataset_tuning(dataset)
    # Create a linear regression model
    model = DecisionTreeRegressor()
    
    # Define a set of hyperparameters to tune
    param_grid = {
        'splitter': ['best', 'random'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'min_weight_fraction_leaf': [0.0, 0.1, 0.2],
        'max_features': [None, 'sqrt', 'log2'],
        'random_state': [None, 42],  # You can set a specific random seed if needed
        'max_leaf_nodes': [None, 50, 100],
        'min_impurity_decrease': [0.0, 0.1, 0.2],
        'ccp_alpha': [0.0, 0.1, 0.2],
    }
    
    # Use GridSearchCV to search for the best hyperparameters
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    print("Best Hyperparameters:", grid_search.best_params_)
    print("Best Model:", grid_search.best_estimator_)
    average_result(grid_search, dataset)

def randomForest_tuning(dataset): 
    print('-----------Random Forest Regression------------')
    X_train, y_train = dataset_tuning(dataset)
    # Create a linear regression model
    model = RandomForestRegressor()
    
    # Define a set of hyperparameters to tune
    param_grid = {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'min_weight_fraction_leaf': [0.0, 0.1, 0.2],
        'max_features': [None, 'sqrt', 'log2'],
        'random_state': [None, 42],  # You can set a specific random seed if needed
        'max_leaf_nodes': [None, 50, 100],
        'min_impurity_decrease': [0.0, 0.1, 0.2],
        'ccp_alpha': [0.0, 0.1, 0.2],
    }
    
    # Use GridSearchCV to search for the best hyperparameters
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    print("Best Hyperparameters:", grid_search.best_params_)
    print("Best Model:", grid_search.best_estimator_)
    average_result(grid_search, dataset)

def main(args):
    # Tuning 
    linear_tuning(args.dataset)
    ridge_tuning(args.dataset)
    lasso_tuning(args.dataset)
    elasticNet_tuning(args.dataset)
    svm_tuning(args.dataset)
    decisionTree_tuning(args.dataset)
    randomForest_tuning(args.dataset)
    
def arg_parser():
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument("--dataset", 
                        default="california_housing", 
                        type=str),
    parser.add_argument("--alg", 
                        default="linear", 
                        type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parser()
    main(args)