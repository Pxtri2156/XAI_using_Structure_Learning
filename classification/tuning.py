import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
from sklearn.svm import SVC
from sklearn import svm
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import random
from sklearn.metrics import classification_report
import statistics as st

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def tuningSVM(args):
    # read data
    data = pd.read_csv(args.data_path)
    data = data.to_numpy()
    X = data[:,1:-2]
    y = data[:,-1]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Define model and training
    clf = svm.SVC()
    # clf = RandomForestClassifier()

    #Tuning svm parameter
    parameters = {
        "C": [1, 10],
        "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
        "gamma": ['scale', 'auto'],
        "probability": [True, False],
        "decision_function_shape": ['ovo', 'ovr'],
        "random_state": [0, 100]
        }

    #Tuning randome forest parameters
    # parameters = {
    #     "n_estimators": [10, 100],
    #     "criterion": ['gini', 'entropy'],
    #     "max_depth": [None, 10, 20],
    #     "min_samples_split": [2, 5],
    #     "min_samples_leaf": [1, 2],
    #     "bootstrap": [True, False],
    #     "random_state": [None, 42],
    #     "n_jobs": [1, -1],
    # }

    clf = GridSearchCV(clf, parameters, verbose=10)
    clf.fit(X_train, y_train)

    best_params = clf.best_params_
    best_model = clf.best_estimator_

    print(f'best_params: {best_params}')
    print(f'best_model: {best_model}')
    train(clf, args)
def tuningRF(args):
    # read data
    data = pd.read_csv(args.data_path)
    data = data.to_numpy()
    X = data[:,1:-2]
    y = data[:,-1]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Define model and training
    # clf = svm.SVC()
    clf = RandomForestClassifier()

    #Tuning svm parameter
    # parameters = {
    #     "C": [1, 10],
    #     "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
    #     "gamma": ['scale', 'auto'],
    #     "probability": [True, False],
    #     "decision_function_shape": ['ovo', 'ovr'],
    #     "random_state": [0, 100]
    #     }

    #Tuning randome forest parameters
    parameters = {
        "n_estimators": [10, 100],
        "criterion": ['gini', 'entropy'],
        "max_depth": [10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "n_jobs": [1, -1],
    }

    clf = GridSearchCV(clf, parameters, verbose=10)
    clf.fit(X_train, y_train)

    best_params = clf.best_params_
    best_model = clf.best_estimator_

    print(f'best_params: {best_params}')
    print(f'best_model: {best_model}')
    train(clf, args)

def train(best_model, args):
    # read data
    acc_results = []
    f1_results = []

    for seed in range(0,20):
        set_random_seed(seed)
        data = pd.read_csv(args.data_path)
        data = data.to_numpy()
        X = data[:,1:-2]
        y = data[:,-1]
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Define model and training
        if type(best_model.best_estimator_) == type((svm.SVC())): 
            model = svm.SVC()
            model.set_params(**best_model.best_params_)
        elif type(best_model.best_estimator_) == type(RandomForestClassifier()): 
            model = RandomForestClassifier()
            model.set_params(**best_model.best_params_)
        else: 
            print('Model training process problem!')

        model.fit(X_train, y_train)

        # Prediction 
        y_pred = model.predict(X_val)
        target_names = ["No", "Yes"]    
        cls_results = classification_report(y_val, y_pred, target_names=target_names)
        acc = cls_results.split('\n', 10)[5][39:43]
        f1 = cls_results.split('\n', 10)[6][39:43]
        acc_results.append(acc)
        f1_results.append(f1)
    #transform 
    acc_results = [float(x) for x in acc_results]
    f1_results = [float(x) for x in f1_results]

    #mean
    mean_acc = sum(acc_results)/len(acc_results)
    mean_f1 = sum(f1_results)/len(f1_results)

    #std
    std_acc = st.pstdev(acc_results)
    std_f1 = st.pstdev(f1_results)

    print(f'Accuracy: {mean_acc}+-{std_acc}')
    print(f'F1-Score: {mean_f1}+-{std_f1}')
        
def main(args):
    tuningSVM(args) 
    tuningRF(args)

def arg_parser():
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument("--data_path", 
                        default="/workspace/tripx/MCS/xai_causality/dataset/breast_cancer_uci.csv", 
                        type=str), 
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parser()
    main(args)
