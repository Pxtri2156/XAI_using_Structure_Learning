import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
from sklearn.svm import SVC
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import random

data_path = "/workspace/tripx/MCS/xai_causality/dataset/adult_income/new_adult_income.csv"
data = pd.read_csv(data_path)
data = data.to_numpy()

for seed in range(15):
    random.seed(seed)
    np.random.seed(seed)
    np.random.shuffle(data)    
    X = data[:,1:-1]
    y = data[:,-1]
    X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)
    # Define model and training
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X_train, y_train)
    # Prediction 
    y_pred = clf.predict(X_val)
    from sklearn.metrics import classification_report
    target_names = ["<=50K", ">50K"]    
    cls_results = classification_report(y_val, y_pred, target_names=target_names)
    print(cls_results)