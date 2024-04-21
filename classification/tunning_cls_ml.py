import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
import sys 
data_path = sys.argv[1]

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def tunning_cls_ml(data_path):
    data = pd.read_csv(data_path)

    # Xác định features (đặc trưng) và target (nhãn)
    data = data.to_numpy()
    X = data[:,1:-1].astype(np.float32)
    y = data[:,-1].astype(np.float32)

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Naive Bayes
    naive_bayes_params = {}  # Không có tham số cần tinh chỉnh cho Naive Bayes

    # Decision Tree
    decision_tree_params = {'max_depth': [None, 5, 10, 20], 'min_samples_split': [2, 5, 10]}

    # SVM
    svm_params = {'C': [0.1, 1, 10], 'kernel': ['linear', 'poly', 'rbf']}

    # Random Forest
    random_forest_params = {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10, 20], 'min_samples_split': [2, 5, 10]}

    # Tạo grid search để tìm các tham số tốt nhất cho mỗi mô hình
    naive_bayes_grid_search = GridSearchCV(GaussianNB(), naive_bayes_params, cv=5, scoring='accuracy')
    decision_tree_grid_search = GridSearchCV(DecisionTreeClassifier(), decision_tree_params, cv=5, scoring='accuracy')
    svm_grid_search = GridSearchCV(SVC(), svm_params, cv=5, scoring='accuracy')
    random_forest_grid_search = GridSearchCV(RandomForestClassifier(), random_forest_params, cv=5, scoring='accuracy')

    # Tuning cho Naive Bayes
    naive_bayes_grid_search.fit(X_train, y_train)
    best_naive_bayes_params = naive_bayes_grid_search.best_params_

    # Tuning cho Decision Tree
    decision_tree_grid_search.fit(X_train, y_train)
    best_decision_tree_params = decision_tree_grid_search.best_params_

    # Tuning cho SVM
    svm_grid_search.fit(X_train, y_train)
    best_svm_params = svm_grid_search.best_params_

    # Tuning cho Random Forest
    random_forest_grid_search.fit(X_train, y_train)
    best_random_forest_params = random_forest_grid_search.best_params_

    # Tạo mô hình Naive Bayes và Decision Tree với các tham số tốt nhất
    best_naive_bayes_model = GaussianNB()
    best_naive_bayes_model.set_params(**best_naive_bayes_params)

    best_decision_tree_model = DecisionTreeClassifier()
    best_decision_tree_model.set_params(**best_decision_tree_params)

    best_svm_model = SVC()
    best_svm_model.set_params(**best_svm_params)

    best_random_forest_model = RandomForestClassifier()
    best_random_forest_model.set_params(**best_random_forest_params)

    # Huấn luyện và đánh giá mô hình trên 20 lần với các random seed khác nhau
    n_runs = 20
    naive_bayes_accuracy_scores = []
    naive_bayes_f1_scores = []
    decision_tree_accuracy_scores = []
    decision_tree_f1_scores = []
    svm_accuracy_scores = []
    svm_f1_scores = []
    random_forest_accuracy_scores = []
    random_forest_f1_scores = []

    for seed in range(n_runs):
        # Phân chia dữ liệu với một random seed mới cho mỗi lần chạy
        X_train_seed, _, y_train_seed, _ = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)
        
        # Naive Bayes
        best_naive_bayes_model.fit(X_train_seed, y_train_seed)
        naive_bayes_pred = best_naive_bayes_model.predict(X_test)
        naive_bayes_accuracy = accuracy_score(y_test, naive_bayes_pred)
        naive_bayes_f1 = f1_score(y_test, naive_bayes_pred)
        naive_bayes_accuracy_scores.append(naive_bayes_accuracy)
        naive_bayes_f1_scores.append(naive_bayes_f1)
        
        # Decision Tree
        best_decision_tree_model.fit(X_train_seed, y_train_seed)
        decision_tree_pred = best_decision_tree_model.predict(X_test)
        decision_tree_accuracy = accuracy_score(y_test, decision_tree_pred)
        decision_tree_f1 = f1_score(y_test, decision_tree_pred)
        decision_tree_accuracy_scores.append(decision_tree_accuracy)
        decision_tree_f1_scores.append(decision_tree_f1)

        # SVM
        best_svm_model.fit(X_train_seed, y_train_seed)
        svm_pred = best_svm_model.predict(X_test)
        svm_accuracy = accuracy_score(y_test, svm_pred)
        svm_f1 = f1_score(y_test, svm_pred)
        svm_accuracy_scores.append(svm_accuracy)
        svm_f1_scores.append(svm_f1)

        # Random Forest
        best_random_forest_model.fit(X_train_seed, y_train_seed)
        random_forest_pred = best_random_forest_model.predict(X_test)
        random_forest_accuracy = accuracy_score(y_test, random_forest_pred)
        random_forest_f1 = f1_score(y_test, random_forest_pred)
        random_forest_accuracy_scores.append(random_forest_accuracy)
        random_forest_f1_scores.append(random_forest_f1)

    # In ra kết quả trung bình và độ lệch chuẩn của 20 lần chạy
    print("Naive Bayes:")
    print("Average Accuracy:", np.mean(naive_bayes_accuracy_scores))
    print("Accuracy Standard Deviation:", np.std(naive_bayes_accuracy_scores))
    print("Average F1-Score:", np.mean(naive_bayes_f1_scores))
    print("F1-Score Standard Deviation:", np.std(naive_bayes_f1_scores))
    print()

    print("Decision Tree:")
    print("Average Accuracy:", np.mean(decision_tree_accuracy_scores))
    print("Accuracy Standard Deviation:", np.std(decision_tree_accuracy_scores))
    print("Average F1-Score:", np.mean(decision_tree_f1_scores))
    print("F1-Score Standard Deviation:", np.std(decision_tree_f1_scores))
    print()

    print("SVM:")
    print("Average Accuracy:", np.mean(svm_accuracy_scores))
    print("Accuracy Standard Deviation:", np.std(svm_accuracy_scores))
    print("Average F1-Score:", np.mean(svm_f1_scores))
    print("F1-Score Standard Deviation:", np.std(svm_f1_scores))
    print()

    print("Random Forest:")
    print("Average Accuracy:", np.mean(random_forest_accuracy_scores))
    print("Accuracy Standard Deviation:", np.std(random_forest_accuracy_scores))
    print("Average F1-Score:", np.mean(random_forest_f1_scores))
    print("F1-Score Standard Deviation:", np.std(random_forest_f1_scores))

tunning_cls_ml(data_path)
