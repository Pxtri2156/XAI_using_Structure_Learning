import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import sys

data_path = sys.argv[1]

def tunning_machine_method(data_path):
  data = pd.read_csv(data_path)
  data = data.to_numpy()
  X = data[:,1:-1].astype(np.float32)
  y = data[:,-1].astype(np.float32)

  # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Khởi tạo các tham số tinh chỉnh cho từng mô hình
  ridge_params = {'alpha': [0.1, 1, 10]}
  lasso_params = {'alpha': [0.1, 1, 10]}
  elasticnet_params = {'alpha': [0.1, 1, 10], 'l1_ratio': [0.1, 0.5, 0.9]}
  svm_params = {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10]}
  decision_tree_params = {'max_depth': [None, 5, 10, 20]}
  random_forest_params = {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10, 20]}
  linear_regression_params = {}

  # Định nghĩa hàm đánh giá sử dụng mean squared error
  scorer = make_scorer(mean_squared_error, greater_is_better=False)

  # Số lần chạy với random seed khác nhau
  n_runs = 20

  # List để lưu kết quả
  results = []

  # Chạy vòng lặp để tinh chỉnh và đánh giá mỗi mô hình
  for params, model in zip([ridge_params, lasso_params, elasticnet_params, svm_params, decision_tree_params, random_forest_params, linear_regression_params], 
                          [Ridge(), Lasso(), ElasticNet(), SVR(), DecisionTreeRegressor(), RandomForestRegressor(), LinearRegression()]):
      # Tạo mô hình GridSearchCV để tìm kiếm tham số tốt nhất
      grid_search = GridSearchCV(model, params, scoring=scorer, cv=5)
      grid_search.fit(X_train, y_train)
      
      # Lấy tham số tốt nhất từ tuning
      best_params = grid_search.best_params_

      # Đánh giá mô hình tốt nhất trên tập kiểm tra với 20 lần chạy với các random seed khác nhau
      mse_scores = []
      for seed in range(n_runs):
          X_train_seed, _, y_train_seed, _ = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)
          best_model = model.set_params(**best_params)
          best_model.fit(X_train_seed, y_train_seed)
          y_pred_test = best_model.predict(X_test)
          mse_score = mean_squared_error(y_test, y_pred_test)
          mse_scores.append(mse_score)
      
      # Tính kết quả trung bình và độ lệch chuẩn trên 20 lần chạy
      test_mse_mean = np.mean(mse_scores)
      test_mse_std = np.std(mse_scores)
      
      # Lưu kết quả
      results.append({'model': model.__class__.__name__, 'best_params': best_params, 'test_mse_mean': test_mse_mean, 'test_mse_std': test_mse_std})

  # In ra kết quả
  for result in results:
      print("Model:", result['model'])
      print("Best Parameters:", result['best_params'])
      print("Mean Squared Error on Test Data (Mean):", result['test_mse_mean'])
      print("Mean Squared Error on Test Data (Standard Deviation):", result['test_mse_std'])
      print()
      
tunning_machine_method(data_path)