import flask
from flask import jsonify, request
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import sys
sys.path.append('../')
from demo_site.api.utils import read_data, pred_cls, pred_reg, plot_classification_results, plot_regression_results, plot_dot_chart, top_values, boston_labels, laml_labels
from causal_xai.notears.xai_cls_demo import NotearsMLPClassification
from causal_xai.notears.xai_reg_demo import NotearsMLPRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score


# reg_model_path = '/workspace/tripx/MCS/xai_causality/run/demo/reg/seed_0/reg_model.pickle'
# reg_model = pickle.load(open(reg_model_path, 'rb'))
# X_test = read_data('/workspace/tripx/MCS/xai_causality/demo_site/datasets/uploaded_datasets/X_test.csv')
# print(X_test)
# pred, tar  = pred_reg(X_test, reg_model)
# print('pred: ', pred)
# test_loss = 0.5 / tar.shape[0] * np.sum((pred - tar) ** 2)
# print('test_loss: ', test_loss)


# # Vẽ Scatter plot
# plt.figure(figsize=(8, 6))
# plt.scatter(tar, pred, color='blue', label='Pred vs Target')
# plt.plot(np.unique(tar), np.poly1d(np.polyfit(tar, pred, 1))(np.unique(tar)), color='red', linestyle='--', label='Best Fit Line')
# plt.title('Predicted vs Target')
# plt.xlabel('Target')
# plt.ylabel('Predicted')
# plt.legend()
# plt.grid(True)
# plt.savefig("test_reg.png")

def evaluation_cls(model, X_test):
    target_names = ["No", "Yes"]
    predict, target = pred_cls(X_test, model)
    f1 = f1_score(target, predict, average='macro') 
    acc = accuracy_score(target, predict)
    # cls_results = classification_report(target, predict, target_names=target_names)
    return acc, f1

cls_model_path = '/workspace/tripx/MCS/xai_causality/run/demo/cls/seed_0/cls_model.pickle'
cls_model = pickle.load(open(cls_model_path, 'rb'))
cls_W_est_path = f'/workspace/tripx/MCS/xai_causality/run/demo/cls/avg_seed/avg_dag_W_est_laml_cancer.csv'


X_test = read_data('/workspace/tripx/MCS/xai_causality/run/demo/cls/seed_0/X_test.csv')
# print(X_test)

# Loading  models
acc, f1 = evaluation_cls(cls_model, X_test)
print(acc, f1)