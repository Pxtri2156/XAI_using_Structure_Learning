import flask
from flask import jsonify, request
import numpy as np
from tqdm import tqdm
import pickle
import sys
sys.path.append('../')
from demo_site.api.utils import read_data, pred_cls, pred_reg, plot_classification_results, plot_dot_chart, top_values, boston_labels, laml_labels, plot_regres_results
from causal_xai.notears.xai_cls_demo import NotearsMLPClassification
from causal_xai.notears.xai_reg_demo import NotearsMLPRegression
# from causal_xai.notears.xai_reg import NotearsMLP

# Functions


# Run application
app = flask.Flask("API: XAI Classification")
app.config["DEBUG"] = False

# Top n contributors to target 
n = 3

# Loading classification models
cls_model_path = '/workspace/tripx/MCS/xai_causality/run/demo/cls/seed_0/cls_model.pickle'
cls_model = pickle.load(open(cls_model_path, 'rb'))
cls_W_est_path = f'/workspace/tripx/MCS/xai_causality/run/demo/cls/avg_seed/avg_dag_W_est_laml_cancer.csv'

cls_W_est = read_data(cls_W_est_path)
cls_top_n = top_values(cls_W_est,n)

# Loading regression models
reg_model_path = '/workspace/tripx/MCS/xai_causality/run/demo/reg/seed_0/reg_model.pickle'

# reg_mse_path = '/workspace/binhtlh/projects/causality/XAI_using_Structure_Learning/demo_site/models/reg/mse.pickle'
# reg_model_list = pickle.load(open(reg_model_path, 'rb'))
# reg_mse_list = pickle.load(open(reg_mse_path, 'rb'))
# reg_model = reg_model_list[np.argmin(reg_mse_list)]

reg_model = pickle.load(open(reg_model_path, 'rb'))

# print("reg_model : ", type(reg_model))
# input("Stop")
reg_W_est_path = f'/workspace/tripx/MCS/xai_causality/run/demo/reg/avg_seed/avg_dag_W_est_boston_housing_v2.csv'
reg_W_est = read_data(reg_W_est_path)
top_n_reg = top_values(reg_W_est, n)


# Savepath 
root_path = '/workspace/tripx/MCS/xai_causality/demo_site/datasets/uploaded_datasets/'
save_path = '/workspace/tripx/MCS/xai_causality/demo_site/graphs/uploaded_datasets/'

@app.route('/predict', methods=['POST', 'GET'])
def updateCurrentCode():
    global model

    if request.method == "POST":
        type_problem = request.json['type']
        X_filename = request.json['filename'] 
    else:
        type_problem = request.args.to_dict()['type'] 
        X_filename = request.args.to_dict()['filename'] 

    X_test = read_data(root_path + X_filename)

    if type_problem == 'cls': 
        pred, tar  = pred_cls(X_test, cls_model)
        plot_classification_results(pred, tar,save_path=save_path+'evaluation.png')
        print_result = 'Classification Task'
    elif type_problem == 'reg':
        pred, tar  = pred_reg(X_test, reg_model)
        print(pred)
        plot_regres_results(pred, tar,save_path=save_path+'evaluation.png')
        plot_dot_chart(X_test, pred, top_n_reg, boston_labels, save_path=save_path)
        print_result = 'Regression Task'
    else:
        print_result = 'Incorrect Type' 
        


    search_results = print_result + ':' + save_path + 'evaluation.png'
    print(search_results)
    response = flask.jsonify(search_results)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.success = True
    return response  

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8505, debug=False)