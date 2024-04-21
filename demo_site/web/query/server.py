# import numpy as np
# from PIL import Image
# from feature_extractor import FeatureExtractor
import math
import requests
from datetime import datetime
from flask import Flask, request, render_template, send_from_directory
import os
import sys
sys.path.append("./")


app = Flask(__name__)
# app.debug = True

UPLOAD_FOLDER = '/workspace/tripx/MCS/xai_causality/demo_site/datasets/uploaded_datasets/'
ALLOWED_EXTENSIONS = {'csv'}
white_img = "/workspace/tripx/MCS/xai_causality/demo_site/web/query/templates/assests/white.png"
graphsicon="/workspace/tripx/MCS/xai_causality/demo_site/web/query/templates/assests/graph.png"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/img/<path:filename>')
def download_file(filename):
    directory = "/".join(filename.split("/")[:-1])
    image_name = filename.split("/")[-1]
    print(directory)
    print(image_name)
    return send_from_directory(directory="/" + directory, path=image_name)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # #GET VALUES
        # # text = request.form['query']
        # image = request.form['fname']
        # url_text = "http://192.168.1.252:{}/predict?text={}&k={}".format(int(model_port),text,k)
        # result = requests.get(url_text).json() ######

        status_file = 'There is no file has been uploaded.'
        if 'file' not in request.files:
            status_file =  'No file part'
    
        file = request.files['file']

        if file.filename == '':
            status_file = 'No selected file'
        
        filename = ''
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            status_file = f'A file named {filename} has been uploaded.'
        else:
            status_file = 'Invalid file format'            
        
        type_problem = request.form['type']

        url_text = "http://192.168.1.252:8505/predict?type={}&filename={}".format(type_problem,filename)
        upload_result = requests.get(url_text).json().split(':')


        logo = '/workspace/binhtlh/projects/causality/XAI_using_Structure_Learning/demo_site/web/query/templates/assests/logo.png'
        home = '/workspace/binhtlh/projects/causality/XAI_using_Structure_Learning/demo_site/web/query/templates/assests/home.png'
        
        cali_icon = '/workspace/binhtlh/projects/causality/XAI_using_Structure_Learning/demo_site/web/query/templates/assests/california.png'
        laml_icon = '/workspace/binhtlh/projects/causality/XAI_using_Structure_Learning/demo_site/web/query/templates/assests/mutation.png'
        data_icon = '/workspace/binhtlh/projects/causality/XAI_using_Structure_Learning/demo_site/web/query/templates/assests/table.png'
        feature_icon = '/workspace/binhtlh/projects/causality/XAI_using_Structure_Learning/demo_site/web/query/templates/assests/feature.png'
        
        tabular = '/workspace/binhtlh/projects/causality/XAI_using_Structure_Learning/demo_site/web/query/templates/assests/datasets.jpg'
        
        graph_icon = '/workspace/binhtlh/projects/causality/XAI_using_Structure_Learning/demo_site/web/query/templates/assests/knowledge-graph.png'
        matrix_icon = '/workspace/binhtlh/projects/causality/XAI_using_Structure_Learning/demo_site/web/query/templates/assests/matrix.png'
        predict_icon = '/workspace/binhtlh/projects/causality/XAI_using_Structure_Learning/demo_site/web/query/templates/assests/network.png'
        
        reg_graph ='/workspace/tripx/MCS/xai_causality/run/demo/reg/avg_seed/vis_avg_dag_graph_boston_housing_v2.png'
        reg_matrix = '/workspace/tripx/MCS/xai_causality/run/demo/reg/avg_seed/mean_dag_array_boston_housing_v2.png'
        reg_perform = '/workspace/tripx/MCS/xai_causality/demo_site/graphs/reg/evaluation.png'

        cls_graph = '/workspace/tripx/MCS/xai_causality/run/demo/cls/avg_seed/vis_avg_dag_graph_laml_cancer.png'
        cls_matrix = '/workspace/tripx/MCS/xai_causality/run/demo/cls/avg_seed/mean_dag_array_laml_cancer.png'
        cls_perform = '/workspace/tripx/MCS/xai_causality/demo_site/graphs/cls/evaluation.png'

        top1 = '/workspace/tripx/MCS/xai_causality/demo_site/graphs/uploaded_datasets/top1.png'
        top2 = '/workspace/tripx/MCS/xai_causality/demo_site/graphs/uploaded_datasets/top2.png'
        top3 = '/workspace/tripx/MCS/xai_causality/demo_site/graphs/uploaded_datasets/top3.png'

        
        if type_problem == 'cls': 
            upload_graph = cls_graph
            upload_matrix = cls_matrix
            top1 = white_img
            top2 = white_img
            top3 = white_img
        else: 
            upload_graph = reg_graph
            upload_matrix = reg_matrix
        upload_perform = upload_result[1]

        return render_template('index.html',    logo_image=logo, 
                                                home_image=home, 
                                                cali_icon=cali_icon,
                                                laml_icon=laml_icon,
                                                data_icon=data_icon, 
                                                tabular = tabular, 
                                                graph_icon=graph_icon,
                                                matrix_icon=matrix_icon, 
                                                predict_icon=predict_icon,
                                                feature_icon=feature_icon, 
                                                reg_graph=reg_graph,
                                                reg_matrix=reg_matrix,
                                                reg_perform=reg_perform,
                                                cls_graph=cls_graph,
                                                cls_matrix=cls_matrix,
                                                cls_perform=cls_perform,
                                                status_file=status_file, 
                                                upload_graph=upload_graph, 
                                                upload_matrix=upload_matrix,
                                                upload_perform=upload_perform, 
                                                top1=top1, 
                                                top2=top2, 
                                                top3=top3,
                                                graphsicon=graphsicon)
    else:
        status_file = 'There is no file has been uploaded.'
        logo = '/workspace/binhtlh/projects/causality/XAI_using_Structure_Learning/demo_site/web/query/templates/assests/logo.png'
        home = '/workspace/binhtlh/projects/causality/XAI_using_Structure_Learning/demo_site/web/query/templates/assests/home.png'
        
        cali_icon = '/workspace/binhtlh/projects/causality/XAI_using_Structure_Learning/demo_site/web/query/templates/assests/california.png'
        laml_icon = '/workspace/binhtlh/projects/causality/XAI_using_Structure_Learning/demo_site/web/query/templates/assests/mutation.png'
        data_icon = '/workspace/binhtlh/projects/causality/XAI_using_Structure_Learning/demo_site/web/query/templates/assests/table.png'
        
        tabular = '/workspace/binhtlh/projects/causality/XAI_using_Structure_Learning/demo_site/web/query/templates/assests/datasets.jpg'
        
        graph_icon = '/workspace/binhtlh/projects/causality/XAI_using_Structure_Learning/demo_site/web/query/templates/assests/knowledge-graph.png'
        matrix_icon = '/workspace/binhtlh/projects/causality/XAI_using_Structure_Learning/demo_site/web/query/templates/assests/matrix.png'
        predict_icon = '/workspace/binhtlh/projects/causality/XAI_using_Structure_Learning/demo_site/web/query/templates/assests/network.png'
        feature_icon = '/workspace/binhtlh/projects/causality/XAI_using_Structure_Learning/demo_site/web/query/templates/assests/feature.png'

        reg_graph ='/workspace/tripx/MCS/xai_causality/run/demo/reg/avg_seed/vis_avg_dag_graph_boston_housing_v2.png'
        reg_matrix = '/workspace/tripx/MCS/xai_causality/run/demo/reg/avg_seed/mean_dag_array_boston_housing_v2.png'
        reg_perform = '/workspace/tripx/MCS/xai_causality/demo_site/graphs/reg/evaluation.png'

        cls_graph = '/workspace/tripx/MCS/xai_causality/run/demo/cls/avg_seed/vis_avg_dag_graph_laml_cancer.png'
        cls_matrix = '/workspace/tripx/MCS/xai_causality/run/demo/cls/avg_seed/mean_dag_array_laml_cancer.png'
        cls_perform = '/workspace/tripx/MCS/xai_causality/demo_site/graphs/cls/evaluation.png'
        
        upload_graph = '/workspace/tripx/MCS/xai_causality/demo_site/web/query/templates/assests/white.png'
        upload_matrix = '/workspace/tripx/MCS/xai_causality/demo_site/web/query/templates/assests/white.png'
        upload_perform = '/workspace/tripx/MCS/xai_causality/demo_site/web/query/templates/assests/white.png'
        return render_template('index.html',    logo_image=logo, 
                                                home_image=home, 
                                                cali_icon=cali_icon,
                                                laml_icon=laml_icon,
                                                data_icon=data_icon, 
                                                tabular = tabular, 
                                                graph_icon=graph_icon,
                                                matrix_icon=matrix_icon, 
                                                predict_icon=predict_icon,
                                                feature_icon=feature_icon, 
                                                reg_graph=reg_graph,
                                                reg_matrix=reg_matrix,
                                                reg_perform=reg_perform,
                                                cls_graph=cls_graph,
                                                cls_matrix=cls_matrix,
                                                cls_perform=cls_perform,
                                                status_file=status_file, 
                                                upload_graph=upload_graph, 
                                                upload_matrix=upload_matrix,
                                                upload_perform=upload_perform, 
                                                top1=upload_perform,
                                                top2=upload_perform,
                                                top3=upload_perform,
                                                graphsicon=graphsicon)
    

if __name__ == "__main__":
    app.run("0.0.0.0", port=8502)

    