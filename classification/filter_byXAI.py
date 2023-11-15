import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import random
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import statistics as st
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def loadDAG(graph_path): 
    dag = pd.read_csv(graph_path, index_col=None, header=None)
    dag = dag.to_numpy() 
    last_column = dag[:,-1]
    feature_selection = np.nonzero(last_column)[0]
    print(last_column)
    print(feature_selection)
    return feature_selection

def loadData(data_path): 
    data = pd.read_csv(data_path)
    return data

def filterData(origin_data, feature_selection):
    column_name = list(origin_data.columns)
    if 'Unnamed: 0' in column_name: 
        column_name.remove('Unnamed: 0')
    filter_column_name = [column_name[i] for i in feature_selection]
    filter_column_name.append(column_name[-1])
    data = origin_data[filter_column_name]
    data = data.to_numpy()
    X = data[:,0:-1]
    y = data[:,-1]
    return X, y, filter_column_name

def dataset_split(X, y): 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return  X_train, X_test, y_train, y_test 

def model_list_by_dataset(dataset_name): 
    if dataset_name == 'adult_income': 
        model_list = [svm.SVC(),
                RandomForestClassifier()]
    elif dataset_name == 'breast_cancer': 
        model_list = [RandomForestClassifier()]
    elif dataset_name == 'laml_cancer': 
        model_list = [svm.SVC(),
                RandomForestClassifier()]
    elif dataset_name == 'ov_cancer': 
        model_list = [svm.SVC(),
                RandomForestClassifier()]
    else: 
        print('Model list problem!')
        
    return model_list

def partial_dependence_plot(X, y, model_list, save_pdp_path, selected_features):
    for model in model_list:
        features = range(0,X.shape[1])
        # Train the model
        model.fit(X, y)
        # Create PartialDependenceDisplay object

        if args.dataset == 'breast_cancer':
            pdp_display = PartialDependenceDisplay.from_estimator(model, X, features=features, categorical_features=features, feature_names=selected_features[:-1], kind='average')
        else:
            pdp_display = PartialDependenceDisplay.from_estimator(model, X, features=features, feature_names=selected_features[:-1], kind='both')

        # Create the partial dependence plot
        
        fig, ax = plt.subplots(figsize=(10, 8))
        pdp_display.plot(ax=ax, line_kw={'alpha': 0.1}, 
                        pd_line_kw={'linewidth': 4, 'linestyle': '-', 'alpha': 0.8})
        plt.tight_layout()
        plt.legend().remove()
        # Save the plot as an image
        file_name = str(model).split('(')[0] + '.png'
        plt.savefig(save_pdp_path+file_name)

def average_result(X, y, model_list):
    for model in model_list:
        acc_results = [] 
        f1_results = []
        for seed in range(0,15): 
            set_random_seed(seed)
            X_train, X_test, y_train, y_test = dataset_split(X, y)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            target_names = ["No", "Yes"]    
            cls_results = classification_report(y_test, y_pred, target_names=target_names)
            acc = cls_results.split('\n', 10)[5][39:43]
            f1 = cls_results.split('\n', 10)[6][39:43]
            acc_results.append(float(acc))
            f1_results.append(float(f1))
        print(model)
        #mean
        mean_acc_results = round(sum(acc_results)/len(acc_results),2) 
        mean_f1_results = round(sum(f1_results)/len(f1_results),2)
        #std
        std_acc_results= round(st.pstdev(acc_results),5) 
        std_f1_results = round(st.pstdev(f1_results),5) 

        print(f'ACC Average: {mean_acc_results}+-{std_acc_results}')
        print(f'F1 Average: {mean_f1_results}+-{std_f1_results}')
        print('---------------------------------------')


def main(args):
    # filter dataset by XAI
    feature_selection = loadDAG(args.graph_path)
    original_data = loadData(args.data_path)
    model_list = model_list_by_dataset(args.dataset)
    X, y, selected_features = filterData(original_data, feature_selection)
    print(f'INFORMATION: {args.dataset.upper()}')
    print(f'ORIGINAL FEATURES: {list(original_data.columns)[1:]}')
    print(f'SELECTED FEATURES: {selected_features}')
    # average_result(X, y, model_list)
    partial_dependence_plot(X, y,model_list, args.save_pdp_path, selected_features)


def arg_parser():
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument("--dataset", 
                        default="breast_cancer", 
                        type=str),
    parser.add_argument("--graph_path", 
                        default="/workspace/tripx/MCS/xai_causality/run_v2/breast_cancer/avg_seed/avg_dag_W_est.csv", 
                        type=str)
    parser.add_argument("--data_path", 
                        default="/workspace/tripx/MCS/xai_causality/dataset/breast_cancer_uci.csv", 
                        type=str)
    parser.add_argument("--save_pdp_path", 
                        default="/workspace/binhtlh/projects/causality/XAI_using_Structure_Learning/classification/pdp/breast_cancer/", 
                        type=str)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parser()
    main(args)