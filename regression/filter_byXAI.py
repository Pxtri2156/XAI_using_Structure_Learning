import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import random
from sklearn import svm
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import statistics as st
from sklearn.metrics import mean_squared_error
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return  X_train, X_test, y_train, y_test 

def model_list_by_dataset(dataset_name): 
    if dataset_name == 'boston_housing': 
        model_list = [LinearRegression(fit_intercept=False),
                Ridge(alpha=1, positive=True), 
                Lasso(alpha=1, positive=True),
                ElasticNet(alpha=1, l1_ratio=0.0),
                svm.SVR(C=3, epsilon=0.0, gamma='auto'), 
                DecisionTreeRegressor(max_leaf_nodes=50, min_samples_leaf=2, min_samples_split=5, random_state=42, splitter='random'), 
                RandomForestRegressor()]
    elif dataset_name == 'cali_housing': 
        model_list = [LinearRegression(fit_intercept=False),
                Ridge(alpha=1, positive=True, fit_intercept=False), 
                Lasso(alpha=1, positive=True),
                ElasticNet(alpha=1, l1_ratio=0.0),
                svm.SVR(C=2), 
                DecisionTreeRegressor(max_leaf_nodes=50, min_samples_leaf=4,min_samples_split=5), 
                RandomForestRegressor()]
    elif dataset_name == 'diabetes': 
        model_list = [LinearRegression(),
                Ridge(alpha=1, positive=True), 
                Lasso(alpha=1, positive=True),
                ElasticNet(alpha=1, l1_ratio=0.9),
                svm.SVR(C=9, epsilon=0.6000000000000001, kernel='sigmoid'), 
                DecisionTreeRegressor(ccp_alpha=0.2, max_depth=20, max_leaf_nodes=50,
                      min_impurity_decrease=0.1, min_samples_leaf=2,
                      min_samples_split=10, min_weight_fraction_leaf=0.1,
                      splitter='random'), 
                RandomForestRegressor()]
    elif dataset_name == 'fried_man1': 
        model_list = [LinearRegression(positive=True),
                Ridge(alpha=1, positive=True), 
                Lasso(alpha=1, positive=True, fit_intercept=False),
                ElasticNet(alpha=1, fit_intercept=False, l1_ratio=0.9, positive=True),
                svm.SVR(C=10, epsilon=0.0), 
                DecisionTreeRegressor(ccp_alpha=0.1, max_depth=30, max_leaf_nodes=50,
                      min_samples_leaf=4, min_samples_split=5,
                      splitter='random'), 
                RandomForestRegressor()]
    else: 
        print('Model list problem!')
        
    return model_list

def partial_dependence_plot(X, y, model_list, save_pdp_path, selected_features):
    for model in model_list:
        features = range(0,X.shape[1])
        # Train the model
        model.fit(X, y)
        # Create PartialDependenceDisplay object

        pdp_display = PartialDependenceDisplay.from_estimator(model, X, features=features, feature_names=selected_features[:-1], kind='both')

        # Create the partial dependence plot
        fig, ax = plt.subplots(figsize=(10, 7))
        pdp_display.plot(ax=ax, line_kw={'alpha': 0.1}, 
                        pd_line_kw={'linewidth': 4, 'linestyle': '-', 'alpha': 0.8})
        plt.tight_layout()
        # Save the plot as an image
        file_name = str(model).split('(')[0] + '.png'
        plt.savefig(save_pdp_path+file_name)

def average_result(X, y, model_list):
    for model in model_list:
        mse = [] 
        for seed in range(0,20): 
            set_random_seed(seed)
            X_train, X_test, y_train, y_test = dataset_split(X, y)
      
            if type(model) == type((LinearRegression())): 
                model = LinearRegression()
            elif type(model) == type(Ridge()): 
                model = Ridge()
            elif type(model) == type(Lasso()): 
                model = Lasso()
            elif type(model) == type(ElasticNet()): 
                model = ElasticNet()
            elif type(model) == type(svm.SVR()): 
                model = svm.SVR()
            elif type(model) == type(DecisionTreeRegressor()): 
                model = DecisionTreeRegressor()
            elif type(model) == type(RandomForestRegressor()): 
                model = RandomForestRegressor()
            else: 
                print('Model training process problem!')
            model.set_params(**model.get_params())

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse.append(mean_squared_error(y_test, y_pred))
        print(model)
        #mean
        mean_mse = sum(mse)/len(mse)
        #std
        std_mse = st.pstdev(mse) 
        print(f'Average: {mean_mse}+-{std_mse}')
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
    average_result(X, y, model_list)
    # partial_dependence_plot(X, y,model_list, args.save_pdp_path, selected_features)


def arg_parser():
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument("--dataset", 
                        default="boston_housing", 
                        type=str),
    parser.add_argument("--graph_path", 
                        default="/workspace/tripx/MCS/xai_causality/run_v2/boston_housing/avg_seed/avg_dag_W_est.csv", 
                        type=str)
    parser.add_argument("--data_path", 
                        default="/workspace/tripx/MCS/xai_causality/dataset/scaled_boston_housing.csv", 
                        type=str)
    parser.add_argument("--save_pdp_path", 
                        default="/workspace/binhtlh/projects/causality/XAI_using_Structure_Learning/regression/run/pdp/boston_housing/", 
                        type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parser()
    main(args)