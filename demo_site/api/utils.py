import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
import random
import os

laml_labels = {0:'0:ASXL1',1:'1:DNMT3A',2:'2:FLT3',3:'3:IDH1',4:'4:IDH2',5:'5:KIT',
                6:'6:KRAS',7:'7:NPM1',8:'8:PTPDC1',9:'9:PTPN11',10:'10:RUNX1',11:'11:SF3B1',
                12:'12:SMC1A',13:'13:TP53', 14:'14:U2AF1', 15:'15:WT1', 16:'16:LAML'}

boston_labels = {0:'0:LSTAT', 1:"1:INDUS", 2:":NOX", 3:"3:PTRATIO", 4:"4:RM", 5:"5:TAX", 
                  6:"6:DIS", 7:"7:AGE", 8:"8:MEDV"}

def read_data(data_path):
    X = pd.read_csv(data_path, header=None) 
    X = X.to_numpy().astype(float)
    return X

def pred_cls(X_test, model):
    X_torch = torch.from_numpy(X_test)
    X_hat = model(X_torch)
    cls_predict = X_hat[:,-1]
    cls_predict = cls_predict.cpu().detach().numpy() > 0.5 
    cls_predict = cls_predict.astype(int)
    target = X_test[:,-1]
    return cls_predict, target

def pred_reg(X_test, model):
    X_torch = torch.from_numpy(X_test)
    X_hat = model(X_torch)
    cls_predict = X_hat[:,-1]
    target = X_test[:,-1]
    return cls_predict.cpu().detach().numpy(), target

def top_values(W, n): 
    W = np.array(W)
    last_column = W[:, -1]  # Extract the last column
    sorted_indices = np.argsort(last_column)[::-1]  # Sort the indices in descending order
    top_n_indices = sorted_indices[:n]  # Get the top 3 indices
    return top_n_indices

def generate_pastel_color():
    """
    Generates a random pastel color.
    """
    # Generate random RGB values within a certain range to ensure pastel tones
    r = random.uniform(0.5, 1)
    g = random.uniform(0.5, 1)
    b = random.uniform(0.5, 1)
    return (r, g, b)

def delete_files_with_format(folder_path, format_string):
    """
    Deletes files in a folder whose filenames match the specified format string.

    Parameters:
    folder_path (str): Path to the folder containing the files.
    format_string (str): Format string used to match filenames.
    """
    for filename in os.listdir(folder_path):
        if filename.startswith(format_string):
            file_path = os.path.join(folder_path, filename)
            os.remove(file_path)
            print(f"Deleted file: {filename}")
            
    plt.close()

def plot_classification_results(cls_predict, target, save_path=None):
    """
    Plot classification results.

    Args:
    cls_predict (list): List of predicted values.
    target (list): List of target values.
    save_path (str, optional): Path to save the plot image. Default is None.
    """
    plt.figure(figsize=(8, 6))  # Set the figure size

    # Plot cls_predict values
    for i, val in enumerate(cls_predict):
        marker = 'o' if val == 0 else '^'  # Circle for 0, triangle for 1
        plt.scatter(i, val, color='#AFEEEE', marker=marker, s=100, label='Predict' if i == 0 else None)

    # Plot target values
    for i, val in enumerate(target):
        marker = 'o' if val == 0 else '^'  # Circle for 0, triangle for 1
        plt.scatter(i, val, color='#FFA07A', marker=marker, s=100, label='Target' if i == 0 else None, alpha=0.7)  # Pastel blue

    plt.title('Predicted vs. Target')  # Set the title
    plt.xlabel('Sample')  # Set the x-axis label
    plt.ylabel('Value')  # Set the y-axis label
    plt.legend()  # Show legend
    plt.grid(True)  # Show grid
    
    if save_path:
        plt.savefig(save_path)  # Save the plot image

def plot_regression_results(reg_predict, target, save_path=None):
    """
    Plot regression results.

    Args:
    predicted (list): List of predicted values.
    target (list): List of target values.
    save_path (str, optional): Path to save the plot image. Default is None.
    """
    plt.figure(figsize=(8, 6))  # Set the figure size

    # Plot predicted and target values
    plt.scatter(range(len(target)), target, label='Target', color='#FFA07A', alpha=0.7)  # Plot target values
    plt.scatter(range(len(reg_predict)), reg_predict, label='Predicted', color='#AFEEEE')  # Plot predicted values


    plt.title('Regression Results')  # Set the title
    plt.xlabel('Sample')  # Set the x-axis label
    plt.ylabel('Value')  # Set the y-axis label

    plt.legend()  # Show legend
    plt.grid(True)  # Show grid
    
    if save_path:
        plt.savefig(save_path)  # Save the plot image
    
    # plt.show()  # Show plot
    plt.close()

def plot_dot_chart(X, predict, indexes, labels, save_path=None):
    """
    Plots a dot chart based on two lists of x and y coordinates.

    Parameters:
    x_values (list): List of x coordinates.
    y_values (list): List of y coordinates.
    """
    delete_files_with_format(save_path,'top')
    for _, i in enumerate(indexes): 
        x_values = X[:,i]
        y_values = X[:,-1]

        #target
        plt.scatter(x_values, y_values, color='#FFA07A', label='Target')

        #predict
        plt.scatter(x_values, predict, color='#AFEEEE', label='Predict')
        
        plt.title(f'Effect of {labels[i]} feature on {labels[X.shape[1] - 1]}')
        plt.xlabel(f'{labels[i]} values')
        plt.ylabel(f'{labels[X.shape[1] - 1]} values')
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path+f'top{_+1}.png')  # Save the plot image

        plt.close()