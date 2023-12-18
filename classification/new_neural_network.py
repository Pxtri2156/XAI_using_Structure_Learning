import torch
import numpy as np
import pandas as pd
import random
import json
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Define your model class
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Define the layers of your neural network
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
result_path = '/workspace/tripx/MCS/xai_causality/classification/run/neural_network/breast_cancer_uci.json'
# data_path = '/dataset/PANCAN/STAD_gene_filter.csv'

data_path = '/workspace/tripx/MCS/xai_causality/dataset/norm_breast_cancer_uci.csv'
data = pd.read_csv(data_path)
data = data.to_numpy()

# Define hyperparameters to tune
hyperparameters = {
    'epochs': [10, 20, 30],
    'batch_size': [32, 64, 128],
    'learning_rate': [0.001, 0.01, 0.1],
    'optimizer': [optim.SGD, optim.Adam, optim.RMSprop],
    'random_seeds': 20
}

best_f1_score = 0
best_params = {}
best_result = {}

for num_epochs in hyperparameters['epochs']:
    print(f"Num of epochs {num_epochs}")
    for batch_size in hyperparameters['batch_size']:
        print(f"Batch size: {batch_size}")
        for learning_rate in hyperparameters['learning_rate']:
            print(f'Learning rate: {learning_rate}')
            for optimizer_class in hyperparameters['optimizer']:
                print(f"Optimization: {optimizer_class}")
                acc_list = []
                f1_list = []
                for seed in range(hyperparameters['random_seeds']):
                    print(f"\t seed: {seed}")
                    random.seed(seed)
                    np.random.seed(seed)
                    np.random.shuffle(data)
                    X = data[:,1:-1]
                    y = data[:,-1]
                    # Load your data, assuming you have X and y for features and labels
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    # Move your data to the GPU if available
                    X_train, X_test, y_train, y_test = (
                        torch.Tensor(X_train).to(device),
                        torch.Tensor(X_test).to(device),
                        torch.Tensor(y_train).to(device),
                        torch.Tensor(y_test).to(device),
                    )
                    model = SimpleNN(input_size=X_train.shape[1], hidden_size=10)
                    model.to(device)  # Move the model to the GPU
                    optimizer = optimizer_class(model.parameters(), lr=learning_rate)
                    criterion = nn.BCELoss()

                    train_dataset = TensorDataset(X_train, y_train)
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                    for epoch in range(num_epochs):
                        model.train()
                        for inputs, labels in train_loader:
                            optimizer.zero_grad()
                            outputs = model(inputs)
                            loss = criterion(outputs, labels.view(-1, 1))
                            loss.backward()
                            optimizer.step()

                    model.eval()
                    with torch.no_grad():
                        predictions = model(X_test)
                        predicted_labels = (predictions > 0.5).float()

                    accuracy = accuracy_score(y_test.cpu().numpy(), predicted_labels.cpu().numpy())
                    f1 = f1_score(y_test.cpu().numpy(), predicted_labels.cpu().numpy())
                    print(f'\t accuracy {accuracy}')
                    print(f'\t f1 score {f1}')
                    f1_list.append(f1)
                    acc_list.append(accuracy)
                acc_list = np.array(acc_list)
                f1_list =  np.array(f1_list)
                
                dict_results = { 'accuracy': {'mean': np.mean(acc_list), 
                                                    'std': np.std(acc_list)},
                                    'f1_score': {'mean': np.mean(f1_list),
                                                'std': np.std(f1_list)}
                }  
                
                if dict_results['f1_score']['mean'] > best_f1_score:
                    best_f1_score =dict_results['f1_score']['mean']
                    best_result = dict_results
                    best_params['epochs'] = num_epochs
                    best_params['batch_size'] = batch_size
                    best_params['learning_rate'] = learning_rate
                    best_params['optimizer'] = optimizer_class

json_object = json.dumps(best_result, indent=4)
# Writing to sample.json
with open(result_path, 'w') as outfile:
    outfile.write(json_object)
    
print("Best Hyperparameters:")
print(best_params)
print("Best Test Accuracy:", best_f1_score)
print("Best avg result: ", best_result)
