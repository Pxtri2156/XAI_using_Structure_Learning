import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random 
import sys

data_path = sys.argv[1]
print('path: ', data_path)
# Generate some random data for demonstration purposes
# Replace this with your actual data loading code
# data_path = '/workspace/tripx/MCS/xai_causality/dataset/scaled_boston_housing.csv'
# data_path = '/workspace/tripx/MCS/xai_causality/dataset/adult_income/new_adult_income.csv'
data = pd.read_csv(data_path)
data = data.to_numpy()


# Define the neural network model
class RegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Hyperparameter tuning grid
learning_rates = [0.001, 0.01, 0.1]
epochs_list = [50, 100, 200]
optimizers_list = ['Adam', 'SGD']

best_model = None
mean_best_loss = float('inf')
best_results = None 

for lr in learning_rates:
    for epochs in epochs_list:
        for optimizer_name in optimizers_list:
            test_loss_list = []
            for seed in range(20):
                np.random.seed(seed)
                random.seed(seed)
                np.random.shuffle(data)
                X = data[:,1:-1].astype(np.float32)
                y = data[:,-1].astype(np.float32)
                
                # Split the data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Convert data to PyTorch tensors
                X_train_tensor = torch.from_numpy(X_train)
                y_train_tensor = torch.from_numpy(y_train)
                X_test_tensor = torch.from_numpy(X_test)
                y_test_tensor = torch.from_numpy(y_test)

                # Create DataLoader for training and testing sets
                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                # Instantiate the model
                model = RegressionModel(input_size=X.shape[1], hidden_size=10, output_size=1)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)

                # Choose the optimizer based on the hyperparameter
                if optimizer_name == 'Adam':
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                elif optimizer_name == 'SGD':
                    optimizer = optim.SGD(model.parameters(), lr=lr)

                # Training loop
                for epoch in range(epochs):
                    model.train()
                    for inputs, targets in train_loader:
                        inputs, targets = inputs.to(device), targets.to(device)

                        # Forward pass
                        outputs = model(inputs)
                        loss = nn.MSELoss()(outputs, targets)

                        # Backward pass and optimization
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # Evaluate on the test set
                model.eval()
                with torch.no_grad():
                    X_test_tensor = X_test_tensor.to(device)
                    y_pred = model(X_test_tensor)

                # Convert predictions and ground truth to numpy arrays
                y_pred = y_pred.cpu().numpy()
                y_test_np = y_test_tensor.numpy()

                # Calculate test set performance metrics (e.g., Mean Squared Error)
                mse = np.mean((y_pred - y_test_np)**2)
                test_loss_list.append(mse)
                
                # Print the results
                # print(f"Learning Rate: {lr}, Epochs: {epochs}, Optimizer: {optimizer_name}, Loss: {mse}")
                
            test_loss_list = np.array(test_loss_list)
            dict_results = { 'test_loss': {'mean': np.mean(test_loss_list), 
                                                'std': np.std(test_loss_list)},} 
            print(f"Learning Rate: {lr}, Epochs: {epochs}, Optimizer: {optimizer_name}, Result: {dict_results}")
            # Update the best model if the current model is better
            if dict_results['test_loss']['mean'] < mean_best_loss:
                mean_best_loss = dict_results['test_loss']['mean'] 
                best_model = model
                best_results = dict_results

print(f"Best Model Parameters - Learning Rate: {lr}, Epochs: {epochs}, Optimizer: {optimizer_name}, Loss: {dict_results}")
