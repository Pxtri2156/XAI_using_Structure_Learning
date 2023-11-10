import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import json
import random 

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

data_path = '/dataset/PANCAN/LAML_gene_filter.csv'
# data_path = '/workspace/tripx/MCS/xai_causality/dataset/adult_income/new_adult_income.csv'
data = pd.read_csv(data_path)
data = data.to_numpy()

for seed in range(15):
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
    # Define the best hyperparameters
    num_epochs = 10
    batch_size = 32
    learning_rate = 0.01
    optimizer_class = optim.Adam

    model = SimpleNN(input_size=X_train.shape[1], hidden_size=10)
    model.to(device)  # Move the model to the GPU
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    fc1_list = []
    acc_list = []
    
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
    print("Test Accuracy:", accuracy)
    print("F1 Score:", f1)
    fc1_list.append(f1)
    acc_list.append(accuracy)
    
acc_list = np.array(acc_list)
f1_list =  np.array(fc1_list) 
final_cls_results = { 'accuracy': {'mean': np.mean(acc_list), 
                                    'std': np.std(acc_list)},
                    'f1_score': {'mean': np.mean(f1_list),
                                'std': np.std(f1_list)}} 

json_object = json.dumps(final_cls_results, indent=4)
# Writing to sample.json
root_path = '/workspace/tripx/MCS/xai_causality/classification/run/neural_network/'
with open(root_path + "laml_final_cls.json", "w") as outfile:
    outfile.write(json_object)