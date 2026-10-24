import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report

# Define the Neural Network for IoT IDS
class IotIDSModel(nn.Module):
    def __init__(self, input_dim):
        super(IotIDSModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.sigmoid(self.output(x))

def load_data(file_path):
    df = pd.read_csv(file_path)
    # The target column is the last one based on our preprocessing script
    target_col = df.columns[-1]
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1)

def train_baseline():
    print("1. Loading centralized training data...")
    X_train, y_train = load_data("processed_data/train_centralized.csv")
    X_test, y_test = load_data("processed_data/test_global.csv")

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    
    # Initialize model using the dynamic input dimension (49 features)
    model = IotIDSModel(input_dim=X_train.shape[1])
    criterion = nn.BCELoss() # Binary Cross Entropy for 0/1 classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("2. Training Centralized Baseline Model (5 Epochs)...")
    model.train()
    for epoch in range(5):
        epoch_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"   Epoch {epoch+1}/5 | Loss: {epoch_loss/len(train_loader):.4f}")

    print("\n3. Evaluating on Global Test Set...")
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        predictions = (predictions >= 0.5).float() # Convert probabilities to binary 0/1
        print(classification_report(y_test, predictions, target_names=["Normal", "Attack"], digits=4))

if __name__ == "__main__":
    train_baseline()