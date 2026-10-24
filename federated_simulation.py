import os
# Force Ray to ignore Windows RAM limitations
os.environ["RAY_memory_monitor_refresh_ms"] = "0" 

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl
from typing import Dict, List, Tuple
from collections import OrderedDict

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
    target_col = df.columns[-1]
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1)

class IotClient(fl.client.NumPyClient):
    def __init__(self, cid):
        self.cid = cid
        client_file = f"processed_data/client_{int(cid) + 1}_data.csv"
        X_train, y_train = load_data(client_file)
        self.train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
        self.model = IotIDSModel(input_dim=X_train.shape[1])
        self.criterion = nn.BCELoss()

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.model.train()
        for _ in range(1):
            for data, target in self.train_loader:
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

def get_evaluate_fn():
    X_test, y_test = load_data("processed_data/test_global.csv")
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]):
        model = IotIDSModel(input_dim=X_test.shape[1])
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        
        criterion = nn.BCELoss()
        model.eval()
        with torch.no_grad():
            output = model(X_test)
            loss = criterion(output, y_test).item()
            predictions = (output >= 0.5).float()
            accuracy = (predictions == y_test).float().mean().item()
            
        print(f"\n[Round {server_round}] Global Model Eval - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}\n")
        return loss, {"accuracy": accuracy}
    return evaluate

def client_fn(cid: str) -> fl.client.Client:
    return IotClient(cid).to_client()

if __name__ == "__main__":
    print("Starting Federated Learning Simulation across 4 Non-IID IoT Nodes...")
    
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=4,
        min_available_clients=4,
        evaluate_fn=get_evaluate_fn()
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=4,
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
        client_resources={"num_cpus": 1},
        ray_init_args={
            "include_dashboard": False,
            "object_store_memory": 1024 * 1024 * 100
        }
    )