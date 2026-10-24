from flask import Flask, render_template, jsonify
import pandas as pd
import torch
import torch.nn as nn
import os
import psutil

app = Flask(__name__)

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

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/inference')
def inference():
    return render_template('inference.html')

@app.route('/orchestrator')
def orchestrator():
    return render_template('orchestrator.html')

@app.route('/api/resources')
def get_resources():
    # Option 2: Live IoT Resource Monitor
    return jsonify({
        "cpu": psutil.cpu_percent(interval=0.1),
        "ram": psutil.virtual_memory().percent
    })

@app.route('/api/intercept')
def intercept_traffic():
    # Option 1: Live Threat Detection Engine
    try:
        df = pd.read_csv("processed_data/test_global.csv")
        sample = df.sample(1)
        target_col = df.columns[-1]
        
        features_df = sample.drop(columns=[target_col])
        ground_truth = int(sample.iloc[0][target_col])
        
        X_tensor = torch.tensor(features_df.values, dtype=torch.float32)
        model = IotIDSModel(input_dim=X_tensor.shape[1])
        model.eval()
        
        with torch.no_grad():
            probability = model(X_tensor).item()
            
        prediction = 1 if probability >= 0.5 else 0
        confidence = probability if prediction == 1 else (1 - probability)
        
        feature_dict = features_df.iloc[0].to_dict()
        display_features = {k: round(v, 4) for k, v in list(feature_dict.items())[:5]}
        
        return jsonify({
            "status": "success",
            "features": display_features,
            "ground_truth": "Attack" if ground_truth == 1 else "Normal",
            "prediction": "Attack" if prediction == 1 else "Normal",
            "confidence": round(confidence * 100, 2)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)