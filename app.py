from flask import Flask, render_template, jsonify, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
import os
import psutil

app = Flask(__name__)
app.config['SECRET_KEY'] = 'super_secret_fedguard_key_2026'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///fedguard.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# --- DATABASE MODELS ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    role = db.Column(db.String(50), default='user') # 'admin' or 'user'
    inference_logs = db.relationship('InferenceLog', backref='user', lazy=True)

class InferenceLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    prediction = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class SimulationLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    rounds = db.Column(db.Integer)
    noise = db.Column(db.Float)
    epsilon = db.Column(db.Float)
    final_accuracy = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- PYTORCH MODEL ---
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

# --- AUTHENTICATION ROUTES ---
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        role = request.form.get('role', 'user')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists.', 'danger')
            return redirect(url_for('signup'))
            
        new_user = User(
            username=username, 
            password=generate_password_hash(password, method='pbkdf2:sha256'),
            role=role
        )
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)
        return redirect(url_for('dashboard'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Login failed. Check username and password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# --- CORE APPLICATION ROUTES ---
@app.route('/')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/inference')
@login_required
def inference():
    # Fetch recent logs for this user to display on the page
    history = InferenceLog.query.filter_by(user_id=current_user.id).order_by(InferenceLog.timestamp.desc()).limit(5).all()
    return render_template('inference.html', history=history)

@app.route('/api/resources')
@login_required
def get_resources():
    # Option 2: Live IoT Resource Monitor (Non-blocking)
    # Note: interval=None returns 0.0 on the very first call, but subsequent calls calculate the delta perfectly.
    return jsonify({
        "cpu": psutil.cpu_percent(interval=0.1),
        "ram": psutil.virtual_memory().percent
    })

@app.route('/orchestrator')
@login_required
def orchestrator():
    history = SimulationLog.query.order_by(SimulationLog.timestamp.desc()).limit(5).all()
    return render_template('orchestrator.html', history=history)

@app.route('/api/intercept')
@login_required
def intercept_traffic():
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
        
        # SAVE TO DATABASE
        log_entry = InferenceLog(
            user_id=current_user.id,
            prediction="Attack" if prediction == 1 else "Normal",
            confidence=round(confidence * 100, 2)
        )
        db.session.add(log_entry)
        db.session.commit()
        
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

@app.route('/api/save_simulation', methods=['POST'])
@login_required
def save_simulation():
    if current_user.role != 'admin':
        return jsonify({"status": "error", "message": "Unauthorized"}), 403
        
    data = request.json
    new_sim = SimulationLog(
        user_id=current_user.id,
        rounds=data['rounds'],
        noise=data['noise'],
        epsilon=data['epsilon'],
        final_accuracy=data['accuracy']
    )
    db.session.add(new_sim)
    db.session.commit()
    return jsonify({"status": "success"})

if __name__ == '__main__':
    with app.app_context():
        db.create_all() # Creates the SQLite database file automatically
    app.run(debug=True, port=5000)