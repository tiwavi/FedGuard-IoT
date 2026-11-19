from flask import Flask, render_template, jsonify, request, redirect, url_for, flash, Response
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
import os
import psutil
import csv
import io

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
    simulation_logs = db.relationship('SimulationLog', backref='user', lazy=True)

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
    # If already logged in, redirect to dashboard
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
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
    # If already logged in, redirect to dashboard
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

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

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username')
        new_password = request.form.get('new_password')
        
        # Check if the user exists in the database
        user = User.query.filter_by(username=username).first()
        
        if user:
            # Hash the new password and save it
            user.password = generate_password_hash(new_password, method='pbkdf2:sha256')
            db.session.commit()
            flash('Password successfully reset. You can now log in with your new credentials.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Username not found in the system.', 'danger')
            
    return render_template('forgot_password.html')

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

@app.route('/api/sample_csv')
@login_required
def download_sample_csv():
    # Generates a 25-row sample CSV from your test set for easy demos
    try:
        df = pd.read_csv("processed_data/test_global.csv")
        sample_df = df.sample(min(25, len(df)))
        return Response(
            sample_df.to_csv(index=False),
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment; filename=sample_iot_traffic.csv'}
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/batch_scan', methods=['POST'])
@login_required
def batch_scan():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded."}), 400
    
    file = request.files['file']
    if file.filename == '' or not file.filename.endswith('.csv'):
        return jsonify({"status": "error", "message": "Please upload a valid .csv file."}), 400

    try:
        df = pd.read_csv(file)
        if df.empty:
            return jsonify({"status": "error", "message": "The uploaded CSV is empty."}), 400

        # Drop target label column if present (e.g., from test_global.csv)
        ref_df = pd.read_csv("processed_data/test_global.csv", nrows=1)
        target_col = ref_df.columns[-1]
        if target_col in df.columns:
            features_df = df.drop(columns=[target_col])
        else:
            features_df = df

        # Ensure numeric data only and cap at 500 rows for fast web response
        features_df = features_df.select_dtypes(include=['number']).head(500)
        X_tensor = torch.tensor(features_df.values, dtype=torch.float32)

        model = IotIDSModel(input_dim=X_tensor.shape[1])
        model.eval()

        with torch.no_grad():
            probs = model(X_tensor).squeeze().tolist()
            if isinstance(probs, float):
                probs = [probs]

        results = []
        attacks_count = 0
        normal_count = 0
        total_conf = 0.0

        for idx, prob in enumerate(probs):
            pred = 1 if prob >= 0.5 else 0
            conf = round((prob if pred == 1 else (1 - prob)) * 100, 2)
            label = "Attack" if pred == 1 else "Normal"
            total_conf += conf

            if pred == 1:
                attacks_count += 1
            else:
                normal_count += 1

            results.append({
                "row_id": idx + 1,
                "prediction": label,
                "confidence": conf
            })

            # Log up to 5 detected threats to SQLite Audit Logs
            if pred == 1 and attacks_count <= 5:
                db.session.add(InferenceLog(
                    user_id=current_user.id,
                    prediction="Attack",
                    confidence=conf
                ))

        db.session.commit()

        return jsonify({
            "status": "success",
            "total": len(results),
            "attacks": attacks_count,
            "normal": normal_count,
            "avg_confidence": round(total_conf / len(results), 2),
            "results": results
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

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

# --- AUDIT LOGS & ADMIN PANEL ROUTES ---
@app.route('/audit')
@login_required
def audit_panel():
    if current_user.role == 'admin':
        inference_logs = InferenceLog.query.order_by(InferenceLog.timestamp.desc()).limit(50).all()
        users = User.query.all()
    else:
        inference_logs = InferenceLog.query.filter_by(user_id=current_user.id).order_by(InferenceLog.timestamp.desc()).limit(50).all()
        users = []

    sim_logs = SimulationLog.query.order_by(SimulationLog.timestamp.desc()).limit(25).all()
    
    # Summary KPIs
    total_packets = InferenceLog.query.count() if current_user.role == 'admin' else InferenceLog.query.filter_by(user_id=current_user.id).count()
    total_attacks = InferenceLog.query.filter_by(prediction='Attack').count() if current_user.role == 'admin' else InferenceLog.query.filter_by(user_id=current_user.id, prediction='Attack').count()
    total_sims = SimulationLog.query.count()
    total_users = User.query.count()

    return render_template(
        'audit.html',
        inference_logs=inference_logs,
        sim_logs=sim_logs,
        users=users,
        stats={
            'packets': total_packets,
            'attacks': total_attacks,
            'sims': total_sims,
            'users': total_users
        }
    )

@app.route('/admin/toggle_role/<int:user_id>', methods=['POST'])
@login_required
def toggle_role(user_id):
    if current_user.role != 'admin':
        flash('Unauthorized: Admin privileges required.', 'danger')
        return redirect(url_for('audit_panel'))
    
    if user_id == current_user.id:
        flash('You cannot change your own role.', 'warning')
        return redirect(url_for('audit_panel'))

    user = User.query.get_or_404(user_id)
    user.role = 'admin' if user.role == 'user' else 'user'
    db.session.commit()
    flash(f"Updated {user.username}'s role to {user.role.title()}.", 'success')
    return redirect(url_for('audit_panel'))

@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
@login_required
def delete_user(user_id):
    if current_user.role != 'admin':
        flash('Unauthorized: Admin privileges required.', 'danger')
        return redirect(url_for('audit_panel'))

    if user_id == current_user.id:
        flash('You cannot delete your own active account.', 'warning')
        return redirect(url_for('audit_panel'))

    user = User.query.get_or_404(user_id)
    InferenceLog.query.filter_by(user_id=user.id).delete()
    SimulationLog.query.filter_by(user_id=user.id).delete()
    db.session.delete(user)
    db.session.commit()
    flash(f"Revoked access and removed user '{user.username}'.", 'info')
    return redirect(url_for('audit_panel'))

@app.route('/api/export_logs')
@login_required
def export_logs():
    logs = InferenceLog.query.order_by(InferenceLog.timestamp.desc()).all() if current_user.role == 'admin' else InferenceLog.query.filter_by(user_id=current_user.id).order_by(InferenceLog.timestamp.desc()).all()
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Log ID', 'Analyst', 'Classification', 'Confidence (%)', 'Timestamp (UTC)'])
    
    for log in logs:
        writer.writerow([log.id, log.user.username, log.prediction, log.confidence, log.timestamp.strftime('%Y-%m-%d %H:%M:%S')])
        
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=fedguard_audit_report.csv'}
    )

if __name__ == '__main__':
    with app.app_context():
        db.create_all() # Creates the SQLite database file automatically
    app.run(debug=True, port=5000)