import pandas as pd
import os

def verify_partitions():
    target_col = 'Attack_label'
    
    # Check Global Test Set
    if os.path.exists("processed_data/test_global.csv"):
        df_test = pd.read_csv("processed_data/test_global.csv")
        print(f"✅ Global Test Set: {df_test.shape[0]} rows, {df_test.shape[1]} features.")
    
    print("\n--- Client Non-IID Partitions ---")
    for i in range(1, 5):
        file_path = f"processed_data/client_{i}_data.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            
            # Identify the target column dynamically if 'Attack_label' is missing
            actual_target = target_col if target_col in df.columns else df.columns[-1]
            attack_ratio = df[actual_target].mean() * 100
            
            print(f"✅ Client {i} Data: {df.shape[0]} rows | Features: {df.shape[1]}")
            print(f"   -> Attack Traffic Ratio: {attack_ratio:.2f}%")
        else:
            print(f"❌ Missing file: {file_path}")

if __name__ == "__main__":
    verify_partitions()