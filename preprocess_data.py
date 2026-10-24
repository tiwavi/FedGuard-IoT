import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def clean_and_partition(file_path, num_clients=4, sample_size=80000):
    print(f"1. Loading dataset from {file_path}...")
    df = pd.read_csv(file_path, low_memory=False)
    print(f"   Original shape: {df.shape}")

    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"   Sampled down to: {df.shape}")

    print("2. Dropping identifiers and preventing data leakage...")
    cols_to_drop = [
        'frame.time', 'ip.src_host', 'ip.dst_host', 'arp.src.hw_mac', 
        'arp.dst.hw_mac', 'http.file_data', 'http.request.full_uri', 
        'icmp.transmit_timestamp', 'http.request.uri.query', 'tcp.options', 
        'tcp.payload', 'tcp.srcport', 'tcp.dstport', 'udp.port', 'mqtt.msg'
    ]
    
    attack_type_col = 'Attack_type' if 'Attack_type' in df.columns else None
    target_col = 'Attack_label' if 'Attack_label' in df.columns else df.columns[-1]

    existing_drops = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_drops)
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in [target_col, attack_type_col]]

    print("3. Encoding categorical columns & scaling features...")
    for col in feature_cols:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna().reset_index(drop=True)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[target_col]
    )

    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    print("4. Creating Non-IID partitions for IoT Clients...")
    if attack_type_col and attack_type_col in train_df.columns:
        train_df = train_df.sort_values(by=attack_type_col).reset_index(drop=True)
        train_df = train_df.drop(columns=[attack_type_col])
        test_df = test_df.drop(columns=[attack_type_col])
    else:
        train_df = train_df.sort_values(by=target_col).reset_index(drop=True)

    os.makedirs("processed_data", exist_ok=True)

    test_df.to_csv("processed_data/test_global.csv", index=False)
    print(f"   -> Saved processed_data/test_global.csv ({len(test_df)} rows)")

    train_df.to_csv("processed_data/train_centralized.csv", index=False)
    print(f"   -> Saved processed_data/train_centralized.csv ({len(train_df)} rows)")

    # Split into Non-IID client datasets using pure Pandas chunking
    chunk_size = len(train_df) // num_clients
    client_splits = [train_df.iloc[i * chunk_size : (i + 1) * chunk_size] for i in range(num_clients)]
    client_splits[-1] = train_df.iloc[(num_clients - 1) * chunk_size : ]

    for i, client_data in enumerate(client_splits):
        client_data = client_data.sample(frac=1, random_state=42).reset_index(drop=True)
        client_path = f"processed_data/client_{i+1}_data.csv"
        client_data.to_csv(client_path, index=False)
        
        attack_ratio = client_data[target_col].mean() * 100
        print(f"   -> Saved {client_path} | Rows: {len(client_data)} | Attack Traffic: {attack_ratio:.1f}%")

    print("\n✅ Dataset cleaning and Non-IID partitioning complete!")

if __name__ == "__main__":
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    if csv_files:
        clean_and_partition(csv_files[0], num_clients=4)
    else:
        print("❌ No CSV file found in the current directory!")