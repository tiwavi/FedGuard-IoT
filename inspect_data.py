import pandas as pd
import os

# List CSV files in your current folder to make sure the name matches
csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
print("Found CSV files:", csv_files)

# Replace with the exact name of the CSV file you moved into the folder
file_name = csv_files[0] 

print(f"\nLoading first 5 rows of {file_name}...")
df = pd.read_csv(file_name, nrows=5, low_memory=False)

print("\n--- Column Names ---")
print(list(df.columns))

print("\n--- Last 5 Columns (Target Labels) ---")
print(df.iloc[:, -5:].head())