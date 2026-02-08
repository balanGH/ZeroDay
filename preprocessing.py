import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

seq_len = 20
max_sequences = 100000

# seq=20, 100k → best balance
# seq=25, 200k → over-smoothed, expensive
# seq=25, 100k → still worse than seq=20

scaler = MinMaxScaler()
protocol_encoder = LabelEncoder()

def load_data(csv_path, dataset_type='auto'):
    print("▶ Loading IoT-23 dataset...")
    
    # Load IoT-23 data
    df = pd.read_csv(csv_path, nrows=200000)
    
    print(f"Columns: {df.columns.tolist()}")
    print(f"Initial shape: {df.shape}")
    
    # Map IoT-23 columns to expected format
    column_mapping = {
        'ts': 'timestamp',
        'id.orig_h': 'src_ip',
        'id.resp_h': 'dst_ip',
        'id.orig_p': 'src_port', 
        'id.resp_p': 'dst_port',
        'proto': 'protocol',
        'duration': 'duration',
        'orig_pkts': 'fwd_pkts',
        'resp_pkts': 'bwd_pkts',
        'orig_bytes': 'fwd_bytes',
        'resp_bytes': 'bwd_bytes',
        'label': 'label'
    }
    
    # Rename columns that exist
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)
    
    # ----- FIX: Handle string labels properly -----
    print("\n▶ Processing labels...")
    
    # Check if label column exists
    if 'label' not in df.columns:
        print("▶ 'label' column not found, checking for alternative label columns...")
        # Look for any column with 'label' in its name
        label_cols = [col for col in df.columns if 'label' in col.lower()]
        if label_cols:
            print(f"  Found label columns: {label_cols}")
            df['label'] = df[label_cols[0]]
        else:
            # Create synthetic labels for testing
            print("  No label column found, creating synthetic labels...")
            df['label'] = np.random.choice(['Benign', 'Malicious'], size=len(df), p=[0.7, 0.3])
    
    # Print sample labels to understand format
    print(f"  Sample labels: {df['label'].unique()[:10]}")
    print(f"  Label types: {df['label'].apply(type).unique()}")
    
    # Convert labels to binary (0 for benign, 1 for malicious)
    def convert_label(label):
        if isinstance(label, str):
            label_lower = str(label).lower()
            # Common benign labels in IoT-23
            benign_indicators = ['benign', 'normal', 'legitimate', '-', 'none', 'background']
            malicious_indicators = ['malicious', 'attack', 'ddos', 'c&c', 'botnet', 'scan', 'spam']
            
            # Check if it's benign
            if any(indicator in label_lower for indicator in benign_indicators):
                return 0
            # Check if it's malicious
            elif any(indicator in label_lower for indicator in malicious_indicators):
                return 1
            # Default to benign if unsure
            else:
                return 0
        else:
            # If it's numeric, convert to binary
            try:
                return int(label) % 2
            except:
                return 0
    
    df['label_binary'] = df['label'].apply(convert_label)
    
    print(f"  Binary label distribution: {df['label_binary'].value_counts()}")
    print(f"  Malicious rate: {df['label_binary'].mean()*100:.2f}%")
    
    # Select and prepare features
    feature_cols = []
    
    # Check each potential feature column
    potential_features = {
        'src_port': 'numeric',
        'dst_port': 'numeric', 
        'protocol': 'categorical',
        'duration': 'numeric',
        'fwd_pkts': 'numeric',
        'bwd_pkts': 'numeric',
        'fwd_bytes': 'numeric',
        'bwd_bytes': 'numeric'
    }
    
    print("\n▶ Preparing features...")
    for col, col_type in potential_features.items():
        if col in df.columns:
            if col_type == 'numeric':
                # Convert to numeric, fill NaN with 0
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(np.float32)
                feature_cols.append(col)
                print(f"  Added numeric feature: {col}")
            elif col_type == 'categorical':
                # Convert protocol strings to numeric codes
                try:
                    df[col] = protocol_encoder.fit_transform(df[col].astype(str))
                    df[col] = df[col].astype(np.float32)
                    feature_cols.append(col)
                    print(f"  Added categorical feature: {col} (mapped to {len(protocol_encoder.classes_)} classes)")
                except:
                    print(f"  Warning: Could not encode {col}, skipping")
    
    print(f"\n▶ Final features: {feature_cols}")
    
    # Handle missing values
    df[feature_cols] = df[feature_cols].fillna(0)
    
    # Scale numeric features
    if feature_cols:
        print("▶ Scaling features...")
        # Only scale if we have features
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        df[feature_cols] = df[feature_cols].astype(np.float32)
    
    # Sort by timestamp if available
    if 'timestamp' in df.columns:
        print("▶ Sorting by timestamp...")
        df = df.sort_values('timestamp')
    
    # Create sequences
    print("▶ Creating sequences...")
    X_seq = []
    y_seq = []
    
    data_array = df[feature_cols].values.astype(np.float32)
    labels_array = df['label_binary'].values.astype(np.int8)
    
    total_sequences = min(len(df) - seq_len + 1, max_sequences)
    
    print(f"▶ Creating {total_sequences} sequences of length {seq_len}...")
    
    for i in range(total_sequences):
        X_seq.append(data_array[i:i + seq_len])
        # Use max label in sequence (if any packet is malicious, sequence is malicious)
        y_seq.append(labels_array[i:i + seq_len].max())
        
        if i % 50000 == 0 and i > 0:
            print(f"  Created {i}/{total_sequences} sequences")
    
    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.int8)
    
    print(f"\n✅ IoT-23 sequences created: {X_seq.shape}")
    unique_labels, counts = np.unique(y_seq, return_counts=True)
    print(f"▶ Label distribution:")
    for label, count in zip(unique_labels, counts):
        label_name = "Benign" if label == 0 else "Malicious"
        print(f"  {label_name} ({label}): {count} samples ({count/len(y_seq)*100:.2f}%)")
    
    return X_seq, y_seq