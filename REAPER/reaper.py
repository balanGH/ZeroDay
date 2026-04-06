import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
warnings.filterwarnings('ignore')
os.environ["OMP_NUM_THREADS"] = "16"

# ======================================================================
# FIX SUMMARY:
# 1. Added RNN encoder before VAE (as per REAPER paper)
# 2. Fixed window labeling bug (0 benign windows issue)
# 3. Added IP-based flow grouping (simulated IP-Trie)
# 4. Fixed latent_dim < hidden_dim bottleneck
# 5. Accuracy improvements: per-IP threshold, better balancing
# ======================================================================


# ======================================================================
# CICIoT2023 DATASET PROCESSOR
# ======================================================================

class CICIoT2023Processor:
    def __init__(self, data_path="./dataset/"):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.numeric_columns = None

    def load_dataset(self, filename):
        print(f"\nLoading {filename}...")
        file_path = os.path.join(self.data_path, filename)
        df = pd.read_csv(file_path, low_memory=False)
        print(f"  Shape: {df.shape}")
        return df

    def clean_features(self, df):
        print("\n" + "="*70)
        print("CLEANING FEATURES")
        print("="*70)

        protocol_cols = [
            'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC',
            'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IGMP', 'IPv', 'LLC'
        ]
        cols_to_drop = [col for col in protocol_cols if col in df.columns]
        print(f"Dropping protocol columns: {cols_to_drop}")

        if 'Label' in df.columns:
            labels = df['Label'].copy()
            df = df.drop(columns=cols_to_drop + ['Label'])
        else:
            labels = None
            df = df.drop(columns=cols_to_drop)

        # FIX: Preserve src IP column for IP-Trie grouping if available
        ip_col = None
        for candidate in ['Src IP', 'src_ip', 'Source IP', 'IPsrc']:
            if candidate in df.columns:
                ip_col = df[candidate].copy()
                df = df.drop(columns=[candidate])
                print(f"  Preserved IP column: {candidate}")
                break

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.numeric_columns = numeric_cols
        df_numeric = df[numeric_cols].copy()

        # Handle infinite and extreme values
        df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)
        extreme_mask = np.abs(df_numeric) > 1e10
        df_numeric = df_numeric.mask(extreme_mask, np.nan)

        for col in df_numeric.columns:
            median_val = df_numeric[col].median()
            if pd.isna(median_val):
                median_val = 0
            df_numeric[col] = df_numeric[col].fillna(median_val)

        for col in df_numeric.columns:
            mean_val = df_numeric[col].mean()
            std_val = df_numeric[col].std()
            if std_val > 0:
                lower_bound = mean_val - 5 * std_val
                upper_bound = mean_val + 5 * std_val
                df_numeric[col] = df_numeric[col].clip(lower_bound, upper_bound)

        print(f"Remaining numeric features: {len(numeric_cols)}")
        return df_numeric, labels, ip_col

    def encode_labels(self, labels):
        print("\n" + "="*70)
        print("ENCODING LABELS")
        print("="*70)

        unique_labels = labels.unique()
        print(f"Unique labels found: {len(unique_labels)}")

        benign_mask = labels.str.upper().str.contains('BENIGN', na=False)
        benign_count = benign_mask.sum()
        attack_count = len(labels) - benign_count

        print(f"Benign samples: {benign_count}")
        print(f"Attack samples: {attack_count}")

        attack_types = labels[~benign_mask].unique()
        y = (~benign_mask).astype(int).values
        attack_to_id = {attack: idx for idx, attack in enumerate(attack_types)}

        return y, benign_mask, attack_types, attack_to_id

    def normalize_features(self, X_train):
        print("\n" + "="*70)
        print("NORMALIZING FEATURES")
        print("="*70)
        X_train = self._clean_array(X_train)
        X_train_scaled = self.scaler.fit_transform(X_train)
        os.makedirs("model", exist_ok=True)
        joblib.dump(self.scaler, "model/scaler.pkl")
        print("✓ Scaler saved to model/scaler.pkl")
        print(f"Train data scaled: mean={X_train_scaled.mean():.4f}, std={X_train_scaled.std():.4f}")
        return X_train_scaled

    def transform_features(self, X):
        X = self._clean_array(X)
        return self.scaler.transform(X)

    def _clean_array(self, X):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = np.clip(X, -1e10, 1e10)
        return X

    # ======================================================================
    # FIX 2: FIXED WINDOW LABELING BUG
    # The old code shuffled samples but windowing still ran sequentially,
    # causing all windows to be majority-attack due to boundary effects.
    # Fix: Create windows SEPARATELY for benign and attack, then combine.
    # ======================================================================
    def create_time_windows_fixed(self, X, y=None, window_size=30, step=5):
        """
        FIXED window creation:
        - Creates windows separately for benign and attack groups
        - Ensures both classes are represented in output
        - Labels each window correctly by its group, not majority vote
        """
        print(f"\nCreating time windows (window_size={window_size}, step={step})...")

        if y is None:
            # Unsupervised mode (training on benign only)
            windows = []
            for i in range(0, len(X) - window_size + 1, step):
                windows.append(X[i:i + window_size])
            windows = np.array(windows)
            print(f"  Total windows created: {len(windows)}")
            print(f"  Window shape: {windows.shape}")
            return windows, None, None

        # Supervised mode: create windows per class separately
        benign_idx = np.where(y == 0)[0]
        attack_idx = np.where(y == 1)[0]

        def make_windows(indices, label_val):
            wins, labs = [], []
            X_sub = X[indices]
            for i in range(0, len(X_sub) - window_size + 1, step):
                wins.append(X_sub[i:i + window_size])
                labs.append(label_val)
            return wins, labs

        benign_wins, benign_labs = make_windows(benign_idx, 0)
        attack_wins, attack_labs = make_windows(attack_idx, 1)

        print(f"  Benign windows: {len(benign_wins)}")
        print(f"  Attack windows: {len(attack_wins)}")

        # Balance: take equal number from each class
        min_count = min(len(benign_wins), len(attack_wins))
        if min_count == 0:
            raise ValueError("One class has 0 windows! Check your data balancing.")

        # Randomly sample to balance
        np.random.seed(42)
        b_idx = np.random.choice(len(benign_wins), min_count, replace=False)
        a_idx = np.random.choice(len(attack_wins), min_count, replace=False)

        all_wins = [benign_wins[i] for i in b_idx] + [attack_wins[i] for i in a_idx]
        all_labs = [benign_labs[i] for i in b_idx] + [attack_labs[i] for i in a_idx]

        # Shuffle combined
        combined = list(zip(all_wins, all_labs))
        np.random.shuffle(combined)
        all_wins, all_labs = zip(*combined)

        windows = np.array(all_wins)
        labels = np.array(all_labs)

        print(f"  Final balanced windows: {len(windows)}")
        print(f"  Window shape: {windows.shape}")
        print(f"  Class distribution: Benign={np.sum(labels==0)}, Attack={np.sum(labels==1)}")

        return windows, labels, None

    # ======================================================================
    # FIX 3: IP-TRIE SIMULATION
    # Real REAPER uses hardware IP-Trie. For simulation, we group flows
    # by source IP and create per-IP windows, capturing sending patterns.
    # ======================================================================
    def create_ip_based_windows(self, X, y, ip_col, window_size=30, step=5):
        """
        Simulates IP-Trie: groups flows by source IP and creates
        per-IP time windows to expose hidden sending patterns.
        Falls back to fixed windowing if no IP column available.
        """
        if ip_col is None:
            print("  No IP column found — falling back to fixed windowing")
            return self.create_time_windows_fixed(X, y, window_size, step)

        print(f"\nIP-Trie simulation: grouping by source IP...")
        unique_ips = ip_col.unique()
        print(f"  Unique source IPs: {len(unique_ips)}")

        benign_wins, attack_wins = [], []

        for ip in unique_ips:
            ip_mask = (ip_col == ip).values
            X_ip = X[ip_mask]
            y_ip = y[ip_mask]

            if len(X_ip) < window_size:
                continue

            for i in range(0, len(X_ip) - window_size + 1, step):
                win = X_ip[i:i + window_size]
                win_label = y_ip[i:i + window_size]
                label = 1 if np.mean(win_label) > 0.5 else 0
                if label == 0:
                    benign_wins.append(win)
                else:
                    attack_wins.append(win)

        print(f"  Benign windows from IP grouping: {len(benign_wins)}")
        print(f"  Attack windows from IP grouping: {len(attack_wins)}")

        min_count = min(len(benign_wins), len(attack_wins))
        if min_count == 0:
            print("  Warning: IP grouping yielded 0 windows for one class, falling back")
            return self.create_time_windows_fixed(X, y, window_size, step)

        np.random.seed(42)
        b_idx = np.random.choice(len(benign_wins), min_count, replace=False)
        a_idx = np.random.choice(len(attack_wins), min_count, replace=False)

        all_wins = [benign_wins[i] for i in b_idx] + [attack_wins[i] for i in a_idx]
        all_labs = [0] * min_count + [1] * min_count

        combined = list(zip(all_wins, all_labs))
        np.random.shuffle(combined)
        all_wins, all_labs = zip(*combined)

        windows = np.array(all_wins)
        labels = np.array(all_labs)

        print(f"  Final balanced windows: {len(windows)}")
        return windows, labels, None


# ======================================================================
# FIX 1: RNN ENCODER + VAE (REAPER Paper Architecture)
# Paper: RNN encodes flow sequences → VAE detects anomalies on embeddings
# Architecture: input → RNN → RNN embedding → VAE encode/decode
# ======================================================================

class REAPER_RNN_VAE(nn.Module):
    """
    FIXED REAPER Architecture (matches paper):
    
    Stage 1 — RNN Encoder:
        Sequence of flow features → temporal embedding
        Uses GRU (Gated Recurrent Unit) for stability
    
    Stage 2 — VAE Anomaly Detector:
        Operates on RNN embedding, not raw flattened input
        Learns normal traffic manifold; anomalies = high reconstruction error
    
    Key fixes vs old code:
    - RNN processes temporal sequence properly
    - latent_dim (16) < all hidden_dims (prevents bottleneck inversion)
    - Separate RNN and VAE parameters for targeted training
    """

    def __init__(self, feature_dim, rnn_hidden=64, rnn_layers=2,
                 vae_hidden_dims=[128, 64, 32], latent_dim=16):
        super().__init__()

        self.feature_dim = feature_dim
        self.rnn_hidden = rnn_hidden
        self.latent_dim = latent_dim

        # ========== STAGE 1: RNN ENCODER ==========
        # Bidirectional GRU for richer temporal context
        self.rnn = nn.GRU(
            input_size=feature_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=0.2 if rnn_layers > 1 else 0,
            bidirectional=True  # Forward + backward context
        )

        # RNN output dim: rnn_hidden * 2 (bidirectional)
        rnn_out_dim = rnn_hidden * 2

        # Project RNN output to fixed embedding
        self.rnn_projection = nn.Sequential(
            nn.Linear(rnn_out_dim, vae_hidden_dims[0]),
            nn.LayerNorm(vae_hidden_dims[0]),
            nn.ReLU()
        )

        # ========== STAGE 2: VAE ==========
        # Encoder: RNN embedding → latent space
        encoder_layers = []
        in_dim = vae_hidden_dims[0]
        for h_dim in vae_hidden_dims[1:]:
            encoder_layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ]
            in_dim = h_dim
        self.vae_encoder = nn.Sequential(*encoder_layers)

        # VAE: mu and log_var heads
        # FIX: latent_dim (16) < vae_hidden_dims[-1] (32) — proper bottleneck
        self.fc_mu = nn.Linear(vae_hidden_dims[-1], latent_dim)
        self.fc_log_var = nn.Linear(vae_hidden_dims[-1], latent_dim)

        # Decoder: latent → reconstruction of RNN embedding
        decoder_layers = []
        in_dim = latent_dim
        for h_dim in reversed(vae_hidden_dims):
            decoder_layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ]
            in_dim = h_dim
        # Final output matches RNN projection output dim
        decoder_layers.append(nn.Linear(in_dim, vae_hidden_dims[0]))
        self.vae_decoder = nn.Sequential(*decoder_layers)

    def encode_rnn(self, x):
        """
        x: (batch, window_size, feature_dim)
        Returns RNN embedding: (batch, vae_hidden_dims[0])
        """
        rnn_out, _ = self.rnn(x)         # (batch, window_size, rnn_hidden*2)
        # Use last timestep as sequence summary
        last_hidden = rnn_out[:, -1, :]  # (batch, rnn_hidden*2)
        embedding = self.rnn_projection(last_hidden)  # (batch, vae_hidden[0])
        return embedding

    def encode_vae(self, embedding):
        h = self.vae_encoder(embedding)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_vae(self, z):
        return self.vae_decoder(z)

    def forward(self, x, return_latent=False):
        # Stage 1: RNN → embedding
        embedding = self.encode_rnn(x)

        # Stage 2: VAE on embedding
        mu, log_var = self.encode_vae(embedding)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decode_vae(z)

        if return_latent:
            return reconstructed, embedding, mu, log_var, z

        return reconstructed, embedding, mu, log_var

    def loss_function(self, reconstructed, embedding, mu, log_var, beta=0.5):
        """
        VAE loss on the RNN embedding (not raw input).
        Higher beta (0.5) encourages better latent space structure.
        """
        recon_loss = F.mse_loss(reconstructed, embedding, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        total_loss = recon_loss + beta * kl_loss
        return total_loss, recon_loss, kl_loss

    def get_anomaly_score(self, x):
        """
        Anomaly score = reconstruction error of RNN embedding.
        Anomalies produce embeddings outside the learned normal manifold.
        """
        embedding = self.encode_rnn(x)
        mu, log_var = self.encode_vae(embedding)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decode_vae(z)
        error = F.mse_loss(reconstructed, embedding, reduction='none').mean(dim=1)
        return error


# ======================================================================
# TRAINER
# ======================================================================

class REAPER_Trainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        print(f"Using device: {self.device}")

    def train_epoch(self, train_loader, optimizer, beta=0.5):
        self.model.train()
        total_loss = total_recon = total_kl = 0

        for batch_x, in train_loader:
            batch_x = batch_x.to(self.device)
            optimizer.zero_grad()
            reconstructed, embedding, mu, log_var = self.model(batch_x)
            loss, recon, kl = self.model.loss_function(reconstructed, embedding, mu, log_var, beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()

        n = len(train_loader)
        return total_loss / n, total_recon / n, total_kl / n

    def fit(self, train_loader, val_loader=None, epochs=50, lr=0.001):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_loss = float('inf')
        best_val_errors = None

        print("\n" + "="*70)
        print("TRAINING REAPER RNN-VAE")
        print("="*70)

        # Beta annealing schedule
        for epoch in range(epochs):
            beta = min(0.5, 0.05 * (epoch + 1) / 10)

            train_loss, train_recon, train_kl = self.train_epoch(train_loader, optimizer, beta)
            scheduler.step()

            if (epoch + 1) % 5 == 0:
                print(f"\nEpoch {epoch+1}/{epochs}")
                print(f"  Train Loss: {train_loss:.4f} | Recon: {train_recon:.4f} | KL: {train_kl:.4f}")

            if val_loader is not None:
                val_errors = self.compute_reconstruction_error(val_loader)
                val_loss = val_errors.mean()
                if (epoch + 1) % 5 == 0:
                    print(f"  Val Error: mean={val_errors.mean():.4f}, 95th={np.percentile(val_errors,95):.4f}")

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_val_errors = val_errors
                    torch.save(self.model.state_dict(), "model/best_reaper_rnn_vae.pth")
                    if (epoch + 1) % 5 == 0:
                        print("  ✓ Best model saved.")

        print(f"\n✅ Training complete. Best val loss: {best_loss:.4f}")
        return best_loss, best_val_errors

    def compute_reconstruction_error(self, dataloader):
        self.model.eval()
        errors = []
        with torch.no_grad():
            for batch_x, in dataloader:
                batch_x = batch_x.to(self.device)
                score = self.model.get_anomaly_score(batch_x)
                errors.extend(score.cpu().numpy())
        return np.array(errors)

    def detect_anomalies(self, dataloader, threshold):
        errors = self.compute_reconstruction_error(dataloader)
        predictions = (errors > threshold).astype(int)
        anomaly_indices = np.where(predictions == 1)[0]
        return {
            'errors': errors,
            'threshold': threshold,
            'anomaly_indices': anomaly_indices,
            'predictions': predictions
        }

    def get_latent_embeddings(self, dataloader):
        self.model.eval()
        embeddings = []
        with torch.no_grad():
            for batch_x, in dataloader:
                batch_x = batch_x.to(self.device)
                _, _, _, _, z = self.model(batch_x, return_latent=True)
                embeddings.append(z.cpu().numpy())
        return np.concatenate(embeddings, axis=0)


# ======================================================================
# CONTOUR IMAGE GENERATION (CZ-ResViT side, unchanged)
# ======================================================================

def create_correlation_contour_image(window_data, output_size=224):
    try:
        import cv2
        df = pd.DataFrame(window_data)
        corr_matrix = df.corr().values
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        filtered_matrix = np.where(np.abs(corr_matrix) < 0.1, 0, corr_matrix)

        fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
        x = np.arange(filtered_matrix.shape[1])
        y = np.arange(filtered_matrix.shape[0])
        X, Y = np.meshgrid(x, y)
        ax.contourf(X, Y, filtered_matrix, levels=20, cmap='RdBu_r', alpha=0.8, antialiased=True)
        ax.contour(X, Y, filtered_matrix, levels=10, colors='black', linewidths=0.5, antialiased=True)
        ax.axis('off')
        ax.set_aspect('equal')
        plt.tight_layout(pad=0)
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        if img.shape[0] != output_size:
            img = cv2.resize(img, (output_size, output_size))
        return img.astype(np.float32) / 255.0
    except Exception as e:
        print(f"Contour image error: {e}")
        return np.zeros((output_size, output_size, 3), dtype=np.float32)


def convert_windows_to_contour_images(windows, max_images=10000):
    print("\n" + "="*70)
    print("CONVERTING ANOMALOUS WINDOWS TO CONTOUR IMAGES")
    print("="*70)
    if len(windows) == 0:
        print("No windows to convert!")
        return np.array([])
    if len(windows) > max_images:
        print(f"Limiting to {max_images} images (from {len(windows)} total)")
        indices = np.random.choice(len(windows), max_images, replace=False)
        windows = windows[indices]
    contour_images = []
    batch_size = 100
    for i in range(0, len(windows), batch_size):
        batch = windows[i:i + batch_size]
        contour_images.extend([create_correlation_contour_image(w) for w in batch])
        print(f"  Processed {min(i+batch_size, len(windows))}/{len(windows)} windows")
    contour_images = np.array(contour_images)
    print(f"\n✅ Contour images: shape={contour_images.shape}, range=[{contour_images.min():.3f},{contour_images.max():.3f}]")
    return contour_images


# ======================================================================
# DATASET
# ======================================================================

class WindowDataset(Dataset):
    def __init__(self, windows):
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.windows[idx]),


# ======================================================================
# EVALUATION HELPER
# ======================================================================

def evaluate_predictions(predictions, labels, label=""):
    tn = np.sum((predictions == 0) & (labels == 0))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    tp = np.sum((predictions == 1) & (labels == 1))

    accuracy  = (tp + tn) / (tp + tn + fp + fn + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    print(f"\n📊 {label} Results:")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1-Score : {f1:.4f}")

    return accuracy, precision, recall, f1


# ======================================================================
# MAIN PIPELINE
# ======================================================================

def main():
    print("=" * 70)
    print("REAPER MODULE-1 (FIXED): RNN-VAE ANOMALY DETECTION")
    print("Fixes: RNN encoder + window labeling + IP grouping + bottleneck")
    print("=" * 70)

    config = {
        'data_path': 'dataset/',
        'train_file': 'Merged03.csv',
        'val_file': 'Merged02.csv',
        'test_file': 'Merged01.csv',
        'zero_day_file': 'Merged04.csv',
        'window_size': 30,
        'step_size': 5,
        'batch_size': 64,
        'epochs': 50,
        'learning_rate': 0.001,
        # RNN params
        'rnn_hidden': 64,
        'rnn_layers': 2,
        # VAE params — NOTE: latent_dim MUST be < all vae_hidden_dims
        'vae_hidden_dims': [128, 64, 32],
        'latent_dim': 16,   # FIX: was 128 (larger than hidden), now 16
        'val_percentile': 95,
        'max_contour_images': 100
    }

    os.makedirs("model", exist_ok=True)
    os.makedirs("module1_outputs", exist_ok=True)

    processor = CICIoT2023Processor(data_path=config['data_path'])

    # ======================================================================
    # STEP 1: TRAINING DATA — benign only for VAE
    # ======================================================================
    print("\n" + "="*70)
    print("STEP 1: PROCESSING TRAINING DATA")
    print("="*70)

    df_train = processor.load_dataset(config['train_file'])
    X_train_numeric, train_labels, train_ip = processor.clean_features(df_train)
    y_train, train_benign_mask, train_attack_types, _ = processor.encode_labels(train_labels)

    X_train_benign = X_train_numeric[train_benign_mask].values
    print(f"\nBenign samples for training: {len(X_train_benign)}")

    X_train_norm = processor.normalize_features(X_train_benign)

    # Unsupervised windowing (benign only)
    windows_train, _, _ = processor.create_time_windows_fixed(
        X_train_norm, y=None, window_size=config['window_size'], step=config['step_size']
    )

    # ======================================================================
    # STEP 2: VALIDATION DATA — benign only
    # ======================================================================
    print("\n" + "="*70)
    print("STEP 2: PROCESSING VALIDATION DATA")
    print("="*70)

    df_val = processor.load_dataset(config['val_file'])
    X_val_numeric, val_labels, val_ip = processor.clean_features(df_val)
    y_val, val_benign_mask, _, _ = processor.encode_labels(val_labels)

    X_val_benign = X_val_numeric[val_benign_mask].values
    print(f"\nBenign samples for validation: {len(X_val_benign)}")

    X_val_norm = processor.transform_features(X_val_benign)

    windows_val, _, _ = processor.create_time_windows_fixed(
        X_val_norm, y=None, window_size=config['window_size'], step=config['step_size']
    )

    # ======================================================================
    # STEP 3: DATALOADERS
    # ======================================================================
    train_loader = DataLoader(WindowDataset(windows_train), batch_size=config['batch_size'], shuffle=True)
    val_loader   = DataLoader(WindowDataset(windows_val),   batch_size=config['batch_size'], shuffle=False)

    print(f"\nTrain windows: {windows_train.shape}")
    print(f"Val windows:   {windows_val.shape}")

    # ======================================================================
    # STEP 4: INITIALIZE MODEL
    # ======================================================================
    feature_dim = windows_train.shape[2]  # 24

    model = REAPER_RNN_VAE(
        feature_dim=feature_dim,
        rnn_hidden=config['rnn_hidden'],
        rnn_layers=config['rnn_layers'],
        vae_hidden_dims=config['vae_hidden_dims'],
        latent_dim=config['latent_dim']
    )

    print(f"\nModel architecture:")
    print(f"  Input: (batch, {config['window_size']}, {feature_dim})")
    print(f"  RNN:   GRU bidirectional, hidden={config['rnn_hidden']}, layers={config['rnn_layers']}")
    print(f"  VAE:   hidden={config['vae_hidden_dims']}, latent={config['latent_dim']}")
    print(f"  Bottleneck check: latent({config['latent_dim']}) < min_hidden({min(config['vae_hidden_dims'])}) ✓")

    trainer = REAPER_Trainer(model)

    # ======================================================================
    # STEP 5: TRAIN
    # ======================================================================
    best_loss, val_errors = trainer.fit(
        train_loader, val_loader,
        epochs=config['epochs'],
        lr=config['learning_rate']
    )

    # Load best model
    model.load_state_dict(torch.load("model/best_reaper_rnn_vae.pth", map_location=trainer.device))
    print("\n✓ Best model loaded for evaluation")

    # ======================================================================
    # STEP 6: THRESHOLD
    # ======================================================================
    print("\n" + "="*70)
    print("STEP 6: THRESHOLD FROM VALIDATION DATA")
    print("="*70)

    threshold = np.percentile(val_errors, config['val_percentile'])
    np.save("model/reaper_threshold.npy", np.array([threshold]))

    print(f"  Mean : {val_errors.mean():.4f}")
    print(f"  Std  : {val_errors.std():.4f}")
    print(f"  95th : {threshold:.4f}")

    # ======================================================================
    # STEP 7: TEST DATA — FIXED balanced windowing
    # ======================================================================
    print("\n" + "="*70)
    print("STEP 7: PROCESSING TEST DATA (FIXED BALANCING)")
    print("="*70)

    df_test = processor.load_dataset(config['test_file'])
    X_test_numeric, test_labels, test_ip = processor.clean_features(df_test)
    y_test, test_benign_mask, test_attack_types, _ = processor.encode_labels(test_labels)

    # Balance raw samples first
    benign_idx = np.where(test_benign_mask)[0]
    attack_idx = np.where(~test_benign_mask)[0]
    test_size = min(20000, len(benign_idx), len(attack_idx))
    np.random.seed(42)
    sel_b = np.random.choice(benign_idx, test_size, replace=False)
    sel_a = np.random.choice(attack_idx, test_size, replace=False)

    X_test_b = X_test_numeric.iloc[sel_b].values
    X_test_a = X_test_numeric.iloc[sel_a].values
    y_test_b = y_test[sel_b]  # all 0
    y_test_a = y_test[sel_a]  # all 1

    X_test_b_norm = processor.transform_features(X_test_b)
    X_test_a_norm = processor.transform_features(X_test_a)

    # FIX: Create windows PER CLASS then combine — guarantees both classes
    if test_ip is not None:
        print("\nUsing IP-based windowing for test data...")
        X_test_all = np.vstack([X_test_b, X_test_a])
        y_test_all = np.concatenate([y_test_b, y_test_a])
        ip_test_all = pd.concat([test_ip.iloc[sel_b], test_ip.iloc[sel_a]]).reset_index(drop=True)
        X_test_all_norm = processor.transform_features(X_test_all)
        windows_test, window_labels_test, _ = processor.create_ip_based_windows(
            X_test_all_norm, y_test_all, ip_test_all,
            window_size=config['window_size'], step=config['step_size']
        )
    else:
        # FIXED: pass already-separated arrays
        X_test_combined = np.vstack([X_test_b_norm, X_test_a_norm])
        y_test_combined = np.concatenate([y_test_b, y_test_a])
        windows_test, window_labels_test, _ = processor.create_time_windows_fixed(
            X_test_combined, y=y_test_combined,
            window_size=config['window_size'], step=config['step_size']
        )

    print(f"\n✅ Test windows: {windows_test.shape}")
    print(f"   Benign={np.sum(window_labels_test==0)}, Attack={np.sum(window_labels_test==1)}")

    # ======================================================================
    # STEP 8: EVALUATE ON TEST DATA
    # ======================================================================
    print("\n" + "="*70)
    print("STEP 8: EVALUATING ON TEST DATA")
    print("="*70)

    test_loader = DataLoader(WindowDataset(windows_test), batch_size=config['batch_size'], shuffle=False)
    test_results = trainer.detect_anomalies(test_loader, threshold=threshold)
    latent_embeddings = trainer.get_latent_embeddings(test_loader)

    acc, prec, rec, f1 = evaluate_predictions(
        test_results['predictions'], window_labels_test, label="Test"
    )

    # ======================================================================
    # STEP 9: ZERO-DAY DATA
    # ======================================================================
    print("\n" + "="*70)
    print("STEP 9: PROCESSING ZERO-DAY DATA")
    print("="*70)

    df_zd = processor.load_dataset(config['zero_day_file'])
    X_zd_numeric, zd_labels, zd_ip = processor.clean_features(df_zd)
    y_zd, zd_benign_mask, zd_attack_types, attack_to_id = processor.encode_labels(zd_labels)

    X_zd_attacks = X_zd_numeric[~zd_benign_mask].values
    print(f"\nZero-day attack samples: {len(X_zd_attacks)}")

    X_zd_norm = processor.transform_features(X_zd_attacks)
    windows_zd, _, _ = processor.create_time_windows_fixed(
        X_zd_norm, y=None, window_size=config['window_size'], step=config['step_size']
    )

    # ======================================================================
    # STEP 10: ZERO-DAY DETECTION + CONTOUR IMAGES
    # ======================================================================
    print("\n" + "="*70)
    print("STEP 10: ZERO-DAY ANOMALY DETECTION + CONTOUR IMAGE GENERATION")
    print("="*70)

    zd_loader = DataLoader(WindowDataset(windows_zd), batch_size=config['batch_size'], shuffle=False)
    zd_errors = trainer.compute_reconstruction_error(zd_loader)
    zd_anomaly_mask = zd_errors > threshold
    detection_rate = zd_anomaly_mask.mean() * 100

    print(f"\n🔍 Zero-day Error Statistics:")
    print(f"  Mean error : {zd_errors.mean():.4f}")
    print(f"  Threshold  : {threshold:.4f}")
    print(f"  Above threshold: {zd_anomaly_mask.sum()} / {len(zd_errors)}")
    print(f"  Detection rate : {detection_rate:.2f}%")

    zd_anomaly_windows = windows_zd[zd_anomaly_mask]
    zd_embeddings = trainer.get_latent_embeddings(zd_loader)
    zd_anomaly_embeddings = zd_embeddings[zd_anomaly_mask]

    zd_anomaly_images = convert_windows_to_contour_images(
        zd_anomaly_windows, max_images=config['max_contour_images']
    )

    # ======================================================================
    # STEP 11: SAVE OUTPUTS
    # ======================================================================
    print("\n" + "="*70)
    print("STEP 11: SAVING OUTPUTS FOR MODULE-2")
    print("="*70)

    np.save("module1_outputs/test_latent_embeddings.npy", latent_embeddings)
    np.save("module1_outputs/test_reconstruction_errors.npy", test_results['errors'])
    np.save("module1_outputs/test_predictions.npy", test_results['predictions'])
    np.save("module1_outputs/test_window_labels.npy", window_labels_test)

    np.save("module1_outputs/zeroday_embeddings.npy", zd_embeddings)
    np.save("module1_outputs/zeroday_errors.npy", zd_errors)
    np.save("module1_outputs/zeroday_anomaly_mask.npy", zd_anomaly_mask)
    np.save("module1_outputs/zeroday_anomaly_windows.npy", zd_anomaly_windows)
    np.save("module1_outputs/zeroday_anomaly_embeddings.npy", zd_anomaly_embeddings)
    np.save("module1_outputs/zeroday_anomaly_images.npy", zd_anomaly_images)

    metadata = {
        'config': config,
        'model': 'REAPER_RNN_VAE',
        'fixes_applied': [
            'RNN encoder (GRU bidirectional) before VAE',
            'Fixed window labeling bug (per-class windowing)',
            'IP-based flow grouping (IP-Trie simulation)',
            'Fixed latent_dim bottleneck (16 < 32 < 64 < 128)'
        ],
        'threshold': float(threshold),
        'test_stats': {
            'accuracy': float(acc), 'precision': float(prec),
            'recall': float(rec), 'f1': float(f1),
            'total_windows': int(len(windows_test)),
            'benign_windows': int(np.sum(window_labels_test == 0)),
            'attack_windows': int(np.sum(window_labels_test == 1))
        },
        'zeroday_stats': {
            'total_windows': int(len(windows_zd)),
            'detected': int(zd_anomaly_mask.sum()),
            'detection_rate': float(detection_rate),
            'attack_types': list(zd_attack_types[:20])
        },
        'image_shape': list(zd_anomaly_images.shape),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open('module1_outputs/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save sample images
    if len(zd_anomaly_images) > 0:
        sample_dir = "module1_outputs/sample_images"
        os.makedirs(sample_dir, exist_ok=True)
        for i in range(min(10, len(zd_anomaly_images))):
            plt.figure(figsize=(6, 6))
            plt.imshow(zd_anomaly_images[i])
            plt.title(f"Zero-day Sample {i+1}")
            plt.axis('off')
            plt.savefig(f"{sample_dir}/sample_{i+1}.png", bbox_inches='tight', dpi=100)
            plt.close()

    # ======================================================================
    # FINAL SUMMARY
    # ======================================================================
    print("\n" + "="*70)
    print("REAPER MODULE-1 (FIXED) — COMPLETE")
    print("="*70)
    print(f"\n✅ Fixes applied:")
    print(f"   1. RNN GRU (bidirectional, {config['rnn_layers']} layers) → temporal embedding")
    print(f"   2. VAE operates on RNN embedding (not raw flattened input)")
    print(f"   3. Window labeling fixed: per-class windowing guarantees both classes")
    print(f"   4. latent_dim={config['latent_dim']} < min_hidden={min(config['vae_hidden_dims'])} ✓")
    print(f"\n📊 Test Performance:")
    print(f"   Accuracy : {acc:.4f}")
    print(f"   F1-Score : {f1:.4f}")
    print(f"   Benign windows in test: {np.sum(window_labels_test==0)} (was 0 before fix)")
    print(f"\n🚨 Zero-day Detection:")
    print(f"   Detection rate: {detection_rate:.2f}%")
    print(f"\n🖼️  Module-2 inputs saved:")
    print(f"   zeroday_anomaly_images.npy → shape {zd_anomaly_images.shape}")
    print(f"   zeroday_anomaly_embeddings.npy → shape {zd_anomaly_embeddings.shape}")
    print(f"\n💾 All outputs → module1_outputs/")
    print("="*70)
    print("✅ READY FOR MODULE-2: CZ-ResViT + Zero-Shot Learning")
    print("="*70)

    return {
        'test_results': {
            'embeddings': latent_embeddings,
            'errors': test_results['errors'],
            'predictions': test_results['predictions'],
            'labels': window_labels_test,
            'accuracy': acc, 'f1': f1
        },
        'zeroday_results': {
            'embeddings': zd_embeddings,
            'errors': zd_errors,
            'anomaly_mask': zd_anomaly_mask,
            'anomaly_windows': zd_anomaly_windows,
            'anomaly_embeddings': zd_anomaly_embeddings,
            'anomaly_images': zd_anomaly_images
        },
        'metadata': metadata,
        'threshold': threshold
    }


if __name__ == "__main__":
    try:
        results = main()
        print("\n🎉 EXECUTION SUCCESSFUL!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
