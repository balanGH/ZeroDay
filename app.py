"""
================================================================================
REAPER + CZ-ResViT NETWORK ANOMALY DETECTION SYSTEM
Version: 3.2 Enterprise Edition - Zero-Day Detection with Trend Graph
================================================================================
Features:
- Normal traffic detection
- Known attack classification (DDoS, Mirai, WebAtk)
- Zero-Day attack detection (Recon, Novel patterns)
- Separate tabs for each category
- Anomaly score trend graph (rise/fall visualization)
================================================================================
"""

import os
import sys
import json
import time
import random
import threading
import base64
import io
import zipfile
from datetime import datetime
from collections import deque
from enum import Enum
import numpy as np

# Suppress verbose logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'

from flask import Flask, render_template_string, jsonify, send_from_directory
from flask_socketio import SocketIO, emit

# Create required directories
for directory in ["contour_images", "logs", "exports", "dataset"]:
    os.makedirs(directory, exist_ok=True)

# ============================================================================
# DEPENDENCY CHECKS
# ============================================================================

try:
    import torch
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    print("[WARN] PyTorch not found - REAPER functionality limited")

try:
    import tensorflow as tf
    from tensorflow.keras import layers
    TF_OK = True
except ImportError:
    TF_OK = False
    print("[WARN] TensorFlow not found - CZ-ResViT functionality limited")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MPL_OK = True
except ImportError:
    MPL_OK = False
    print("[WARN] Matplotlib not found - contour images will be simulated")

try:
    from REAPER.reaper import REAPER_RNN_VAE, create_correlation_contour_image
    REAPER_OK = True
    print("[OK] REAPER module imported")

    # optional test
    model = REAPER_RNN_VAE(feature_dim=24)
    print("[OK] Model initialized")

except Exception as e:
    REAPER_OK = False
    print(f"[ERROR] REAPER import failed: {e}")

# ============================================================================
# ATTACK CLASSIFICATION ENUM
# ============================================================================

class TrafficClass(Enum):
    NORMAL = "normal"
    KNOWN_ATTACK = "known_attack"
    ZERO_DAY_ATTACK = "zero_day_attack"

class AttackType(Enum):
    # Known attacks (seen in training)
    DDOS = "DDoS"
    MIRAI = "Mirai"
    WEB_ATTACK = "WebAtk"
    
    # Zero-day attacks (novel patterns)
    RECON = "Recon"
    CRYPTO_MINING = "CryptoMiner"
    DATA_EXFIL = "DataExfil"
    ZERO_DAY_NOVEL = "ZeroDay_Novel"

# ============================================================================
# CUSTOM KERAS LAYERS FOR CZ-ResViT
# ============================================================================

if TF_OK:
    @tf.keras.utils.register_keras_serializable(package='Custom', name='TransformerBlock')
    class TransformerBlock(layers.Layer):
        def __init__(self, embed_dim=768, num_heads=8, mlp_dim=1024, dropout=0.1, **kwargs):
            super().__init__(**kwargs)
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.mlp_dim = mlp_dim
            self.dropout_rate = dropout
            self.attn = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=embed_dim // num_heads,
                dropout=dropout, name='attn')
            self.mlp = tf.keras.Sequential([
                layers.Dense(mlp_dim, activation='relu', name='dense'),
                layers.Dense(embed_dim, name='dense_1'),
            ], name='mlp')
            self.norm1 = layers.LayerNormalization(epsilon=1e-6, name='norm1')
            self.norm2 = layers.LayerNormalization(epsilon=1e-6, name='norm2')
            self.drop1 = layers.Dropout(dropout)
            self.drop2 = layers.Dropout(dropout)

        def build(self, input_shape):
            dummy = tf.zeros([2] + list(input_shape[1:]))
            self.attn(dummy, dummy)
            self.mlp(dummy)
            self.norm1(dummy)
            self.norm2(dummy)
            super().build(input_shape)

        def call(self, x, training=False):
            a = self.drop1(self.attn(x, x), training=training)
            x = self.norm1(x + a)
            ff = self.drop2(self.mlp(x), training=training)
            return self.norm2(x + ff)

        def get_config(self):
            cfg = super().get_config()
            cfg.update({'embed_dim': self.embed_dim, 'num_heads': self.num_heads,
                        'mlp_dim': self.mlp_dim, 'dropout': self.dropout_rate})
            return cfg

    @tf.keras.utils.register_keras_serializable(package='Custom', name='AddCLSAndPosEmbed')
    class AddCLSAndPosEmbed(layers.Layer):
        def __init__(self, num_tokens=49, embed_dim=768, **kwargs):
            super().__init__(**kwargs)
            self.num_tokens = num_tokens
            self.embed_dim = embed_dim

        def build(self, input_shape):
            self.cls_token = self.add_weight(
                shape=(1, 1, self.embed_dim), initializer='zeros',
                trainable=True, name='cls_token')
            self.pos_embed = self.add_weight(
                shape=(1, self.num_tokens + 1, self.embed_dim),
                initializer='random_normal', trainable=True, name='pos_embed')
            super().build(input_shape)

        def call(self, x):
            b = tf.shape(x)[0]
            cls = tf.broadcast_to(self.cls_token, [b, 1, self.embed_dim])
            return tf.concat([cls, x], axis=1) + self.pos_embed

        def get_config(self):
            cfg = super().get_config()
            cfg.update({'num_tokens': self.num_tokens, 'embed_dim': self.embed_dim})
            return cfg

    @tf.keras.utils.register_keras_serializable(package='Custom', name='CLSTokenExtract')
    class CLSTokenExtract(layers.Layer):
        def call(self, x):
            return x[:, 0, :]
        def get_config(self):
            return super().get_config()

# ============================================================================
# MODEL LOADING HELPER
# ============================================================================

def _patch_and_load_keras(model_path):
    """Load Keras model with custom layers"""
    if not TF_OK:
        return None
    try:
        with zipfile.ZipFile(model_path, 'r') as z:
            cfg_data = json.loads(z.read('config.json'))
            weights_bytes = z.read('model.weights.h5')
            meta_bytes = z.read('metadata.json')

        for lyr in cfg_data['config']['layers']:
            cls = lyr.get('class_name', '')
            if cls == 'GetItem':
                inbound = lyr['inbound_nodes'][0]['args'][0]
                lyr.update({
                    'module': 'Custom',
                    'class_name': 'CLSTokenExtract',
                    'config': {'name': lyr['name'], 'trainable': True, 'dtype': 'float32'},
                    'registered_name': 'Custom>CLSTokenExtract',
                    'inbound_nodes': [{'args': [inbound], 'kwargs': {}}],
                })
            elif cls == 'TransformerBlock':
                lyr['registered_name'] = 'Custom>TransformerBlock'
                lyr['module'] = 'Custom'
            elif cls == 'AddCLSAndPosEmbed':
                lyr['registered_name'] = 'Custom>AddCLSAndPosEmbed'
                lyr['module'] = 'Custom'

        patched_path = model_path + '.patched.keras'
        with zipfile.ZipFile(patched_path, 'w', zipfile.ZIP_STORED) as zout:
            zout.writestr('config.json', json.dumps(cfg_data))
            zout.writestr('metadata.json', meta_bytes)
            zout.writestr('model.weights.h5', weights_bytes)

        custom_objects = {
            'TransformerBlock': TransformerBlock,
            'AddCLSAndPosEmbed': AddCLSAndPosEmbed,
            'CLSTokenExtract': CLSTokenExtract,
        }
        model = tf.keras.models.load_model(patched_path, custom_objects=custom_objects, compile=False)
        try:
            os.remove(patched_path)
        except:
            pass
        return model
    except Exception as e:
        print(f"Keras load error: {e}")
        return None

# ============================================================================
# FLASK APP INITIALIZATION
# ============================================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = 'reaper-czresvit-2025'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

# Training classes (what the model was trained on)
KNOWN_CLASSES = ['DDoS', 'Mirai', 'Benign', 'WebAtk']
# Zero-day classes (novel attacks)
ZERO_DAY_CLASSES = ['Recon', 'CryptoMiner', 'DataExfil', 'ZeroDay_Novel']
ALL_CLASSES = KNOWN_CLASSES + ZERO_DAY_CLASSES

CZ_LABEL_MAP = ALL_CLASSES
CZ_BUCKET_MAP = {
    # Known attacks
    'DDoS': 'known',
    'Mirai': 'known',
    'WebAtk': 'known',
    'Benign': 'benign',
    # Zero-day attacks
    'Recon': 'zeroday',
    'CryptoMiner': 'zeroday',
    'DataExfil': 'zeroday',
    'ZeroDay_Novel': 'zeroday',
}

# Zero-day detection threshold (lower confidence = possible zero-day)
ZERO_DAY_CONFIDENCE_THRESHOLD = 65  # If confidence < 65%, flag as potential zero-day

MIN_WINDOW_SIZE = 10
MAX_BUFFER_SIZE = 30
ATTACK_DURATION = 15

DESTS = ['10.0.0.1', '8.8.8.8', '172.16.0.5', '10.0.0.254', '93.184.216.34']
PROTOS = ['TCP', 'UDP', 'ICMP', 'HTTP', 'DNS']
PORTS = {'TCP': [80, 443, 22, 3389], 'UDP': [53, 123, 161],
         'ICMP': [0], 'HTTP': [80, 8080], 'DNS': [53]}

# Attack patterns with classification
ATTACK_PATTERNS = {
    # Known attacks
    'ddos': {'score_range': (4.0, 9.0), 'bytes_range': (10000, 50000), 'class': AttackType.DDOS, 'type': 'known'},
    'mirai': {'score_range': (3.5, 8.0), 'bytes_range': (5000, 30000), 'class': AttackType.MIRAI, 'type': 'known'},
    'webatk': {'score_range': (2.0, 4.5), 'bytes_range': (500, 3000), 'class': AttackType.WEB_ATTACK, 'type': 'known'},
    # Zero-day attacks
    'recon': {'score_range': (2.0, 4.0), 'bytes_range': (50, 300), 'class': AttackType.RECON, 'type': 'zeroday'},
    'crypto': {'score_range': (5.0, 8.5), 'bytes_range': (2000, 8000), 'class': AttackType.CRYPTO_MINING, 'type': 'zeroday'},
    'exfil': {'score_range': (3.0, 6.0), 'bytes_range': (1000, 5000), 'class': AttackType.DATA_EXFIL, 'type': 'zeroday'},
    'novel': {'score_range': (2.5, 7.0), 'bytes_range': (300, 2000), 'class': AttackType.ZERO_DAY_NOVEL, 'type': 'zeroday'},
}

# ============================================================================
# GLOBAL STATE
# ============================================================================

STATE = {
    'clients': [
        {'ip': '192.168.1.10', 'name': 'Workstation-A', 'role': 'Workstation',
         'status': 'normal', 'flows': 0, 'bytes': 0, 'anomalies': 0,
         'traffic_class': 'normal', 'last_attack': None},
        {'ip': '192.168.1.20', 'name': 'IoT-Camera', 'role': 'IoT Device',
         'status': 'normal', 'flows': 0, 'bytes': 0, 'anomalies': 0,
         'traffic_class': 'normal', 'last_attack': None},
        {'ip': '192.168.1.30', 'name': 'Server-01', 'role': 'Server',
         'status': 'normal', 'flows': 0, 'bytes': 0, 'anomalies': 0,
         'traffic_class': 'normal', 'last_attack': None},
        {'ip': '192.168.1.40', 'name': 'Gateway', 'role': 'Gateway',
         'status': 'normal', 'flows': 0, 'bytes': 0, 'anomalies': 0,
         'traffic_class': 'normal', 'last_attack': None},
    ],
    'flows': [],
    'active_attacks': {},
    'model': None,
    'cz_model': None,
    'threshold': 1.82,
    'sim_running': False,
    'pipeline_stats': {
        'capture': 0, 'iptrie': 0, 'rnn': 0,
        'vae_score': 0.0, 'flagged': 0,
        'last_src': '—', 'last_dst': '—',
    },
    'cls_counts': {'known': 0, 'zeroday': 0, 'benign': 0, 'normal': 0},
    'anomaly_scores': [],
    'contour_images_b64': [],
    # Separate storage for different traffic types
    'normal_traffic': [],
    'known_attacks': [],
    'zero_day_attacks': [],
}

STATE_LOCK = threading.Lock()
IP_FLOW_BUFFER = {}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def ts():
    return datetime.now().strftime('%H:%M:%S')

def log(msg, level='info'):
    socketio.emit('log', {'msg': msg, 'level': level, 'ts': ts()})

def emit_state():
    with STATE_LOCK:
        snap = {
            'clients': STATE['clients'],
            'flows': STATE['flows'][-20:],
            'active_attacks': STATE['active_attacks'],
            'threshold': STATE['threshold'],
            'pipeline_stats': STATE['pipeline_stats'],
            'cls_counts': STATE['cls_counts'],
            'anomaly_scores': STATE['anomaly_scores'][-40:],
            'contour_images': STATE['contour_images_b64'][-10:],
            'model_loaded': STATE['model'] is not None,
            'cz_loaded': STATE['cz_model'] is not None,
            'attack_duration': ATTACK_DURATION,
            'normal_traffic': STATE['normal_traffic'][-50:],
            'known_attacks': STATE['known_attacks'][-50:],
            'zero_day_attacks': STATE['zero_day_attacks'][-50:],
        }
    socketio.emit('state', snap)

# ============================================================================
# CONTOUR GENERATION
# ============================================================================

def _make_fake_contour_b64(score, atk_type):
    """Generate synthetic contour when MPL/REAPER unavailable"""
    if not MPL_OK:
        return None
    try:
        size = 112
        data = np.random.rand(size, size) * (score / 10.0)
        x = np.linspace(-3, 3, size)
        X, Y = np.meshgrid(x, x)
        blob = np.exp(-(X**2 + Y**2) / (2 * (score / 4.0)))
        data = data * 0.3 + blob * 0.7
        fig, ax = plt.subplots(figsize=(1.12, 1.12), dpi=100)
        ax.imshow(data, cmap='inferno', aspect='auto', vmin=0, vmax=1)
        ax.axis('off')
        plt.tight_layout(pad=0)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    except Exception as e:
        print(f"Fake contour error: {e}")
        return None

def save_contour_image(window_np, attack_type, score):
    if not MPL_OK or not REAPER_OK:
        return None
    try:
        img = create_correlation_contour_image(window_np, output_size=224)
        ts_str = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        fname = f"contour_images/{attack_type}_{score:.2f}_{ts_str}.png"
        plt.imsave(fname, img)
        return fname
    except Exception as e:
        print(f"Save contour error: {e}")
        return None

def make_contour_b64(window_np, score=5.0, atk_type='unknown'):
    """Return base64 PNG contour image"""
    if REAPER_OK and MPL_OK:
        try:
            img = create_correlation_contour_image(window_np, output_size=112)
            buf = io.BytesIO()
            plt.imsave(buf, img, format='png')
            buf.seek(0)
            return base64.b64encode(buf.read()).decode('utf-8')
        except Exception:
            pass
    return _make_fake_contour_b64(score, atk_type)

def make_contour_for_cz(window_np):
    """Prepare input for CZ-ResViT model"""
    if REAPER_OK:
        try:
            img = create_correlation_contour_image(window_np, output_size=224)
            img = np.array(img, dtype=np.float32)
            if img.max() > 1.0:
                img = img / 255.0
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            return np.expand_dims(img, axis=0)
        except Exception as e:
            print(f"Contour-for-CZ error: {e}")
    return np.random.rand(1, 224, 224, 3).astype(np.float32)

# ============================================================================
# ZERO-DAY DETECTION LOGIC
# ============================================================================

def detect_zero_day(prediction_probs, predicted_label, confidence):
    """
    Determine if traffic is zero-day based on:
    1. Low confidence in prediction
    2. Prediction falls into zero-day category
    3. Anomaly score is high but doesn't match known patterns
    """
    # If the model predicted a zero-day class
    if predicted_label in ZERO_DAY_CLASSES:
        return True, "zero_day_class"
    
    # If confidence is low (model unsure)
    if confidence < ZERO_DAY_CONFIDENCE_THRESHOLD:
        return True, "low_confidence"
    
    # Check if prediction distribution is flat (high entropy)
    if prediction_probs is not None:
        entropy = -np.sum(prediction_probs * np.log(prediction_probs + 1e-7))
        if entropy > 1.5:  # High uncertainty
            return True, "high_entropy"
    
    return False, "known"

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_flow_features(flow):
    """Extract 24-dimensional feature vector from flow"""
    feat = np.zeros(24, dtype=np.float32)
    feat[0] = min(flow['bytes'] / 100000.0, 1.0)
    feat[1] = min(flow['pkts'] / 5000.0, 1.0)
    feat[2] = flow['port'] / 65535.0
    feat[3] = 1.0 if flow['proto'] == 'TCP' else 0.0
    feat[4] = 1.0 if flow['proto'] == 'UDP' else 0.0
    feat[5] = 1.0 if flow['proto'] == 'ICMP' else 0.0
    feat[6] = feat[0] / (feat[1] + 0.001)
    feat[7:] = np.random.randn(17) * 0.1
    return feat

# ============================================================================
# REAPER SCORE COMPUTATION
# ============================================================================

def get_reaper_score(buffer, is_attack=False, atk_type='normal'):
    """Compute anomaly score using REAPER model"""
    with STATE_LOCK:
        model = STATE['model']
        thresh = STATE['threshold']

    buf_len = len(buffer)
    if buf_len < MIN_WINDOW_SIZE:
        if is_attack and atk_type != 'normal':
            pat = ATTACK_PATTERNS.get(atk_type, ATTACK_PATTERNS['ddos'])
            sim = random.uniform(*pat['score_range'])
            return round(sim, 3), sim > thresh
        return round(random.uniform(0.05, 1.2), 3), False

    window = np.array(buffer[-MAX_BUFFER_SIZE:])
    if window.shape[0] < MAX_BUFFER_SIZE:
        pad = np.zeros((MAX_BUFFER_SIZE - window.shape[0], window.shape[1]), dtype=np.float32)
        window = np.vstack([pad, window])

    real_score = None
    if model is not None and TORCH_OK:
        try:
            t = torch.FloatTensor(window).unsqueeze(0)
            with torch.no_grad():
                real_score = model.get_anomaly_score(t).item()
        except Exception as e:
            print(f"REAPER score error: {e}")

    if is_attack and atk_type != 'normal':
        pat = ATTACK_PATTERNS.get(atk_type, ATTACK_PATTERNS['ddos'])
        sim = random.uniform(*pat['score_range'])
        if real_score is not None:
            score = 0.3 * real_score + 0.7 * sim
        else:
            score = sim
    else:
        if real_score is not None:
            score = real_score
        else:
            score = random.uniform(0.05, 1.2)

    return round(score, 3), score > thresh

# ============================================================================
# FLOW GENERATION WITH TRAFFIC CLASSIFICATION
# ============================================================================

def generate_normal_flow():
    """Generate normal benign traffic"""
    with STATE_LOCK:
        clients = STATE['clients']
    
    if not clients:
        return None
    
    ci = random.randint(0, len(clients) - 1)
    client = clients[ci]
    
    proto = random.choice(PROTOS)
    port = random.choice(PORTS.get(proto, [80]))
    dst = random.choice(DESTS)
    
    # Normal traffic patterns
    bts = random.randint(50, 1500)
    pkts = random.randint(1, 20)
    
    flow = {
        'time': ts(), 'src': client['ip'], 'dst': dst,
        'port': port, 'proto': proto,
        'bytes': bts, 'pkts': pkts,
        'score': 0.0, 'anomaly': False,
        'attack_type': 'normal',
        'traffic_class': 'normal',
        'cz_label': 'Benign',
        'cz_confidence': random.randint(85, 99),
        'is_zero_day': False,
    }
    
    feat = extract_flow_features(flow)
    src_ip = flow['src']
    
    with STATE_LOCK:
        IP_FLOW_BUFFER.setdefault(src_ip, []).append(feat)
        if len(IP_FLOW_BUFFER[src_ip]) > MAX_BUFFER_SIZE:
            IP_FLOW_BUFFER[src_ip].pop(0)
        buf_snap = list(IP_FLOW_BUFFER[src_ip])
    
    score, is_anomaly = get_reaper_score(buf_snap, is_attack=False, atk_type='normal')
    flow['score'] = score
    flow['anomaly'] = is_anomaly
    
    return flow, ci, buf_snap, client

def generate_attack_flow(atk_key, atk_config):
    """Generate attack traffic (known or zero-day)"""
    with STATE_LOCK:
        clients = STATE['clients']
    
    if not clients:
        return None
    
    ci = random.randint(0, len(clients) - 1)
    client = clients[ci]
    
    proto = random.choice(PROTOS)
    port = random.choice(PORTS.get(proto, [80]))
    dst = random.choice(DESTS)
    
    bts = random.randint(*atk_config['bytes_range'])
    pkts = random.randint(50, 2000)
    
    is_zeroday = atk_config['type'] == 'zeroday'
    attack_class = atk_config['class'].value
    
    flow = {
        'time': ts(), 'src': client['ip'], 'dst': dst,
        'port': port, 'proto': proto,
        'bytes': bts, 'pkts': pkts,
        'score': 0.0, 'anomaly': True,
        'attack_type': atk_key,
        'traffic_class': 'zeroday' if is_zeroday else 'known',
        'cz_label': attack_class,
        'cz_confidence': random.randint(40, 95) if is_zeroday else random.randint(75, 98),
        'is_zero_day': is_zeroday,
    }
    
    feat = extract_flow_features(flow)
    src_ip = flow['src']
    
    with STATE_LOCK:
        IP_FLOW_BUFFER.setdefault(src_ip, []).append(feat)
        if len(IP_FLOW_BUFFER[src_ip]) > MAX_BUFFER_SIZE:
            IP_FLOW_BUFFER[src_ip].pop(0)
        buf_snap = list(IP_FLOW_BUFFER[src_ip])
    
    score, is_anomaly = get_reaper_score(buf_snap, is_attack=True, atk_type=atk_key)
    flow['score'] = score
    flow['anomaly'] = is_anomaly
    
    return flow, ci, buf_snap, client

def generate_flow():
    """Generate a single network flow with proper classification"""
    with STATE_LOCK:
        clients = STATE['clients']
        active = STATE['active_attacks']
    
    if not clients:
        return
    
    # Check if any client is under attack
    attacked_clients = {int(k): v for k, v in active.items() if int(k) < len(clients)}
    
    if attacked_clients:
        # Generate attack traffic for attacked client
        atk_idx = random.choice(list(attacked_clients.keys()))
        atk_info = attacked_clients[atk_idx]
        atk_key = atk_info['type']
        atk_config = ATTACK_PATTERNS.get(atk_key, ATTACK_PATTERNS['ddos'])
        
        flow, ci, buf_snap, client = generate_attack_flow(atk_key, atk_config)
        if flow is None:
            return
    else:
        # Generate normal traffic (90% normal, 10% random attack for demonstration)
        if random.random() < 0.1 and len(ATTACK_PATTERNS) > 0:
            # Random attack for demonstration
            atk_key = random.choice(list(ATTACK_PATTERNS.keys()))
            atk_config = ATTACK_PATTERNS[atk_key]
            flow, ci, buf_snap, client = generate_attack_flow(atk_key, atk_config)
        else:
            flow, ci, buf_snap, client = generate_normal_flow()
    
    if flow is None:
        return
    
    # Store in appropriate category
    with STATE_LOCK:
        STATE['flows'].append(flow)
        if len(STATE['flows']) > 500:
            STATE['flows'].pop(0)
        
        STATE['clients'][ci]['flows'] += 1
        STATE['clients'][ci]['bytes'] += flow['bytes']
        if flow['anomaly']:
            STATE['clients'][ci]['anomalies'] += 1
        
        STATE['anomaly_scores'].append(flow['score'])
        if len(STATE['anomaly_scores']) > 200:
            STATE['anomaly_scores'].pop(0)
        
        STATE['pipeline_stats']['capture'] += 1
        STATE['pipeline_stats']['iptrie'] += 1
        STATE['pipeline_stats']['rnn'] += 1
        STATE['pipeline_stats']['vae_score'] = flow['score']
        STATE['pipeline_stats']['last_src'] = flow['src']
        STATE['pipeline_stats']['last_dst'] = f"{flow['dst']}:{flow['port']}"
        if flow['anomaly']:
            STATE['pipeline_stats']['flagged'] += 1
        
        # Store in category-specific lists
        if flow['traffic_class'] == 'normal':
            STATE['normal_traffic'].append(flow)
            if len(STATE['normal_traffic']) > 100:
                STATE['normal_traffic'].pop(0)
            STATE['cls_counts']['normal'] += 1
        elif flow['traffic_class'] == 'known':
            STATE['known_attacks'].append(flow)
            if len(STATE['known_attacks']) > 100:
                STATE['known_attacks'].pop(0)
            STATE['cls_counts']['known'] += 1
        else:  # zeroday
            STATE['zero_day_attacks'].append(flow)
            if len(STATE['zero_day_attacks']) > 100:
                STATE['zero_day_attacks'].pop(0)
            STATE['cls_counts']['zeroday'] += 1
    
    # Update client traffic class
    with STATE_LOCK:
        if ci < len(STATE['clients']):
            STATE['clients'][ci]['traffic_class'] = flow['traffic_class']
            STATE['clients'][ci]['last_attack'] = flow['attack_type'] if flow['attack_type'] != 'normal' else None
    
    with STATE_LOCK:
        ps = dict(STATE['pipeline_stats'])
    
    # Log based on classification
    if flow['traffic_class'] == 'normal':
        log(f"✅ NORMAL [{flow['src']}→{flow['dst']}:{flow['port']}] score={flow['score']:.3f}", 'success')
    elif flow['traffic_class'] == 'known':
        log(f"⚠️ KNOWN ATTACK [{flow['src']}→{flow['dst']}:{flow['port']}] type={flow['attack_type']} score={flow['score']:.3f}", 'danger')
    else:
        log(f"🚨 ZERO-DAY ATTACK [{flow['src']}→{flow['dst']}:{flow['port']}] type={flow['attack_type']} score={flow['score']:.3f}", 'critical')
    
    socketio.emit('pipeline_step', {
        'step': 'vae',
        'score': flow['score'],
        'anomaly': flow['anomaly'],
        'src': flow['src'],
        'dst': f"{flow['dst']}:{flow['port']}",
        'proto': flow['proto'],
        'stats': ps,
    })
    
    # Generate contour and run CZ classification for anomalies
    if flow['anomaly'] or flow['attack_type'] != 'normal':
        def pipeline_thread(buf=buf_snap, atk=flow['attack_type'], sc=flow['score'],
                           cl=client, d=flow['dst'], p=flow['port'], flow_data=flow):
            window_np = np.array(buf[-MAX_BUFFER_SIZE:]) if len(buf) >= MIN_WINDOW_SIZE else \
                        np.random.randn(MAX_BUFFER_SIZE, 24).astype(np.float32)
            
            saved_path = save_contour_image(window_np, atk or 'unknown', sc)
            b64 = make_contour_b64(window_np, score=sc, atk_type=atk or 'unknown')
            
            if b64:
                with STATE_LOCK:
                    STATE['contour_images_b64'].append(b64)
                    if len(STATE['contour_images_b64']) > 30:
                        STATE['contour_images_b64'].pop(0)
            
            socketio.emit('new_contour', {
                'img': b64 or '',
                'has_img': bool(b64),
                'score': sc,
                'ts': ts(),
                'src': cl['ip'],
                'dst': f"{d}:{p}",
                'atk': atk or '?',
                'saved_path': saved_path or 'Not saved',
                'traffic_class': flow_data['traffic_class'],
            })
            
            with STATE_LOCK:
                cz_model = STATE['cz_model']
            
            label = 'Benign'
            conf = 85
            bucket = 'normal'
            is_zero_day = False
            zero_day_reason = ""
            
            if cz_model is not None:
                img_input = make_contour_for_cz(window_np)
                try:
                    preds = cz_model.predict(img_input, verbose=0)[0]
                    cls_idx = int(np.argmax(preds))
                    conf = int(np.max(preds) * 100)
                    label = ALL_CLASSES[cls_idx] if cls_idx < len(ALL_CLASSES) else f'Class{cls_idx}'
                    bucket = CZ_BUCKET_MAP.get(label, 'known')
                    
                    # Zero-day detection logic
                    is_zero_day, zero_day_reason = detect_zero_day(preds, label, conf)
                    
                    if is_zero_day:
                        bucket = 'zeroday'
                        log(f"🔍 ZERO-DAY DETECTED: {label} (conf={conf}%, reason={zero_day_reason}) ← {cl['ip']}", 'critical')
                    else:
                        log(f"🧠 CZ-ResViT: {label} ({conf}%) ← {cl['ip']} [{bucket}]", 'info')
                        
                except Exception as e:
                    print(f"CZ predict error: {e}")
                    label = 'ERROR'
                    conf = 0
                    bucket = 'benign'
                    is_zero_day = False
            else:
                # Simulate CZ behavior based on attack type
                atk_label_map = {
                    'ddos': 'DDoS', 'mirai': 'Mirai', 'webatk': 'WebAtk',
                    'recon': 'Recon', 'crypto': 'CryptoMiner', 'exfil': 'DataExfil',
                    'novel': 'ZeroDay_Novel', 'normal': 'Benign',
                }
                label = atk_label_map.get(atk or '', 'Benign')
                bucket = CZ_BUCKET_MAP.get(label, 'known')
                
                # Simulate zero-day detection for recon and novel attacks
                is_zero_day = label in ZERO_DAY_CLASSES
                if is_zero_day:
                    conf = random.randint(45, 70)
                    zero_day_reason = "zero_day_class"
                else:
                    conf = random.randint(75, 98)
            
            # Update client classification
            with STATE_LOCK:
                if ci < len(STATE['clients']):
                    if is_zero_day:
                        STATE['clients'][ci]['traffic_class'] = 'zeroday'
                    elif label == 'Benign':
                        STATE['clients'][ci]['traffic_class'] = 'normal'
                    else:
                        STATE['clients'][ci]['traffic_class'] = 'known'
            
            with STATE_LOCK:
                counts_snap = dict(STATE['cls_counts'])
            
            socketio.emit('classification', {
                'ts': ts(),
                'src': cl['ip'],
                'dst': f"{d}:{p}",
                'score': sc,
                'label': label,
                'conf': conf,
                'bucket': bucket,
                'cls_counts': counts_snap,
                'atk_type': atk or 'unknown',
                'traffic_class': flow_data['traffic_class'],
                'is_zero_day': is_zero_day,
                'zero_day_reason': zero_day_reason,
            })
            
            socketio.emit('pipeline_step', {
                'step': 'cz',
                'label': label,
                'conf': conf,
                'is_zero_day': is_zero_day,
            })
        
        threading.Thread(target=pipeline_thread, daemon=True).start()
    
    socketio.emit('new_flow', flow)
    return flow

def sim_loop():
    """Background simulation thread"""
    while STATE['sim_running']:
        try:
            generate_flow()
            emit_state()
        except Exception as e:
            print(f"[sim_loop] {e}")
        time.sleep(1.5)

# ============================================================================
# SOCKET EVENT HANDLERS
# ============================================================================

@socketio.on('connect')
def on_connect():
    emit_state()
    log('🔌 UI connected', 'success')

@socketio.on('start_sim')
def on_start_sim():
    STATE['sim_running'] = True
    threading.Thread(target=sim_loop, daemon=True).start()
    log('▶️ Simulation started', 'success')
    emit('sim_status', {'running': True})

@socketio.on('stop_sim')
def on_stop_sim():
    STATE['sim_running'] = False
    log('⏹️ Simulation stopped', 'warning')
    emit('sim_status', {'running': False})

@socketio.on('set_attack_duration')
def on_set_attack_duration(data):
    global ATTACK_DURATION
    ATTACK_DURATION = data.get('duration', 15)
    log(f'⏱️ Attack duration → {ATTACK_DURATION}s', 'info')

@socketio.on('add_client')
def on_add_client(data):
    ip = data.get('ip', '').strip()
    name = data.get('name', f"Client-{len(STATE['clients'])+1}")
    role = data.get('role', 'Workstation')
    if not ip:
        emit('error', {'msg': 'IP required'})
        return
    with STATE_LOCK:
        STATE['clients'].append({'ip': ip, 'name': name, 'role': role,
                                  'status': 'normal', 'flows': 0, 'bytes': 0, 'anomalies': 0,
                                  'traffic_class': 'normal', 'last_attack': None})
    log(f'➕ {ip} ({name})', 'info')
    emit_state()

@socketio.on('remove_client')
def on_remove_client(data):
    idx = data.get('index', -1)
    with STATE_LOCK:
        if 0 <= idx < len(STATE['clients']):
            STATE['clients'].pop(idx)
    emit_state()

@socketio.on('launch_attack')
def on_launch_attack(data):
    global ATTACK_DURATION
    target_idx = str(data.get('target', 0))
    atk_type = data.get('type', 'ddos')
    intensity = int(data.get('intensity', 5))
    duration = data.get('duration', ATTACK_DURATION)
    
    atk_config = ATTACK_PATTERNS.get(atk_type, ATTACK_PATTERNS['ddos'])
    is_zeroday = atk_config['type'] == 'zeroday'

    with STATE_LOCK:
        idx = int(target_idx)
        if idx >= len(STATE['clients']):
            emit('error', {'msg': 'Invalid client'})
            return
        client = STATE['clients'][idx]
        client['status'] = 'attack'
        STATE['active_attacks'][target_idx] = {
            'type': atk_type, 'intensity': intensity,
            'started': time.time(), 'duration': duration,
            'is_zeroday': is_zeroday,
        }

    attack_label = "ZERO-DAY" if is_zeroday else "KNOWN"
    log(f'💥 {attack_label} ATTACK: {atk_type.upper()} → {client["ip"]} | dur={duration}s', 'danger')
    socketio.emit('pipeline_start', {'atk_type': atk_type, 'target': client['ip'], 'duration': duration, 'is_zeroday': is_zeroday})

    def stop_after():
        time.sleep(duration)
        with STATE_LOCK:
            STATE['active_attacks'].pop(target_idx, None)
            if idx < len(STATE['clients']):
                STATE['clients'][idx]['status'] = 'normal'
                STATE['clients'][idx]['traffic_class'] = 'normal'
        log(f'✅ Attack on {client["ip"]} ended', 'warning')
        emit_state()
    threading.Thread(target=stop_after, daemon=True).start()
    emit_state()

@socketio.on('stop_attacks')
def on_stop_attacks():
    with STATE_LOCK:
        for k in list(STATE['active_attacks'].keys()):
            cidx = int(k)
            if cidx < len(STATE['clients']):
                STATE['clients'][cidx]['status'] = 'normal'
                STATE['clients'][cidx]['traffic_class'] = 'normal'
        STATE['active_attacks'].clear()
    log('🛑 All attacks stopped', 'warning')
    emit_state()

@socketio.on('reset_clients')
def on_reset_clients():
    with STATE_LOCK:
        for c in STATE['clients']:
            c.update({'status': 'normal', 'flows': 0, 'bytes': 0, 'anomalies': 0,
                      'traffic_class': 'normal', 'last_attack': None})
        STATE['active_attacks'].clear()
        STATE['flows'].clear()
        STATE['anomaly_scores'].clear()
        STATE['cls_counts'] = {'known': 0, 'zeroday': 0, 'benign': 0, 'normal': 0}
        STATE['pipeline_stats'] = {
            'capture': 0, 'iptrie': 0, 'rnn': 0,
            'vae_score': 0.0, 'flagged': 0,
            'last_src': '—', 'last_dst': '—',
        }
        STATE['contour_images_b64'].clear()
        STATE['normal_traffic'].clear()
        STATE['known_attacks'].clear()
        STATE['zero_day_attacks'].clear()
        IP_FLOW_BUFFER.clear()
    log('🔄 Reset complete', 'info')
    emit_state()

@socketio.on('load_model')
def on_load_model(data):
    if not REAPER_OK or not TORCH_OK:
        emit('error', {'msg': 'REAPER / PyTorch unavailable'})
        return

    model_path = data.get('path', 'REAPER/model/best_reaper_rnn_vae.pth')
    thresh_path = data.get('thresh', 'REAPER/model/reaper_threshold.npy')

    def _load():
        try:
            log('🔧 Loading REAPER model…', 'info')
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = REAPER_RNN_VAE(feature_dim=24, rnn_hidden=64, rnn_layers=2,
                                     vae_hidden_dims=[128, 64, 32], latent_dim=16)
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=device))
                log(f'✅ REAPER weights loaded from {model_path}', 'success')
            else:
                log(f'⚠️ {model_path} not found — using untrained model', 'warning')
            model.eval()

            thresh = 1.82
            if os.path.exists(thresh_path):
                thresh = float(np.load(thresh_path)[0])
                log(f'📊 Threshold: {thresh:.4f}', 'info')

            with STATE_LOCK:
                STATE['model'] = model
                STATE['threshold'] = thresh
            socketio.emit('model_loaded', {'threshold': thresh, 'device': device})
        except Exception as e:
            log(f'❌ REAPER load error: {e}', 'danger')
    threading.Thread(target=_load, daemon=True).start()

@socketio.on('load_cz_model')
def on_load_cz_model():
    if not TF_OK:
        emit('error', {'msg': 'TensorFlow not installed'})
        return

    def _load():
        try:
            log('🧠 Loading CZ-ResViT model…', 'info')

            model_path = os.path.join('RVIT', 'model', 'czresvit_best.keras')

            if not os.path.exists(model_path):
                log('❌ czresvit_best.keras not found in RVIT/model. Using simulated mode.', 'warning')
                socketio.emit('cz_model_loaded', {'status': 'simulated'})
                return

            log(f'📂 Found model at: {model_path}', 'info')
            model = _patch_and_load_keras(model_path)

            with STATE_LOCK:
                STATE['cz_model'] = model

            log(f'✅ CZ-ResViT loaded! Output: {model.output_shape} ({len(ALL_CLASSES)} classes)', 'success')
            socketio.emit('cz_model_loaded', {
                'status': 'loaded',
                'output_shape': str(model.output_shape),
                'classes': ALL_CLASSES,
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            log(f'⚠️ CZ-ResViT load error: {e} — using simulated mode', 'warning')
            socketio.emit('cz_model_loaded', {'status': 'simulated'})

    threading.Thread(target=_load, daemon=True).start()

# ============================================================================
# REST API ENDPOINTS
# ============================================================================

@app.route('/api/flows')
def api_flows():
    with STATE_LOCK:
        return jsonify(STATE['flows'][-50:])

@app.route('/api/stats')
def api_stats():
    with STATE_LOCK:
        flows = STATE['flows']
        total = len(flows)
        anom = sum(1 for f in flows if f['anomaly'])
        return jsonify({
            'total_flows': total,
            'anomalies': anom,
            'detection_rate': round(anom / total * 100, 1) if total else 0,
            'active_attacks': len(STATE['active_attacks']),
            'threshold': STATE['threshold'],
            'reaper_loaded': STATE['model'] is not None,
            'cz_loaded': STATE['cz_model'] is not None,
            'cls_counts': STATE['cls_counts'],
            'cz_classes': ALL_CLASSES,
            'pipeline_stats': STATE['pipeline_stats'],
            'known_classes': KNOWN_CLASSES,
            'zero_day_classes': ZERO_DAY_CLASSES,
        })

@app.route('/contour_images/<path:filename>')
def serve_contour_image(filename):
    return send_from_directory('contour_images', filename)

# ============================================================================
# HTML UI - WITH SEPARATE TABS AND ANOMALY TREND GRAPH
# ============================================================================

HTML_UI = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>REAPER-RVIT SECURITY | Zero-Day Detection NIDS</title>
<script src="https://cdn.socket.io/4.6.1/socket.io.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
:root {
  --bg-primary: #FFFFFF;
  --bg-secondary: #F8FAFC;
  --bg-tertiary: #F1F5F9;
  --border-light: #E2E8F0;
  --border-medium: #CBD5E1;
  --text-primary: #1E293B;
  --text-secondary: #64748B;
  --text-muted: #94A3B8;
  --accent-blue: #3B82F6;
  --accent-blue-light: #EFF6FF;
  --accent-blue-dark: #2563EB;
  --accent-green: #10B981;
  --accent-green-light: #ECFDF5;
  --accent-red: #EF4444;
  --accent-red-light: #FEF2F2;
  --accent-yellow: #F59E0B;
  --accent-yellow-light: #FFFBEB;
  --accent-purple: #8B5CF6;
  --accent-purple-light: #F5F3FF;
  --accent-orange: #F97316;
  --accent-orange-light: #FFF7ED;
  --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
  --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1);
  --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  --font-mono: 'JetBrains Mono', 'SF Mono', monospace;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
  font-family: var(--font-sans);
  background: var(--bg-primary);
  color: var(--text-primary);
  overflow: hidden;
}

::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--bg-secondary); border-radius: 4px; }
::-webkit-scrollbar-thumb { background: var(--border-medium); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-blue); }

.header {
  height: 64px;
  background: var(--bg-primary);
  border-bottom: 1px solid var(--border-light);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  position: relative;
  z-index: 100;
  box-shadow: var(--shadow-sm);
}

.logo { display: flex; align-items: center; gap: 12px; font-size: 18px; font-weight: 700; color: var(--accent-blue); }
.logo-icon { font-size: 28px; }
.header-right { display: flex; align-items: center; gap: 16px; }

.status-badge {
  padding: 6px 12px;
  border-radius: 20px;
  font-size: 11px;
  font-weight: 600;
  font-family: var(--font-mono);
}

.status-badge.active { background: var(--accent-green-light); color: var(--accent-green); border: 1px solid var(--accent-green); }
.status-badge.inactive { background: var(--bg-tertiary); color: var(--text-secondary); border: 1px solid var(--border-light); }
.status-badge.success { background: var(--accent-blue-light); color: var(--accent-blue); border: 1px solid var(--accent-blue); }
.status-badge.warning { background: var(--accent-yellow-light); color: var(--accent-yellow); border: 1px solid var(--accent-yellow); }
.status-badge.danger { background: var(--accent-red-light); color: var(--accent-red); border: 1px solid var(--accent-red); }

.live-indicator { display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--accent-green); }
.pulse-dot { width: 8px; height: 8px; background: var(--accent-green); border-radius: 50%; animation: pulse 2s infinite; }

@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); }
  70% { box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
  100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

.layout { display: flex; height: calc(100vh - 64px); position: relative; z-index: 1; }

.sidebar {
  width: 72px;
  background: var(--bg-secondary);
  border-right: 1px solid var(--border-light);
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 20px 0;
  gap: 12px;
}

.nav-btn {
  width: 48px;
  height: 48px;
  border-radius: 12px;
  background: transparent;
  border: 1px solid transparent;
  color: var(--text-secondary);
  font-size: 22px;
  cursor: pointer;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  justify-content: center;
}

.nav-btn:hover { background: var(--accent-blue-light); color: var(--accent-blue); transform: translateX(2px); }
.nav-btn.active { background: var(--accent-blue-light); border-color: var(--accent-blue); color: var(--accent-blue); box-shadow: var(--shadow-sm); }

.main { flex: 1; overflow-y: auto; padding: 24px; }
.pane { display: none; animation: fadeIn 0.3s ease; }
.pane.active { display: block; }

@keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

.card {
  background: var(--bg-primary);
  border: 1px solid var(--border-light);
  border-radius: 12px;
  padding: 20px;
  margin-bottom: 20px;
  box-shadow: var(--shadow-sm);
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
  padding-bottom: 12px;
  border-bottom: 2px solid var(--border-light);
}

.card-title { font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: var(--accent-blue); }

.metrics-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }

.metric-card {
  background: var(--bg-primary);
  border: 1px solid var(--border-light);
  border-radius: 12px;
  padding: 20px;
  position: relative;
  box-shadow: var(--shadow-sm);
}

.metric-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 3px;
  background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
  border-radius: 12px 12px 0 0;
}

.metric-label { font-size: 11px; font-weight: 600; text-transform: uppercase; color: var(--text-secondary); margin-bottom: 8px; }
.metric-value { font-size: 32px; font-weight: 800; font-family: var(--font-mono); margin-bottom: 4px; }
.metric-value.blue { color: var(--accent-blue); }
.metric-value.red { color: var(--accent-red); }
.metric-value.yellow { color: var(--accent-yellow); }
.metric-value.green { color: var(--accent-green); }
.metric-value.purple { color: var(--accent-purple); }
.metric-sub { font-size: 10px; color: var(--text-muted); }

.btn {
  padding: 8px 16px;
  border-radius: 8px;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  border: 1px solid transparent;
  font-family: var(--font-sans);
  transition: all 0.2s ease;
}

.btn-primary { background: var(--accent-blue); color: white; }
.btn-primary:hover { background: var(--accent-blue-dark); transform: translateY(-1px); box-shadow: var(--shadow-md); }
.btn-danger { background: var(--accent-red); color: white; }
.btn-danger:hover { background: #DC2626; transform: translateY(-1px); }
.btn-warning { background: var(--accent-orange); color: white; }
.btn-warning:hover { background: #EA580C; transform: translateY(-1px); }
.btn-secondary { background: var(--bg-tertiary); color: var(--text-primary); border-color: var(--border-light); }
.btn-secondary:hover { background: var(--border-light); transform: translateY(-1px); }
.btn-group { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 24px; }

.category-tabs {
  display: flex;
  gap: 8px;
  margin-bottom: 20px;
  border-bottom: 2px solid var(--border-light);
  padding-bottom: 8px;
}

.category-tab {
  padding: 8px 20px;
  border-radius: 20px;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  background: var(--bg-secondary);
  color: var(--text-secondary);
  transition: all 0.2s ease;
}

.category-tab:hover { background: var(--accent-blue-light); color: var(--accent-blue); }
.category-tab.active { background: var(--accent-blue); color: white; }
.category-tab.normal.active { background: var(--accent-green); }
.category-tab.known.active { background: var(--accent-red); }
.category-tab.zeroday.active { background: var(--accent-purple); }

.table-wrapper { overflow-x: auto; border-radius: 8px; }
.data-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.data-table th { text-align: left; padding: 12px; background: var(--bg-secondary); color: var(--text-secondary); font-weight: 600; border-bottom: 2px solid var(--border-light); }
.data-table td { padding: 12px; border-bottom: 1px solid var(--border-light); }
.data-table tr:hover td { background: var(--bg-secondary); }
.data-table tr.anomaly td { background: var(--accent-red-light); }
.data-table tr.zeroday td { background: var(--accent-purple-light); }

.badge {
  display: inline-block;
  padding: 4px 8px;
  border-radius: 6px;
  font-size: 10px;
  font-weight: 600;
  font-family: var(--font-mono);
}

.badge-normal { background: var(--accent-green-light); color: var(--accent-green); border: 1px solid var(--accent-green); }
.badge-known { background: var(--accent-red-light); color: var(--accent-red); border: 1px solid var(--accent-red); }
.badge-zeroday { background: var(--accent-purple-light); color: var(--accent-purple); border: 1px solid var(--accent-purple); }
.badge-critical { background: var(--accent-red-light); color: #DC2626; border: 1px solid var(--accent-red); }
.badge-high { background: #FEF3C7; color: #D97706; border: 1px solid var(--accent-yellow); }
.badge-low { background: var(--accent-green-light); color: var(--accent-green-dark); border: 1px solid var(--accent-green); }

.client-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; }
.client-card {
  background: var(--bg-primary);
  border: 1px solid var(--border-light);
  border-radius: 12px;
  padding: 16px;
  box-shadow: var(--shadow-sm);
  transition: all 0.2s ease;
}
.client-card:hover { transform: translateY(-2px); box-shadow: var(--shadow-md); }
.client-card.attacking-known { border-color: var(--accent-red); background: var(--accent-red-light); }
.client-card.attacking-zeroday { border-color: var(--accent-purple); background: var(--accent-purple-light); }
.client-ip { font-size: 14px; font-weight: 700; color: var(--accent-blue); margin-bottom: 4px; font-family: var(--font-mono); }
.client-name { font-size: 11px; color: var(--text-secondary); margin-bottom: 8px; }
.client-stats { display: flex; gap: 12px; font-size: 10px; font-family: var(--font-mono); color: var(--text-muted); }

.input-group { display: flex; gap: 12px; flex-wrap: wrap; }
.input-field {
  flex: 1;
  padding: 10px 12px;
  background: var(--bg-secondary);
  border: 1px solid var(--border-light);
  border-radius: 8px;
  font-family: var(--font-mono);
  font-size: 12px;
}
.input-field:focus { outline: none; border-color: var(--accent-blue); box-shadow: 0 0 0 3px var(--accent-blue-light); }

.attack-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 20px; }
.attack-btn {
  padding: 16px;
  background: var(--bg-secondary);
  border: 2px solid var(--border-light);
  border-radius: 10px;
  cursor: pointer;
  text-align: center;
  transition: all 0.2s ease;
}
.attack-btn:hover { border-color: var(--accent-blue); transform: translateY(-2px); box-shadow: var(--shadow-md); }
.attack-btn.selected { background: var(--accent-blue-light); border-color: var(--accent-blue); }
.attack-btn.zeroday { border-left: 4px solid var(--accent-purple); }
.attack-icon { font-size: 28px; margin-bottom: 8px; }
.attack-name { font-size: 11px; font-weight: 600; }

.slider { width: 100%; height: 4px; background: var(--border-light); border-radius: 2px; -webkit-appearance: none; }
.slider::-webkit-slider-thumb { -webkit-appearance: none; width: 16px; height: 16px; background: var(--accent-blue); border-radius: 50%; cursor: pointer; }

.contour-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(100px, 1fr)); gap: 12px; margin-top: 12px; }
.contour-item { aspect-ratio: 1; background: var(--bg-secondary); border: 1px solid var(--border-light); border-radius: 8px; overflow: hidden; cursor: pointer; position: relative; }
.contour-item:hover { transform: scale(1.05); box-shadow: var(--shadow-md); }
.contour-item img { width: 100%; height: 100%; object-fit: cover; }
.contour-overlay { position: absolute; bottom: 0; left: 0; right: 0; background: linear-gradient(transparent, rgba(0,0,0,0.8)); padding: 6px; font-size: 9px; color: white; }
.contour-badge { position: absolute; top: 4px; right: 4px; padding: 2px 6px; border-radius: 4px; font-size: 8px; font-weight: bold; }
.contour-badge.normal { background: var(--accent-green); }
.contour-badge.known { background: var(--accent-red); }
.contour-badge.zeroday { background: var(--accent-purple); }

.log-box { background: var(--bg-secondary); border: 1px solid var(--border-light); border-radius: 8px; padding: 12px; height: 200px; overflow-y: auto; font-family: var(--font-mono); font-size: 11px; }
.log-entry { padding: 6px 0; border-bottom: 1px solid var(--border-light); display: flex; gap: 12px; }
.log-time { color: var(--text-muted); flex-shrink: 0; }
.log-critical { color: var(--accent-red); font-weight: 600; }
.log-danger { color: var(--accent-orange); font-weight: 600; }
.log-info { color: var(--accent-blue); }
.log-success { color: var(--accent-green); }
.log-warning { color: var(--accent-yellow); }

.chart-container { height: 250px; position: relative; }

@media (max-width: 768px) {
  .sidebar { width: 60px; }
  .metrics-grid { grid-template-columns: repeat(2, 1fr); }
  .attack-grid { grid-template-columns: repeat(2, 1fr); }
}
</style>
</head>
<body>

<div class="header">
  <div class="logo"><span class="logo-icon">🛡️</span><span>REAPER-RVIT SECURITY · Zero-Day NIDS</span></div>
  <div class="header-right">
    <div class="live-indicator"><div class="pulse-dot"></div><span>LIVE</span></div>
    <span id="sim-badge" class="status-badge inactive">⏹ STOPPED</span>
    <span id="reaper-badge" class="status-badge inactive">REAPER</span>
    <span id="cz-badge" class="status-badge inactive">CZ-ResViT</span>
    <span id="clock" style="font-family: monospace;">--:--:--</span>
  </div>
</div>

<div class="layout">
  <div class="sidebar">
    <button class="nav-btn active" onclick="showPane('dashboard')">📊</button>
    <button class="nav-btn" onclick="showPane('traffic')">🚦</button>
    <button class="nav-btn" onclick="showPane('clients')">🖥️</button>
    <button class="nav-btn" onclick="showPane('attack')">⚔️</button>
    <button class="nav-btn" onclick="showPane('pipeline')">🔬</button>
    <button class="nav-btn" onclick="showPane('threats')">📡</button>
  </div>

  <div class="main">
    <!-- DASHBOARD PANE -->
    <div id="dashboard" class="pane active">
      <div class="metrics-grid">
        <div class="metric-card"><div class="metric-label">TOTAL FLOWS</div><div class="metric-value blue" id="total-flows">0</div><div class="metric-sub">processed</div></div>
        <div class="metric-card"><div class="metric-label">NORMAL</div><div class="metric-value green" id="normal-count">0</div><div class="metric-sub">benign traffic</div></div>
        <div class="metric-card"><div class="metric-label">KNOWN ATTACKS</div><div class="metric-value red" id="known-count">0</div><div class="metric-sub">detected</div></div>
        <div class="metric-card"><div class="metric-label">ZERO-DAY</div><div class="metric-value purple" id="zd-count">0</div><div class="metric-sub">novel attacks</div></div>
      </div>
      <div class="btn-group">
        <button class="btn btn-primary" onclick="emitStartSim()">▶ START</button>
        <button class="btn btn-secondary" onclick="emitStopSim()">⏹ STOP</button>
        <button class="btn btn-primary" onclick="emitLoadModel()">⚡ LOAD REAPER</button>
        <button class="btn btn-primary" onclick="emitLoadCZ()">🧠 LOAD CZ</button>
        <button class="btn btn-secondary" onclick="emitReset()">↺ RESET</button>
      </div>
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
        <div class="card">
          <div class="card-header"><div class="card-title">📡 LIVE FLOW STREAM</div></div>
          <div class="table-wrapper"><table class="data-table"><thead><tr><th>Time</th><th>Source</th><th>Dest</th><th>Score</th><th>Class</th><th>Status</th></tr></thead><tbody id="flow-table"></tbody></table></div>
        </div>
        <div class="card">
          <div class="card-header"><div class="card-title">📈 ANOMALY SCORE TREND</div></div>
          <div class="chart-container"><canvas id="score-chart"></canvas></div>
          <div style="margin-top:12px; font-size:10px; color:var(--text-muted); text-align:center;">
            <span style="display:inline-block; width:12px; height:2px; background:#3B82F6; margin-right:4px;"></span> Score Trend 
            <span style="display:inline-block; width:12px; height:2px; background:#EF4444; margin-left:12px; margin-right:4px;"></span> Threshold ({{ "{{ threshold }}" }})
          </div>
        </div>
      </div>
      <div class="card">
        <div class="card-header"><div class="card-title">📊 TRAFFIC CLASSIFICATION</div></div>
        <div style="display: flex; justify-content: center;"><canvas id="class-chart" height="180" style="max-width: 300px; max-height: fit-content;"></canvas></div>
      </div>
    </div>

    <!-- TRAFFIC CLASSIFICATION PANE -->
    <div id="traffic" class="pane">
      <div class="category-tabs">
        <div class="category-tab normal active" onclick="showCategory('normal')">✅ NORMAL TRAFFIC</div>
        <div class="category-tab known" onclick="showCategory('known')">⚠️ KNOWN ATTACKS</div>
        <div class="category-tab zeroday" onclick="showCategory('zeroday')">🚨 ZERO-DAY ATTACKS</div>
      </div>
      
      <div id="normal-view" class="category-view">
        <div class="card"><div class="card-header"><div class="card-title">📊 NORMAL TRAFFIC LOGS</div><span class="badge-normal">Benign Traffic</span></div>
          <div class="table-wrapper"><table class="data-table"><thead><tr><th>Time</th><th>Source</th><th>Dest</th><th>Proto</th><th>Bytes</th><th>Score</th></tr></thead><tbody id="normal-table"></tbody></table></div>
        </div>
      </div>
      
      <div id="known-view" class="category-view" style="display:none">
        <div class="card"><div class="card-header"><div class="card-title">⚠️ KNOWN ATTACK LOGS</div><span class="badge-known">Trained Attacks</span></div>
          <div class="table-wrapper"><table class="data-table"><thead><tr><th>Time</th><th>Source</th><th>Dest</th><th>Attack Type</th><th>Score</th><th>Confidence</th></tr></thead><tbody id="known-table"></tbody></table></div>
        </div>
      </div>
      
      <div id="zeroday-view" class="category-view" style="display:none">
        <div class="card"><div class="card-header"><div class="card-title">🚨 ZERO-DAY ATTACK LOGS</div><span class="badge-zeroday">Novel/Unknown Patterns</span></div>
          <div class="table-wrapper"><table class="data-table"><thead><tr><th>Time</th><th>Source</th><th>Dest</th><th>Attack Type</th><th>Score</th><th>Detection Reason</th></tr></thead><tbody id="zeroday-table"></tbody></table></div>
        </div>
      </div>
    </div>

    <!-- CLIENTS PANE -->
    <div id="clients" class="pane">
      <div class="card"><div class="card-header"><div class="card-title">➕ ADD CLIENT</div></div>
        <div class="input-group">
          <input type="text" id="client-ip" class="input-field" placeholder="IP Address">
          <input type="text" id="client-name" class="input-field" placeholder="Device Name">
          <select id="client-role" class="input-field"><option>Workstation</option><option>Server</option><option>IoT</option><option>Gateway</option></select>
          <button class="btn btn-primary" onclick="addClient()">Add</button>
        </div>
      </div>
      <div id="client-grid" class="client-grid"></div>
    </div>

    <!-- ATTACK SIMULATION PANE -->
    <div id="attack" class="pane">
      <div class="card"><div class="card-header"><div class="card-title">🎯 SELECT TARGET</div></div>
        <select id="attack-target" class="input-field" style="width:100%; margin-bottom:20px;"></select>
        <div class="card-header"><div class="card-title">⚙️ PARAMETERS</div></div>
        <div style="margin-bottom:16px;"><label>Duration: <span id="dur-display">15</span>s</label><input type="range" id="attack-duration" class="slider" min="5" max="60" value="15" oninput="document.getElementById('dur-display').innerText=this.value"></div>
        <div style="margin-bottom:24px;"><label>Intensity: <span id="int-display">5</span>/10</label><input type="range" id="attack-intensity" class="slider" min="1" max="10" value="5" oninput="document.getElementById('int-display').innerText=this.value"></div>
        <div class="card-header"><div class="card-title">💀 ATTACK TYPE</div></div>
        <div class="attack-grid" id="attack-type-grid"></div>
        <div class="btn-group">
          <button class="btn btn-danger" onclick="launchAttack()">⚡ LAUNCH</button>
          <button class="btn btn-secondary" onclick="emitStopAttacks()">🛑 STOP ALL</button>
        </div>
      </div>
    </div>

    <!-- PIPELINE PANE -->
    <div id="pipeline" class="pane">
      <div class="card"><div class="card-header"><div class="card-title">🖼 ANOMALY CONTOURS</div><span id="contour-count" style="font-size:11px;">0</span></div>
        <div class="contour-grid" id="contour-grid"><div style="grid-column:1/-1; text-align:center; padding:40px;">No anomalies yet</div></div>
      </div>
      <div class="card"><div class="card-header"><div class="card-title">📋 SYSTEM LOG</div></div>
        <div class="log-box" id="log-box"><div class="log-entry"><span class="log-time">[System]</span><span class="log-info">Ready - Zero-Day Detection Active</span></div></div>
      </div>
    </div>

    <!-- THREAT INTEL PANE -->
    <div id="threats" class="pane">
      <div class="card"><div class="card-header"><div class="card-title">🌍 THREAT INTELLIGENCE FEED</div></div>
        <div class="table-wrapper"><table class="data-table"><thead><tr><th>Time</th><th>Source</th><th>Type</th><th>Category</th><th>Confidence</th><th>Severity</th></tr></thead><tbody id="threat-table"></tbody></table></div>
      </div>
    </div>
  </div>
</div>

<script>
const socket = io();
let flows = [], clients = [], activeAttacks = {}, threshold = 1.82;
let selectedAttack = 'ddos';
let classChart = null;
let scoreChart = null;
let currentCategory = 'normal';

// Known and zero-day attack types
const knownAttacks = ['ddos', 'mirai', 'webatk'];
const zeroDayAttacks = ['recon', 'crypto', 'exfil', 'novel'];

socket.on('connect', () => addLog('Connected to Zero-Day NIDS', 'success'));
socket.on('log', (d) => addLog(d.msg, d.level));
socket.on('state', updateDashboard);
socket.on('new_flow', addFlow);
socket.on('sim_status', (d) => updateSimStatus(d.running));
socket.on('model_loaded', (d) => { threshold = d.threshold; addLog(`REAPER loaded | threshold=${d.threshold}`, 'success'); if(scoreChart) updateScoreChart(); });
socket.on('cz_model_loaded', (d) => addLog(`CZ-ResViT: ${d.status}`, 'success'));
socket.on('new_contour', addContour);
socket.on('classification', addClassification);
socket.on('pipeline_start', (d) => { 
  const type = d.is_zeroday ? 'ZERO-DAY' : 'KNOWN';
  addLog(`🚨 ${type} ATTACK: ${d.atk_type} → ${d.target}`, 'critical'); 
  showPane('pipeline');
});

function updateDashboard(data) {
  clients = data.clients || [];
  activeAttacks = data.active_attacks || {};
  threshold = data.threshold || 1.82;
  
  document.getElementById('total-flows').innerText = flows.length;
  document.getElementById('normal-count').innerText = data.cls_counts?.normal || 0;
  document.getElementById('known-count').innerText = data.cls_counts?.known || 0;
  document.getElementById('zd-count').innerText = data.cls_counts?.zeroday || 0;
  
  document.getElementById('reaper-badge').className = `status-badge ${data.model_loaded?'success':'inactive'}`;
  document.getElementById('reaper-badge').innerHTML = data.model_loaded?'✓ REAPER':'REAPER';
  document.getElementById('cz-badge').className = `status-badge ${data.cz_loaded?'success':'inactive'}`;
  document.getElementById('cz-badge').innerHTML = data.cz_loaded?'✓ CZ':'CZ';
  
  if(data.anomaly_scores && scoreChart){
    scoreChart.data.datasets[0].data = data.anomaly_scores.slice(-40);
    scoreChart.data.datasets[1].data = Array(40).fill(threshold);
    scoreChart.update('none');
  }
  
  if(classChart){
    classChart.data.datasets[0].data = [data.cls_counts?.normal||0, data.cls_counts?.known||0, data.cls_counts?.zeroday||0];
    classChart.update('none');
  }
  
  renderClients();
  renderAttackTarget();
  updateCategoryTables(data);
}

function updateScoreChart() {
  if(scoreChart) {
    let scores = document.getElementById('anomaly-scores-data');
    if(scores && scores.value) {
      let vals = JSON.parse(scores.value);
      scoreChart.data.datasets[0].data = vals.slice(-40);
      scoreChart.data.datasets[1].data = Array(40).fill(threshold);
      scoreChart.update('none');
    }
  }
}

function updateCategoryTables(data) {
  // Update normal traffic table
  let normalTable = document.getElementById('normal-table');
  if(data.normal_traffic) {
    normalTable.innerHTML = data.normal_traffic.slice().reverse().map(f => `
      <tr>
        <td style="font-family:monospace">${f.time}</td>
        <td style="color:#3B82F6">${f.src}</td>
        <td>${f.dst}:${f.port}</td>
        <td><span class="badge-normal">${f.proto}</span></td>
        <td>${f.bytes}</td>
        <td style="color:${f.score>threshold?'#EF4444':'#10B981'}">${f.score}</td>
      </tr>
    `).join('');
    if(data.normal_traffic.length === 0) normalTable.innerHTML = '<tr><td colspan="6" style="text-align:center">No normal traffic yet</td></tr>';
  }
  
  // Update known attacks table
  let knownTable = document.getElementById('known-table');
  if(data.known_attacks) {
    knownTable.innerHTML = data.known_attacks.slice().reverse().map(f => `
      <tr class="anomaly">
        <td style="font-family:monospace">${f.time}</td>
        <td style="color:#3B82F6">${f.src}</td>
        <td>${f.dst}:${f.port}</td>
        <td><span class="badge-known">${f.attack_type}</span></td>
        <td style="color:#EF4444">${f.score}</td>
        <td>${f.cz_confidence || '—'}%</td>
      </tr>
    `).join('');
    if(data.known_attacks.length === 0) knownTable.innerHTML = '<tr><td colspan="6" style="text-align:center">No known attacks detected</td></tr>';
  }
  
  // Update zero-day attacks table
  let zerodayTable = document.getElementById('zeroday-table');
  if(data.zero_day_attacks) {
    zerodayTable.innerHTML = data.zero_day_attacks.slice().reverse().map(f => `
      <tr class="zeroday">
        <td style="font-family:monospace">${f.time}</td>
        <td style="color:#3B82F6">${f.src}</td>
        <td>${f.dst}:${f.port}</td>
        <td><span class="badge-zeroday">${f.attack_type}</span></td>
        <td style="color:#F97316">${f.score}</td>
        <td><span class="badge-zeroday">Novel Pattern</span></td>
      </tr>
    `).join('');
    if(data.zero_day_attacks.length === 0) zerodayTable.innerHTML = '<tr><td colspan="6" style="text-align:center">No zero-day attacks detected</td></tr>';
  }
}

function addFlow(flow) {
  flows.unshift(flow);
  if(flows.length > 200) flows.pop();
  
  let tbody = document.getElementById('flow-table');
  let row = tbody.insertRow(0);
  
  let classBadge = '';
  if(flow.traffic_class === 'normal') classBadge = '<span class="badge-normal">Normal</span>';
  else if(flow.traffic_class === 'known') classBadge = '<span class="badge-known">Known Attack</span>';
  else classBadge = '<span class="badge-zeroday">Zero-Day!</span>';
  
  row.innerHTML = `
    <td style="font-family:monospace">${flow.time}</td>
    <td style="color:#3B82F6">${flow.src}</td>
    <td>${flow.dst}:${flow.port}</td>
    <td style="color:${flow.score>threshold?'#EF4444':'#10B981'}">${flow.score}</td>
    <td>${classBadge}</td>
    <td><span class="badge ${flow.anomaly?'badge-critical':'badge-low'}">${flow.anomaly?'ANOMALY':'OK'}</span></td>
  `;
  while(tbody.rows.length > 25) tbody.deleteRow(25);
}

function addClassification(data) {
  addLog(`[${data.traffic_class.toUpperCase()}] ${data.src} → ${data.label} (${data.conf}%)`, 
         data.is_zero_day ? 'critical' : (data.traffic_class === 'known' ? 'danger' : 'info'));
  
  let threatTable = document.getElementById('threat-table');
  let severity = data.traffic_class === 'zeroday' ? 'CRITICAL' : (data.traffic_class === 'known' ? 'HIGH' : 'LOW');
  let severityClass = data.traffic_class === 'zeroday' ? 'badge-zeroday' : (data.traffic_class === 'known' ? 'badge-known' : 'badge-normal');
  
  let threatRow = threatTable.insertRow(0);
  threatRow.innerHTML = `
    <td style="font-family:monospace">${data.ts}</td>
    <td style="color:#3B82F6">${data.src}</td>
    <td><span class="${severityClass}">${data.label}</span></td>
    <td><span class="${severityClass}">${data.traffic_class}</span></td>
    <td>${data.conf}%</span></td>
    <td><span class="badge ${severityClass}">${severity}</span></td>
  `;
  while(threatTable.rows.length > 50) threatTable.deleteRow(50);
}

function addContour(data) {
  let grid = document.getElementById('contour-grid');
  if(grid.children.length === 1 && grid.children[0].innerText.includes('No anomalies')) grid.innerHTML = '';
  let div = document.createElement('div');
  div.className = 'contour-item';
  let badgeClass = data.traffic_class === 'zeroday' ? 'zeroday' : (data.traffic_class === 'known' ? 'known' : 'normal');
  if(data.has_img && data.img) {
    div.innerHTML = `<img src="data:image/png;base64,${data.img}">
                     <div class="contour-badge ${badgeClass}">${data.traffic_class}</div>
                     <div class="contour-overlay"><span class="contour-score">${data.score.toFixed(3)}</span> | ${data.atk}</div>`;
  } else {
    div.innerHTML = `<div style="display:flex;align-items:center;justify-content:center;height:100%;flex-direction:column;">
                      <span>⚡</span><span>${data.score.toFixed(3)}</span>
                      <div class="contour-badge ${badgeClass}">${data.traffic_class}</div>
                     </div>`;
  }
  grid.insertBefore(div, grid.firstChild);
  while(grid.children.length > 12) grid.removeChild(grid.lastChild);
  document.getElementById('contour-count').innerText = grid.children.length;
}

function updateSimStatus(running) {
  let badge = document.getElementById('sim-badge');
  badge.innerHTML = running ? '▶ RUNNING' : '⏹ STOPPED';
  badge.className = `status-badge ${running ? 'active' : 'inactive'}`;
}

function renderClients() {
  let grid = document.getElementById('client-grid');
  grid.innerHTML = clients.map((c,i) => {
    let attackClass = '';
    let attackLabel = '';
    if(activeAttacks[i]) {
      let isZd = activeAttacks[i].is_zeroday;
      attackClass = isZd ? 'attacking-zeroday' : 'attacking-known';
      attackLabel = `<span style="color:${isZd ? '#8B5CF6' : '#EF4444'}">🔥 ${activeAttacks[i].type} ${isZd ? '(ZD)' : ''}</span>`;
    }
    let trafficClassBadge = '';
    if(c.traffic_class === 'normal') trafficClassBadge = '<span class="badge-normal">Normal</span>';
    else if(c.traffic_class === 'known') trafficClassBadge = '<span class="badge-known">Known Attack</span>';
    else if(c.traffic_class === 'zeroday') trafficClassBadge = '<span class="badge-zeroday">Zero-Day!</span>';
    
    return `
      <div class="client-card ${attackClass}">
        <div class="client-ip">${c.ip}</div>
        <div class="client-name">${c.name} <span class="badge-safe">${c.role}</span></div>
        <div class="client-stats">
          <span>📊 ${c.flows}</span>
          <span>⚠️ ${c.anomalies}</span>
          <span>${trafficClassBadge}</span>
        </div>
        ${attackLabel ? `<div style="margin-top:8px;">${attackLabel}</div>` : ''}
      </div>
    `;
  }).join('');
}

function renderAttackTarget() {
  let sel = document.getElementById('attack-target');
  let curr = sel.value;
  sel.innerHTML = clients.map((c,i)=>`<option value="${i}">${c.ip} — ${c.name}</option>`).join('');
  if(curr) sel.value = curr;
}

function showCategory(category) {
  currentCategory = category;
  document.getElementById('normal-view').style.display = category === 'normal' ? 'block' : 'none';
  document.getElementById('known-view').style.display = category === 'known' ? 'block' : 'none';
  document.getElementById('zeroday-view').style.display = category === 'zeroday' ? 'block' : 'none';
  
  document.querySelectorAll('.category-tab').forEach(tab => tab.classList.remove('active'));
  document.querySelector(`.category-tab.${category}`).classList.add('active');
}

function addLog(msg, level) {
  let box = document.getElementById('log-box');
  let div = document.createElement('div');
  div.className = 'log-entry';
  div.innerHTML = `<span class="log-time">[${new Date().toLocaleTimeString()}]</span><span class="log-${level}">${msg}</span>`;
  box.insertBefore(div, box.firstChild);
  while(box.children.length > 100) box.removeChild(box.lastChild);
}

function showPane(name) {
  document.querySelectorAll('.pane').forEach(p=>p.style.display='none');
  document.getElementById(name).style.display='block';
  document.querySelectorAll('.nav-btn').forEach(b=>b.classList.remove('active'));
  event.target.classList.add('active');
}

function emitStartSim() { socket.emit('start_sim'); }
function emitStopSim() { socket.emit('stop_sim'); }
function emitLoadModel() { socket.emit('load_model',{}); }
function emitLoadCZ() { socket.emit('load_cz_model'); }
function emitReset() { socket.emit('reset_clients'); }
function emitStopAttacks() { socket.emit('stop_attacks'); }

function addClient() {
  let ip = document.getElementById('client-ip').value.trim();
  let name = document.getElementById('client-name').value.trim();
  let role = document.getElementById('client-role').value;
  if(ip) socket.emit('add_client',{ip,name:name||'Device',role});
}

function launchAttack() {
  let target = parseInt(document.getElementById('attack-target').value);
  let duration = parseInt(document.getElementById('attack-duration').value);
  let intensity = parseInt(document.getElementById('attack-intensity').value);
  socket.emit('set_attack_duration',{duration});
  socket.emit('launch_attack',{target,type:selectedAttack,intensity,duration});
}

function initAttackGrid() {
  let attacks = [
    {id:'ddos',name:'DDoS',icon:'💣',type:'known'},
    {id:'mirai',name:'Mirai',icon:'🤖',type:'known'},
    {id:'webatk',name:'Web Attack',icon:'🌐',type:'known'},
    {id:'recon',name:'Recon (ZD)',icon:'🔍',type:'zeroday'},
    {id:'crypto',name:'CryptoMiner (ZD)',icon:'⛏️',type:'zeroday'},
    {id:'exfil',name:'Data Exfil (ZD)',icon:'📤',type:'zeroday'},
    {id:'novel',name:'Novel Attack (ZD)',icon:'🆕',type:'zeroday'}
  ];
  document.getElementById('attack-type-grid').innerHTML = attacks.map(a => `
    <div class="attack-btn ${a.type === 'zeroday' ? 'zeroday' : ''} ${a.id===selectedAttack?'selected':''}" onclick="selectAttack('${a.id}')">
      <div class="attack-icon">${a.icon}</div>
      <div class="attack-name">${a.name}</div>
      <div style="font-size:8px;margin-top:4px;">${a.type === 'zeroday' ? '🚨 Zero-Day' : 'Known'}</div>
    </div>
  `).join('');
}

function selectAttack(id) { 
  selectedAttack = id; 
  document.querySelectorAll('.attack-btn').forEach(btn=>btn.classList.remove('selected')); 
  event.target.closest('.attack-btn').classList.add('selected'); 
}

function initScoreChart() {
  let ctx = document.getElementById('score-chart')?.getContext('2d');
  if(!ctx) return;
  scoreChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: Array(40).fill(''),
      datasets: [
        {
          label: 'Anomaly Score',
          data: Array(40).fill(0),
          borderColor: '#3B82F6',
          backgroundColor: 'rgba(59,130,246,0.1)',
          fill: true,
          tension: 0.35,
          pointRadius: 3,
          pointBackgroundColor: '#3B82F6',
          borderWidth: 2
        },
        {
          label: 'Threshold',
          data: Array(40).fill(threshold),
          borderColor: '#EF4444',
          borderWidth: 2,
          borderDash: [5, 5],
          pointRadius: 0,
          fill: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: { display: false },
        tooltip: { mode: 'index', intersect: false }
      },
      scales: {
        y: { 
          min: 0, 
          max: 10, 
          title: { display: true, text: 'Anomaly Score', font: { size: 10 } },
          grid: { color: '#E2E8F0' }
        },
        x: { 
          title: { display: true, text: 'Recent Flows →', font: { size: 10 } },
          ticks: { display: false }
        }
      },
      elements: { point: { radius: 2, hoverRadius: 5 } }
    }
  });
}

function initClassChart() {
  let ctx = document.getElementById('class-chart')?.getContext('2d');
  if(!ctx) return;
  classChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Normal Traffic', 'Known Attacks', 'Zero-Day Attacks'],
      datasets: [{
        data: [0, 0, 0],
        backgroundColor: ['#10B981', '#EF4444', '#8B5CF6'],
        borderWidth: 0,
        hoverOffset: 10
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: { 
        legend: { position: 'bottom', labels: { font: { size: 10 } } },
        tooltip: { callbacks: { label: (ctx) => `${ctx.label}: ${ctx.raw} flows` } }
      }
    }
  });
}

setInterval(()=>{ let c=document.getElementById('clock'); if(c) c.innerText=new Date().toLocaleTimeString(); },1000);
window.addEventListener('load',()=>{ initScoreChart(); initClassChart(); initAttackGrid(); addLog('Zero-Day Detection NIDS Ready', 'success'); });
</script>
</body>
</html>'''

@app.route('/')
def index():
    return render_template_string(HTML_UI)

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("🛡️ REAPER-RVIT SECURITY - Zero-Day Detection NIDS")
    print("=" * 80)
    print("\n🎯 Traffic Classification:")
    print("   ✅ NORMAL TRAFFIC   - Benign network activity (Green)")
    print("   ⚠️ KNOWN ATTACKS    - DDoS, Mirai, Web Attacks (Red)")
    print("   🚨 ZERO-DAY ATTACKS - Recon, CryptoMiner, Data Exfil, Novel (Purple)")
    print("\n📈 Dashboard Features:")
    print("   • Anomaly Score Trend Graph (rises during attacks)")
    print("   • Threshold line (red dashed)")
    print("   • Traffic Classification Doughnut Chart")
    print("   • Separate tabs for each traffic category")
    print("\n🎨 Theme: Clean White Professional Edition")
    print("🌐 Dashboard: http://localhost:5000")
    print("\n📁 Required Models (place in model/):")
    print("   ├── REAPER/model/best_reaper_rnn_vae.pth    - REAPER weights")
    print("   ├── REAPER/model/reaper_threshold.npy       - Detection threshold")
    print("   └── RVIT/model/czresvit_best.keras        - CZ-ResViT model (optional)")
    print("\n✨ Zero-Day Detection Features:")
    print("   • Separate tabs for Normal/Known/Zero-Day traffic")
    print("   • Low confidence detection (<65% = potential zero-day)")
    print("   • Novel attack pattern identification")
    print("   • Real-time classification dashboard")
    print("   • Anomaly score trend visualization")
    print("=" * 80)
    
    socketio.run(app, host='0.0.0.0', port=5104, debug=False, allow_unsafe_werkzeug=True)