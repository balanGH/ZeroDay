# REAPER + Residual Vision Transformer with Zero-Shot Learning

## Zero-Day Attack Detection in IoT Networks

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Project Overview

This repository presents a **hybrid Intrusion Detection System (IDS)** for IoT networks that integrates three cutting-edge approaches:

| Component | Function |
|-----------|----------|
| **REAPER** | Real-time malicious traffic detection using deep time-series embeddings (RNN-VAE) |
| **CZ-ResViT** | Contour-based Zero-day Residual Vision Transformer for global traffic pattern representation |
| **Zero-Shot Learning (ZSL)** | Detection of previously unseen (zero-day) attacks using semantic attribute vectors |

The system is implemented as a **complete end-to-end pipeline** with:
- Real-time traffic simulation and monitoring
- Web-based dashboard with anomaly visualization
- Separate classification tabs for Normal, Known, and Zero-Day traffic
- Anomaly score trend graph with threshold visualization
- Gradio interface for interactive testing

---

## 🎯 Problem Statement

IoT networks are increasingly vulnerable to:
- **Zero-day attacks** - Novel exploits with no known signatures
- **Rapidly evolving malware** - Polymorphic and metamorphic variants
- **Scarcity of labeled attack data** - Limited training examples for rare attacks

Traditional signature-based and supervised IDS solutions struggle to generalize to unseen attacks. This project addresses the problem by combining:
- Temporal traffic modeling (REAPER)
- Transformer-based visual representation learning (CZ-ResViT)
- Semantic inference via Zero-Shot Learning

---

## 📄 Referenced Research Papers

| Paper | Venue | Link |
|-------|-------|------|
| **REAPER**: Real-Time Detection of Malicious Traffic via Deep Time-Series Embedding Analysis | IEEE Transactions on Networking, 2025 | [IEEE Xplore](https://ieeexplore.ieee.org/document/11192781) |
| **Zero-Day Attack Detection** using Residual Vision Transformer with Zero-Shot Learning | IEEE Open Journal of the Communications Society, 2025 | [IEEE Xplore](https://ieeexplore.ieee.org/document/11151630) |

---

## 🏗️ System Architecture
[]
---

## 📊 Traffic Classification Categories

| Category | Description | Color | Examples |
|----------|-------------|-------|----------|
| **NORMAL** | Benign network activity | 🟢 Green | Regular HTTP/HTTPS, DNS queries |
| **KNOWN ATTACK** | Attacks seen during training | 🔴 Red | DDoS, Mirai, Web Attacks |
| **ZERO-DAY ATTACK** | Novel/Unseen attack patterns | 🟣 Purple | Recon, CryptoMiner, Data Exfiltration |

---

## 🚀 Features

### Core Capabilities
- ✅ **Real-time traffic monitoring** with WebSocket streaming
- ✅ **REAPER RNN-VAE** for temporal anomaly detection with contour generation
- ✅ **CZ-ResViT (ResNet-50 + Transformer)** for contour image classification
- ✅ **Zero-Shot Learning** for detecting unseen attack classes
- ✅ **Interactive Web Dashboard** with live updates
- ✅ **Gradio Interface** for easy testing and demonstration

### Dashboard Features
- 📈 **Anomaly Score Trend Graph** - Visualizes score changes with threshold line
- 🥧 **Traffic Classification Chart** - Doughnut chart showing class distribution
- 🖼️ **Contour Image Gallery** - Correlation contour visualizations of anomalies
- 📋 **Separate Category Tabs** - Dedicated views for Normal/Known/Zero-Day traffic
- 🖥️ **Client Management** - Add/remove monitored devices
- ⚔️ **Attack Simulation** - Launch known or zero-day attacks on specific targets

### Attack Types Supported

| Attack ID | Name | Category | Description |
|-----------|------|----------|-------------|
| `ddos` | DDoS | Known | Distributed Denial of Service |
| `mirai` | Mirai | Known | IoT Botnet Malware |
| `webatk` | Web Attack | Known | SQLi, XSS, Web Exploits |
| `recon` | Reconnaissance | Zero-Day | Port scanning, network mapping |
| `crypto` | CryptoMiner | Zero-Day | Cryptocurrency mining malware |
| `exfil` | Data Exfiltration | Zero-Day | Unauthorized data transfer |
| `novel` | Novel Attack | Zero-Day | Generic unknown attack pattern |

---

## 📁 Project Structure

```
ZeroDay/
│
├── app.py                      # Main Flask web application
├── setup.py                    # Environment setup script
├── requirements.txt            # Python dependencies
│
├── REAPER/                     # Module 1: REAPER Implementation
│   ├── __init__.py            # Package exports
│   ├── reaper.py              # RNN-VAE model + training pipeline
│   ├── reaper_output.txt      # Training output log
│   └── model/                 # Trained model artifacts
│       ├── best_reaper_rnn_vae.pth   # PyTorch model weights
│       ├── reaper_threshold.npy      # Detection threshold
│       └── scaler.pkl                # Feature scaler
│
├── RVIT/                       # Module 2: CZ-ResViT + ZSL
│   ├── RVIT.ipynb             # Complete training notebook
│   │   ├── Dataset preprocessing (IoT-23, CIC IoT 2023, IoTID20)
│   │   ├── SHAP feature selection (Top 15 features)
│   │   ├── Contour image generation
│   │   ├── CNN training
│   │   ├── ResNet-50 training
│   │   ├── Vision Transformer (ViT) training
│   │   └── CZ-ResViT training with two-phase fine-tuning
│   └── model/
│       └── czresvit_best.keras       # Trained CZ-ResViT model
│
├── dataset/                   # dataset files
├── contour_images/            # Generated contour images (auto-created)
├── logs/                      # System logs (auto-created)
└── exports/                   # Exported data (auto-created)

```

---

## 🔧 Installation

### Prerequisites
- **Python 3.10 or 3.11** (strict requirement)
- pip package manager
- CUDA-capable GPU (recommended for training)

### Step 1: Clone Repository
```bash
git clone https://github.com/balanGH/ZeroDay.git
cd ZeroDay-Detection
```

### Step 2: Run Setup Script
```bash
python setup.py
```

This will:
- Create a virtual environment (`venv/`)
- Install all dependencies
- Setup TensorFlow Metal support (macOS)

### Step 3: Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### Step 4: Install Dependencies Manually (if needed)
```bash
pip install -r requirements.txt
```

### Step 5: Download Pre-trained Models

Place the following files in their respective directories:

| File | Location | Purpose |
|------|----------|---------|
| `best_reaper_rnn_vae.pth` | `REAPER/model/` | REAPER RNN-VAE weights |
| `reaper_threshold.npy` | `REAPER/model/` | Anomaly detection threshold |
| `scaler.pkl` | `REAPER/model/` | Feature normalization scaler |
| `czresvit_best.keras` | `RVIT/model/` | CZ-ResViT model |

---

## 🚀 Running the Application

### Start the Web Server
```bash
python app.py
```

### Access Dashboard
Open your browser and navigate to: **http://localhost:5000**

### Dashboard Controls

| Button | Function |
|--------|----------|
| **START** | Begin traffic simulation |
| **STOP** | Halt traffic generation |
| **LOAD REAPER** | Load REAPER model weights |
| **LOAD CZ** | Load CZ-ResViT model |
| **RESET** | Clear all data and reset state |

---

## 🧪 Usage Guide

### 1. Start Simulation
Click **START** to begin generating synthetic network traffic. The system will:
- Generate normal traffic (90% of flows)
- Randomly inject attacks (10% of flows)
- Display flows in real-time table

### 2. Load Models
- Click **LOAD REAPER** to activate the RNN-VAE anomaly detector
- Click **LOAD CZ** to activate the CZ-ResViT zero-shot classifier

### 3. Monitor Traffic
- **Dashboard Tab**: View overall statistics and anomaly score trend
- **Traffic Tab**: Filter by Normal/Known/Zero-Day categories
- **Clients Tab**: Monitor individual device status
- **Pipeline Tab**: View contour images of detected anomalies

### 4. Launch Attacks
1. Navigate to **Attack Tab**
2. Select a target client
3. Choose attack type (Known or Zero-Day)
4. Set duration and intensity
5. Click **LAUNCH**

### 5. View Zero-Day Detection
When a zero-day attack is launched:
- Purple badges appear in the UI
- Contour images are generated
- Threat intelligence feed updates
- Zero-Day tab shows detailed logs

---

## 📊 Model Training

### REAPER Training
```python
from REAPER.reaper import main as train_reaper
results = train_reaper()  # Trains on CICIoT2023 dataset
```

Training outputs:
- `model/best_reaper_rnn_vae.pth` - Model weights
- `model/reaper_threshold.npy` - Detection threshold
- `module1_outputs/` - Embeddings and contour images for Module 2

### CZ-ResViT Training (Two-Phase)

The CZ-ResViT training follows a two-phase approach as described in the paper:

**Phase 1: Train Transformer Head (ResNet Frozen)**
- ResNet-50 backbone frozen (ImageNet pre-trained)
- Only transformer encoder + MLP head trained
- Learning rate: 1e-4
- Epochs: 10

**Phase 2: Fine-tune Top 50 ResNet Layers**
- Unfreeze top 50 layers of ResNet-50
- Fine-tune entire model with lower learning rate (1e-5)
- Epochs: 20

```python
# Run the complete training pipeline in RVIT.ipynb
# Includes:
# 1. IoT-23 dataset preprocessing
# 2. SHAP feature selection (Top 15 features)
# 3. Contour image generation
# 4. CNN, ResNet, ViT, and CZ-ResViT training
# 5. Zero-day evaluation on CIC IoT 2023 and IoTID20
```

---

## 📈 Performance Metrics

### REAPER (CICIoT2023 Test Results)

| Metric | Value |
|--------|-------|
| Test Accuracy | **97.27%** |
| Test Precision | **94.82%** |
| Test Recall | **100.00%** |
| Test F1-Score | **97.34%** |
| Zero-Day Detection Rate | **100.00%** |

### CZ-ResViT (IoT-23 Test Results)

| Metric | Value | Paper |
|--------|-------|-------|
| Accuracy | **97.75%** | 98% |

### Zero-Day Detection Results

| Dataset | Model | Accuracy (Ours) | Paper |
|---------|-------|-----------------|-------|
| CIC IoT 2023 | CNN | 44.02% | 47% |
| CIC IoT 2023 | ResNet-50 | 38.70% | 77% |
| CIC IoT 2023 | ViT | 80.00% | 72% |
| CIC IoT 2023 | **CZ-ResViT** | 44.32% | 82% |
| IoTID20 | CNN | 63.54% | 49% |
| IoTID20 | ResNet-50 | 55.40% | 75% |
| IoTID20 | ViT | 75.00% | 71% |
| IoTID20 | **CZ-ResViT** | 54.44% | 81% |

**Note:** Zero-day detection treats all non-Benign classes as "Malicious" (binary classification).

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| **PyTorch** | REAPER RNN-VAE implementation |
| **TensorFlow/Keras** | CZ-ResViT, CNN, ResNet-50, ViT models |
| **Flask + SocketIO** | Web server and real-time updates |
| **Matplotlib** | Contour image generation |
| **Pandas/NumPy** | Data processing |
| **scikit-learn** | Feature scaling, SHAP analysis, metrics |
| **Chart.js** | Dashboard visualizations |
| **Gradio** | Interactive testing interface |
| **OpenCV** | Image processing |

---

## 📋 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main dashboard UI |
| `/api/flows` | GET | Get recent flows (JSON) |
| `/api/stats` | GET | Get system statistics (JSON) |
| `/contour_images/<filename>` | GET | Serve contour images |

### WebSocket Events

| Event | Direction | Description |
|-------|-----------|-------------|
| `connect` | Server → Client | Initial connection |
| `start_sim` | Client → Server | Start traffic simulation |
| `stop_sim` | Client → Server | Stop simulation |
| `load_model` | Client → Server | Load REAPER model |
| `load_cz_model` | Client → Server | Load CZ-ResViT model |
| `launch_attack` | Client → Server | Launch attack on target |
| `state` | Server → Client | Full system state update |
| `new_flow` | Server → Client | New traffic flow detected |
| `classification` | Server → Client | CZ-ResViT classification result |
| `new_contour` | Server → Client | New contour image generated |

---

## 📝 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 5000 | Flask server port |
| `HOST` | 0.0.0.0 | Server host address |
| `ZERO_DAY_CONFIDENCE_THRESHOLD` | 65 | Confidence threshold for zero-day detection |

---

## 🔬 Zero-Day Detection Logic

The system flags traffic as **Zero-Day** under these conditions:

| Condition | Description | Threshold |
|-----------|-------------|-----------|
| **Zero-Day Class** | Model predicts a zero-day class | Label in `ZERO_DAY_CLASSES` |
| **Low Confidence** | Model is uncertain about prediction | Confidence < 65% |
| **High Entropy** | Flat prediction distribution | Entropy > 1.5 |

```python
def detect_zero_day(prediction_probs, predicted_label, confidence):
    if predicted_label in ZERO_DAY_CLASSES:
        return True, "zero_day_class"
    if confidence < ZERO_DAY_CONFIDENCE_THRESHOLD:
        return True, "low_confidence"
    entropy = -np.sum(prediction_probs * np.log(prediction_probs + 1e-7))
    if entropy > 1.5:
        return True, "high_entropy"
    return False, "known"
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

**Project Lead:** balanGH
**Project Link:** [https://github.com/balanGH/ZeroDay](https://github.com/balanGH/ZeroDay)

---

## 🙏 Acknowledgments

- IEEE for publishing the foundational research papers
- CICIoT2023, IoT-23, and IoTID20 dataset providers
- Open-source community for PyTorch, TensorFlow, Flask, and Gradio

---

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{tang2025reaper,
  title={REAPER: Real-Time Detection of Malicious Traffic via Deep Time-Series Embedding Analysis},
  author={Tang, Dan and Liu, Boru and Qin, Zheng and Liang, Wei and Li, Keqin and Jin, Wenqiang},
  journal={IEEE Transactions on Networking},
  year={2025}
}

@article{nitrat2025zeroday,
  title={Zero-Day Attack Detection in IoT Networks Using a Residual Vision Transformer-Based Approach With Zero-Shot Learning},
  author={Nitrat, Komcharn and Suetrong, Nopparuj and Promsuk, Natthanan},
  journal={IEEE Open Journal of the Communications Society},
  year={2025}
}
```

---

## ⚠️ Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'torch'` | Run `pip install torch` |
| `REAPER import failed` | Ensure REAPER module is in PYTHONPATH |
| `CZ model load error` | Check that `czresvit_best.keras` exists in `RVIT/model/` |
| `Port 5000 already in use` | Change port in app.py or kill existing process |
| `Matplotlib backend error` | Set `matplotlib.use('Agg')` before importing pyplot |
| `Out of memory during training` | Reduce batch size or use gradient accumulation |

### Debug Mode
```bash
python app.py --debug
```

---

## 🔮 Future Work

- [ ] Integrate with real PCAP capture (Scapy)
- [ ] Add more zero-day attack patterns
- [ ] Implement incremental learning for new attack classes
- [ ] Add REST API for external integration
- [ ] Deploy with Docker containerization
- [ ] Add Prometheus metrics export
- [ ] Implement multi-tenant support
- [ ] Optimize CZ-ResViT for edge deployment
- [ ] Add explainable AI (XAI) for detection decisions

---


# Contributors

Thanks to these amazing contributors:

* **lashmie** – https://github.com/lashmie
* **vivamuss** – https://github.com/vivamuss
* **shrinigashthiyagarajan** – https://github.com/shrinigashthiyagarajan


<a href="https://github.com/balanGH/ZeroDay/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=balanGH/ZeroDay" />
</a>

**Built with ❤️ for IoT Security Research**