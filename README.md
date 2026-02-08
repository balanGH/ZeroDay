# Zero-Day Attack Detection in IoT Networks  
### REAPER + Residual Vision Transformer with Zero-Shot Learning

This repository presents a **hybrid Intrusion Detection System (IDS)** for IoT networks that integrates:

- **REAPER** – Real-time malicious traffic detection using deep time-series embeddings  
- **Residual Vision Transformer (ViT)** – Global traffic pattern representation learning  
- **Zero-Shot Learning (ZSL)** – Detection of previously unseen (zero-day) attacks  

The project is implemented as a **semester-long academic research project**, inspired by recent **IEEE conference and journal papers**, with full experimentation and evaluation.

---

## 📌 Project Motivation

IoT networks are increasingly vulnerable to:

- Zero-day attacks  
- Rapidly evolving malware  
- Scarcity of labeled attack data  

Traditional signature-based and supervised IDS solutions struggle to generalize to unseen attacks.  
This project addresses the problem by combining:

- **Temporal traffic modeling**
- **Transformer-based visual representation learning**
- **Semantic inference via Zero-Shot Learning**

---

## 📄 Referenced Research Papers

1. **REAPER: Real-Time Detection of Malicious Traffic via Deep Time-Series Embedding Analysis**  
   Dan Tang, Boru Liu, Zheng Qin, Wei Liang, Keqin Li & Wenqiang Jin,  
   *REAPER: Real-Time Detection of Malicious Traffic via Deep Time-Series Embedding Analysis*,  
   **IEEE Transactions on Networking**, 2025.  
   🔗 https://ieeexplore.ieee.org/document/11192781

2. **Zero-Day Attack Detection in IoT Networks Using a Residual Vision Transformer-Based Approach With Zero-Shot Learning**  
   Komcharn Nitrat, Nopparuj Suetrong & Natthanan Promsuk,  
   *Zero-Day Attack Detection in IoT Networks Using a Residual Vision Transformer-Based Approach With Zero-Shot Learning*,  
   **IEEE Open Journal of the Communications Society**, 2025.  
   🔗 https://ieeexplore.ieee.org/document/11151630


---

## 🏗️ System Architecture

```

PCAP Traffic
↓
Flow Extraction (NFStream)
↓
Time-Series Construction
↓
REAPER (LSTM + GRU)
↓
Latent Traffic Embeddings
↓
Traffic Image Representation
↓
Residual Vision Transformer
↓
Zero-Shot Learning Module
↓
Zero-Day Attack Detection

```

---

## 📁 Repository Structure

```

ZeroDay/
├── data/
│   └── flows.csv
├── scripts/
│   └── pcap_to_flow.py
├── preprocessing.py
├── reaper_model.py
├── train.py
├── evaluate.py
├── model/
│   └── reaper_model.h5
├── requirements.txt
└── README.md

````

---

## ⚙️ Technologies Used

- **Python 3**
- **TensorFlow / Keras**
- **NFStream**
- **NumPy, Pandas**
- **Scikit-learn**
- **Vision Transformers (ViT)** *(Phase-2)*

---

## 🚀 Phase-1: REAPER Implementation

### Key Features

- Flow-based traffic representation
- Sequential time-series modeling
- Deep embedding generation
- Binary malicious traffic classification
- Real-time capable architecture

### Model Architecture

- LSTM (64 units)
- GRU (32 units) → **REAPER Embedding**
- Fully Connected Layer
- Sigmoid Output Layer

---

## 🔥 Phase-2: Zero-Day Attack Detection (Ongoing)

- Transform REAPER embeddings into 2D traffic images
- Apply a **Residual Vision Transformer**
- Use **Zero-Shot Learning** to classify unseen attacks
- Semantic similarity-based inference for zero-day detection

---

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Zero-Day Detection Rate *(Phase-2)*

---

## 🧪 Dataset

- PCAP-based IoT network traffic
- Flow extraction using **NFStream**
- Planned datasets:
  - **CIC-IDS2017**
  - **IoT-23**

---

## ▶️ How to Run (Phase-1)

### Install Dependencies
```bash
pip install -r requirements.txt
````

### Convert PCAP to Network Flows

```bash
python scripts/pcap_to_flow.py
```

### Train the REAPER Model

```bash
python train.py
```

### Evaluate the Model

```bash
python evaluate.py
```

---

## 📌 Project Status

* ✅ Flow extraction & preprocessing
* ✅ REAPER time-series model
* ✅ Traffic embedding generation
* ⏳ Vision Transformer integration
* ⏳ Zero-Shot Learning module
* ⏳ Full zero-day attack evaluation

---

## 👨‍🎓 Author

**Balan**
Final Year B.Tech / B.Tech Project

GitHub: [https://github.com/balanGH](https://github.com/balanGH)

---

## 📜 License

This project is intended **strictly for academic and research purposes only**.

---

## 🔧 GitHub Branch Fix (Important)

If you encounter a branch mismatch error:

* Local branch: `master`
* GitHub default branch: `main`

Run the following commands **once**:

```bash
git branch -M main
git remote add origin https://github.com/balanGH/ZeroDay.git
git push -u origin main
```