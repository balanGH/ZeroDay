# extract_embeddings.py (updated)
import numpy as np
from tensorflow.keras.models import load_model, Model
from preprocessing import load_data

print("▶ Loading IoT-23 trained REAPER...")
reaper = load_model("model/reaper_model_iot23.h5")

# Create embedding extractor
embedding_model = Model(
    inputs=reaper.input,
    outputs=reaper.get_layer("reaper_embedding").output
)

print("▶ Loading full IoT-23 dataset for embedding extraction...")
X_full, y_full = load_data("dataset/iot23_combined_new.csv", dataset_type='iot23')

print("▶ Running REAPER inference on IoT traffic...")
p_mal = reaper.predict(X_full, batch_size=32).flatten()

print("▶ Extracting embeddings...")
embeddings = embedding_model.predict(X_full, batch_size=32)

# Filter with adaptive threshold
threshold = np.percentile(p_mal, 95)  # Top 5% suspicious
suspicious_idx = p_mal >= threshold

suspicious_embeddings = embeddings[suspicious_idx]
suspicious_labels = y_full[suspicious_idx]

print(f"▶ Found {suspicious_embeddings.shape[0]} suspicious IoT flows")
print(f"▶ Suspicious rate: {suspicious_embeddings.shape[0]/len(X_full)*100:.2f}%")

# Save for Paper 2
np.save("data/iot23_reaper_embeddings.npy", suspicious_embeddings)
np.save("data/iot23_reaper_labels.npy", suspicious_labels)

print("✅ IoT embeddings saved for Residual Vision Transformer!")