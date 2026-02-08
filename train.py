# train.py (updated)
import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

import tensorflow as tf
from reaper_model import build_reaper
from preprocessing import load_data

print("▶ Loading IoT-23 data...")
# Note: load_iot23_data returns (X, y, feature_cols) but we only need X, y
X, y = load_data("dataset/iot23_combined_new.csv", dataset_type='iot23')

# If load_data returns 3 values, extract only X, y
if isinstance(X, tuple) and len(X) == 3:
    X, y, _ = X

print(f"▶ Data loaded: X shape = {X.shape}, y shape = {y.shape}")

print("▶ Building REAPER model...")
model = build_reaper(input_shape=(X.shape[1], X.shape[2]))

print(f"▶ Training on {X.shape[0]} IoT sequences...")

# Split for validation
split_idx = int(0.8 * len(X))
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

print(f"▶ Train samples: {len(X_train)}, Validation samples: {len(X_val)}")

# Use TF Dataset for efficiency
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(10000).batch(32).prefetch(2)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_ds = val_ds.batch(32).prefetch(2)

# Train with callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )
]

print("▶ Starting training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=callbacks,
    verbose=1
)

# Save the model
model.save("model/reaper_model_iot23.h5")
print("✅ IoT-23 REAPER model saved!")

# Print training summary
print(f"\n▶ Training Summary:")
print(f"   Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"   Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"   Final Training Loss: {history.history['loss'][-1]:.4f}")
print(f"   Final Validation Loss: {history.history['val_loss'][-1]:.4f}")