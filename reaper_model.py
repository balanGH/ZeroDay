# reaper_model.py
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Dropout

def build_reaper(input_shape):
    inputs = Input(shape=input_shape)

    x = LSTM(64, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)

    # 🔥 REAPER embedding layer
    embedding = GRU(32, name="reaper_embedding")(x)
    x = Dropout(0.2)(embedding)

    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model
