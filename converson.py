import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import tensorflow as tf

layers = tf.keras.layers
models = tf.keras.models
# ---------------------------
# Config
# ---------------------------
DATASET_DIR = ""
MODEL_OUT_DIR = "sign_numbers_model"
NUM_CLASSES = 10
RANDOM_SEED = 42

# ---------------------------
# Load dataset
# Supports:
# 1) processed_dataset/X.npy + y.npy
# 2) processed_dataset/0.npy ... 9.npy
# ---------------------------
def load_dataset(dataset_dir):
    X_list, y_list = [], []

    for label in range(10):
        class_dir = os.path.join(dataset_dir, str(label))
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Missing folder {class_dir}")

        for fname in os.listdir(class_dir):
            if fname.endswith(".npy"):
                arr = np.load(os.path.join(class_dir, fname))
                X_list.append(arr)
                y_list.append(label)

    X = np.stack(X_list)
    y = np.array(y_list)

    assert X.shape[1] == 63
    return X.astype(np.float32), y.astype(np.int32)
# ---------------------------
# Build model
# ---------------------------
def build_model():
    model = models.Sequential([
        layers.Input(shape=(63,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# ---------------------------
# Main
# ---------------------------
def main():
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # Load data
    X, y = load_dataset(DATASET_DIR)
    print(f"Loaded dataset: X={X.shape}, y={y.shape}")

    # Train / validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    # Build + train
    model = build_model()
    model.summary()

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
    )

    # Save model
    model.save(MODEL_OUT_DIR)
    print(f"Model saved to: {MODEL_OUT_DIR}/")

if __name__ == "__main__":
    main()
