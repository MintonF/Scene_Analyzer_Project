import os
import hashlib
import shutil
import math
from sklearn.model_selection import train_test_split
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import Precision, Recall
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

# --- Global SEED for Reproducibility ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# --- Updated Dataset Splitter ---
def split_dataset(input_dirs, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=SEED):
    """
    Splits datasets from input folders into Train/Validation/Test, ensuring no duplicates.
    """
    if not (train_ratio + val_ratio + test_ratio == 1.0):
        raise ValueError("Train, validation, and test ratios must sum to 1.")

    # Initialize tracking for unique file hashes
    unique_hashes = set()

    for class_name, input_dir in input_dirs.items():
        if not os.path.exists(input_dir):
            print(f"Input directory for '{class_name}' does not exist: {input_dir}")
            continue

        # Get all files in the directory
        files = [
            os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))
        ]

        deduplicated_files = []
        for file in files:
            with open(file, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            # Avoid duplicates
            if file_hash not in unique_hashes:
                deduplicated_files.append(file)
                unique_hashes.add(file_hash)

        if not deduplicated_files:
            print(f"No unique files found in '{input_dir}' for class '{class_name}'. Skipping...")
            continue

        # Randomly split the data
        train_files, temp_files = train_test_split(deduplicated_files, train_size=train_ratio, random_state=seed)
        val_files, test_files = train_test_split(
            temp_files, test_size=test_ratio / (val_ratio + test_ratio), random_state=seed
        )

        # Organize splits
        splits = {"Train": train_files, "Validation": val_files, "Test": test_files}

        for split_name, split_files in splits.items():
            split_dir = os.path.join(output_dir, split_name, class_name)
            os.makedirs(split_dir, exist_ok=True)

            for file_path in split_files:
                shutil.copy(file_path, os.path.join(split_dir, os.path.basename(file_path)))

            print(f"Copied {len(split_files)} files to {split_dir}")

    print(f"Dataset successfully split into Train, Validation, and Test splits under: {output_dir}")


# --- Updated Data Leakage Verifier ---
def verify_split_integrity(output_dir):
    """
    Verify no files overlap between Train, Validation, and Test splits.
    """
    sets = ["Train", "Validation", "Test"]
    all_files = {}

    for dataset in sets:
        dataset_dir = os.path.join(output_dir, dataset)
        dataset_files = set()

        for class_dir in os.listdir(dataset_dir):
            class_path = os.path.join(dataset_dir, class_dir)
            if os.path.isdir(class_path):
                # Use full file paths for comparison to ensure no duplicates
                dataset_files.update(os.path.join(class_path, f) for f in os.listdir(class_path))

        all_files[dataset] = dataset_files

    # Compare all pairs of sets
    for i, set1 in enumerate(sets):
        for set2 in sets[i + 1:]:
            intersection = all_files[set1] & all_files[set2]
            if intersection:
                print(f"\nERROR: Data leakage detected between {set1} and {set2}.")
                print(f"Number of overlapping files: {len(intersection)}")
                for file in intersection:
                    print(f" - {file}")
                return False

    print("No data leakage detected between Train, Validation, and Test splits.")
    return True


# --- Training Function ---
def train_mobilenet(train_dir, val_dir, test_dir, output_dir):
    """
    Train a MobileNetV2 model and evaluate it with Precision, Recall, and F1-score.
    """
    # Define parameters
    img_size = (224, 224)
    batch_size = 16
    epochs = 20

    # Augmentation for training and rescaling for validation/test
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        zoom_range=0.3,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    val_test_gen = ImageDataGenerator(rescale=1.0 / 255)

    # Load datasets
    train_flow = train_gen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size,
                                               class_mode="binary", shuffle=True, seed=SEED)
    val_flow = val_test_gen.flow_from_directory(val_dir, target_size=img_size, batch_size=batch_size,
                                                class_mode="binary", shuffle=False, seed=SEED)
    test_flow = val_test_gen.flow_from_directory(test_dir, target_size=img_size, batch_size=batch_size,
                                                 class_mode="binary", shuffle=False, seed=SEED)

    # Compute class weights
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(train_flow.classes),
                                         y=train_flow.classes)
    class_weights_dict = dict(enumerate(class_weights))

    # Load MobileNetV2 base
    mobilenet_base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(img_size[0], img_size[1], 3))

    # Add custom layers
    x = mobilenet_base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.6)(x)
    output = Dense(1, activation="sigmoid")(x)

    # Compile model
    model = Model(inputs=mobilenet_base.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss="binary_crossentropy",
                  metrics=["accuracy", Precision(), Recall()])

    # Callbacks for better training
    early_stop = EarlyStopping(monitor="val_loss", patience=5, verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)

    def cosine_decay(epoch, lr):
        initial_lr = 0.0001
        decay = 0.5 * (1 + math.cos(epoch / epochs * math.pi))
        return initial_lr * decay

    lr_scheduler = LearningRateScheduler(cosine_decay)

    # Train the model
    history = model.fit(
        train_flow, validation_data=val_flow, epochs=epochs,
        callbacks=[early_stop, reduce_lr, lr_scheduler],
        class_weight=class_weights_dict
    )

    # Save model
    model.save(os.path.join(output_dir, "mobilenet_fire_model.keras"))

    # --- Test the Model ---
    # Predict on test set
    predictions = model.predict(test_flow)
    predicted_classes = (predictions > 0.5).astype("int32")

    # Confusion Matrix and Precision
    true_labels = test_flow.classes
    conf_matrix = confusion_matrix(true_labels, predicted_classes)
    precision = precision_score(true_labels, predicted_classes)
    print(f"\nTesting Precision: {precision}")

    # Confusion Matrix Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=test_flow.class_indices.keys(),
                yticklabels=test_flow.class_indices.keys())
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    # --- Accuracy and Loss Plots ---
    # Accuracy plot
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    # Loss plot
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


# --- Main Program ---
if __name__ == "__main__":
    input_dirs = {
        "Fire": r"C:\Users\fmint\Desktop\Kaggle_Fire_Images",
        "Not_Fire": r"C:\Users\fmint\Desktop\Kaggle_Not_Fire_Images",
    }
    output_dir = r"C:\Users\fmint\Desktop\Kaggle_Cleaned_Fire_Dataset"

    # Step 1: Split dataset
    split_dataset(input_dirs, output_dir)

    # Step 2: Check for data leakage
    if not verify_split_integrity(output_dir):
        raise Exception("Data leakage detected. Please review your dataset configuration.")

    # Step 3: Define paths for train, validation, and test sets
    train_dir = os.path.join(output_dir, "Train")
    val_dir = os.path.join(output_dir, "Validation")
    test_dir = os.path.join(output_dir, "Test")

    # Step 4: Train and evaluate the model
    train_mobilenet(train_dir, val_dir, test_dir, output_dir)