import os
import shutil
import numpy as np
import math
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers  # Import regularizers
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# --- Dataset Splitter ---
def split_dataset(input_dirs, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Splits datasets from input folders into a Train/Validation/Test directory structure.

    Parameters:
        input_dirs (dict): Dictionary with class names as keys and image folder paths as values.
                           Example: {"Fire": "path-to-fire-images", "Not_Fire": "path-to-not-fire-images"}
        output_dir (str): Base output directory for the split dataset.
        train_ratio (float): Ratio of training data. Default is 0.7 (70%).
        val_ratio (float): Ratio of validation data. Default is 0.15 (15%).
        test_ratio (float): Ratio of test data. Default is 0.15 (15%).
    """
    if not (train_ratio + val_ratio + test_ratio == 1.0):
        raise ValueError("Train, validation, and test ratios must sum to 1.")

    for class_name, input_dir in input_dirs.items():
        if not os.path.exists(input_dir):
            print(f"Input directory for '{class_name}' does not exist: {input_dir}")
            continue

        # Get all files from the directory
        files = [
            os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))
        ]

        # Split into train, validation, and test
        train_files, temp_files = train_test_split(files, train_size=train_ratio, random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=test_ratio / (val_ratio + test_ratio),
                                                 random_state=42)

        # Define the splits
        splits = {
            "Train": train_files,
            "Validation": val_files,
            "Test": test_files,
        }

        # Copy files to the output directory
        for split_name, split_files in splits.items():
            split_dir = os.path.join(output_dir, split_name, class_name)
            os.makedirs(split_dir, exist_ok=True)

            for file_path in split_files:
                shutil.copy(file_path, os.path.join(split_dir, os.path.basename(file_path)))

            print(f"Copied {len(split_files)} files to {split_dir}")

    print(f"\nDataset split into Train, Validation, and Test sets under: {output_dir}")


# --- Confusion Matrix Plot ---
def plot_confusion_matrix(test_flow, model):
    """
    Generate and plot a confusion matrix for the model's predictions on the test data.
    """
    # Get ground-truth labels
    true_labels = test_flow.classes

    # Get predicted labels
    preds = model.predict(test_flow, verbose=1)
    predicted_labels = (preds > 0.5).astype(int).flatten()

    # Generate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_flow.class_indices.keys())

    # Plot confusion matrix
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()


# --- Training Function ---
def train_mobilenet(train_dir, val_dir, test_dir, output_dir):
    """
    Train a MobileNetV2 model using transfer learning with enhancements for generalization and resource efficiency.
    """
    # Define parameters
    img_size = (224, 224)  # Larger input size for better feature extraction
    batch_size = 16
    epochs = 20

    # Augmentation for training and preprocessing for validation/test
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=45,
        zoom_range=0.5,
        brightness_range=[0.5, 1.5],
        width_shift_range=0.4,
        height_shift_range=0.4,
        shear_range=0.4,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    val_test_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_flow = train_gen.flow_from_directory(
        train_dir, target_size=img_size, batch_size=batch_size, class_mode="binary"
    )
    val_flow = val_test_gen.flow_from_directory(
        val_dir, target_size=img_size, batch_size=batch_size, class_mode="binary"
    )
    test_flow = val_test_gen.flow_from_directory(
        test_dir, target_size=img_size, batch_size=batch_size, class_mode="binary", shuffle=False
    )

    # Compute class weights
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_flow.classes),
        y=train_flow.classes
    )
    class_weights_dict = dict(enumerate(class_weights))

    # Load MobileNetV2 base model
    mobilenet_base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(img_size[0], img_size[1], 3))

    # Add custom layers
    x = mobilenet_base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
    x = Dropout(0.6)(x)  # Stronger dropout for better generalization
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=mobilenet_base.input, outputs=output)

    # Freeze base model except the last 20 layers
    for layer in mobilenet_base.layers[:-20]:
        layer.trainable = False

    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])

    # Callbacks for efficient training
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1, min_lr=1e-6)

    # Cosine decay learning rate scheduler
    def cosine_decay(epoch, lr):
        initial_lr = 0.0001
        decay = 0.5 * (1 + math.cos(epoch / epochs * math.pi))
        return initial_lr * decay

    lr_scheduler = LearningRateScheduler(cosine_decay, verbose=1)

    # Train the model
    history = model.fit(
        train_flow,
        validation_data=val_flow,
        epochs=epochs,
        callbacks=[early_stop, reduce_lr, lr_scheduler],
        class_weight=class_weights_dict
    )

    # Save trained model
    model_path = os.path.join(output_dir, "mobilenet_fire_model.keras")
    model.save(model_path)
    print(f"Model saved to: {model_path}")

    # Evaluate on test dataset
    test_loss, test_acc = model.evaluate(test_flow)
    print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

    # Plot confusion matrix
    plot_confusion_matrix(test_flow, model)

    # Visualize training
    visualize_training(history)


# --- Visualization Function ---
def visualize_training(history):
    """
    Visualize training and validation loss/accuracy over epochs.
    """
    plt.figure(figsize=(12, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

    plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    # Paths to raw dataset
    input_dirs = {
        "Fire": r"C:\Users\fmint\Desktop\Kaggle_Fire_Images",
        "Not_Fire": r"C:\Users\fmint\Desktop\Kaggle_Not_Fire_Images",
    }
    output_dir = r"C:\Users\fmint\Desktop\Kaggle_Cleaned_Fire_Dataset"

    # Split dataset into Train/Validation/Test
    split_dataset(input_dirs, output_dir)

    # Define directories for subsets
    train_dir = os.path.join(output_dir, "Train")
    val_dir = os.path.join(output_dir, "Validation")
    test_dir = os.path.join(output_dir, "Test")

    # Train the model
    train_mobilenet(train_dir, val_dir, test_dir, output_dir)