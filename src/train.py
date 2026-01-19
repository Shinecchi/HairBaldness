import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from load_data import load_data
from model import build_model


def train_model():
    # --- 1. LOAD DATA ---
    print("--- Loading Data ---")
    try:
        train_images, train_labels, val_images, val_labels, test_images, test_labels = load_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if len(train_images) == 0:
        print("Error: No data loaded.")
        return

    n_total = len(train_images) + len(val_images) + len(test_images)
    print(f"\n--- DATASET SPLIT ---")
    print(f"Train: {len(train_images)} ({len(train_images) / n_total * 100:.1f}%)")
    print(f"Valid: {len(val_images)} ({len(val_images) / n_total * 100:.1f}%)")

    # --- 2. AUGMENTATION (Medium Strength) ---
    # We increased this slightly to fight the overfitting
    datagen = ImageDataGenerator(
        rotation_range=15,  # Increased from 5 to 15
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,  # Increased from 0.1 to 0.15
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(train_images)

    # --- 3. STAGE 1: WARMUP ---
    print("\n=== STAGE 1: WARMUP (Frozen Weights) ===")
    model = build_model()

    model.fit(
        datagen.flow(train_images, train_labels, batch_size=32),
        validation_data=(val_images, val_labels),
        epochs=12,
        verbose=1
    )

    # --- 4. STAGE 2: FINE TUNING ---
    print("\n=== STAGE 2: FINE TUNING (Unlocking Top 20 Layers) ===")

    # Unfreeze only the top 20 layers (Less aggressive than before)
    total_layers = len(model.layers)

    # EfficientNet is wrapped inside the model, so we find it first
    # (The Sequential model wraps layers in a list)
    # We will just unfreeze the base_model inside the Sequential wrapper
    base_model = model.layers[0]
    base_model.trainable = True

    # Freeze all layers in EfficientNet EXCEPT the last 20
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    print(f"Total layers in Base Model: {len(base_model.layers)}")
    print(f"Fine-tuning the last 20 layers only.")

    # Recompile with LOW Learning Rate
    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Callbacks
    checkpoint = ModelCheckpoint('hair_loss_model.h5', monitor='val_accuracy', save_best_only=True, mode='max',
                                 verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
    early_stop = EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1)

    model.fit(
        datagen.flow(train_images, train_labels, batch_size=32),
        validation_data=(val_images, val_labels),
        epochs=40,
        callbacks=[checkpoint, reduce_lr, early_stop]
    )

    # --- 5. FINAL TEST ---
    print("\n=== Final Test Evaluation ===")
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Final Test Accuracy: {test_acc * 100:.2f}%")


if __name__ == '__main__':
    train_model()