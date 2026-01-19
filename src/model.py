from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2  # Import L2 Regularization


def build_model():
    # 1. Load EfficientNetB0 (Smarter than MobileNet)
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # 2. Start Frozen
    base_model.trainable = False

    # 3. Build Architecture with Strong Regularization
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),

        # Dense Layer with L2 Regularization (The "Anti-Cheat" mechanism)
        Dense(256, activation='relu', kernel_regularizer=l2(0.01)),

        # Increased Dropout from 0.4 to 0.5 (Drops 50% of neurons to force learning)
        Dropout(0.5),

        # Output Layer
        Dense(6, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model