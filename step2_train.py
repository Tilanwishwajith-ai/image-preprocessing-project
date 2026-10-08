import os

try:
    import tensorflow as tf  # type: ignore
    from tensorflow.keras import layers, models  # type: ignore
except ImportError as exc:
    raise ImportError(
        "TensorFlow is required to run this script. Install it with: pip install tensorflow"
    ) from exc

# 1. Locate dataset folder
data_dir = os.path.join(os.path.expanduser('~'), '.keras', 'datasets', 'flower_photos')

IMG_HEIGHT = 180
IMG_WIDTH = 180
BATCH_SIZE = 32

# 2. AUTOMATIC PREPROCESSING PIPELINE
# Splits into 80% training and 20% validation, resizes every image to 180x180
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print(f"\nClasses to learn: {class_names}")

# 3. BUILD THE MODEL WITH PREPROCESSING INSIDE IT
model = models.Sequential([
    # PREPROCESSING LAYER: Scales pixel numbers from [0, 255] down to [0.0, 1.0]
    layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

    # Convolutional layers (find petals, colors, shapes)
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    # Classifier (makes the final guess between 5 flowers)
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names))
])

# 4. COMPILE MODEL
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# 5. TRAIN FOR 3 ROUNDS (EPOCHS)
print("\n--- Training Model Now ---")
model.fit(train_ds, validation_data=val_ds, epochs=3)

print("\nModel training finished successfully!")