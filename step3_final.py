import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Direct path to flower photos
data_dir = os.path.join(os.path.expanduser('~'), '.keras', 'datasets', 'flower_photos')

# Check if there is a nested flower_photos folder
nested_dir = os.path.join(data_dir, 'flower_photos')
if os.path.exists(nested_dir):
    data_dir = nested_dir

print(f"Loading data from: {data_dir}")

IMG_HEIGHT = 180
IMG_WIDTH = 180
BATCH_SIZE = 32

# 1. Load & Preprocess (Resize to 180x180)
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
print(f"Classes found ({len(class_names)}): {class_names}\n")

# 2. Build Model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    tf.keras.layers.Rescaling(1./255),  # Normalization [0, 255] -> [0, 1]

    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_names))
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# 3. Train
print("--- Training for 3 epochs ---")
model.fit(train_ds, validation_data=val_ds, epochs=3)

# 4. Predict One Image
print("\n--- Making a Prediction ---")
for images, labels in val_ds.take(1):
    test_img = images[0]
    real_name = class_names[labels[0]]

    predictions = model.predict(tf.expand_dims(test_img, 0))
    score = tf.nn.softmax(predictions[0])
    predicted_name = class_names[np.argmax(score)]

    print(f"\nReal Flower:      {real_name}")
    print(f"Computer Guessed: {predicted_name} ({100 * np.max(score):.1f}% confidence)")

    plt.imshow(test_img.numpy().astype("uint8"))
    plt.title(f"Real: {real_name}\nGuessed: {predicted_name}")
    plt.axis("off")
    plt.show()
    break