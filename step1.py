import os
import glob
import tensorflow as tf
import matplotlib.pyplot as plt

# Search recursively for any .jpg file in the downloaded keras folder
base_search_dir = os.path.join(os.path.expanduser('~'), '.keras', 'datasets')
all_photos = glob.glob(os.path.join(base_search_dir, '**', '*.jpg'), recursive=True)

if not all_photos:
    print(f"No JPG files found inside {base_search_dir}")
    exit()

sample_photo_path = all_photos[0]
print(f"Found photo: {sample_photo_path}\n")

# 1. Read the RAW photo from disk
raw_file = tf.io.read_file(sample_photo_path)
raw_image = tf.io.decode_jpeg(raw_file, channels=3)

print("-------------------------------------------")
print("BEFORE PREPROCESSING (RAW):")
print(f"Shape: {raw_image.shape} (Height, Width, Colors)")
print(f"Pixel values: min={tf.reduce_min(raw_image).numpy()} to max={tf.reduce_max(raw_image).numpy()}")
print("-------------------------------------------\n")

# 2. APPLY PREPROCESSING
# Step A: Resize to standard 180x180
resized_image = tf.image.resize(raw_image, [180, 180])

# Step B: Normalize from [0, 255] down to [0.0, 1.0]
clean_image = resized_image / 255.0

print("-------------------------------------------")
print("AFTER PREPROCESSING (CLEAN):")
print(f"Shape: {clean_image.shape} (Uniform size)")
print(f"Pixel values: min={tf.reduce_min(clean_image).numpy():.2f} to max={tf.reduce_max(clean_image).numpy():.2f}")
print("-------------------------------------------\n")

# 3. View Side-by-Side Comparison
plt.figure(figsize=(10, 5))

# Raw
plt.subplot(1, 2, 1)
plt.title(f"RAW IMAGE\nShape: {raw_image.shape}\nValues: [0 to 255]")
plt.imshow(raw_image.numpy())
plt.axis("off")

# Preprocessed
plt.subplot(1, 2, 2)
plt.title(f"PREPROCESSED\nShape: {clean_image.shape}\nValues: [0.0 to 1.0]")
plt.imshow(clean_image.numpy())
plt.axis("off")

plt.tight_layout()
plt.show()