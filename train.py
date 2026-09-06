import tensorflow as tf

print("TensorFlow Version:", tf.__version__)

# Image size
IMG_SIZE = (160, 160)

# Batch size
BATCH_SIZE = 32

# Load training dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Load validation dataset
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset/validation",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

print("Datasets loaded successfully!")