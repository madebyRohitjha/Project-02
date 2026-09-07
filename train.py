import tensorflow as tf

print("TensorFlow Version:", tf.__version__)

IMG_SIZE = (160, 160)
BATCH_SIZE = 32

train_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset/PetImages",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset/PetImages",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

print("Datasets loaded successfully!")