import tensorflow as tf
import matplotlib.pyplot as plt


# -----------------------------
# 1. TensorFlow version
# -----------------------------

print("TensorFlow Version:", tf.__version__)


# -----------------------------
# 2. Dataset settings
# -----------------------------

IMG_SIZE = (160, 160)
BATCH_SIZE = 32


# -----------------------------
# 3. Load training dataset
# -----------------------------

train_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset/PetImages",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)


# -----------------------------
# 4. Load validation dataset
# -----------------------------

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset/PetImages",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

print("Datasets loaded successfully!")


# -----------------------------
# 5. Show some images
# -----------------------------

class_names = train_dataset.class_names

plt.figure(figsize=(10, 10))

for images, labels in train_dataset.take(1):
    for i in range(9):
        plt.subplot(3, 3, i + 1)

        plt.imshow(images[i].numpy().astype("uint8"))

        plt.title(class_names[labels[i]])

        plt.axis("off")

plt.show()


# -----------------------------
# 6. Check image shape
# -----------------------------

for images, labels in train_dataset.take(1):
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)


# -----------------------------
# 7. Normalize images
# -----------------------------

normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

train_dataset = train_dataset.map(
    lambda images, labels: (normalization_layer(images), labels)
)

validation_dataset = validation_dataset.map(
    lambda images, labels: (normalization_layer(images), labels)
)


# Check normalized pixel values

for images, labels in train_dataset.take(1):
    print("Minimum pixel value:", tf.reduce_min(images).numpy())
    print("Maximum pixel value:", tf.reduce_max(images).numpy())


# -----------------------------
# 8. Build CNN model
# -----------------------------

model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(160, 160, 3)
    ),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])


# -----------------------------
# 9. Show model structure
# -----------------------------

model.summary()

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10
)