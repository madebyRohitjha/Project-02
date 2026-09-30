import tensorflow as tf
import matplotlib.pyplot as plt


# Load trained model
model = tf.keras.models.load_model("cats_vs_dogs_cnn.keras")

print("Model loaded successfully!")


# Images to test
image_paths = [
    "dataset/PetImages/Cat/0.jpg",
    "dataset/PetImages/Cat/1.jpg",
    "dataset/PetImages/Dog/0.jpg",
    "dataset/PetImages/Dog/1.jpg"
]


# Test each image
for image_path in image_paths:

    image = tf.keras.utils.load_img(
        image_path,
        target_size=(160, 160)
    )

    image_array = tf.keras.utils.img_to_array(image)

    image_array = image_array / 255.0

    image_array = tf.expand_dims(image_array, 0)

    prediction = model.predict(image_array, verbose=0)

    score = prediction[0][0]

    if score < 0.5:
        result = "Cat 🐱"
        confidence = (1 - score) * 100
    else:
        result = "Dog 🐶"
        confidence = score * 100

    print("\nImage:", image_path)
    print("Prediction:", result)
    print(f"Confidence: {confidence:.2f}%")