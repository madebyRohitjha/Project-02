import tensorflow as tf
import matplotlib.pyplot as plt


# Load the trained model
model = tf.keras.models.load_model("cats_vs_dogs_cnn.keras")

print("Model loaded successfully!")


# Image we want to predict
image_path = "dataset/PetImages/Cat/0.jpg"


# Load and resize the image
image = tf.keras.utils.load_img(
    image_path,
    target_size=(160, 160)
)


# Convert image to numbers
image_array = tf.keras.utils.img_to_array(image)


# Normalize pixel values
image_array = image_array / 255.0


# Add batch dimension
image_array = tf.expand_dims(image_array, 0)


# Make prediction
prediction = model.predict(image_array)

print("Prediction score:", prediction[0][0])


# Convert prediction into Cat or Dog
if prediction[0][0] < 0.5:
    result = "Cat 🐱"
else:
    result = "Dog 🐶"


print("Prediction:", result)


# Show image
plt.imshow(image)
plt.title(result)
plt.axis("off")
plt.show()