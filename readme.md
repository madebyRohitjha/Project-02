# 🐱🐶 Cats vs Dogs Classification using CNN

A Deep Learning project that uses a **Convolutional Neural Network (CNN)** to classify images as either a **Cat** or a **Dog**.

This is **Project 02** in my Deep Learning learning journey.

---

## 🎯 Project Goal

Build a CNN that can look at an image and predict whether it is:

- 🐱 Cat
- 🐶 Dog

The project covers the complete image-classification workflow:

```text
Dataset
   ↓
Preprocessing
   ↓
CNN
   ↓
Training
   ↓
Evaluation
   ↓
Prediction
```

---

## 🛠️ Technologies Used

- Python
- TensorFlow 2.21.0
- Keras
- NumPy
- Matplotlib

---

## 📂 Dataset

The project uses the Cats vs Dogs `PetImages` dataset.

Structure:

```text
dataset/
└── PetImages/
    ├── Cat/
    └── Dog/
```

The dataset contains:

```text
25,000 images
2 classes
```

The data is automatically split into:

```text
20,000 images → Training
5,000 images  → Validation
```

---

## 🧠 CNN Architecture

The final CNN contains:

```text
Input Image
160 × 160 × 3
       ↓
Conv2D
32 filters
       ↓
MaxPooling
       ↓
Conv2D
64 filters
       ↓
MaxPooling
       ↓
Flatten
       ↓
Dense
128 neurons
       ↓
Dropout
50%
       ↓
Dense
1 neuron
       ↓
Sigmoid
       ↓
Cat 🐱 / Dog 🐶
```

---

## 🔬 Preprocessing

### Image Resizing

Every image is resized to:

```text
160 × 160 pixels
```

### Normalization

Pixel values are converted from:

```text
0–255
```

to:

```text
0–1
```

using:

```python
tf.keras.layers.Rescaling(1.0 / 255)
```

---

## 🧠 Model Training

The model uses:

```text
Optimizer: Adam
Loss: Binary Crossentropy
Metric: Accuracy
Epochs: 10
```

Training and validation performance are stored in the `history` object.

---

## 📊 Evaluation

The model is evaluated using the validation dataset.

The project also generates:

- Training vs Validation Accuracy graph
- Training vs Validation Loss graph

These graphs help identify whether the model is learning properly or overfitting.

---

## 🛡️ Dropout

A Dropout layer is included:

```python
tf.keras.layers.Dropout(0.5)
```

Dropout helps reduce overfitting by temporarily disabling some neurons during training.

---

## 💾 Saved Model

The trained model is saved as:

```text
cats_vs_dogs_cnn.keras
```

The model file is ignored by Git using:

```text
*.keras
```

---

## 🔮 Prediction

`predict.py` loads the saved model and allows the user to provide an image path.

Example:

```text
Enter image path:
```

The program returns:

```text
Prediction: Dog 🐶
Confidence: 94.70%
```

---

## 🧪 Multiple Image Testing

`test_predictions.py` allows several images to be tested automatically.

It reports:

```text
Image
Prediction
Confidence
```

for each image.

---

## 📁 Project Structure

```text
project02/
│
├── dataset/
│   └── PetImages/
│       ├── Cat/
│       └── Dog/
│
├── train.py
├── predict.py
├── test_predictions.py
├── requirements.txt
├── .gitignore
├── README.md
└── venv/
```

The `venv/` folder is kept locally and ignored by Git.

---

## ▶️ How to Run

### 1. Create virtual environment

```powershell
python -m venv venv
```

### 2. Activate it

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Train the model

```powershell
python train.py
```

### 5. Make a prediction

```powershell
python predict.py
```

### 6. Test multiple images

```powershell
python test_predictions.py
```

---

## 📚 What I Learned

Through this project I practiced:

- Loading image datasets
- Training/validation splitting
- Image visualization
- Image normalization
- Convolution layers
- Filters and feature detection
- Max pooling
- Flattening feature maps
- Dense layers
- ReLU activation
- Sigmoid activation
- Binary classification
- Binary crossentropy
- Adam optimizer
- Model training
- Model evaluation
- Accuracy and loss visualization
- Dropout
- Saving and loading models
- Making predictions
- Prediction confidence
- Testing multiple images
- Managing Python dependencies
- Using Git and GitHub

---

## 🚀 Future Improvements

Possible improvements for this project:

- Data augmentation
- More CNN layers
- Batch normalization
- Transfer learning
- Use a pretrained model such as MobileNet or ResNet
- Improve validation accuracy
- Test on completely new external images
- Build a simple web interface

---

## 👨‍💻 Author

**Rohit Jha**

Deep Learning Learning Journey

---

## ✅ Project Status

**Project 02 completed: CNN Cats vs Dogs Classification** 🐱🐶