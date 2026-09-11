# 🐱🐶 Cats vs Dogs Classification using CNN

A Deep Learning project that uses a **Convolutional Neural Network (CNN)** to classify images as either a **cat** or a **dog**.

This is **Project 02** in my Deep Learning learning journey.

---

## 🎯 Project Goal

Build a CNN that can look at an image and predict:

- 🐱 Cat
- 🐶 Dog

The project will cover the complete image-classification workflow, from loading and preprocessing images to training, evaluation, and prediction.

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib

---

## 📂 Dataset

The project uses the **Microsoft Cats vs Dogs (PetImages)** dataset.

Dataset structure:

```text
dataset/
└── PetImages/
    ├── Cat/
    └── Dog/
```

The dataset contains:

- **25,000 images**
- **2 classes**
  - Cat
  - Dog

For training, TensorFlow automatically splits the dataset into:

- **20,000 images for training**
- **5,000 images for validation**

This uses an **80/20 split**.

---

## 🧠 What I Have Learned

### 1. Virtual Environment

Created and activated a Python virtual environment to keep project dependencies isolated.

### 2. TensorFlow Setup

Installed and verified TensorFlow.

Current TensorFlow version:

```text
2.21.0
```

### 3. Dataset Loading

Used:

```python
tf.keras.utils.image_dataset_from_directory()
```

to load images directly from the Cat and Dog folders.

### 4. Training and Validation Split

Used:

```python
validation_split=0.2
```

to automatically divide the dataset:

```text
80% → Training
20% → Validation
```

### 5. Image Visualization

Used Matplotlib to display sample images from the dataset and their corresponding labels.

---

## 🏗️ Current Project Structure

```text
project02/
│
├── dataset/
│   └── PetImages/
│       ├── Cat/
│       └── Dog/
│
├── train.py
├── venv/
└── README.md
```

---

# ✅ Checkpoint 1

### Project Setup

- [x] Created Project 02
- [x] Created virtual environment
- [x] Activated virtual environment
- [x] Installed TensorFlow
- [x] Installed NumPy
- [x] Installed Matplotlib
- [x] Verified TensorFlow 2.21.0
- [x] Downloaded/extracted Cats vs Dogs dataset
- [x] Loaded 25,000 images
- [x] Created 80/20 training-validation split
- [x] Visualized sample images

### Dataset Result

```text
Total images:       25,000
Training images:    20,000
Validation images:  5,000
Classes:            2
```

---

## 🚀 Next Steps

- [ ] Understand image dimensions and RGB channels
- [ ] Preprocess images
- [ ] Build the first CNN
- [ ] Understand Convolution layers
- [ ] Understand Filters/Kernels
- [ ] Understand ReLU
- [ ] Understand Max Pooling
- [ ] Add Dense layers
- [ ] Train the CNN
- [ ] Evaluate the model
- [ ] Plot training/validation accuracy
- [ ] Add Dropout
- [ ] Improve the model
- [ ] Save the trained model
- [ ] Create `predict.py`
- [ ] Test the model on new images

---

## 📚 Learning Approach

I am building this project step-by-step while learning the concepts behind each part of the code.

The goal is not only to make the model work, but also to understand **why each component is used**.

---

## 👨‍💻 Author

**Rohit Jha**

Deep Learning Learning Journey