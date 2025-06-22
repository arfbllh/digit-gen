Here’s a clean and minimal set of files for your **Conditional VAE on MNIST**:

---

### ✅ `README.md`

````md
# Conditional Variational Autoencoder (CVAE) on MNIST

This project trains a Conditional Variational Autoencoder (CVAE) to generate handwritten digits from the MNIST dataset, conditioned on class labels.

## 🔧 Features
- Trains a CVAE model on MNIST
- Generates digit-specific images (0–9)
- Saves sample images and trained weights

## 🧰 Requirements

Install dependencies:

```bash
pip install tensorflow matplotlib numpy
````

## 🚀 Usage

### 1. Train the model

```bash
python train.py
```

This will:

* Train the CVAE model for 10 epochs
* Save model weights to `models/cvae.weights.h5`
* Save generated digit images to `models/sample_*.png`

### 2. Run the generation app

```bash
python app.py
```

This will:

* Load trained weights
* Generate and show 5 samples per digit (0–9) using matplotlib

## 📁 Project Structure

```
.
├── train.py       # Training the CVAE model
├── app.py         # Image generation using trained model
├── models/        # Saved weights and generated samples
└── README.md
```

## 📸 Samples

Generated samples will look like this:

```
models/sample_0.png
models/sample_1.png
...
models/sample_9.png
```

---

Created with ❤️ using TensorFlow

````

---

### ✅ `train.py`

```python
# train.py
from cvae_model import build_model, generate_digits
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

# Load and preprocess data
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = (x_train.astype("float32") / 255.0)[..., None]
y_train = y_train.astype("int32")

# Create dataset
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(128)

# Build and train
cvae, encoder, decoder = build_model()
cvae.fit(train_ds, epochs=10)
_ = cvae((x_train[:1], y_train[:1]))  # build model before saving

# Save weights
os.makedirs("models", exist_ok=True)
cvae.save_weights("models/cvae.weights.h5")

# Generate and save sample images
for d in range(10):
    imgs = generate_digits(decoder, d, 5)
    plt.figure(figsize=(10, 2))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(imgs[i].squeeze(), cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"models/sample_{d}.png")
    plt.close()

print("Training complete. Weights and samples saved.")
````

---

### ✅ `app.py`

```python
# app.py
from cvae_model import build_model, generate_digits
import matplotlib.pyplot as plt
import tensorflow as tf

# Build model and load weights
cvae, encoder, decoder = build_model()
cvae((tf.zeros((1, 28, 28, 1)), tf.zeros((1,), dtype=tf.int32)))  # build
cvae.load_weights("models/cvae.weights.h5")

# Display samples
for d in range(10):
    imgs = generate_digits(decoder, d, 5)
    plt.figure(figsize=(10, 2))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(imgs[i].squeeze(), cmap="gray")
        plt.axis("off")
    plt.suptitle(f"Generated Samples for Digit {d}")
    plt.tight_layout()
    plt.show()
```

---

### ✅ Suggested (optional) file: `cvae_model.py`

To keep `train.py` and `app.py` clean, place your model code in `cvae_model.py`.

Would you like me to extract that from your current code and format it properly?
