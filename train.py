import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os

# ───────────────────────────────
# 1. Reproducibility
# ───────────────────────────────
tf.random.set_seed(42)
np.random.seed(42)

# ───────────────────────────────
# 2. Data
# ───────────────────────────────
print("Loading MNIST dataset …")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = (x_train.astype("float32") / 255.0)[..., None]   # (60000, 28, 28, 1)
x_test  = (x_test.astype("float32")  / 255.0)[..., None]
print(f"Training data shape : {x_train.shape}")
print(f"Test data shape     : {x_test.shape}")

# ───────────────────────────────
# 3. Model parts
# ───────────────────────────────
latent_dim, num_classes, input_shape = 64, 10, (28, 28, 1)

def build_conditional_encoder():
    inp = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, 2, "same", activation="relu")(inp)
    x = layers.Conv2D(64, 3, 2, "same", activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    z_mean    = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    return keras.Model(inp, [z_mean, z_log_var], name="encoder")

def build_conditional_decoder():
    inp = keras.Input(shape=(latent_dim + num_classes,))
    x = layers.Dense(7 * 7 * 64, activation="relu")(inp)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, 3, 2, "same", activation="relu")(x)
    x = layers.Conv2DTranspose(32, 3, 2, "same", activation="relu")(x)
    out = layers.Conv2DTranspose(1, 3, padding="same", activation="sigmoid")(x)
    return keras.Model(inp, out, name="decoder")

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps

class ConditionalVAE(keras.Model):
    def __init__(self, encoder, decoder, num_classes, **kw):
        super().__init__(**kw)
        self.encoder, self.decoder, self.num_classes = encoder, decoder, num_classes
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.rec_tracker  = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_tracker   = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.loss_tracker, self.rec_tracker, self.kl_tracker]

    def call(self, inputs):
        x, y = inputs
        y_1h = tf.one_hot(y, self.num_classes)
        z_m, z_lv = self.encoder(x)
        z   = Sampling()([z_m, z_lv])
        z_c = tf.concat([z, y_1h], axis=1)
        return self.decoder(z_c)

    def train_step(self, data):
        x, y = data
        y_1h = tf.one_hot(y, self.num_classes)
        with tf.GradientTape() as tape:
            z_m, z_lv = self.encoder(x, training=True)
            z = Sampling()([z_m, z_lv])
            z_c = tf.concat([z, y_1h], axis=1)
            recon = self.decoder(z_c, training=True)

            rec_loss = tf.reduce_mean(
                tf.reduce_sum(keras.losses.binary_crossentropy(x, recon), axis=(1, 2))
            )
            kl = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_lv - tf.square(z_m) - tf.exp(z_lv), axis=1)
            )
            loss = rec_loss + kl

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.loss_tracker.update_state(loss)
        self.rec_tracker.update_state(rec_loss)
        self.kl_tracker.update_state(kl)
        return {m.name: m.result() for m in self.metrics}

# ───────────────────────────────
# 4. Build, compile, train
# ───────────────────────────────
encoder = build_conditional_encoder()
decoder = build_conditional_decoder()
cvae = ConditionalVAE(encoder, decoder, num_classes)
cvae.compile(optimizer=keras.optimizers.Adam(1e-3))

train_ds = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .shuffle(10000).batch(128))

print("Training …")
cvae.fit(train_ds, epochs=100)

# Run the model once to mark it built
_ = cvae((x_train[:1], y_train[:1]))

# ───────────────────────────────
# 5. Save weights
# ───────────────────────────────
os.makedirs("models", exist_ok=True)
cvae.save_weights("models/cvae.weights.h5")   # ‘.weights.h5’ required
print("Weights saved to models/cvae.weights.h5")

# ───────────────────────────────
# 6. Generation helper
# ───────────────────────────────
def generate_digits(digit, n=5):
    z   = tf.random.normal((n, latent_dim))
    lbl = tf.one_hot([digit] * n, num_classes)
    imgs = decoder(tf.concat([z, lbl], axis=1), training=False)
    return imgs.numpy()

print("Generating sample images …")
for d in range(10):
    imgs = generate_digits(d, 5)
    plt.figure(figsize=(10, 2))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(imgs[i].squeeze(), cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"models/sample_{d}.png")
    plt.close()
print("Samples saved in models/")
