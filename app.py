import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import os

# Set page config
st.set_page_config(
    page_title="Handwritten Digit Generator",
    page_icon="🔢",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1e3a8a;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .description {
        text-align: center;
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .digit-images {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin: 2rem 0;
    }
    .digit-card {
        border: 2px solid #e5e7eb;
        border-radius: 10px;
        padding: 10px;
        background-color: #f9fafb;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        # Try to load saved models
        if os.path.exists('models/encoder.h5') and os.path.exists('models/decoder.h5'):
            encoder = keras.models.load_model('models/encoder.h5')
            decoder = keras.models.load_model('models/decoder.h5')
            return encoder, decoder
        else:
            st.error("Model files not found! Please run the training script first.")
            return None, None
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

# Sampling layer for generation
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Generation function
def generate_digit_images(decoder, digit, num_samples=5, latent_dim=64):
    """Generate images for a specific digit"""
    try:
        # Sample from latent space
        z_sample = tf.random.normal(shape=(num_samples, latent_dim))
        
        # Create one-hot encoded labels for the desired digit
        labels = tf.one_hot([digit] * num_samples, 10)
        
        # Concatenate latent vector with labels
        z_cond = tf.concat([z_sample, labels], axis=1)
        
        # Generate images
        generated_images = decoder(z_cond)
        
        return generated_images.numpy()
    except Exception as e:
        st.error(f"Error generating images: {str(e)}")
        return None

# Alternative simple generation function if models fail
def generate_simple_digit_pattern(digit, num_samples=5):
    """Generate simple digit patterns as fallback"""
    images = []
    for i in range(num_samples):
        # Create a simple pattern based on digit
        img = np.zeros((28, 28))
        
        # Simple patterns for each digit (this is a fallback)
        if digit == 0:
            img[8:20, 8:12] = 0.8
            img[8:20, 16:20] = 0.8
            img[8:12, 8:20] = 0.8
            img[16:20, 8:20] = 0.8
        elif digit == 1:
            img[6:22, 12:16] = 0.8
            img[10:14, 8:12] = 0.8
        elif digit == 2:
            img[8:12, 8:20] = 0.8
            img[12:16, 16:20] = 0.8
            img[14:18, 8:20] = 0.8
            img[16:20, 8:12] = 0.8
            img[18:22, 8:20] = 0.8
        # Add more patterns for other digits...
        else:
            # Generic pattern
            img[8:20, 10:18] = 0.7
            img[10:18, 8:20] = 0.7
        
        # Add some noise for variation
        noise = np.random.normal(0, 0.1, (28, 28))
        img = np.clip(img + noise * (i * 0.1), 0, 1)
        images.append(img)
    
    return np.array(images)

def main():
    # Header
    st.markdown('<h1 class="main-header">🔢 Handwritten Digit Image Generator</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="description">Generate synthetic MNIST-like images using your trained model</p>', 
                unsafe_allow_html=True)
    
    # Load models
    encoder, decoder = load_models()
    
    # User interface
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### Choose a digit to generate (0-9)")
        selected_digit = st.selectbox(
            "Select digit:",
            options=list(range(10)),
            index=2,
            help="Choose which digit you want to generate"
        )
        
        generate_button = st.button(
            "🎨 Generate Images",
            type="primary",
            use_container_width=True
        )
    
    # Generate and display images
    if generate_button:
        with st.spinner('Generating handwritten digit images...'):
            if decoder is not None:
                # Use trained model
                generated_images = generate_digit_images(decoder, selected_digit, 5)
            else:
                # Use fallback pattern generation
                st.warning("Using fallback pattern generation since trained model is not available.")
                generated_images = generate_simple_digit_pattern(selected_digit, 5)
            
            if generated_images is not None:
                st.markdown(f"### Generated images of digit **{selected_digit}**")
                
                # Display images in a row
                cols = st.columns(5)
                for i, col in enumerate(cols):
                    with col:
                        # Convert to PIL Image for display
                        img_array = generated_images[i].reshape(28, 28)
                        
                        # Create matplotlib figure
                        fig, ax = plt.subplots(figsize=(3, 3))
                        ax.imshow(img_array, cmap='gray')
                        ax.set_title(f'Sample {i+1}', fontsize=12, fontweight='bold')
                        ax.axis('off')
                        
                        # Convert to image
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png', bbox_inches='tight', 
                                  facecolor='white', dpi=100)
                        buf.seek(0)
                        img = Image.open(buf)
                        plt.close()
                        
                        # Display in Streamlit
                        st.image(img, use_container_width=True)
                
                # Add some statistics or info
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Digit Generated", selected_digit)
                with col2:
                    st.metric("Number of Images", 5)
                with col3:
                    st.metric("Image Size", "28×28")
    
    # Sidebar information
    with st.sidebar:
        st.markdown("## ℹ️ About")
        st.markdown("""
        This app generates handwritten digit images using a Variational Autoencoder (VAE) 
        trained on the MNIST dataset.
        
        **Features:**
        - Generate 5 unique images per digit
        - MNIST-style 28×28 grayscale images
        - Trained using TensorFlow/Keras
        - Conditional generation for specific digits
        """)
        
        st.markdown("## 🛠️ Technical Details")
        st.markdown("""
        **Model Architecture:**
        - Conditional Variational Autoencoder
        - Latent dimension: 64
        - Input/Output: 28×28×1
        - Framework: TensorFlow 2.x
        
        **Training:**
        - Dataset: MNIST (60,000 training images)
        - Epochs: 30
        - Optimizer: Adam
        - Loss: Reconstruction + KL Divergence
        """)
        
        st.markdown("## 📊 Model Status")
        if decoder is not None:
            st.success("✅ Model loaded successfully")
        else:
            st.error("❌ Model not available")
            st.info("Run `python train_digit_generator.py` to train the model first.")

if __name__ == "__main__":
    main()