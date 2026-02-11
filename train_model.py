"""
Train autoencoder model for AI vs Real Image Detection
Uses unsupervised learning on RealArt images only
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import glob

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_and_preprocess_images(image_paths, target_size=(224, 224)):
    """
    Load and preprocess images from given paths
    Returns normalized image array
    """
    images = []
    
    print(f"Loading {len(image_paths)} images...")
    
    for i, path in enumerate(image_paths):
        if i % 100 == 0:
            print(f"Processed {i}/{len(image_paths)} images")
        
        try:
            # Load image
            image = Image.open(path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize image
            image = image.resize(target_size)
            
            # Convert to array and normalize
            image_array = np.array(image, dtype=np.float32) / 255.0
            
            images.append(image_array)
            
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            continue
    
    return np.array(images)

def create_autoencoder(input_shape=(224, 224, 3)):
    """
    Create autoencoder model for anomaly detection
    Smaller bottleneck forces learning of important features
    """
    # Encoder
    encoder_input = keras.Input(shape=input_shape, name='encoder_input')
    
    # Convolutional layers with decreasing dimensions
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(encoder_input)
    x = layers.MaxPooling2D(2, padding='same')(x)
    
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2, padding='same')(x)
    
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2, padding='same')(x)
    
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2, padding='same')(x)
    
    # Bottleneck (compressed representation)
    encoded = layers.Conv2D(512, 3, activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D(2, padding='same')(encoded)
    
    # Decoder (mirror of encoder)
    x = layers.Conv2D(512, 3, activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D(2)(x)
    
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D(2)(x)
    
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D(2)(x)
    
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D(2)(x)
    
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D(2)(x)
    
    # Output layer
    decoder_output = layers.Conv2D(3, 3, activation='sigmoid', padding='same', name='decoder_output')(x)
    
    # Create autoencoder model
    autoencoder = keras.Model(encoder_input, decoder_output, name='autoencoder')
    
    return autoencoder

def calculate_threshold(model, validation_data):
    """
    Calculate optimal threshold based on validation data
    Uses 95th percentile of reconstruction errors
    """
    print("Calculating optimal threshold...")
    
    # Get reconstruction errors on validation data
    reconstructions = model.predict(validation_data, verbose=0)
    reconstruction_errors = np.mean(np.square(validation_data - reconstructions), axis=(1, 2, 3))
    
    # Use 95th percentile as threshold
    threshold = np.percentile(reconstruction_errors, 95)
    
    print(f"Threshold calculated: {threshold:.6f}")
    print(f"Mean reconstruction error: {np.mean(reconstruction_errors):.6f}")
    print(f"Std reconstruction error: {np.std(reconstruction_errors):.6f}")
    
    return threshold

def main():
    """Main training function"""
    print("Starting AI vs Real Image Detector Training")
    print("=" * 50)
    
    # Create model directory
    os.makedirs("model", exist_ok=True)
    
    # Load RealArt images for training
    real_art_path = "../RealArt/RealArt"
    if not os.path.exists(real_art_path):
        print(f"ERROR: RealArt directory not found at {real_art_path}")
        return
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    real_image_paths = []
    
    for ext in image_extensions:
        real_image_paths.extend(glob.glob(os.path.join(real_art_path, ext)))
        real_image_paths.extend(glob.glob(os.path.join(real_art_path, ext.upper())))
    
    print(f"Found {len(real_image_paths)} real art images")
    
    if len(real_image_paths) == 0:
        print("ERROR: No images found in RealArt directory")
        return
    
    # Load and preprocess images
    real_images = load_and_preprocess_images(real_image_paths)
    print(f"Loaded {real_images.shape[0]} real images with shape {real_images.shape[1:]}")
    
    # Split into train and validation
    train_images, val_images = train_test_split(
        real_images, 
        test_size=0.2, 
        random_state=42
    )
    
    print(f"Training set: {train_images.shape[0]} images")
    print(f"Validation set: {val_images.shape[0]} images")
    
    # Create autoencoder model
    print("\nCreating autoencoder model...")
    autoencoder = create_autoencoder()
    
    # Compile model
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    # Print model summary
    autoencoder.summary()
    
    # Training callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            'model/ai_detector_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    print("\nTraining autoencoder...")
    history = autoencoder.fit(
        train_images, train_images,  # Autoencoder learns to reconstruct input
        epochs=50,
        batch_size=32,
        validation_data=(val_images, val_images),
        callbacks=callbacks,
        verbose=1
    )
    
    # Calculate threshold
    threshold = calculate_threshold(autoencoder, val_images)
    
    # Save threshold
    np.save('model/threshold.npy', threshold)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('model/training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nTraining completed!")
    print(f"Model saved to: model/ai_detector_model.h5")
    print(f"Threshold saved to: model/threshold.npy")
    print(f"Training history saved to: model/training_history.png")
    
    # Test on some sample images
    print("\nTesting model on sample images...")
    sample_indices = np.random.choice(len(val_images), 5, replace=False)
    
    for i, idx in enumerate(sample_indices):
        sample_image = val_images[idx:idx+1]
        reconstruction = autoencoder.predict(sample_image, verbose=0)
        error = np.mean(np.square(sample_image - reconstruction))
        
        print(f"Sample {i+1}: Reconstruction error = {error:.6f} ({'AI' if error > threshold else 'Real'})")

if __name__ == "__main__":
    main()
