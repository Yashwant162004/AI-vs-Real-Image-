"""
Evaluate the trained autoencoder model and generate confusion matrix
Tests on both RealArt and AiArtData images
"""

import os
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import glob

def load_and_preprocess_images(image_paths, target_size=(224, 224)):
    """
    Load and preprocess images from given paths
    Returns normalized image array and labels
    """
    images = []
    labels = []
    
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
            
            # Determine label based on directory
            if 'RealArt' in path:
                labels.append(0)  # Real
            elif 'AiArtData' in path:
                labels.append(1)  # AI-generated
            else:
                labels.append(0)  # Default to real
            
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            continue
    
    return np.array(images), np.array(labels)

def evaluate_model():
    """Evaluate the trained model on test data"""
    print("Evaluating AI vs Real Image Detector")
    print("=" * 50)
    
    # Load model
    model_path = "model/ai_detector_model.h5"
    threshold_path = "model/threshold.npy"
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}. Please train the model first.")
        return
    
    if not os.path.exists(threshold_path):
        print(f"ERROR: Threshold not found at {threshold_path}. Please train the model first.")
        return
    
    autoencoder = tf.keras.models.load_model(model_path)
    threshold = float(np.load(threshold_path))
    
    print(f"Model loaded successfully")
    print(f"Threshold: {threshold:.6f}")
    
    # Load test data
    real_art_path = "../RealArt/RealArt"
    ai_art_path = "../AiArtData/AiArtData"
    
    # Get image paths
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    
    real_image_paths = []
    for ext in image_extensions:
        real_image_paths.extend(glob.glob(os.path.join(real_art_path, ext)))
        real_image_paths.extend(glob.glob(os.path.join(real_art_path, ext.upper())))
    
    ai_image_paths = []
    for ext in image_extensions:
        ai_image_paths.extend(glob.glob(os.path.join(ai_art_path, ext)))
        ai_image_paths.extend(glob.glob(os.path.join(ai_art_path, ext.upper())))
    
    print(f"Found {len(real_image_paths)} real images")
    print(f"Found {len(ai_image_paths)} AI images")
    
    # Sample equal number of images for evaluation
    min_samples = min(len(real_image_paths), len(ai_image_paths), 200)  # Limit for faster evaluation
    
    real_sample_paths = np.random.choice(real_image_paths, min_samples, replace=False)
    ai_sample_paths = np.random.choice(ai_image_paths, min_samples, replace=False)
    
    all_paths = list(real_sample_paths) + list(ai_sample_paths)
    
    # Load and preprocess images
    test_images, true_labels = load_and_preprocess_images(all_paths)
    
    print(f"Loaded {test_images.shape[0]} test images")
    
    # Get predictions
    print("🔮 Making predictions...")
    reconstructions = autoencoder.predict(test_images, verbose=0)
    reconstruction_errors = np.mean(np.square(test_images - reconstructions), axis=(1, 2, 3))
    
    # Predictions based on threshold
    predictions = (reconstruction_errors > threshold).astype(int)
    
    # Calculate metrics
    accuracy = np.mean(predictions == true_labels)
    
    print(f"\nEvaluation Results:")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Confusion Matrix
    cm = confusion_matrix(true_labels, predictions)
    
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Real  AI")
    print(f"Actual Real      {cm[0,0]:4d} {cm[0,1]:4d}")
    print(f"       AI        {cm[1,0]:4d} {cm[1,1]:4d}")
    
    # Classification Report
    print(f"\nClassification Report:")
    print(classification_report(true_labels, predictions, 
                              target_names=['Real', 'AI-generated']))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'AI-generated'],
                yticklabels=['Real', 'AI-generated'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Plot reconstruction errors distribution
    plt.subplot(2, 2, 2)
    real_errors = reconstruction_errors[true_labels == 0]
    ai_errors = reconstruction_errors[true_labels == 1]
    
    plt.hist(real_errors, bins=50, alpha=0.7, label='Real Images', color='green')
    plt.hist(ai_errors, bins=50, alpha=0.7, label='AI Images', color='red')
    plt.axvline(threshold, color='black', linestyle='--', label=f'Threshold: {threshold:.4f}')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title('Reconstruction Error Distribution')
    plt.legend()
    
    # Plot accuracy by threshold
    plt.subplot(2, 2, 3)
    thresholds = np.linspace(0, np.max(reconstruction_errors), 100)
    accuracies = []
    
    for t in thresholds:
        pred = (reconstruction_errors > t).astype(int)
        acc = np.mean(pred == true_labels)
        accuracies.append(acc)
    
    plt.plot(thresholds, accuracies)
    plt.axvline(threshold, color='red', linestyle='--', label=f'Current Threshold: {threshold:.4f}')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Threshold')
    plt.legend()
    
    # Sample predictions visualization
    plt.subplot(2, 2, 4)
    sample_indices = np.random.choice(len(test_images), 8, replace=False)
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    
    for i, idx in enumerate(sample_indices):
        axes[i].imshow(test_images[idx])
        true_label = 'Real' if true_labels[idx] == 0 else 'AI'
        pred_label = 'Real' if predictions[idx] == 0 else 'AI'
        error = reconstruction_errors[idx]
        
        color = 'green' if true_label == pred_label else 'red'
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}\nError: {error:.4f}', 
                         color=color, fontsize=8)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('model/evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Error analysis
    print(f"\nError Analysis:")
    real_correct = np.sum((true_labels == 0) & (predictions == 0))
    real_total = np.sum(true_labels == 0)
    ai_correct = np.sum((true_labels == 1) & (predictions == 1))
    ai_total = np.sum(true_labels == 1)
    
    print(f"Real images correctly identified: {real_correct}/{real_total} ({real_correct/real_total*100:.2f}%)")
    print(f"AI images correctly identified: {ai_correct}/{ai_total} ({ai_correct/ai_total*100:.2f}%)")
    
    # False positives and negatives
    false_positives = np.sum((true_labels == 0) & (predictions == 1))
    false_negatives = np.sum((true_labels == 1) & (predictions == 0))
    
    print(f"False positives (Real classified as AI): {false_positives}")
    print(f"False negatives (AI classified as Real): {false_negatives}")
    
    print(f"\nEvaluation completed!")
    print(f"Results saved to: model/evaluation_results.png")

if __name__ == "__main__":
    evaluate_model()
