# AI vs Real Image Detector

A complete full-stack application that detects whether an uploaded image is **AI-generated or Real** using an unsupervised deep-learning Autoencoder built with TensorFlow/Keras, featuring Grad-CAM visualization.

## 🎯 Project Overview

This project uses an **unsupervised autoencoder** approach where the model is trained only on real images. When a new image is uploaded, the model calculates the reconstruction error - AI-generated images typically have higher reconstruction errors because they contain patterns the model hasn't seen during training.

### Key Features

- **Unsupervised Learning**: Model trained only on real images
- **Grad-CAM Visualization**: Shows which parts of the image contributed to the decision
- **Modern UI**: Beautiful React frontend with TailwindCSS
- **REST API**: FastAPI backend with CORS support
- **Real-time Analysis**: Upload and get instant results

## 🏗️ Project Structure

```
yashu1/
├── backend/
│   ├── app.py                  # FastAPI backend for predictions
│   ├── train_model.py          # Train autoencoder on RealArt images
│   ├── evaluate_model.py       # Evaluate model & plot confusion matrix
│   ├── model/                  # Model storage directory
│   │   ├── ai_detector_model.h5
│   │   ├── threshold.npy
│   │   ├── training_history.png
│   │   └── evaluation_results.png
│   ├── utils/
│   │   ├── preprocess.py       # Image preprocessing utilities
│   │   └── gradcam.py         # Grad-CAM visualization
│   └── requirements.txt
├── frontend/
│   ├── package.json
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   ├── public/
│   │   ├── index.html
│   │   └── manifest.json
│   └── src/
│       ├── App.jsx
│       ├── index.js
│       ├── index.css
│       └── components/
│           ├── UploadForm.jsx
│           └── ResultCard.jsx
├── AiArtData/                 # AI-generated images dataset
├── RealArt/                   # Real images dataset
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 16+
- npm or yarn

### 1. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Train the autoencoder model (this may take some time)
python train_model.py

# Start the FastAPI server
uvicorn app:app --reload
```

The backend will be available at `http://localhost:8000`

### 2. Frontend Setup

```bash
# Navigate to frontend directory (in a new terminal)
cd frontend

# Install dependencies
npm install

# Start the React development server
npm start
```

The frontend will be available at `http://localhost:3000`

## 📊 Dataset

The project uses two existing folders:

- **`./AiArtData`** → AI-generated images (test set)
- **`./RealArt`** → Real images (training set)

### Data Preprocessing

- Images are resized to (224, 224)
- Normalized to [0, 1] range
- Autoencoder is trained only on `RealArt` images
- Train/validation split: 80/20 from RealArt
- Test set includes both RealArt and AiArtData images

## 🧠 Model Architecture

### Autoencoder Design

The autoencoder uses a convolutional architecture with:

- **Encoder**: Progressive downsampling with Conv2D + MaxPooling2D layers
- **Bottleneck**: Compressed representation (512 channels)
- **Decoder**: Progressive upsampling with Conv2D + UpSampling2D layers
- **Output**: Sigmoid activation for pixel values in [0, 1]

### Training Process

1. **Unsupervised Learning**: Model learns to reconstruct real images
2. **Threshold Calculation**: Uses 95th percentile of validation reconstruction errors
3. **Anomaly Detection**: Higher reconstruction error = more likely AI-generated

## 🔧 API Endpoints

### POST `/predict/`

Upload an image and get prediction results.

**Request:**
- `file`: Image file (multipart/form-data)

**Response:**
```json
{
  "prediction": "AI-generated" | "Real",
  "confidence": 0.85,
  "reconstruction_error": 0.023456,
  "threshold": 0.015000,
  "heatmap": "base64_encoded_image"
}
```

### GET `/`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "message": "AI vs Real Image Detector API is running"
}
```

## 🎨 Frontend Features

### Upload Interface
- Drag & drop file upload
- Image preview
- File validation (type and size)
- Loading states

### Results Display
- **Prediction**: Real (green) or AI-generated (red)
- **Confidence Score**: Visual progress bar
- **Technical Details**: Reconstruction error, threshold, model info
- **Grad-CAM Heatmap**: Visual explanation of the decision
- **Educational Content**: How the system works

### Responsive Design
- Mobile-friendly interface
- Modern gradient backgrounds
- Smooth animations and transitions

## 🔍 Grad-CAM Implementation

Grad-CAM (Gradient-weighted Class Activation Mapping) visualizes which parts of the image contributed most to the reconstruction error:

- **Red areas**: High importance for the model's prediction
- **Blue areas**: Lower importance for the model's prediction
- **Overlay**: Blended with original image for better visualization

## 📈 Model Evaluation

Run the evaluation script to see model performance:

```bash
cd backend
python evaluate_model.py
```

This generates:
- Confusion matrix
- Accuracy metrics
- Reconstruction error distributions
- Sample predictions visualization
- Classification report

## 🛠️ Development

### Backend Development

```bash
# Install development dependencies
pip install -r requirements.txt

# Train model with custom parameters
python train_model.py

# Evaluate model performance
python evaluate_model.py

# Run with auto-reload
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Development

```bash
# Install dependencies
npm install

# Start development server
npm start

# Build for production
npm run build
```

## 🔧 Configuration

### Model Parameters

Edit `train_model.py` to modify:
- Image size: `target_size=(224, 224)`
- Batch size: `batch_size=32`
- Epochs: `epochs=50`
- Learning rate: `learning_rate=0.001`

### Threshold Calculation

The threshold is calculated as the 95th percentile of reconstruction errors on validation data. Modify the percentile in `train_model.py`:

```python
threshold = np.percentile(reconstruction_errors, 95)  # Change 95 to desired percentile
```

## 🐛 Troubleshooting

### Common Issues

1. **Model not found error**
   - Run `python train_model.py` first to train the model

2. **CORS errors**
   - Ensure backend is running on `http://localhost:8000`
   - Check CORS settings in `app.py`

3. **Image upload fails**
   - Check file size (max 10MB)
   - Ensure file is a valid image format

4. **Training takes too long**
   - Reduce image size or batch size
   - Use fewer epochs for testing

### Performance Optimization

- **GPU Support**: Install TensorFlow with GPU support for faster training
- **Batch Processing**: Increase batch size if you have more memory
- **Image Resolution**: Reduce image size for faster processing

## 📚 Technical Details

### How It Works

1. **Training Phase**:
   - Autoencoder learns to reconstruct real images
   - Model compresses images to a bottleneck representation
   - Decoder reconstructs images from compressed representation

2. **Detection Phase**:
   - Uploaded image is preprocessed and fed to the model
   - Model generates reconstruction
   - Reconstruction error is calculated (MSE between input and output)
   - Error is compared to threshold to make prediction

3. **Grad-CAM Generation**:
   - Gradients of reconstruction error w.r.t. feature maps are computed
   - Feature maps are weighted by gradient importance
   - Heatmap is generated and overlaid on original image

### Model Performance

The model's effectiveness depends on:
- Quality and diversity of training data
- Model architecture complexity
- Threshold selection
- Image preprocessing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- TensorFlow/Keras for the deep learning framework
- FastAPI for the web framework
- React for the frontend
- TailwindCSS for styling
- OpenCV for image processing

---

**Note**: This is a demonstration project. For production use, consider additional validation, error handling, and security measures.
