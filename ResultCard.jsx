import React from 'react';

const ResultCard = ({ result, onReset }) => {
  const { prediction, confidence, reconstruction_error, threshold, heatmap } = result;
  
  const isReal = prediction === 'Real';
  const confidencePercentage = Math.round(confidence * 100);

  return (
    <div className="gradient-card rounded-2xl p-8 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-800">Analysis Results</h2>
        <button
          onClick={onReset}
          className="px-4 py-2 rounded-lg bg-gray-200 text-gray-700 hover:bg-gray-300 transition-colors duration-300"
        >
          Analyze Another Image
        </button>
      </div>

      {/* Prediction Result */}
      <div className="text-center">
        <div className={`inline-flex items-center px-6 py-3 rounded-full text-lg font-semibold ${
          isReal ? 'prediction-real' : 'prediction-ai'
        }`}>
          <div className="flex items-center space-x-2">
            {isReal ? (
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            ) : (
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
              </svg>
            )}
            <span>{prediction}</span>
          </div>
        </div>
      </div>

      {/* Confidence Score */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-gray-600">Confidence Score</span>
          <span className="text-sm font-bold text-gray-800">{confidencePercentage}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className={`confidence-bar h-2 rounded-full ${
              isReal ? 'confidence-real' : 'confidence-ai'
            }`}
            style={{ width: `${confidencePercentage}%` }}
          ></div>
        </div>
      </div>

      {/* Technical Details */}
      <div className="bg-gray-50 rounded-lg p-4 space-y-3">
        <h3 className="font-semibold text-gray-800">Technical Details</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-gray-600">Reconstruction Error:</span>
            <span className="ml-2 font-mono text-gray-800">
              {reconstruction_error.toFixed(6)}
            </span>
          </div>
          <div>
            <span className="text-gray-600">Detection Threshold:</span>
            <span className="ml-2 font-mono text-gray-800">
              {threshold.toFixed(6)}
            </span>
          </div>
          <div>
            <span className="text-gray-600">Model Type:</span>
            <span className="ml-2 text-gray-800">Autoencoder</span>
          </div>
          <div>
            <span className="text-gray-600">Method:</span>
            <span className="ml-2 text-gray-800">Unsupervised Learning</span>
          </div>
        </div>
      </div>

      {/* Grad-CAM Heatmap */}
      {heatmap && (
        <div className="space-y-4">
          <h3 className="font-semibold text-gray-800">Grad-CAM Visualization</h3>
          <div className="text-sm text-gray-600 mb-3">
            This heatmap shows which parts of the image contributed most to the detection decision.
            Red areas indicate higher importance for the model's prediction.
          </div>
          <div className="heatmap-container">
            <img
              src={`data:image/png;base64,${heatmap}`}
              alt="Grad-CAM Heatmap"
              className="heatmap-image"
            />
          </div>
          <div className="text-xs text-gray-500 text-center">
            <p>🔴 Red areas: High importance for prediction</p>
            <p>🔵 Blue areas: Lower importance for prediction</p>
          </div>
        </div>
      )}

      {/* Explanation */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <h4 className="font-semibold text-blue-800 mb-2">How it works:</h4>
        <p className="text-sm text-blue-700">
          This detector uses an unsupervised autoencoder trained only on real images. 
          When you upload an image, it calculates the reconstruction error - how well 
          the model can recreate the image. AI-generated images typically have higher 
          reconstruction errors because they contain patterns the model hasn't seen during training.
        </p>
      </div>
    </div>
  );
};

export default ResultCard;
