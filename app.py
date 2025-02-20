from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from flask_cors import CORS
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)  # Allow cross-origin requests

# Define gesture labels
GESTURES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

# Load model with absolute path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "gesture_model.h5")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logging.info("✅ Model loaded successfully")
except Exception as e:
    logging.error(f"❌ Error loading model: {e}")
    model = None

@app.route('/')
def index():
    """Serves the frontend page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Receives hand landmarks, predicts gesture, and returns result"""
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # Receive hand landmarks from frontend
        data = request.json.get('landmarks', [])
        if not data:
            return jsonify({'error': 'No landmarks received'}), 400
        
        input_data = np.array([data])  # Convert to NumPy array

        # Ensure correct input shape
        expected_shape = model.input_shape[1]
        if len(input_data[0]) != expected_shape:
            return jsonify({'error': f'Invalid input shape, expected {expected_shape} values'}), 400

        # Make prediction
        prediction = model.predict(input_data)
        predicted_class = np.argmax(prediction)  # Get highest probability index
        predicted_gesture = GESTURES[predicted_class]

        logging.info(f"✅ Prediction: {predicted_gesture} ({np.max(prediction):.2f})")

        return jsonify({'gesture': predicted_gesture, 'confidence': float(np.max(prediction))})
    
    except Exception as e:
        logging.error(f"❌ Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
