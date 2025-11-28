from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize variables
model = None
scaler = None
feature_columns = None

def load_models():
    """Load models with comprehensive error handling"""
    global model, scaler, feature_columns
    
    try:
        logger.info("Attempting to load models...")
        
        # Try different loading strategies
        try:
            model = joblib.load('price_optimization_model.pkl')
            logger.info("✓ Main model loaded successfully")
        except Exception as e:
            logger.error(f"✗ Failed to load main model: {e}")
            # Try with allow_pickle
            try:
                model = joblib.load('price_optimization_model.pkl', allow_pickle=True)
                logger.info("✓ Main model loaded with allow_pickle=True")
            except Exception as e2:
                logger.error(f"✗ Second attempt failed: {e2}")
                model = None
        
        # Load scaler
        try:
            scaler = joblib.load('scaler.pkl')
            logger.info("✓ Scaler loaded successfully")
        except Exception as e:
            logger.error(f"✗ Failed to load scaler: {e}")
            scaler = None
        
        # Load feature columns
        try:
            feature_columns = joblib.load('feature_columns.pkl')
            logger.info("✓ Feature columns loaded successfully")
        except Exception as e:
            logger.error(f"✗ Failed to load feature columns: {e}")
            feature_columns = None
            
    except Exception as e:
        logger.error(f"✗ Critical error during model loading: {e}")

# Load models on startup
load_models()

@app.route('/')
def home():
    models_status = {
        'model': model is not None,
        'scaler': scaler is not None,
        'feature_columns': feature_columns is not None
    }
    
    status = 'active' if all(models_status.values()) else 'degraded'
    
    return jsonify({
        'message': 'Bosch Price Optimization API',
        'status': status,
        'models_loaded': models_status,
        'endpoints': {
            '/predict': 'POST - Predict optimal price',
            '/health': 'GET - Health check',
            '/reload': 'POST - Reload models'
        }
    })

@app.route('/health')
def health():
    if all([model, scaler, feature_columns]):
        return jsonify({'status': 'healthy'})
    else:
        return jsonify({
            'status': 'degraded',
            'message': 'Some models failed to load'
        }), 503

@app.route('/reload', methods=['POST'])
def reload_models():
    """Endpoint to reload models"""
    load_models()
    return jsonify({
        'message': 'Models reload attempted',
        'models_loaded': {
            'model': model is not None,
            'scaler': scaler is not None,
            'feature_columns': feature_columns is not None
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if all models are loaded
        if None in [model, scaler, feature_columns]:
            missing = []
            if model is None: missing.append('model')
            if scaler is None: missing.append('scaler')
            if feature_columns is None: missing.append('feature_columns')
            
            return jsonify({
                'error': f'Models not loaded: {", ".join(missing)}',
                'status': 'failed'
            }), 500
        
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'No JSON data provided',
                'status': 'failed'
            }), 400
        
        input_df = pd.DataFrame([data])
        
        # Ensure all required columns are present
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Reorder columns to match training data
        input_df = input_df[feature_columns]
        
        # Scale features and predict
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        
        return jsonify({
            'predicted_price': float(prediction),
            'status': 'success'
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'error': str(e),
            'status': 'failed'
        }), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port)
