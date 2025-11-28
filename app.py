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
    """Load models with fallback to new versions"""
    global model, scaler, feature_columns
    
    try:
        logger.info("Attempting to load models...")
        
        # Try to load new models first, then fallback to old ones
        model_files = [
            ('price_optimization_model_new.pkl', 'price_optimization_model.pkl'),
            ('scaler_new.pkl', 'scaler.pkl'), 
            ('feature_columns_new.pkl', 'feature_columns.pkl')
        ]
        
        for new_file, old_file in model_files:
            try:
                if os.path.exists(new_file):
                    if 'model' in new_file:
                        model = joblib.load(new_file)
                        logger.info(f"✓ Loaded {new_file}")
                    elif 'scaler' in new_file:
                        scaler = joblib.load(new_file)
                        logger.info(f"✓ Loaded {new_file}")
                    elif 'feature' in new_file:
                        feature_columns = joblib.load(new_file)
                        logger.info(f"✓ Loaded {new_file}")
                elif os.path.exists(old_file):
                    if 'model' in old_file:
                        model = joblib.load(old_file)
                        logger.info(f"✓ Loaded {old_file}")
                    elif 'scaler' in old_file:
                        scaler = joblib.load(old_file)
                        logger.info(f"✓ Loaded {old_file}")
                    elif 'feature' in old_file:
                        feature_columns = joblib.load(old_file)
                        logger.info(f"✓ Loaded {old_file}")
            except Exception as e:
                logger.error(f"✗ Failed to load {new_file} or {old_file}: {e}")
                
    except Exception as e:
        logger.error(f"✗ Critical error during model loading: {e}")

# Load models on startup
load_models()

@app.route('/')
def home():
    models_loaded = {
        'model': model is not None,
        'scaler': scaler is not None,
        'feature_columns': feature_columns is not None
    }
    
    status = 'active' if all(models_loaded.values()) else 'degraded'
    
    return jsonify({
        'message': 'Bosch Price Optimization API',
        'status': status,
        'models_loaded': models_loaded,
        'endpoints': {
            '/predict': 'POST - Predict optimal price',
            '/health': 'GET - Health check',
            '/retrain': 'GET - Retrain models (development)'
        }
    })

@app.route('/health')
def health():
    if all([model, scaler, feature_columns]):
        return jsonify({'status': 'healthy'})
    else:
        return jsonify({
            'status': 'degraded',
            'message': 'Some models failed to load',
            'loaded_models': {
                'model': model is not None,
                'scaler': scaler is not None,
                'feature_columns': feature_columns is not None
            }
        }), 503

@app.route('/retrain')
def retrain_models():
    """Endpoint to retrain models (for development)"""
    try:
        from retrain_models import retrain_models as retrain
        retrain()
        # Reload models after retraining
        load_models()
        return jsonify({'message': 'Models retrained successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
                'status': 'failed',
                'solution': 'Run /retrain endpoint to create new models'
            }), 500
        
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'No JSON data provided',
                'status': 'failed'
            }), 400
        
        # Create input DataFrame
        input_df = pd.DataFrame([data])
        
        # Ensure all required columns are present
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0  # or appropriate default value
        
        # Reorder columns to match training data
        input_df = input_df[feature_columns]
        
        # Scale features and predict
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        
        return jsonify({
            'predicted_price': float(prediction),
            'status': 'success',
            'features_used': len(feature_columns)
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
