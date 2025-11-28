from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Try to load models, but don't crash if they don't exist yet
try:
    model = joblib.load('price_optimization_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    models_loaded = True
    print("✅ All models loaded successfully!")
except Exception as e:
    model = None
    scaler = None
    feature_columns = None
    models_loaded = False
    print(f"⚠️ Models not loaded: {e}")
    print("💡 Visit /create-models to generate new models")

@app.route('/')
def home():
    return jsonify({
        'message': 'Bosch Price Optimization API',
        'status': 'ready' if models_loaded else 'setup_required',
        'models_loaded': models_loaded,
        'endpoints': {
            '/': 'GET - API info',
            '/health': 'GET - Health check',
            '/create-models': 'GET - Create new models',
            '/predict': 'POST - Predict optimal price'
        }
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy' if models_loaded else 'setup_required',
        'models_ready': models_loaded
    })

@app.route('/create-models')
def create_models():
    """Create new compatible models"""
    try:
        from retrain_models import create_compatible_models
        success = create_compatible_models()
        
        if success:
            # Reload models
            global model, scaler, feature_columns, models_loaded
            model = joblib.load('price_optimization_model.pkl')
            scaler = joblib.load('scaler.pkl')
            feature_columns = joblib.load('feature_columns.pkl')
            models_loaded = True
            
            return jsonify({
                'status': 'success',
                'message': 'Models created and loaded successfully!'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to create models'
            }), 500
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error creating models: {str(e)}'
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    if not models_loaded:
        return jsonify({
            'error': 'Models not loaded',
            'solution': 'Visit /create-models first to generate models',
            'status': 'setup_required'
        }), 503
    
    try:
        data = request.get_json()
        
        # Create input DataFrame
        input_df = pd.DataFrame([data])
        
        # Add missing columns with default values
        for col in feature_columns:
            if col not in input_df.columns:
                if 'price' in col or 'comp' in col:
                    input_df[col] = 100.0  # Reasonable default for prices
                elif 'qty' in col or 'customers' in col:
                    input_df[col] = 50     # Reasonable default for quantities
                else:
                    input_df[col] = 0
        
        # Ensure correct column order
        input_df = input_df[feature_columns]
        
        # Scale and predict
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        
        return jsonify({
            'predicted_price': round(float(prediction), 2),
            'status': 'success',
            'features_used': len(feature_columns)
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
