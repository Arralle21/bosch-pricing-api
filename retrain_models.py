import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os

def create_compatible_models():
    print("=== Creating Compatible Model Files ===")
    
    # Create realistic sample data
    np.random.seed(42)
    n_samples = 500
    
    # Generate features that match your original structure
    features = {
        'qty': np.random.randint(1, 100, n_samples),
        'freight_price': np.random.uniform(10, 100, n_samples),
        'product_name_lenght': np.random.randint(10, 100, n_samples),
        'product_description_lenght': np.random.randint(50, 500, n_samples),
        'product_photos_qty': np.random.randint(1, 10, n_samples),
        'product_weight_g': np.random.randint(100, 5000, n_samples),
        'product_score': np.random.uniform(1, 5, n_samples),
        'customers': np.random.randint(100, 10000, n_samples),
        'weekday': np.random.randint(0, 7, n_samples),
        'weekend': np.random.randint(0, 2, n_samples),
        'holiday': np.random.randint(0, 2, n_samples),
        'month': np.random.randint(1, 13, n_samples),
        'year': np.random.randint(2023, 2025, n_samples),
        's': np.random.uniform(0, 1, n_samples),
        'volume': np.random.uniform(100, 10000, n_samples),
        'comp_1': np.random.uniform(50, 200, n_samples),
        'ps1': np.random.uniform(1, 5, n_samples),
        'fp1': np.random.uniform(10, 50, n_samples),
        'comp_2': np.random.uniform(50, 200, n_samples),
        'ps2': np.random.uniform(1, 5, n_samples),
        'fp2': np.random.uniform(10, 50, n_samples),
        'comp_3': np.random.uniform(50, 200, n_samples),
        'ps3': np.random.uniform(1, 5, n_samples),
        'fp3': np.random.uniform(10, 50, n_samples),
        'lag_price': np.random.uniform(50, 200, n_samples),
        'product_id_encoded': np.random.randint(0, 50, n_samples),
        'product_category_encoded': np.random.randint(0, 20, n_samples),
        'price_to_comp1_ratio': np.random.uniform(0.8, 1.2, n_samples),
        'price_to_comp2_ratio': np.random.uniform(0.8, 1.2, n_samples),
        'price_to_comp3_ratio': np.random.uniform(0.8, 1.2, n_samples),
        'avg_competitor_price': np.random.uniform(80, 150, n_samples),
        'price_vs_avg_comp': np.random.uniform(0.9, 1.1, n_samples),
        'total_freight': np.random.uniform(20, 80, n_samples),
        'avg_competitor_score': np.random.uniform(3, 5, n_samples),
        'price_per_gram': np.random.uniform(0.02, 0.08, n_samples),
        'demand_indicator': np.random.uniform(0.3, 0.9, n_samples)
    }
    
    X = pd.DataFrame(features)
    
    # Create realistic target variable (price)
    y = (
        X['comp_1'] * 0.25 +
        X['comp_2'] * 0.15 + 
        X['comp_3'] * 0.10 +
        X['product_score'] * 8 +
        X['freight_price'] * 0.5 +
        np.random.normal(0, 5, n_samples)
    )
    
    # Save feature columns
    feature_columns = X.columns.tolist()
    joblib.dump(feature_columns, 'feature_columns.pkl')
    print(f"✓ Feature columns saved ({len(feature_columns)} features)")
    
    # Create and save scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'scaler.pkl')
    print("✓ Scaler saved")
    
    # Create and save model
    model = LinearRegression()
    model.fit(X_scaled, y)
    joblib.dump(model, 'price_optimization_model.pkl')
    print("✓ Model saved")
    
    # Test the model
    test_prediction = model.predict(X_scaled[:1])[0]
    print(f"✓ Test prediction: ${test_prediction:.2f}")
    print(f"✓ Model score: {model.score(X_scaled, y):.4f}")
    
    print("=== All models created successfully! ===")
    return True

if __name__ == "__main__":
    create_compatible_models()
