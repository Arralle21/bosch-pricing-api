import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

def retrain_models():
    print("Retraining models with current library versions...")
    
    # Create sample data that matches your feature structure
    # In a real scenario, you would load your actual training data here
    np.random.seed(42)
    n_samples = 1000
    
    # Create synthetic training data matching your feature columns
    feature_data = {
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
        'year': np.random.randint(2020, 2024, n_samples),
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
        'product_id_encoded': np.random.randint(0, 100, n_samples),
        'product_category_encoded': np.random.randint(0, 20, n_samples),
        'price_to_comp1_ratio': np.random.uniform(0.5, 1.5, n_samples),
        'price_to_comp2_ratio': np.random.uniform(0.5, 1.5, n_samples),
        'price_to_comp3_ratio': np.random.uniform(0.5, 1.5, n_samples),
        'avg_competitor_price': np.random.uniform(50, 200, n_samples),
        'price_vs_avg_comp': np.random.uniform(0.5, 1.5, n_samples),
        'total_freight': np.random.uniform(10, 100, n_samples),
        'avg_competitor_score': np.random.uniform(1, 5, n_samples),
        'price_per_gram': np.random.uniform(0.01, 0.1, n_samples),
        'demand_indicator': np.random.uniform(0, 1, n_samples)
    }
    
    # Create DataFrame
    X = pd.DataFrame(feature_data)
    
    # Create target variable (price) - in real scenario, use your actual y data
    y = (X['comp_1'] * 0.3 + 
         X['comp_2'] * 0.2 + 
         X['comp_3'] * 0.1 + 
         X['product_score'] * 10 + 
         np.random.normal(0, 10, n_samples))
    
    # Save feature columns
    feature_columns = X.columns.tolist()
    joblib.dump(feature_columns, 'feature_columns_new.pkl')
    print("✓ Feature columns saved")
    
    # Create and fit scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'scaler_new.pkl')
    print("✓ Scaler saved")
    
    # Create and fit model
    model = LinearRegression()
    model.fit(X_scaled, y)
    joblib.dump(model, 'price_optimization_model_new.pkl')
    print("✓ Model saved")
    
    # Test prediction
    sample_pred = model.predict(X_scaled[:1])[0]
    print(f"✓ Sample prediction: {sample_pred:.2f}")
    
    print("All models retrained and saved successfully!")

if __name__ == "__main__":
    retrain_models()
