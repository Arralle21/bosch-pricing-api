# Bosch Product Pricing Optimization

Machine learning solution for optimizing product pricing based on demand elasticity, competitor pricing, and customer preferences.

## 🎯 Project Overview

This project develops and deploys a pricing optimization model for Bosch Corporation to enable data-driven pricing decisions in real-time.

**Live API:** https://bosch-pricing-api-1.onrender.com

## 📊 Dataset

- **Source:** [Kaggle - Retail Price Optimization](https://www.kaggle.com/datasets/bhanupratapbiswas/retail-price-optimization-case-study)
- **Records:** 676 transactions
- **Features:** 30 variables (product, competitor, customer data)

## 🚀 Model Performance

| Model | Test R² | RMSE | MAE |
|-------|---------|------|-----|
| **Linear Regression** | **1.0000** | **0.0000** | **0.0000** |
| Lasso Regression | 0.9999 | 0.5477 | 0.3056 |
| Ridge Regression | 0.9997 | 1.2092 | 0.7068 |

## 🛠️ Technology Stack

- Python 3.11
- Scikit-learn 1.6.1
- Pandas 2.0.3
- Flask 3.0.0
- Gunicorn 21.2.0
- Deployed on Render

## 📁 Repository Structure
```
├── app.py                          # Flask API application
├── requirements.txt                # Python dependencies
├── .python-version                 # Python version specification
├── price_optimization_model.pkl    # Trained model
├── scaler.pkl                      # Feature scaler
├── feature_columns.pkl             # Feature list
├── retail_price.csv               # Dataset
└── README.md                       # Documentation
```

## 🔗 API Endpoints

### Base URL
```
https://bosch-pricing-api-1.onrender.com
```

### Endpoints

**1. Home**
```
GET /
```

**2. Health Check**
```
GET /health
```

**3. Price Prediction**
```
POST /predict
Content-Type: application/json

{
  "qty": 10,
  "freight_price": 15.0,
  "product_weight_g": 1000,
  "product_score": 4.1,
  "customers": 50,
  "comp_1": 89.9,
  "comp_2": 95.0,
  "comp_3": 85.0,
  "lag_price": 90.0
}
```

**Response:**
```json
{
  "predicted_price": 89.95,
  "status": "success"
}
```

## 💻 Local Setup

1. **Clone repository**
```bash
git clone https://github.com/Arralle21/bosch-pricing-api.git
cd bosch-pricing-api
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run application**
```bash
python app.py
```

4. **Access API**
```
http://localhost:5000
```

## 🧪 Test the API

**Using cURL:**
```bash
curl -X POST https://bosch-pricing-api-1.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"qty":10,"customers":50,"comp_1":89.9,"comp_2":95.0,"comp_3":85.0,"product_score":4.1}'
```

**Using Python:**
```python
import requests

url = "https://bosch-pricing-api-1.onrender.com/predict"
data = {
    "qty": 10,
    "customers": 50,
    "comp_1": 89.9,
    "comp_2": 95.0,
    "comp_3": 85.0,
    "product_score": 4.1
}

response = requests.post(url, json=data)
print(response.json())
```

## 📈 Key Features

- ✅ Real-time price prediction
- ✅ 99.9%+ accuracy
- ✅ REST API with JSON responses
- ✅ Cloud-deployed solution
- ✅ Scalable architecture

## 👤 Author

**Your Name**  
Nexford University - Business Analytics Final Project

## 📄 License

This project is for educational purposes.

---

**Live Demo:** https://bosch-pricing-api-1.onrender.com
