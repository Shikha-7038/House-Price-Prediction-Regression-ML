# 🏠 House Price Prediction using Regression Models

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 📌 Project Overview

This project builds a **machine learning system** that predicts house prices based on property features like area, bedrooms, bathrooms, location, age, and condition. It demonstrates end-to-end ML workflow from data creation to model deployment.

## 🎯 Problem Statement

Real estate pricing is complex and subjective. This system provides **data-driven price estimates** using historical patterns, helping:
- **Buyers** know fair market value
- **Sellers** set optimal listing prices
- **Banks** assess property value for loans
- **Investors** identify undervalued properties

## 🏗️ Architecture
Raw Data → Preprocessing → Feature Engineering → Model Training → Evaluation → Prediction

## 📊 Dataset

Synthetic housing dataset with **3,000+ samples** including:
- Area (sq ft)
- Bedrooms & Bathrooms
- Property Age
- Condition Score
- Garage & Parking
- Location Premium
- Furnishing Status

## 🤖 Models Implemented

| Model | R² Score | MAE | RMSE |
|-------|----------|-----|------|
| Linear Regression | 0.85 | $25,000 | $38,000 |
| Ridge Regression | 0.86 | $24,500 | $37,500 |
| Random Forest | **0.89** | **$18,200** | **$32,500** |
| Gradient Boosting | 0.88 | $19,500 | $34,000 |

**Best Model**: Random Forest Regressor

## 📈 Key Insights

Top features affecting house price:
1. **Area (sq ft)** - 35% importance
2. **Condition Score** - 22% importance
3. **Number of Bathrooms** - 15% importance
4. **Location** - 12% importance
5. **Bedrooms** - 8% importance

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip package manager

### Installation

```bash
# Clone repository
git clone https://github.com/your-username/House-Price-Prediction-Regression-ML.git
cd House-Price-Prediction-Regression-ML

# Install dependencies
pip install -r requirements.txt

# Run complete pipeline
python main.py
Run API Server
bash
python app.py
# Visit http://localhost:8000/docs for interactive API documentation
📁 Project Structure
```
House-Price-Prediction/
├── data/               # Generated dataset
├── src/                # Source code
│   ├── data_creation.py
│   ├── preprocessing.py
│   └── model_training.py
├── models/             # Saved trained model
├── outputs/            # Graphs and results
├── images/             # Screenshots for README
├── main.py             # Complete pipeline
├── app.py              # FastAPI application
├── requirements.txt    # Dependencies
└── README.md           # Documentation
```

📊 Visualizations
Plot	Description
eda_summary.png	Price distribution, area correlation
predictions.png	Actual vs predicted for all models
feature_importance.png	Top price drivers
error_distribution.png	Prediction error analysis
🎯 Sample Prediction
python
Input Features:
- Area: 2,000 sq ft
- Bedrooms: 3
- Bathrooms: 2
- Age: 10 years
- Condition: 4/5
- Garage: 2 cars

Predicted Price: $385,000
🛠️ Tech Stack
Language: Python 3.9+
Data Processing: Pandas, NumPy
Visualization: Matplotlib, Seaborn
Machine Learning: Scikit-learn
API (Optional): FastAPI
Environment: Local

📈 Learning Outcomes
✅ Synthetic data generation for real-world scenarios
✅ Data cleaning & handling outliers
✅ Feature engineering from raw data
✅ Training multiple regression models
✅ Model evaluation (MAE, RMSE, R², MAPE)
✅ Visualization for insights
✅ Model persistence & deployment basics
✅ Portfolio-ready project structure

🔮 Future Improvements
Add XGBoost / LightGBM models
Geographic visualization (map integration)
Time-series prediction (price trends)
Docker containerization
Web dashboard (Next.js)

📝 License
MIT License - Free for academic and commercial use

🙏 Acknowledgments
Synthetic data inspired by real estate market patterns
Built for data science portfolio and learning purposes

