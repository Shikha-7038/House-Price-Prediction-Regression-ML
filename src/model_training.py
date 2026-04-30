"""
MODEL TRAINING & EVALUATION
Trains multiple regression models and compares performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import preprocessing functions
import sys
sys.path.append('..')
from src.preprocessing import *

def train_models(X_train, y_train):
    """
    Train multiple regression models
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.001),
        'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    trained_models = {}
    
    print("\n" + "=" * 60)
    print("TRAINING MODELS")
    print("=" * 60)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"✅ {name} trained")
    
    return trained_models

def evaluate_models(models, X_train, X_test, y_train, y_test):
    """
    Evaluate all models and return results dataframe
    """
    results = []
    
    for name, model in models.items():
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Metrics for test set
        mae = mean_absolute_error(y_test, y_test_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        r2 = r2_score(y_test, y_test_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
        
        # Training score (for overfitting check)
        train_r2 = r2_score(y_train, y_train_pred)
        
        results.append({
            'Model': name,
            'MAE ($)': f"${mae:,.0f}",
            'RMSE ($)': f"${rmse:,.0f}",
            'R² Score': f"{r2:.4f}",
            'MAPE (%)': f"{mape:.1f}%",
            'Train R²': f"{train_r2:.4f}"
        })
        
        print(f"\n📊 {name}:")
        print(f"   MAE: ${mae:,.0f}")
        print(f"   RMSE: ${rmse:,.0f}")
        print(f"   R²: {r2:.4f}")
        print(f"   MAPE: {mape:.1f}%")
    
    return pd.DataFrame(results)

def plot_predictions(models, X_test, y_test, save_path='outputs/predictions.png'):
    """
    Create actual vs predicted price plots
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (name, model) in enumerate(models.items()):
        if idx >= 6:
            break
        
        y_pred = model.predict(X_test)
        
        axes[idx].scatter(y_test, y_pred, alpha=0.5, s=10)
        axes[idx].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                      'r--', lw=2, label='Perfect Prediction')
        axes[idx].set_xlabel('Actual Price ($)')
        axes[idx].set_ylabel('Predicted Price ($)')
        axes[idx].set_title(f'{name}')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        
        # Add R² score
        r2 = r2_score(y_test, y_pred)
        axes[idx].text(0.05, 0.95, f'R² = {r2:.3f}', 
                      transform=axes[idx].transAxes, fontsize=10,
                      verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Predictions plot saved to {save_path}")

def plot_feature_importance(model, feature_names, save_path='outputs/feature_importance.png'):
    """
    Plot feature importance for tree-based models
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importance (Random Forest)')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"✅ Feature importance plot saved to {save_path}")
        
        # Print top features
        print("\n🔝 Top 5 Most Important Features:")
        for i in range(min(5, len(indices))):
            print(f"   {i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

def plot_error_distribution(models, X_test, y_test, save_path='outputs/error_distribution.png'):
    """
    Plot error distribution for the best model
    """
    best_model = models['Random Forest']
    y_pred = best_model.predict(X_test)
    errors = y_test - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram of errors
    axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    axes[0].set_xlabel('Prediction Error ($)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Prediction Errors')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Q-Q plot (for normality check)
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot (Normality Check)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Error distribution plot saved to {save_path}")
    
    print(f"\n📊 Error Statistics:")
    print(f"   Mean Error: ${errors.mean():,.0f}")
    print(f"   Std Error: ${errors.std():,.0f}")
    print(f"   90% of errors between: ${errors.quantile(0.05):,.0f} and ${errors.quantile(0.95):,.0f}")

def save_best_model(models, scaler, feature_names, filepath='models/best_house_price_model.pkl'):
    """
    Save the best performing model
    """
    # Random Forest is usually best for tabular data
    best_model = models['Random Forest']
    
    # Save model, scaler, and feature names together
    model_package = {
        'model': best_model,
        'scaler': scaler,
        'feature_names': feature_names,
        'model_type': 'Random Forest Regressor'
    }
    
    joblib.dump(model_package, filepath)
    print(f"✅ Best model saved to {filepath}")
    
    return model_package

def predict_price(model_package, input_features):
    """
    Predict price for a single house
    """
    # Scale input features
    scaled_input = model_package['scaler'].transform([input_features])
    
    # Predict
    prediction = model_package['model'].predict(scaled_input)[0]
    
    return prediction

if __name__ == "__main__":
    print("=" * 60)
    print("HOUSE PRICE PREDICTION - MODEL TRAINING")
    print("=" * 60)
    
    # Load and preprocess data
    print("\n📂 Loading and preprocessing data...")
    df = load_and_clean_data()
    df = engineer_features(df)
    df, encoders = encode_categorical_features(df)
    X, y, feature_names = prepare_features_for_model(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y)
    
    # Train models
    models = train_models(X_train, y_train)
    
    # Evaluate models
    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    results_df = evaluate_models(models, X_train, X_test, y_train, y_test)
    print("\n📊 Complete Results:")
    print(results_df.to_string(index=False))
    
    # Save results to CSV
    results_df.to_csv('outputs/model_comparison.csv', index=False)
    
    # Create visualizations
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    plot_predictions(models, X_test, y_test)
    plot_feature_importance(models['Random Forest'], feature_names)
    plot_error_distribution(models, X_test, y_test)
    
    # Save best model
    model_package = save_best_model(models, scaler, feature_names)
    
    # Test prediction
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTION")
    print("=" * 60)
    
    # Create a sample house
    sample_house = [
        2000,    # area_sqft
        3,       # bedrooms
        2,       # bathrooms
        10,      # age_years
        4,       # condition_score
        2,       # garage_cars
        1,       # parking_spots
        2,       # floor_number
        2000/3.5,  # sqft_per_room (calculated)
        (2*2 + 4)/3,  # luxury_index
        100,     # age_squared
        0,       # needs_renovation
        4,       # location_score
        1        # furnishing_score
    ]
    
    predicted_price = predict_price(model_package, sample_house)
    print(f"🏠 Sample House Prediction: ${predicted_price:,.0f}")
    
    print("\n✅ Model training complete!")