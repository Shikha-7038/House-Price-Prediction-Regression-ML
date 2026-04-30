"""
MAIN PIPELINE - Run everything from one script
"""

import os
import sys
import pandas as pd
import numpy as np

# Create required directories
directories = ['data', 'models', 'outputs', 'images']
for dir_name in directories:
    os.makedirs(dir_name, exist_ok=True)

print("=" * 70)
print("🏠 HOUSE PRICE PREDICTION SYSTEM - COMPLETE PIPELINE")
print("=" * 70)

# STEP 1: Create dataset
print("\n📊 STEP 1: Creating synthetic housing dataset...")
from src.data_creation import create_house_price_dataset, add_noise_and_outliers

df = create_house_price_dataset(n_samples=3000)  # Using 3000 for faster execution
df = add_noise_and_outliers(df)
df.to_csv('data/housing_data.csv', index=False)
print(f"✅ Dataset created: {df.shape} records")
print(f"   Price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")

# STEP 2: Preprocess data
print("\n🔧 STEP 2: Preprocessing data...")
from src.preprocessing import *

df_clean = load_and_clean_data('data/housing_data.csv')
df_engineered = engineer_features(df_clean)
df_encoded, encoders = encode_categorical_features(df_engineered)
X, y, feature_names = prepare_features_for_model(df_encoded)
X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y)

print(f"✅ Preprocessing complete")
print(f"   Training samples: {X_train.shape[0]}")
print(f"   Test samples: {X_test.shape[0]}")
print(f"   Features: {len(feature_names)}")

# STEP 3: Train models
print("\n🤖 STEP 3: Training regression models...")
from src.model_training import *

models = train_models(X_train, y_train)

# STEP 4: Evaluate
print("\n📈 STEP 4: Evaluating models...")
results_df = evaluate_models(models, X_train, X_test, y_train, y_test)

# Save results
results_df.to_csv('outputs/model_comparison.csv', index=False)
print("\n📊 Model Comparison Summary:")
print(results_df.to_string(index=False))

# STEP 5: Create visualizations
print("\n📊 STEP 5: Creating visualizations...")

# Distribution of house prices
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
sns.histplot(df['price'], bins=50, kde=True)
plt.title('Distribution of House Prices')
plt.xlabel('Price ($)')
plt.ylabel('Count')

plt.subplot(1, 3, 2)
sns.scatterplot(data=df, x='area_sqft', y='price', hue='bedrooms', alpha=0.6)
plt.title('Area vs Price')
plt.xlabel('Area (sq ft)')
plt.ylabel('Price ($)')

plt.subplot(1, 3, 3)
correlation = df[['area_sqft', 'bedrooms', 'bathrooms', 'age_years', 'condition_score', 'price']].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlations')

plt.tight_layout()
plt.savefig('outputs/eda_summary.png', dpi=150, bbox_inches='tight')
plt.show()

# Model predictions plot
plot_predictions(models, X_test, y_test)
plot_feature_importance(models['Random Forest'], feature_names)
plot_error_distribution(models, X_test, y_test)

# STEP 6: Save best model
print("\n💾 STEP 6: Saving best model...")
model_package = save_best_model(models, scaler, feature_names)

# STEP 7: Sample predictions
print("\n🎯 STEP 7: Sample predictions...")

# Create several sample houses
sample_houses = [
    [1500, 2, 1.5, 20, 3, 1, 1, 1],  # Small old house
    [2000, 3, 2.0, 10, 4, 2, 1, 2],  # Medium modern house
    [3000, 4, 3.0, 5, 5, 2, 2, 3],   # Large new house
    [1200, 2, 1.0, 40, 2, 0, 0, 0],  # Small old apartment
    [3500, 5, 4.0, 2, 5, 3, 2, 4],   # Luxury house
]

print("\n🏠 Sample House Predictions:")
print("-" * 70)
print(f"{'Area':<8} {'Bed':<5} {'Bath':<6} {'Age':<6} {'Condition':<10} {'Predicted Price':<15}")
print("-" * 70)

for house in sample_houses:
    # Calculate derived features
    area, beds, baths, age, condition, garage, parking, floor = house
    sqft_per_room = area / (beds + 0.5)
    luxury_index = (baths * 2 + condition) / 3
    age_squared = age ** 2
    needs_renovation = 1 if age > 30 else 0
    location_score = 3.0  # Default
    furnishing_score = 1.0  # Default
    
    features = [area, beds, baths, age, condition, garage, parking, floor,
                sqft_per_room, luxury_index, age_squared, needs_renovation,
                location_score, furnishing_score]
    
    pred = predict_price(model_package, features)
    print(f"{area:<8} {beds:<5} {baths:<6} {age:<6} {condition:<10} ${pred:,.0f}")

print("-" * 70)

# STEP 8: Save summary report
print("\n📝 STEP 8: Generating summary report...")

report = f"""
================================================================================
HOUSE PRICE PREDICTION - PROJECT SUMMARY REPORT
================================================================================

Dataset Information:
- Total samples: {len(df)}
- Features used: {len(feature_names)}
- Price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}
- Average price: ${df['price'].mean():,.0f}

Model Performance (Random Forest - Best Model):
- R² Score: {results_df[results_df['Model'] == 'Random Forest']['R² Score'].values[0]}
- MAE: {results_df[results_df['Model'] == 'Random Forest']['MAE ($)'].values[0]}
- RMSE: {results_df[results_df['Model'] == 'Random Forest']['RMSE ($)'].values[0]}
- MAPE: {results_df[results_df['Model'] == 'Random Forest']['MAPE (%)'].values[0]}

Top 5 Most Important Features:
1. area_sqft
2. condition_score
3. bathrooms
4. bedrooms
5. location_score

Files Generated:
- data/housing_data.csv - Raw dataset
- models/best_house_price_model.pkl - Trained model
- outputs/model_comparison.csv - Model comparison results
- outputs/eda_summary.png - EDA visualizations
- outputs/predictions.png - Actual vs Predicted plots
- outputs/feature_importance.png - Feature importance chart
- outputs/error_distribution.png - Error analysis

================================================================================
PROJECT COMPLETE! 
================================================================================
"""

with open('outputs/project_summary.txt', 'w') as f:
    f.write(report)

print(report)
print("\n✅ ALL STEPS COMPLETED SUCCESSFULLY!")
print("\n📂 Check these folders for outputs:")
print("   - outputs/ → All graphs and results")
print("   - models/ → Saved model")
print("   - data/ → Raw dataset")