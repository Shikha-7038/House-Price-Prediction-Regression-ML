"""
DATA PREPROCESSING & FEATURE ENGINEERING
Handles missing values, outliers, encoding, and feature creation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_and_clean_data(filepath='data/housing_data.csv'):
    """
    Load and perform initial cleaning
    """
    df = pd.read_csv(filepath)
    
    print(f"Initial shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # Remove duplicate rows if any
    df = df.drop_duplicates()
    
    # Remove unrealistic outliers (top 1% and bottom 1% of price)
    price_upper = df['price'].quantile(0.99)
    price_lower = df['price'].quantile(0.01)
    df = df[(df['price'] >= price_lower) & (df['price'] <= price_upper)]
    
    # Remove unrealistic area outliers
    area_upper = df['area_sqft'].quantile(0.99)
    df = df[df['area_sqft'] <= area_upper]
    
    print(f"After cleaning shape: {df.shape}")
    
    return df

def engineer_features(df):
    """
    Create new features from existing ones
    """
    df = df.copy()
    
    # Square footage per room
    df['sqft_per_room'] = df['area_sqft'] / (df['bedrooms'] + 0.5)
    
    # Luxury index (combination of bathroom count and condition)
    df['luxury_index'] = (df['bathrooms'] * 2 + df['condition_score']) / 3
    
    # Age squared (non-linear relationship)
    df['age_squared'] = df['age_years'] ** 2
    
    # Renovation needed flag
    df['needs_renovation'] = (df['age_years'] > 30).astype(int)
    
    # Location encoding
    location_mapping = {
        'Downtown': 5, 'Riverside': 4.5, 'Lakeside': 4.5, 'Hillview': 4,
        'Northside': 3.5, 'Southside': 3, 'Greenfield': 2.5,
        'Suburb_A': 2, 'Suburb_B': 1.5, 'OldTown': 1
    }
    df['location_score'] = df['location'].map(location_mapping)
    
    # Furnishing encoding
    furnishing_mapping = {
        'Fully Furnished': 2,
        'Semi-Furnished': 1,
        'Unfurnished': 0
    }
    df['furnishing_score'] = df['furnishing_status'].map(furnishing_mapping)
    
    # Property value tier (derived from price)
    df['price_tier'] = pd.qcut(df['price'], q=4, labels=['Budget', 'Standard', 'Premium', 'Luxury'])
    
    return df

def encode_categorical_features(df, fit_encoders=True, saved_encoders=None):
    """
    Encode categorical variables for machine learning
    """
    df = df.copy()
    
    categorical_cols = ['location', 'furnishing_status', 'age_category']
    encoders = {}
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            # Handle NaN values
            df[col] = df[col].fillna('Unknown')
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    
    return df, encoders

def prepare_features_for_model(df):
    """
    Select and prepare final feature set for training
    """
    # Features to use for prediction
    feature_cols = [
        'area_sqft',
        'bedrooms',
        'bathrooms',
        'age_years',
        'condition_score',
        'garage_cars',
        'parking_spots',
        'floor_number',
        'sqft_per_room',
        'luxury_index',
        'age_squared',
        'needs_renovation',
        'location_score',
        'furnishing_score'
    ]
    
    # Ensure all features exist
    available_features = [col for col in feature_cols if col in df.columns]
    
    X = df[available_features]
    y = df['price']
    
    print(f"Features used: {available_features}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    return X, y, available_features

def split_and_scale_data(X, y, test_size=0.2, random_state=42):
    """
    Split into train/test and scale features
    """
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features (important for linear models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == "__main__":
    # Test preprocessing pipeline
    print("=" * 50)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 50)
    
    df = load_and_clean_data()
    df = engineer_features(df)
    df, encoders = encode_categorical_features(df)
    X, y, features = prepare_features_for_model(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y)
    
    print("\n✅ Preprocessing complete!")
    print(f"Training: {X_train.shape[0]} samples")
    print(f"Testing: {X_test.shape[0]} samples")