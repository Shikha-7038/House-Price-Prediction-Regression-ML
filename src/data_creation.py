"""
SYNTHETIC HOUSING DATA CREATION
Creates realistic house price dataset without real estate data access
"""

import pandas as pd
import numpy as np
from datetime import datetime

def create_house_price_dataset(n_samples=5000, random_seed=42):
    """
    Create synthetic housing dataset with realistic patterns
    
    Parameters:
    n_samples: number of houses to generate
    random_seed: for reproducible results
    
    Returns:
    pandas DataFrame with house features and prices
    """
    np.random.seed(random_seed)
    
    # ============== LOCATION FEATURES ==============
    neighborhoods = ['Downtown', 'Suburb_A', 'Suburb_B', 'Riverside', 
                     'Lakeside', 'Hillview', 'Greenfield', 'OldTown', 
                     'Northside', 'Southside']
    # Higher score = more premium location
    location_premium = {
        'Downtown': 1.4,    # Most expensive
        'Riverside': 1.3,
        'Lakeside': 1.3,
        'Hillview': 1.2,
        'Northside': 1.1,
        'Southside': 1.0,
        'Greenfield': 0.9,
        'Suburb_A': 0.85,
        'Suburb_B': 0.8,
        'OldTown': 0.75     # Least expensive
    }
    
    # ============== GENERATE FEATURES ==============
    # Area features (in sq ft)
    area_base = np.random.normal(1800, 500, n_samples)
    area_base = np.clip(area_base, 600, 5000)
    
    # Number of bedrooms (correlated with area)
    bedrooms = np.random.choice([1, 2, 3, 4, 5], n_samples, 
                                 p=[0.05, 0.25, 0.40, 0.20, 0.10])
    # Adjust based on area (larger houses = more bedrooms)
    bedrooms = np.where(area_base > 3000, bedrooms + 1, bedrooms)
    bedrooms = np.clip(bedrooms, 1, 6)
    
    # Number of bathrooms
    bathrooms = np.round(np.random.normal(bedrooms * 0.8, 0.3), 1)
    bathrooms = np.clip(bathrooms, 1, 5)
    
    # Property age - FIXED: probabilities now sum to 1
    # Creating a probability distribution that sums to 1
    age_probs = []
    for i in range(100):
        if i == 0:  # Age 0 (new house)
            prob = 0.02
        elif i <= 40:  # Age 1-40
            prob = 0.015
        else:  # Age 41-99
            prob = 0.005
        age_probs.append(prob)
    
    # Normalize to ensure sum = 1
    age_probs = np.array(age_probs) / sum(age_probs)
    
    age = np.random.choice(range(0, 100), n_samples, p=age_probs)
    current_year = 2024
    year_built = current_year - age
    
    # Condition (1=Poor, 5=Excellent) - better for newer houses
    condition = np.where(age < 5, 5, 
                        np.where(age < 15, 4,
                                np.where(age < 30, 3,
                                        np.where(age < 50, 2, 1))))
    # Add some randomness
    condition = condition + np.random.choice([-1, 0, 1], n_samples, p=[0.1, 0.8, 0.1])
    condition = np.clip(condition, 1, 5)
    
    # Garage (cars that fit)
    garage_cars = np.where(area_base > 2500, 
                          np.random.choice([0, 1, 2, 3], n_samples, p=[0.1, 0.3, 0.4, 0.2]),
                          np.random.choice([0, 1, 2], n_samples, p=[0.2, 0.5, 0.3]))
    
    # Parking (additional open parking spots)
    parking = np.where(garage_cars > 0, 
                      np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.4, 0.2]),
                      np.random.choice([1, 2, 3], n_samples, p=[0.3, 0.5, 0.2]))
    
    # Furnishing status
    furnishing = np.random.choice(['Fully Furnished', 'Semi-Furnished', 'Unfurnished'], 
                                  n_samples, p=[0.3, 0.4, 0.3])
    furnishing_multiplier = {'Fully Furnished': 1.15, 'Semi-Furnished': 1.05, 'Unfurnished': 1.0}
    
    # Location
    location = np.random.choice(neighborhoods, n_samples)
    
    # Floor (Ground=0, etc.)
    floor_probs = [0.3] + [0.07] * 9  # 0.3 + 0.07*9 = 0.93, need to adjust to sum to 1
    floor_probs = np.array(floor_probs) / sum(floor_probs)  # Normalize
    floor = np.random.choice(range(0, 10), n_samples, p=floor_probs)
    
    # Number of floors in building
    total_floors = np.random.choice(range(1, 6), n_samples, p=[0.3, 0.3, 0.2, 0.1, 0.1])
    
    # ============== CALCULATE PRICE ==============
    # Base price formula (in USD or local currency)
    base_price_per_sqft = 200  # $200 per sq ft base
    
    # Calculate components
    area_component = area_base * base_price_per_sqft
    
    bedroom_component = bedrooms * 15000
    bathroom_component = bathrooms * 10000
    
    # Age discount (newer houses cost more)
    age_discount = np.where(age < 5, 1.2,
                           np.where(age < 10, 1.1,
                                   np.where(age < 20, 1.0,
                                           np.where(age < 40, 0.9, 0.8))))
    
    # Condition multiplier
    condition_multiplier = {1: 0.7, 2: 0.85, 3: 1.0, 4: 1.15, 5: 1.3}
    condition_component = [condition_multiplier[c] for c in condition]
    
    # Garage value
    garage_component = garage_cars * 8000
    parking_component = parking * 3000
    
    # Location premium
    location_component = [location_premium[loc] for loc in location]
    
    # Furnishing multiplier
    furnishing_component = [furnishing_multiplier[f] for f in furnishing]
    
    # Floor premium (higher floors cost more)
    floor_premium = 1 + (floor / total_floors) * 0.1
    
    # Calculate final price
    price = (area_component + 
             bedroom_component + 
             bathroom_component +
             garage_component +
             parking_component)
    
    price = price * np.array(age_discount)
    price = price * np.array(condition_component)
    price = price * np.array(location_component)
    price = price * np.array(furnishing_component)
    price = price * np.array(floor_premium)
    
    # Add random noise (real-world variation)
    noise = np.random.normal(1, 0.1, n_samples)
    price = price * noise
    
    # Round and ensure positive
    price = np.round(price, -2)  # Round to nearest 100
    price = np.maximum(price, 20000)  # Minimum price
    
    # ============== CREATE DATAFRAME ==============
    df = pd.DataFrame({
        'house_id': range(1, n_samples + 1),
        'area_sqft': np.round(area_base, 0),
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'age_years': age,
        'year_built': year_built,
        'condition_score': condition,
        'garage_cars': garage_cars,
        'parking_spots': parking,
        'furnishing_status': furnishing,
        'location': location,
        'floor_number': floor,
        'total_floors': total_floors,
        'price': price
    })
    
    # Add some derived/engineering features
    df['price_per_sqft'] = df['price'] / df['area_sqft']
    df['rooms_per_bath'] = df['bedrooms'] / df['bathrooms']
    df['age_category'] = pd.cut(df['age_years'], 
                                  bins=[-1, 5, 15, 30, 50, 100],
                                  labels=['New', 'Recent', 'Moderate', 'Old', 'Very Old'])
    
    return df

def add_noise_and_outliers(df, outlier_percentage=0.02):
    """
    Add realistic outliers to the dataset
    """
    n_outliers = int(len(df) * outlier_percentage)
    outlier_indices = np.random.choice(df.index, n_outliers, replace=False)
    
    # Mark outliers with extreme prices
    df.loc[outlier_indices, 'price'] = df.loc[outlier_indices, 'price'] * np.random.uniform(1.5, 2.5, n_outliers)
    
    # Also add some extreme area outliers
    area_outliers = int(len(df) * 0.01)
    area_outlier_indices = np.random.choice(df.index, area_outliers, replace=False)
    df.loc[area_outlier_indices, 'area_sqft'] = df.loc[area_outlier_indices, 'area_sqft'] * np.random.uniform(1.5, 2.0, area_outliers)
    
    print(f"Added {n_outliers} price outliers and {area_outliers} area outliers")
    return df

if __name__ == "__main__":
    # Create the dataset
    print("🏠 Creating synthetic housing dataset...")
    df = create_house_price_dataset(n_samples=3000)
    df = add_noise_and_outliers(df)
    
    # Create data directory if it doesn't exist
    import os
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV
    df.to_csv('data/housing_data.csv', index=False)
    print(f"✅ Dataset saved! Shape: {df.shape}")
    print(f"📊 Price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
    print(f"💰 Average price: ${df['price'].mean():,.0f}")
    
    # Preview
    print("\n📋 First 5 rows:")
    print(df.head())
    
    # Quick statistics
    print("\n📈 Key statistics:")
    print(df[['area_sqft', 'bedrooms', 'bathrooms', 'age_years', 'price']].describe())