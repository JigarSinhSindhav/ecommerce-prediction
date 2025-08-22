"""
Data Generator for E-commerce Sales Prediction
Generates synthetic e-commerce data with 50,000+ records
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class EcommerceDataGenerator:
    def __init__(self, n_records=50000):
        self.n_records = n_records
        self.categories = ['Electronics', 'Clothing', 'Home & Garden', 'Books', 'Sports', 'Beauty', 'Automotive']
        self.brands = ['Brand_A', 'Brand_B', 'Brand_C', 'Brand_D', 'Brand_E', 'Brand_F', 'Brand_G']
        self.seasons = ['Spring', 'Summer', 'Fall', 'Winter']
        
    def generate_dataset(self):
        """Generate comprehensive e-commerce dataset"""
        np.random.seed(42)
        random.seed(42)
        
        # Generate time series data
        start_date = datetime(2021, 1, 1)
        end_date = datetime(2023, 12, 31)
        dates = pd.date_range(start_date, end_date, freq='D')
        
        data = []
        
        for i in range(self.n_records):
            # Basic product info
            product_id = f"PROD_{i+1:05d}"
            category = np.random.choice(self.categories)
            brand = np.random.choice(self.brands)
            
            # Date and seasonality
            date = np.random.choice(dates)
            date_obj = pd.to_datetime(date)  # Convert to datetime object
            season = self._get_season(date_obj)
            month = date_obj.month
            day_of_week = date_obj.weekday()
            is_weekend = 1 if day_of_week >= 5 else 0
            
            # Product attributes
            price = np.random.lognormal(mean=3, sigma=0.8)  # Log-normal distribution for prices
            discount_pct = np.random.choice([0, 5, 10, 15, 20, 25, 30], p=[0.3, 0.2, 0.15, 0.15, 0.1, 0.05, 0.05])
            final_price = price * (1 - discount_pct/100)
            
            # Customer behavior features
            avg_rating = np.random.normal(4.0, 0.8)
            avg_rating = np.clip(avg_rating, 1, 5)
            num_reviews = np.random.poisson(lam=20)
            
            # Marketing and promotion features
            is_promoted = np.random.choice([0, 1], p=[0.7, 0.3])
            social_media_mentions = np.random.poisson(lam=5) if is_promoted else np.random.poisson(lam=1)
            
            # Website traffic features
            page_views = np.random.poisson(lam=100)
            time_on_page = np.random.exponential(scale=180)  # seconds
            bounce_rate = np.random.beta(2, 8)  # Typical bounce rate distribution
            
            # Inventory features
            stock_level = np.random.randint(0, 1000)
            days_since_launch = (date_obj - start_date).days + np.random.randint(0, 365)
            
            # Competition features
            competitor_price = price * np.random.normal(1, 0.2)
            competitor_price = max(competitor_price, 0)
            price_advantage = (competitor_price - final_price) / competitor_price if competitor_price > 0 else 0
            
            # Calculate sales (target variable)
            # Sales influenced by multiple factors
            base_sales = 10  # base sales
            
            # Price effect (negative correlation)
            price_effect = max(0, 100 - final_price) * 0.5
            
            # Rating effect
            rating_effect = (avg_rating - 1) * 10
            
            # Promotion effect
            promotion_effect = 20 if is_promoted else 0
            
            # Seasonality effect
            seasonality_effect = self._get_seasonality_effect(category, season)
            
            # Weekend effect
            weekend_effect = 5 if is_weekend else 0
            
            # Stock effect
            stock_effect = min(stock_level / 10, 50)
            
            # Page views effect
            traffic_effect = page_views * 0.1
            
            # Price advantage effect
            advantage_effect = price_advantage * 30
            
            # Random noise
            noise = np.random.normal(0, 10)
            
            # Calculate final sales
            sales = (base_sales + price_effect + rating_effect + promotion_effect + 
                    seasonality_effect + weekend_effect + stock_effect + 
                    traffic_effect + advantage_effect + noise)
            
            sales = max(0, int(sales))  # Ensure non-negative integer sales
            
            data.append({
                'product_id': product_id,
                'date': date,
                'category': category,
                'brand': brand,
                'season': season,
                'month': month,
                'day_of_week': day_of_week,
                'is_weekend': is_weekend,
                'original_price': round(price, 2),
                'discount_pct': discount_pct,
                'final_price': round(final_price, 2),
                'avg_rating': round(avg_rating, 2),
                'num_reviews': num_reviews,
                'is_promoted': is_promoted,
                'social_media_mentions': social_media_mentions,
                'page_views': page_views,
                'time_on_page': round(time_on_page, 2),
                'bounce_rate': round(bounce_rate, 3),
                'stock_level': stock_level,
                'days_since_launch': days_since_launch,
                'competitor_price': round(competitor_price, 2),
                'price_advantage': round(price_advantage, 3),
                'sales': sales
            })
        
        df = pd.DataFrame(data)
        return df
    
    def _get_season(self, date):
        """Determine season based on date"""
        month = date.month
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    def _get_seasonality_effect(self, category, season):
        """Get seasonality effect based on category and season"""
        effects = {
            'Electronics': {'Winter': 15, 'Spring': 5, 'Summer': 5, 'Fall': 10},
            'Clothing': {'Winter': 10, 'Spring': 15, 'Summer': 5, 'Fall': 20},
            'Home & Garden': {'Winter': 0, 'Spring': 25, 'Summer': 20, 'Fall': 5},
            'Books': {'Winter': 10, 'Spring': 5, 'Summer': 0, 'Fall': 15},
            'Sports': {'Winter': 5, 'Spring': 20, 'Summer': 25, 'Fall': 10},
            'Beauty': {'Winter': 5, 'Spring': 10, 'Summer': 15, 'Fall': 5},
            'Automotive': {'Winter': 0, 'Spring': 15, 'Summer': 10, 'Fall': 5}
        }
        return effects.get(category, {}).get(season, 0)

def main():
    """Generate and save the dataset"""
    print("Generating synthetic e-commerce dataset...")
    generator = EcommerceDataGenerator(n_records=50000)
    df = generator.generate_dataset()
    
    # Save the dataset
    output_path = '../data/ecommerce_sales_data.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Dataset generated successfully!")
    print(f"Shape: {df.shape}")
    print(f"Saved to: {output_path}")
    print("\nDataset Overview:")
    print(df.head())
    print("\nBasic Statistics:")
    print(df.describe())

if __name__ == "__main__":
    main()
