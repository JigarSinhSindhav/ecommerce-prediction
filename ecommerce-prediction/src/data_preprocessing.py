"""
Data Preprocessing and Cleaning Module
Handles data cleaning, feature engineering, and preparation for ML models
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        
    def load_data(self, file_path):
        """Load the dataset from CSV file"""
        print("Loading dataset...")
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        print(f"Dataset loaded: {df.shape}")
        return df
    
    def clean_data(self, df):
        """Clean the dataset by handling missing values and outliers"""
        print("Cleaning data...")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print("Missing values found:")
            print(missing_values[missing_values > 0])
            
            # Fill missing values based on data type
            for column in df.columns:
                if df[column].dtype in ['float64', 'int64']:
                    df[column].fillna(df[column].median(), inplace=True)
                else:
                    df[column].fillna(df[column].mode()[0], inplace=True)
        
        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed_duplicates = initial_rows - len(df)
        if removed_duplicates > 0:
            print(f"Removed {removed_duplicates} duplicate rows")
        
        # Handle outliers using IQR method for numerical columns
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        outliers_removed = 0
        
        for column in numerical_columns:
            if column not in ['sales', 'product_id']:  # Don't remove outliers from target or ID
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
                outliers_removed += len(outliers)
                df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        
        if outliers_removed > 0:
            print(f"Removed {outliers_removed} outlier records")
        
        print(f"Data cleaning completed. Final shape: {df.shape}")
        return df.reset_index(drop=True)
    
    def engineer_features(self, df):
        """Create new features from existing data"""
        print("Engineering features...")
        
        # Create date-based features
        df['year'] = df['date'].dt.year
        df['quarter'] = df['date'].dt.quarter
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week
        
        # Create price-related features
        df['discount_amount'] = df['original_price'] - df['final_price']
        df['price_per_rating'] = df['final_price'] / (df['avg_rating'] + 0.001)  # Avoid division by zero
        df['reviews_per_day'] = df['num_reviews'] / (df['days_since_launch'] + 1)
        
        # Create engagement features
        df['engagement_score'] = (df['page_views'] * (1 - df['bounce_rate']) * df['time_on_page']) / 1000
        df['social_engagement'] = df['social_media_mentions'] * df['avg_rating']
        
        # Create inventory features
        df['stock_turnover'] = df['sales'] / (df['stock_level'] + 1)
        df['low_stock'] = (df['stock_level'] < 50).astype(int)
        
        # Create competition features
        df['price_competitiveness'] = (df['competitor_price'] - df['final_price']) / df['competitor_price']
        
        # Create interaction features
        df['promotion_rating_interaction'] = df['is_promoted'] * df['avg_rating']
        df['weekend_promotion'] = df['is_weekend'] * df['is_promoted']
        
        print(f"Feature engineering completed. New shape: {df.shape}")
        return df
    
    def encode_categorical_features(self, df):
        """Encode categorical variables"""
        print("Encoding categorical features...")
        
        categorical_columns = ['category', 'brand', 'season']
        
        for column in categorical_columns:
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
                df[f'{column}_encoded'] = self.label_encoders[column].fit_transform(df[column])
            else:
                df[f'{column}_encoded'] = self.label_encoders[column].transform(df[column])
        
        # Create one-hot encoding for categories (alternative approach)
        df_encoded = pd.get_dummies(df, columns=['category', 'brand', 'season'], prefix=['cat', 'brand', 'season'])
        
        print("Categorical encoding completed")
        return df_encoded
    
    def select_features(self, df):
        """Select relevant features for modeling"""
        print("Selecting features...")
        
        # Define feature columns (exclude target and non-predictive columns)
        exclude_columns = ['product_id', 'date', 'sales', 'category', 'brand', 'season']
        
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        X = df[feature_columns]
        y = df['sales']
        
        print(f"Selected {len(feature_columns)} features for modeling")
        print("Features:", feature_columns[:10], "..." if len(feature_columns) > 10 else "")
        
        return X, y
    
    def scale_features(self, X_train, X_test=None):
        """Scale features using StandardScaler"""
        print("Scaling features...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        print(f"Splitting data (test_size={test_size})...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )
        
        print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def preprocess_pipeline(self, file_path, test_size=0.2):
        """Complete preprocessing pipeline"""
        print("Starting preprocessing pipeline...")
        
        # Load and clean data
        df = self.load_data(file_path)
        df_clean = self.clean_data(df)
        
        # Engineer features
        df_featured = self.engineer_features(df_clean)
        
        # Encode categorical variables
        df_encoded = self.encode_categorical_features(df_featured)
        
        # Select features
        X, y = self.select_features(df_encoded)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y, test_size)
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        print("Preprocessing pipeline completed!")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def get_feature_importance_data(self):
        """Return data for feature importance analysis"""
        return {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler
        }

def main():
    """Main function to test the preprocessing pipeline"""
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Run preprocessing pipeline
    data_path = '../data/ecommerce_sales_data.csv'
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(data_path)
    
    print("\nPreprocessing Summary:")
    print(f"Training features shape: {X_train.shape}")
    print(f"Testing features shape: {X_test.shape}")
    print(f"Training target shape: {y_train.shape}")
    print(f"Testing target shape: {y_test.shape}")
    
    print(f"\nTarget variable statistics:")
    print(f"Training - Mean: {y_train.mean():.2f}, Std: {y_train.std():.2f}")
    print(f"Testing - Mean: {y_test.mean():.2f}, Std: {y_test.std():.2f}")

if __name__ == "__main__":
    main()
