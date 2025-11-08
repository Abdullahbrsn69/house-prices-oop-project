import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from category_encoders import TargetEncoder
from sklearn.model_selection import KFold
from typing import List, Tuple

class FeatureEngineer:
    """
    Feature Engineering class: Interactions, target encoding, time features.
    """
    
    def __init__(self, target_col: str, cat_features: List[str], num_features: List[str], date_col: str = 'DateSold'):
        """
        :param target_col: Hedef
        :param cat_features: Target enc için kategorik listesi (['Neighborhood']).
        :param num_features: Polinom için sayısal listesi (['LotArea', 'OverallQual']).
        :param date_col: Zaman kolonu (parse için).
        """
        self.target_col = target_col
        self.cat_features = cat_features
        self.num_features = num_features
        self.date_col = date_col
        self.encoders = {}  
    
    def polynomial_interactions(self, df: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
        """
        :param degree: Polinom derecesi 
        Sayısal için polinom özellikler üret, non-lineer yakala.
        """
        if not self.num_features:
            return df
    
        poly = PolynomialFeatures(degree=degree, interaction_only=True, include_bias=False)
        poly_features = poly.fit_transform(df[self.num_features])
        # Kolon isimleri üret.
        poly_cols = poly.get_feature_names_out(self.num_features)
        # Yeni df'ye ekle.
        poly_df = pd.DataFrame(poly_features, columns=poly_cols, index=df.index)
        df = pd.concat([df, poly_df], axis=1)
        print(f"Polynomial features eklendi: {len(poly_cols)} yeni kolon.")
        return df
    
    def target_encoding(self, df: pd.DataFrame, smoothing: int = 10) -> pd.DataFrame:
        """
        :param smoothing: Overfit önleme
        Kategorik'leri target ile encode et
        """
        if not self.cat_features:
            return df
    
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        encoded_df = df.copy()
        for cat in self.cat_features:
            for train_idx, val_idx in kf.split(df):
                train_fold = df.iloc[train_idx]
                val_fold = df.iloc[val_idx]
            
                encoder = TargetEncoder(cols=[cat], smoothing=smoothing)
                encoder.fit(train_fold[[cat]], train_fold[self.target_col])
                
                val_encoded = encoder.transform(val_fold[[cat]])
                encoded_df.iloc[val_idx, encoded_df.columns.get_loc(cat)] = val_encoded[cat].values
            # Test için sakla
            self.encoders[cat] = encoder
        print(f"Target encoding uygulandı: {self.cat_features}")
        return encoded_df
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tarih kolonunu parse et, yıl/ay/quarter ekle
        House Prices'ta seasonality için (yaz satışları yüksek olur genelde)
        """
        if self.date_col not in df.columns:
            return df
        
        df[self.date_col] = pd.to_datetime(df[self.date_col], errors='coerce')
        # Zaman özellikleri çıkaralım
        df['year'] = df[self.date_col].dt.year
        df['month'] = df[self.date_col].dt.month
        df['quarter'] = df[self.date_col].dt.quarter
        df['is_summer'] = (df['month'].isin([6,7,8])).astype(int)
        print("Zaman özellikleri eklendi.")
        return df
    
    def transform_test(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """Test için transform """
        X_test = self.polynomial_interactions(X_test)
        for cat in self.cat_features:
            if cat in self.encoders:
                encoder = self.encoders[cat]
                X_test[cat] = encoder.transform(X_test[[cat]])[cat].values
        X_test = self.add_time_features(X_test)
        return X_test
    
    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Train fit + transform, test transform.
        :return: (X_train_eng, X_test_eng)
        """
        # Train'e uygula.
        X_train_eng = self.polynomial_interactions(X_train)
        X_train_eng = self.target_encoding(X_train_eng)  # y_train ile.
        X_train_eng = self.add_time_features(X_train_eng)
        
        # Test'e uygula
        # Not: X_test main'de geçilir.
        return X_train_eng