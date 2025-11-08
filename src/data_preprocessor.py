import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from typing import Tuple, Dict

class DataPreprocessor:
    """
    Veri ön işleme sınıfı: Yükleme, missing doldurma, outlier clipping, scaling.
    House Prices için optimize: Sayısal log transform, kategorik one-hot uygulandı.
    """
    
    def __init__(self, target_col: str = 'SalePrice'):
        """
        :param target_col: Hedef kolon ('SalePrice' – log transform uygulanır).
        """
        self.target_col = target_col
        self.scaler = StandardScaler()  
        self.label_encoders = {}  
        self.imputers_num = SimpleImputer(strategy='median')  # Sayısal missing için.
        self.imputers_cat = SimpleImputer(strategy='most_frequent')  # Kategorik için.
    
    def load_data(self, train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        :param train_path: Train CSV yolu.
        :param test_path: Test CSV yolu.
        :return: (train_df, test_df).
        """
        
        train_df = pd.read_csv(train_path)
       
        test_df = pd.read_csv(test_path)
        print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        return train_df, test_df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Missing'leri doldur: Sayısal median, kategorik mode.
        House Prices özel: LotFrontage median, Garage* kategorik 'None' uygulandı.
        """
        #Sayısal kolonlar için median imput
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = self.imputers_num.fit_transform(df[num_cols])
        
        # Kategorik kolonlar
        cat_cols = df.select_dtypes(include=['object']).columns
        df[cat_cols] = self.imputers_cat.fit_transform(df[cat_cols])
        
        #Garage kolonlarında missing -> 'None'.
        garage_cols = [col for col in df.columns if 'Garage' in col and df[col].dtype == 'object']
        for col in garage_cols:
            df[col] = df[col].fillna('None')
        
        print(f"Missing values dolduruldu. Kalan NaN: {df.isnull().sum().sum()}")
        return df
    
    def clip_outliers(self, df: pd.DataFrame, clip_quantile: float = 0.99) -> pd.DataFrame:
        """
        :param clip_quantile: Outlier clipping quantile (0.99 – %1 üst/alt).
        Sayısal kolonlarda IQR bazlı değil, quantile ile basitçe clip etme
        """
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if col != self.target_col:  # Hedef clip etme
                lower = df[col].quantile(1 - clip_quantile)
                upper = df[col].quantile(clip_quantile)
                df[col] = df[col].clip(lower, upper)
        print(f"Outliers clipped (quantile={clip_quantile}).")
        return df
    
    def log_transform_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Hedef (SalePrice) için log1p transform, skewness azaltmak için.
        """
        df[self.target_col] = np.log1p(df[self.target_col])
        print("Target log transform uygulandı.")
        return df
    
    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        LabelEncoder, yüksek eleman sayısı için, one-hot overfit yapar
        """
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le  # Test için sakladık.
        print(f"Kategorik kolonlar encode edildi: {cat_cols.tolist()}")
        return df
    
    def scale_numerical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sayısal kolonları scale et (fit train'de, transform test'te).
        """
        num_cols = df.select_dtypes(include=[np.number]).columns
        num_cols = [col for col in num_cols if col != self.target_col] 
        df[num_cols] = self.scaler.fit_transform(df[num_cols])
        print("Sayısal kolonlar scale edildi.")
        return df
    
    def preprocess(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Tüm preprocessing zinciri: Missing -> Outliers -> Log target -> Encode -> Scale.
        :return: (X_train, X_test) – y_train ayrı döner.
        """
        # Train ve test'e ortak uygula (leakage önlemek için).
        train_df = self.handle_missing_values(train_df)
        test_df = self.handle_missing_values(test_df)
        
        train_df = self.clip_outliers(train_df)
        test_df = self.clip_outliers(test_df)
        
        train_df = self.log_transform_target(train_df)  
        
        # Kolonları senkronize et. Test'te ekstra kolon olmasın diye.
        train_df, test_df = self._align_columns(train_df, test_df)
        
        train_df = self.encode_categorical(train_df)
        test_df = self.encode_categorical(test_df)  
        
        train_df = self.scale_numerical(train_df)
        test_df = self.scale_numerical(test_df)  
        
        # X, y ayıralım.
        y_train = train_df[self.target_col]
        X_train = train_df.drop(columns=[self.target_col, 'Id'])
        X_test = test_df.drop(columns=['Id'])
        
        print("Preprocessing tamamlandı.")
        return X_train, y_train, X_test

    def _align_columns(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Train ve test kolonlarını eşitle."""
        common_cols = train_df.columns.intersection(test_df.columns)
        train_df = train_df[common_cols]
        test_df = test_df[common_cols]
        return train_df, test_df