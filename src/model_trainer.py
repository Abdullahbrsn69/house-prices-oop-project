import xgboost as xgb
from typing import Tuple
import numpy as np

class ModelTrainer:
    """
    Model train ve predict sınıfı: XGBoost regression.

    """
    
    def __init__(self):
        """XGBoost regressor initialize (tuned params)."""
        self.model = xgb.XGBRegressor(
            n_estimators=1000,  # Ağaç sayısı (hedef -> HIGH Accuracy).
            learning_rate=0.05,  # Yavaş öğren (hedef -> STOP Overfitting).
            max_depth=3,  
            subsample=0.8,  
            colsample_bytree=0.8,  # Feature subsample.
            random_state=42
        )
        self.is_fitted = False
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Model fit et.
        :param X_train: Özellik matrisi.
        :param y_train: Hedef vektör.
        """
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        print("Model train edildi.")
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict: Expm1 ile log inverse ile orjinal scale'e döndür.
        """
        if not self.is_fitted:
            raise ValueError("Model trained değil.")
        preds = self.model.predict(X_test)
        preds = np.expm1(preds)  # Log inverse.
        return preds
    
    def get_feature_importance(self) -> dict:
        """Evaluation için feature importance dict döner """
        return dict(zip(['feature_' + str(i) for i in range(len(self.model.feature_importances_))], 
                        self.model.feature_importances_))