import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

class Evaluator:
    """
    Evaluation sınıfı: CV skorlar, metrikler (RMSE, MAE, R2), importance plot.
    """
    
    def __init__(self, model):
        """
        :param model: Eğitilmiş model
        """
        self.model = model
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> dict:
        """
        :param cv_folds: KFold sayısı.
        :return: {'rmse': mean, 'mae': mean, 'r2': mean}
        CV skorları hesaplama
        """
        # RMSE için neg_mse skor.
        rmse_scores = np.sqrt(-cross_val_score(self.model, X, y, cv=cv_folds, scoring='neg_mean_squared_error'))
        mae_scores = -cross_val_score(self.model, X, y, cv=cv_folds, scoring='neg_mean_absolute_error')
        r2_scores = cross_val_score(self.model, X, y, cv=cv_folds, scoring='r2')
        
        metrics = {
            'rmse': rmse_scores.mean(),
            'mae': mae_scores.mean(),
            'r2': r2_scores.mean()
        }
        print(f"CV RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, R2: {metrics['r2']:.4f}")
        return metrics
    
    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Train seti predictions için metrikler.
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        metrics = {'rmse': rmse, 'mae': mae, 'r2': r2}
        print(f"Train RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
        return metrics
    
    def plot_feature_importance(self, importance_dict: dict, top_n: int = 10):
        """
        :param top_n: En önemli top N feature ı plot ediyoruz.
        Bar plot ile importance göster (debug için).
        """
        top_features = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n])
        plt.figure(figsize=(10, 6))
        plt.barh(list(top_features.keys()), list(top_features.values()))
        plt.title('Top Feature Importances')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')  
        plt.show()
        print("Importance plot kaydedildi: feature_importance.png")