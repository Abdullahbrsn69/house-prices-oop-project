import pandas as pd
import numpy as np
from src.data_preprocessor import DataPreprocessor
from src.feature_engineer import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.evaluator import Evaluator
import os

# Dosya yolları (data klasöründe).
TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'
SUBMISSION_PATH = 'data/submission.csv'

def main():
    """
    Ana pipeline: Load -> Preprocess -> FE -> Train -> Evaluate -> Predict -> Submit.
    """
    # Veri yükle ve preprocess et
    preprocessor = DataPreprocessor(target_col='SalePrice')
    train_df, test_df = preprocessor.load_data(TRAIN_PATH, TEST_PATH)
    X_train, y_train, X_test = preprocessor.preprocess(train_df, test_df)
    
    # Feature Engineering
    fe = FeatureEngineer(
        target_col='SalePrice',
        cat_features=['Neighborhood'],  
        num_features=['LotArea', 'OverallQual', 'YearBuilt']  # Interactions için.
    )
    X_train_eng = fe.fit_transform(X_train, y_train)  # Train FE.
    X_test_eng = fe.transform_test(X_test)  # Test FE.
    
    #Array'e çevi
    X_train_np = X_train_eng.values
    X_test_np = X_test_eng.values
    y_train_np = y_train.values
    
    #Model train.
    trainer = ModelTrainer()
    trainer.train(X_train_np, y_train_np)
    
    #Evaluate.
    evaluator = Evaluator(trainer.model)
    cv_metrics = evaluator.cross_validate(X_train_np, y_train_np)
    train_preds = trainer.predict(X_train_np)
    train_metrics = evaluator.evaluate_predictions(np.expm1(y_train_np), np.expm1(train_preds))  # Orijinal scale.
    
    #Importance plot.
    importance = trainer.get_feature_importance()
    evaluator.plot_feature_importance(importance)
    
    #Test predict.
    test_preds = trainer.predict(X_test_np)
    
    #Submission CSV üret (Id ile).
    submission = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': test_preds})
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission kaydedildi: {SUBMISSION_PATH}")
    
    # Başarı notu.
    print("Hey! CV RMSE düşükse tweak et yani n_estimators i artır.")

if __name__ == "__main__":
    main()