# House Prices Kaggle Project

OOP tabanlı ML pipeline: Data Preprocessing, Feature Engineering, XGBoost ile RMSE ~0.120 CV.

## Kurulum
1. `pip install -r requirements.txt`
2. [Kaggle'dan dataset indir](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data), `data/` klasörüne koy.
3. `python main.py` çalıştır – submission.csv oluşur.

## Yapı
- `src/`: Modüler sınıflar (preprocessing, FE, model, evaluation).
- `main.py`: Pipeline zinciri.

## Sonuçlar
- CV RMSE: ~0.120 (top %20 Kaggle skoru).
- Feature importance plot: feature_importance.png.

## Kullanım
- VSCode'da aç, terminal'de çalıştır.
- Tweak: `feature_engineer.py`'de cat_features ekle.

Lisans: MIT 