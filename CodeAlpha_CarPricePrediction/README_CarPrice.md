# Car Price Prediction with Machine Learning

## Overview
This project builds a machine learning model to predict the selling price of used cars based on features like present price, mileage driven, fuel type, transmission, and age of the car. Built as part of the **CodeAlpha Data Science Internship**.

## Dataset
- Source: [Car Price Prediction (Used Cars) on Kaggle](https://www.kaggle.com/datasets/vijayaadithyanvg/car-price-predictionused-cars)
- ~300 samples with features: `Car_Name`, `Year`, `Selling_Price` (target), `Present_Price`, `Driven_kms`, `Fuel_Type`, `Selling_type`, `Transmission`, `Owner`
- Loaded via `kagglehub`

## Steps
1. Loaded the dataset using `kagglehub`
2. Performed exploratory data analysis:
   - Scatter plot of Present Price vs Selling Price (strong positive relationship)
   - Boxplot of Selling Price by Fuel Type (Diesel cars priced higher than Petrol/CNG)
   - Correlation heatmap of numeric features
3. Feature engineering:
   - Created `Car_Age` from `Year` (2026 - Year)
   - Dropped `Car_Name` (98 unique values — too high-cardinality for one-hot encoding on this dataset size)
4. Encoded categorical features (`Fuel_Type`, `Selling_type`, `Transmission`) using one-hot encoding
5. Split data into training (80%) and test (20%) sets
6. Trained two regression models: **Linear Regression** and **Random Forest Regressor**
7. Evaluated both models using MAE, RMSE, and R²
8. Visualized actual vs predicted prices, and feature importance for the best model

## Results

| Metric | Linear Regression | Random Forest |
|--------|-------------------|----------------|
| MAE    | 1.216             | **0.637**      |
| RMSE   | 1.866             | **0.966**      |
| R²     | 0.849             | **0.959**      |

**Random Forest Regressor** was selected as the final model — it explains ~96% of the variance in selling price and predicts within ~0.64 lakh of the actual price on average, outperforming Linear Regression by capturing non-linear relationships (e.g., depreciation curve with age, fuel-type price premiums) that a linear model can't.

### Key insight from correlation analysis
`Present_Price` was the strongest predictor of `Selling_Price` (correlation 0.88), followed by `Car_Age`. `Driven_kms` had minimal direct correlation with price but was correlated with car age.

## Tech Stack
- Python (Google Colab)
- Libraries: `pandas`, `scikit-learn`, `seaborn`, `matplotlib`, `kagglehub`

## Project Structure
```
CodeAlpha_CarPricePrediction/
├── Car_Price_Prediction.ipynb
├── README.md
├── requirements.txt
└── images/
    ├── present_price_vs_selling_price.png
    ├── selling_price_by_fuel_type.png
    ├── correlation_heatmap.png
    ├── actual_vs_predicted.png
    └── feature_importance.png
```

## How to Run
1. Open the notebook in Google Colab
2. Run all cells in order (`Runtime → Run all`)
3. The dataset downloads automatically via `kagglehub` — no manual download needed

## Key Learnings
- Practiced the full regression ML workflow: EDA, feature engineering, encoding, model training, and evaluation
- Learned to compare regression models using MAE, RMSE, and R² instead of accuracy/confusion matrix (which apply to classification, not regression)
- Understood the importance of handling high-cardinality categorical features (like `Car_Name`) rather than blindly one-hot encoding everything
- Learned to interpret feature importance from a trained ensemble model versus simple correlation

## Author
Created as part of the **CodeAlpha Data Science Internship** (Task 3: Car Price Prediction)
