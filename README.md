# mlops-project
# Crop Yield Prediction using Machine Learning

## About the Project

This project is about predicting crop yield using machine learning.

The model takes information about the crop, soil, weather conditions,
fertilizer, irrigation and other farming factors and predicts the
expected crop yield in tons per hectare.

## Problem Statement

Crop yield depends on many different factors such as rainfall, soil
conditions, temperature, fertilizer usage, irrigation and the type of
crop.

The main aim of this project is to predict the crop yield in tons per
hectare using different agricultural, soil, weather, and farming-related
factors.

## Dataset

The dataset used in this project is `crop_yield_dataset.csv`.

It contains 10,000 rows and 13 columns.

### Features used for prediction

- Crop
- Region
- Soil_Type
- Soil_pH
- Rainfall_mm
- Temperature_C
- Humidity_pct
- Fertilizer_Used_kg
- Irrigation
- Pesticides_Used_kg
- Planting_Density
- Previous_Crop

### Output

The model predicts:

`Yield_ton_per_ha`

This is the expected crop yield in tons per hectare.

## Models Used

I compared three machine learning models:

1. Linear Regression
2. Random Forest
3. Gradient Boosting

The models were compared using:

- MAE
- RMSE
- R² Score

## Results
| Model | MAE | RMSE | R² |
| Linear Regression | 4.0765 | 5.0806 | 0.9821 |
| Random Forest | 4.3150 | 5.3678 | 0.9800 |
| Gradient Boosting | 4.1972 | 5.2041 | 0.9812 |

Linear Regression gave the best result with an R² score of **0.9821**.

The selected model is saved in:

`models/crop_yield_model.pkl`

