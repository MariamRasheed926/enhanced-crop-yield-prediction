# Enhanced Crop Yield Prediction

This project presents a machine learning model designed to predict crop yield based on historical crop and year data.

The system uses a Random Forest Regressor to model the relationship between crop type and yield over time, providing a data-driven approach to crop productivity analysis.

## Dataset

The dataset contains historical crop yield records with information about the year, crop type, and crop yield measured in hectograms per hectare (hg/ha).

## Model

The project uses a Random Forest Regressor to predict crop yield.

The input features include:

* Year
* Crop Type

Crop Type is one-hot encoded before being used by the model.

The preprocessing pipeline includes categorical encoding, feature standardization, and a train/test split for model evaluation.

## Performance

The model achieved a Mean Squared Error (MSE) of approximately **2.41 × 10⁹** on the available dataset.

The performance can be further improved by incorporating additional agricultural and environmental features.

## Development Tools

Python
Scikit-learn
Pandas
NumPy
* Random Forest
* Machine Learning
