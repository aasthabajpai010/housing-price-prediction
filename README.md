#  Housing Price Prediction System

## Project Overview

This project predicts house prices using Machine Learning and Linear Regression. It provides an interactive Streamlit web application where users can enter house details and get an estimated property price instantly.

The project includes data preprocessing, feature encoding, feature scaling, model training, evaluation, and visualization.

---

## Features

* Predict house prices using Linear Regression
* Interactive Streamlit user interface
* Data preprocessing and cleaning
* Categorical feature encoding
* Feature scaling using StandardScaler
* Model performance evaluation
* Actual vs Predicted price visualization
* Dataset preview option

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Streamlit
* Matplotlib

---

## Project Structure

```text
AI_project_Linear-regression/

├── dataset/
├── python_code/
│   ├── app.py
│   ├── preprocess.py
│   ├── train_model.py
│   ├── predict_prices.py
│   ├── evaluate_model.py
│   ├── visualize.py
│   └── graph.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Dataset

The project uses a housing dataset containing information such as:

* Area
* Bedrooms
* Bathrooms
* Stories
* Parking
* Main Road Access
* Guest Room
* Basement
* Air Conditioning
* Preferred Area
* Furnishing Status

Target Variable:

* House Price

---

## Machine Learning Workflow

1. Load Housing Dataset
2. Handle Missing Values
3. Encode Categorical Features
4. Scale Features
5. Split Dataset into Training and Testing Sets
6. Train Linear Regression Model
7. Evaluate Model Performance
8. Predict House Prices
9. Visualize Results

---

## Model Evaluation Metrics

The model is evaluated using:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

---

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
streamlit run python_code/app.py
```

---

## User Inputs

The application accepts:

* Area
* Bedrooms
* Bathrooms
* Stories
* Main Road Access
* Guest Room
* Basement
* Hot Water Heating
* Air Conditioning
* Parking
* Preferred Area
* Furnishing Status

---

## Output

The application displays:

* Predicted House Price
* Price in Lakhs
* Price in Crores
* Model Evaluation Metrics
* Actual vs Predicted Price Graph

---

## Future Improvements

* Support multiple machine learning algorithms
* Add model comparison dashboard
* Deploy application online
* Add advanced feature engineering
* Improve prediction accuracy

---

## Author

**Aastha Bajpai**
MCA Student, NIT Jamshedpur
