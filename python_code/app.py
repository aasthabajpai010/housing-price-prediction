# import streamlit as st

# import pandas as pd
# import numpy as np

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression

# from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import r2_score

# # -------------------------------------------------
# # PAGE TITLE
# # -------------------------------------------------

# st.title("House Price Prediction System")

# st.write("Machine Learning Project using Linear Regression")

# # -------------------------------------------------
# # LOAD DATASET
# # -------------------------------------------------

# data = pd.read_csv("Housing.csv")

# # Handle missing values

# data = data.fillna(method='ffill')

# # -------------------------------------------------
# # ENCODE CATEGORICAL DATA
# # -------------------------------------------------

# encoder = LabelEncoder()

# categorical_columns = [
#     'mainroad',
#     'guestroom',
#     'basement',
#     'hotwaterheating',
#     'airconditioning',
#     'prefarea',
#     'furnishingstatus'
# ]

# for column in categorical_columns:

#     data[column] = encoder.fit_transform(data[column])

# # -------------------------------------------------
# # FEATURES AND TARGET
# # -------------------------------------------------

# X = data.drop("price", axis=1)

# y = data["price"]

# # -------------------------------------------------
# # FEATURE SCALING
# # -------------------------------------------------

# scaler = StandardScaler()

# X_scaled = scaler.fit_transform(X)

# # -------------------------------------------------
# # SPLIT DATASET
# # -------------------------------------------------

# X_train, X_test, y_train, y_test = train_test_split(
#     X_scaled,
#     y,
#     test_size=0.20,
#     random_state=42
# )

# # -------------------------------------------------
# # TRAIN MODEL
# # -------------------------------------------------

# model = LinearRegression()

# model.fit(X_train, y_train)

# # -------------------------------------------------
# # SIDEBAR INPUTS
# # -------------------------------------------------

# st.sidebar.header("Enter House Details")

# area = st.sidebar.number_input("Area", value=5000)

# bedrooms = st.sidebar.number_input("Bedrooms", value=3)

# bathrooms = st.sidebar.number_input("Bathrooms", value=2)

# stories = st.sidebar.number_input("Stories", value=2)

# mainroad = st.sidebar.selectbox("Main Road", [0,1])

# guestroom = st.sidebar.selectbox("Guest Room", [0,1])

# basement = st.sidebar.selectbox("Basement", [0,1])

# hotwaterheating = st.sidebar.selectbox(
#     "Hot Water Heating",
#     [0,1]
# )

# airconditioning = st.sidebar.selectbox(
#     "Air Conditioning",
#     [0,1]
# )

# parking = st.sidebar.number_input("Parking", value=1)

# prefarea = st.sidebar.selectbox(
#     "Preferred Area",
#     [0,1]
# )

# furnishingstatus = st.sidebar.selectbox(
#     "Furnishing Status",
#     [0,1,2]
# )

# # -------------------------------------------------
# # CUSTOM INPUT DATA
# # -------------------------------------------------

# custom_data = [[
#     area,
#     bedrooms,
#     bathrooms,
#     stories,
#     mainroad,
#     guestroom,
#     basement,
#     hotwaterheating,
#     airconditioning,
#     parking,
#     prefarea,
#     furnishingstatus
# ]]

# # Scale custom data

# custom_scaled = scaler.transform(custom_data)

# # -------------------------------------------------
# # PREDICTION BUTTON
# # -------------------------------------------------

# if st.button("Predict House Price"):

#     prediction = model.predict(custom_scaled)

#     st.success(
#         f"Predicted House Price: ₹ {prediction[0]:,.2f}"
#     )

# # -------------------------------------------------
# # MODEL EVALUATION
# # -------------------------------------------------

# predictions = model.predict(X_test)

# mae = mean_absolute_error(y_test, predictions)

# mse = mean_squared_error(y_test, predictions)

# rmse = np.sqrt(mse)

# r2 = r2_score(y_test, predictions)

# st.subheader("Model Evaluation")

# st.write("MAE:", mae)

# st.write("MSE:", mse)

# st.write("RMSE:", rmse)

# st.write("R2 Score:", r2)

# # -------------------------------------------------
# # ACTUAL VS PREDICTED GRAPH
# # -------------------------------------------------

# chart_data = pd.DataFrame({
#     'Actual Prices': y_test.values[:50],
#     'Predicted Prices': predictions[:50]
# })

# st.subheader("Actual vs Predicted Prices")

# st.line_chart(chart_data)

# # -------------------------------------------------
# # SHOW DATASET
# # -------------------------------------------------

# if st.checkbox("Show Dataset"):

#     st.write(data.head())


import os

import streamlit as st

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="Dream Home Price Predictor",
    page_icon="🏡",
    layout="wide"
)

# -------------------------------------------------
# CUSTOM CSS STYLING
# -------------------------------------------------

st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7fa;
    }

    .title {
        font-size: 45px;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 10px;
    }

    .subtitle {
        font-size: 20px;
        color: #555555;
        text-align: center;
        margin-bottom: 30px;
    }

    .stButton>button {
        background-color: #1f77b4;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }

    .prediction-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 20px;
    }

    .metric-box {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0px 0px 8px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# PAGE TITLE
# -------------------------------------------------

st.markdown(
    '<div class="title">🏡 Dream Home Price Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI Powered Real Estate Price Prediction using Linear Regression</div>',
    unsafe_allow_html=True
)

st.image(
    "https://images.unsplash.com/photo-1560518883-ce09059eeffa?q=80&w=1200&auto=format&fit=crop",
    use_container_width=True
)

# -------------------------------------------------
# LOAD DATASET
# -------------------------------------------------
# NOTE: Streamlit Cloud runs the app from the repo root, not from
# python_code/, so a bare "Housing.csv" path breaks deployment.
# Build an absolute path relative to this file instead, and go up
# one directory into dataset/.

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..", "dataset", "Housing.csv")

data = pd.read_csv(CSV_PATH)

# Handle missing values
# (fillna(method="ffill") was removed in newer pandas versions;
# use .ffill() instead)

data = data.ffill()

# -------------------------------------------------
# ENCODE CATEGORICAL DATA
# -------------------------------------------------

encoder = LabelEncoder()

categorical_columns = [
    'mainroad',
    'guestroom',
    'basement',
    'hotwaterheating',
    'airconditioning',
    'prefarea',
    'furnishingstatus'
]

for column in categorical_columns:

    data[column] = encoder.fit_transform(data[column])

# -------------------------------------------------
# FEATURES AND TARGET
# -------------------------------------------------

X = data.drop("price", axis=1)

y = data["price"]

# -------------------------------------------------
# SCALING
# -------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# -------------------------------------------------
# SPLIT DATASET
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=42
)

# -------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------

model = LinearRegression()

model.fit(X_train, y_train)

# -------------------------------------------------
# SIDEBAR INPUTS
# -------------------------------------------------

st.sidebar.header("Enter House Details")

area = st.sidebar.number_input("Area", value=5000)

bedrooms = st.sidebar.number_input("Bedrooms", value=3)

bathrooms = st.sidebar.number_input("Bathrooms", value=2)

stories = st.sidebar.number_input("Stories", value=2)

mainroad = st.sidebar.selectbox("Main Road", [0, 1])

guestroom = st.sidebar.selectbox("Guest Room", [0, 1])

basement = st.sidebar.selectbox("Basement", [0, 1])

hotwaterheating = st.sidebar.selectbox("Hot Water Heating", [0, 1])

airconditioning = st.sidebar.selectbox("Air Conditioning", [0, 1])

parking = st.sidebar.number_input("Parking", value=1)

prefarea = st.sidebar.selectbox("Preferred Area", [0, 1])

furnishingstatus = st.sidebar.selectbox(
    "Furnishing Status",
    [0, 1, 2]
)

# -------------------------------------------------
# CUSTOM INPUT DATA
# -------------------------------------------------

custom_data = [[
    area,
    bedrooms,
    bathrooms,
    stories,
    mainroad,
    guestroom,
    basement,
    hotwaterheating,
    airconditioning,
    parking,
    prefarea,
    furnishingstatus
]]

# Scale custom data

custom_scaled = scaler.transform(custom_data)

# -------------------------------------------------
# PREDICTION BUTTON
# -------------------------------------------------

if st.button("Predict House Price"):

    prediction = model.predict(custom_scaled)

    price = prediction[0]
    lakh = price / 100000
    crore = price / 10000000

    st.success(
        f"Predicted House Price: ₹ {price:,.2f}"
    )
    st.info(
        f"Approx Price: {lakh:.2f} Lakh"
    )
    st.info(
        f"Approx Price: {crore:.2f} Crore"
    )

# -------------------------------------------------
# MODEL EVALUATION
# -------------------------------------------------

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(y_test, predictions)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, predictions)

st.subheader("Model Evaluation")

st.write("MAE:", mae)

st.write("MSE:", mse)

st.write("RMSE:", rmse)

st.write("R2 Score:", r2)

# -------------------------------------------------
# ACTUAL VS PREDICTED GRAPH
# -------------------------------------------------

chart_data = pd.DataFrame({
    'Actual Prices': y_test.values[:50],
    'Predicted Prices': predictions[:50]
})

st.subheader("Actual vs Predicted Prices")

st.line_chart(chart_data)

# -------------------------------------------------
# SHOW DATASET
# -------------------------------------------------

if st.checkbox("Show Dataset"):

    st.write(data.head())