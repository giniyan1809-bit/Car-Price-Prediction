import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Car Price Prediction", page_icon="🚗")

st.title("🚗 Car Price Prediction")
st.write
(
    "Predict the estimated market price of a used car "
    "using Machine Learning."
)

df = pd.read_csv("car_data.csv")

df = df.dropna()

encoders = {}
for col in ["Car_Name", "Fuel_Type", "Transmission"]:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col].astype(str))

X = df
[[
    "Car_Name",
    "Year",
    "Present_Price",
    "Kms_Driven",
    "Fuel_Type",
    "Transmission",
    "Owner"
]]
y = df["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

st.subheader("Enter Car Details")

car_name = st.selectbox(
    "Car Name",
    encoders["Car_Name"].classes_
)

year = st.number_input(
    "Manufacturing Year",
    min_value=2000,
    max_value=2026,
    value=2020
)

present_price = st.number_input(
    "Current Showroom Price (Lakhs)",
    min_value=0.1,
    value=8.0
)

kms_driven = st.number_input(
    "Kilometers Driven",
    min_value=0,
    value=30000
)

fuel_type = st.selectbox(
    "Fuel Type",
    encoders["Fuel_Type"].classes_
)

transmission = st.selectbox(
    "Transmission",
    encoders["Transmission"].classes_
)

owner = st.selectbox("Previous Owners", [0, 1, 2, 3])

if st.button("Predict Car Price"):
    input_data = pd.DataFrame([{
        "Car_Name": encoders["Car_Name"].transform([car_name])[0],
        "Year": year,
        "Present_Price": present_price,
        "Kms_Driven": kms_driven,
        "Fuel_Type": encoders["Fuel_Type"].transform([fuel_type])[0],
        "Transmission": encoders["Transmission"].transform(
            [transmission]
        )[0],
        "Owner": owner
    }])

    prediction = model.predict(input_data)[0]

    st.success(
        f"Estimated Selling Price: ₹{prediction:.2f} Lakhs"
    )

    st.caption(
        "This is a model-based estimate, not a guaranteed market price."
    )