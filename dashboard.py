import streamlit as st
import pandas as pd
import joblib

# Load pre-trained KNN model and preprocessing pipeline
knn_model = joblib.load('trained_knn_model.pkl')
encoder = joblib.load('target_encoder.pkl')

# Load data
data = pd.read_csv("C:/Users/aminah/OneDrive/Desktop/final project @UTeM/test_train_files/actual_data.csv")
data['BizDate'] = pd.to_datetime(data['BizDate'])  # Ensure BizDate is in datetime format

# Streamlit UI
st.title("Dynamic Pricing Dashboard")

# 1. Date Selection
selected_date = st.date_input("Select Date")
st.write(f"Selected Date: {selected_date}")

# Filter data by date
filtered_data = data[data['BizDate'] == pd.Timestamp(selected_date)]

if filtered_data.empty:
    st.warning("No data available for the selected date.")
else:
    # 2. Branch Selection
    branches = filtered_data['Loc_group'].unique()
    selected_branch = st.selectbox("Select Branch", branches)

    # 3. Product Type Selection
    product_types = filtered_data['SubDept'].unique()
    selected_type = st.selectbox("Select Product Type", product_types)

    # 4. Item Selection
    filtered_items = filtered_data[filtered_data['SubDept'] == selected_type]['Description'].unique()
    selected_item = st.selectbox("Select Item", filtered_items)

    # Filter the selected item data
    item_data = filtered_data[
        (filtered_data['Loc_group'] == selected_branch) &
        (filtered_data['SubDept'] == selected_type) &
        (filtered_data['Description'] == selected_item)
    ]

    if not item_data.empty:
        # Display Unit Price
        unit_price = item_data['UnitPrice'].iloc[0]
        st.write(f"Unit Price: {unit_price:.2f} MYR")

        # Prepare features for prediction
        features = item_data[["Qty", "DayOfWeek", "HolidayPeriod", "IsWeekend", "Loc_group", "UnitPrice", "Dept", "SubDept", "Category"]]
        
        # Apply preprocessing using encoder
        features_preprocessed = encoder.transform(features)

        # Predict using the trained model
        prediction = knn_model.predict(features_preprocessed)
        predicted_discount = prediction[0]

        # Calculate Predicted Price
        predicted_price = unit_price * (1 - predicted_discount / 100)
        st.write(f"Predicted Discount: {predicted_discount:.2f}%")
        st.write(f"Predicted Price: {predicted_price:.2f} MYR")
    else:
        st.warning("No data available for the selected item.")