# # 27

# שלב 6: בניית ממשק Streamlit
# משימות לביצוע:
# 1. ליצור טופס קלט (בהתאם למאפיינים שבחרתם במודל)
# 2. להמיר את הקלט לפורמט שהמודל מבין
# 3. להשתמש במודל ולחזות את המחיר
# 4. להציג את המחיר בצורה ברורה

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
model = joblib.load("C:/Users/Danel Wittner/Desktop/ai/airbnb_new/model/price_trained_model.pkl")

st.title(" Airbnb Property Price Prediction")
st.write("Enter property details to get the estimated price per night.")

# --- Input Form ---
with st.form("prediction_form"):
    host_is_superhost = st.selectbox("Host is Superhost", ["f", "t"])
    host_listings_count = st.number_input("Host Listings Count", min_value=1, max_value=100, value=1)
    availability_30 = st.number_input("Availability in next 30 days", min_value=0, max_value=30, value=10)
    review_scores_rating = st.slider("Review Score Rating", 0.0, 100.0, 80.0)
    review_scores_location = st.slider("Review Score Location", 0.0, 100.0, 80.0)
    calculated_host_listings_count_private_rooms = st.number_input("Calculated Host Listings Count Private Rooms", min_value=0, max_value=100, value=0)
    bedrooms = st.slider("Number of bedrooms", 0, 10, 1)
    bathrooms = st.slider("Number of bathrooms", 0.0, 5.0, 1.0)
    room_type = st.selectbox("Room type", ["Entire home/apt", "Private room", "Shared room"])

    submitted = st.form_submit_button("Predict Price")

# --- Process input and predict ---
# our df includes:
# 'bedrooms', 'bathrooms', 'price', 'host_listings_count',
#        'availability_30', 'review_scores_rating', 'review_scores_location',
#        'calculated_host_listings_count_private_rooms', 'room_type_Hotel room',
#        'room_type_Private room', 'room_type_Shared room',
#        'host_is_superhost_t'

if submitted:
    try:
        # Create a DataFrame from user input
        input_data = pd.DataFrame({
            "bedrooms": [bedrooms],
            "bathrooms": [bathrooms],
            "room_type": [room_type],
            "host_is_superhost": [host_is_superhost],
            "host_listings_count": [host_listings_count],
            "availability_30": [availability_30],
            "review_scores_rating": [review_scores_rating],
            "review_scores_location": [review_scores_location],
            "calculated_host_listings_count_private_rooms": [calculated_host_listings_count_private_rooms]
        })

        # מהמיר את הקלט לפורמט שהמודל מבין
        input_encoded = pd.get_dummies(input_data)

        # טוען את נתוני האימון כדי לוודא שלקלט החדש יש את אותן העמודות, באותו הסדר
        X_train = pd.read_csv("C:/Users/Danel Wittner/Desktop/ai/airbnb_new/notebooks/X_train.csv")
        missing_cols = set(X_train.columns) - set(input_encoded.columns)
        for col in missing_cols:
            input_encoded[col] = 0
        input_encoded = input_encoded[X_train.columns]

        # משתמש במודל ומחזיר את המחיר
        prediction = model.predict(input_encoded)[0]
        st.success(f"Estimated price: ${prediction:.2f} per night")

        # --- Plot: Price distribution ---
        st.subheader("Price Distribution in Training Data")
        y_train = pd.read_csv("C:/Users/Danel Wittner/Desktop/ai/airbnb_new/notebooks/y_train.csv")
        fig, ax = plt.subplots()
        sns.histplot(y_train, bins=30, kde=True, ax=ax)
        ax.axvline(prediction, color='red', linestyle='--', label='Your prediction')
        ax.set_title("Price Distribution")
        ax.set_xlabel("Price")
        ax.legend()
        st.pyplot(fig) 

    except Exception as e:
        st.error(f"Error during prediction: {e}")
