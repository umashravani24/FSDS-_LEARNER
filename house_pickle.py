import streamlit as st
import pickle
import numpy as np

# Load the saved model
model = pickle.load(open(r'C:\Users\sss\Documents\GitHub\FSDS-_LEARNER\linear_regression_housemodel.pkl', 'rb'))

# Set the title of the Streamlit app
st.title("House Price Prediction App ")

# Add a brief description
st.write("This app predicts the price of a house based on its living space using a simple linear regression model.")

# Add input widget for user to enter living space
living_space = st.number_input("Enter Living Space (in sqft):", min_value=0.0, max_value=10000.0, value=1000.0, step=100.0)

# When the button is clicked, make predictions
if st.button("Predict Price"):
    # Make a prediction using the trained model
    living_space_input = np.array([[living_space]])  # Convert the input to a 2D array for prediction
    prediction = model.predict(living_space_input)

    # Display the result
    st.success(f"The predicted price for a house with {living_space} sqft is: ${prediction[0]:,.2f}")

# Display information about the model
st.write("The model was trained using a dataset of house prices and living space BY UMASHRAVANI.")