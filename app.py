import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load('best_model.pkl')

# Load known districts from the dataset
file_path = 'Monthly_house_rental_prices_in_Sri_Lanka.csv'
df = pd.read_csv(file_path)
known_districts = df['District'].unique().tolist()

st.title("House Rental Price Prediction in Sri Lanka")

st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode", ["Home", "About"])

if app_mode == "About":
    st.subheader("About this App")
    st.write("""
    This application predicts the monthly rental prices of houses in various districts in Sri Lanka. 
    It uses a machine learning model trained on historical rental data to estimate prices based on 
    factors like the number of baths, number of beds, land size, house size, and district.
    """)
else:

    st.subheader("Enter House Details for Price Prediction")

    # Inputs for the user
    district = st.selectbox("Select District", sorted(known_districts))
    baths = st.number_input("Number of Baths", min_value=1, max_value=10, value=1)
    beds = st.number_input("Number of Beds", min_value=1, max_value=10, value=1)
    house_size = st.number_input("House Size (Square Feet)", min_value=500.0, value=1000.0)
    land_size = st.number_input("Land Size (Perches)", min_value=1.0, value=10.0)
    
    if st.button("Predict"):
        try:
            # Normalize the district name
            normalized_district = district.title()

            # Create a DataFrame for the new input
            new_data = pd.DataFrame([[baths, beds, land_size, house_size, normalized_district]],
                                    columns=['Baths', 'Beds', 'Land size', 'House size', 'District'])

            # Make the prediction
            predicted_price = model.predict(new_data)

            # Display the result
            st.success(f"The predicted monthly rental price is Rs: {predicted_price[0]:,.2f}")
        except ValueError as e:
            st.error(f"Error: {e}")

# Footer
st.sidebar.markdown("""
**Developed by:**
- Gihan Pramod
- Janaka Nupehewage
- Kavishka Jayasinghe
- Pawan Perera
- Pasindu Malshan
""")
