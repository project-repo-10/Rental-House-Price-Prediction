# Sri Lanka House Rental Price Prediction

This application predicts monthly rental prices of houses in various districts across Sri Lanka. The predictions are based on a machine learning model trained on historical data, taking into account key features such as the number of bathrooms, bedrooms, house size, and land size.

## Key Features

- **District**: Select from the available districts in Sri Lanka.
- **Number of Beds**: Input the number of bedrooms.
- **Number of Baths**: Input the number of bathrooms.
- **House Size**: Input the size of the house in square feet.
- **Land Size**: Input the land size in perches.

The machine learning model estimates the rental price using the above features, providing an accurate and data-driven prediction.

## Files in the Repository

- **`app.py`**: This is the main file that contains the Streamlit app code.
- **`best_model.pkl`**: This is the trained machine learning model used for predicting rental prices.
- **`Monthly_house_rental_prices_in_Sri_Lanka.csv`**: The dataset used to train the model, containing historical house rental data for different districts.
- **`requirements.txt`**: A list of dependencies required to run the application.
