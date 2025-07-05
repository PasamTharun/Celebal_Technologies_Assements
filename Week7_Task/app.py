import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px

# Function to generate house data for visualization
def generate_house_data(n_samples=100):
    np.random.seed(42)
    sizes = np.random.normal(loc=1500, scale=500, size=n_samples)
    prices = 100 * sizes + np.random.normal(loc=0, scale=30000, size=n_samples)
    return pd.DataFrame({'size': sizes, 'price': prices})

# Load the pre-trained model
model = joblib.load('house_price_model.pkl')

def main():
    st.title('🏠 Simple House Pricing Predictor')
    st.write("This app predicts house prices based on size using a pre-trained linear regression model.")

    # Generate data for visualization
    data = generate_house_data()

    # User input
    size = st.number_input('Enter the size of the house (in sq ft)', min_value=500, max_value=5000, value=1500)

    if st.button('Predict Price'):
        prediction = model.predict([[size]])
        st.write(f'Predicted Price: ${prediction[0]:,.2f}')

    # Visualization
    fig = px.scatter(x=data['size'], y=data['price'], title='House Size vs Price',
                     labels={'x': 'Size (sq ft)', 'y': 'Price ($)'})
    data['predicted_price'] = model.predict(data[['size']])
    fig.add_scatter(x=data['size'], y=data['predicted_price'], mode='lines', name='Regression Line')
    st.plotly_chart(fig)

if __name__ == '__main__':
    main()