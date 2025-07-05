import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

def generate_house_data(n_samples=100):
    np.random.seed(42)
    sizes = np.random.normal(loc=1500, scale=500, size=n_samples)
    prices = 100 * sizes + np.random.normal(loc=0, scale=30000, size=n_samples)
    return pd.DataFrame({'size': sizes, 'price': prices})

# Generate data and train model
data = generate_house_data()
X = data[['size']]
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'house_price_model.pkl')