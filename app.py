from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__,template_folder='templates',static_folder='static')

# Load the dataset and preprocess it
# (You can place this code outside of the Flask app if you want to load the dataset only once)
df = pd.read_csv("data (2).csv")
df['bathrooms'] = df['bathrooms'].round().astype(int)

# Define the target variable (house price) and features
target = 'price'
features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view',
            'condition', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated']

# Split the data into training and testing sets
X = df[features]
y = df[target]
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train a Linear Regression model
lr = LinearRegression()
lr.fit(X, y)

# Train a Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=123)
rf.fit(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'submit' in request.form:
            # Get input values from the form
            input_data = {
                'bedrooms': int(request.form['bedrooms']),
                'bathrooms': int(request.form['bathrooms']),
                'sqft_living': int(request.form['sqft_living']),
                'sqft_lot': int(request.form['sqft_lot']),
                'floors': int(request.form['floors']),
                'waterfront': int(request.form['waterfront']),
                'view': int(request.form['view']),
                'condition': int(request.form['condition']),
                'sqft_above': int(request.form['sqft_above']),
                'sqft_basement': int(request.form['sqft_basement']),
                'yr_built': int(request.form['yr_built']),
                'yr_renovated': int(request.form['yr_renovated'])
            }

            # Create a DataFrame from the input data
            input_df = pd.DataFrame([input_data])

            # Preprocess the input data
            input_df = scaler.transform(input_df)

            # Predict house price using Random Forest
            predicted_price = rf.predict(input_df)

            return render_template('result.html', predicted_price=predicted_price[0])

if __name__ == '__main__':
    app.run(debug=True)