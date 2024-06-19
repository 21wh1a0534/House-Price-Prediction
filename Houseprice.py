import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Load the new dataset
df = pd.read_csv("data (2).csv")
df['bedrooms'] = df['bedrooms'].round().astype(int)
df['bathrooms'] = df['bathrooms'].round().astype(int)
df['floors'] = df['floors'].round().astype(int)
# Define the target variable (house price) and features
target = 'price'
features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view',
            'condition', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated']
df
df.head()
df.tail()
df.info()
# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
X
y
train_data = X_train.join(y_train)
train_data
train_data.hist(figsize = (15, 10))
#correlation with each column
train_data.corr()
#heatmap on correalation and annot to see the numbers
plt.figure(figsize =  (15, 8))
sns.heatmap(train_data.corr(), annot=True, cmap= 'Blues')
plt.figure(figsize=(15, 8))
sns.scatterplot(x='sqft_living', y='price', data=df, palette='coolwarm')
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Train a Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=123)
rf.fit(X_train, y_train)
# Make predictions
y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)
# Evaluate the model
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
print('Training MAE: ', mae_train)
print('Testing MAE: ', mae_test)
print('Training RMSE: ', rmse_train)
print('Testing RMSE: ', rmse_test)
# Define a function for predicting house prices
def predict_house_price(input_data):
    input_df = pd.DataFrame(input_data, columns=features)
    input_df = scaler.transform(input_df)
    predicted_price = rf.predict(input_df)
    return predicted_price[0]
# Example input data
input_data = {
    'bedrooms': [3],
    'bathrooms': [2],
    'sqft_living': [2000],
    'sqft_lot': [8030],
    'floors': [1],
    'waterfront': [0],
    'view': [0],
    'condition': [4],
    'sqft_above': [1000],
    'sqft_basement': [1000],
    'yr_built': [1963],
    'yr_renovated': [0]
}
# Make predictions with the model
predicted_price = predict_house_price(input_data)
# Print the predicted price
print("Predicted house price: ", predicted_price)