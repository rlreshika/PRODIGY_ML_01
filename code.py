import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
train_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

# Select features and target variable
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
target = 'SalePrice'

# Handle missing values (if any) directly on the DataFrame
train_df[features] = train_df[features].fillna(train_df[features].mean())

# Split data into features (X) and target variable (y)
X = train_df[features]
y = train_df[target]

# Initialize the linear regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X, y)

# Predict on training data
train_predictions = model.predict(X)

# Calculate Mean Squared Error
mse = mean_squared_error(y, train_predictions)
print(f"Mean Squared Error: {mse}")

# Plotting the scatter plot
plt.figure(figsize=(10, 6))

# Scatter plot of actual vs predicted prices
plt.scatter(y, train_predictions, color='blue', alpha=0.5)
plt.title('Actual vs Predicted Prices (Linear Regression)')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')

# Overlay the regression line using sns.regplot
sns.regplot(x=y, y=train_predictions, scatter=False, color='red', line_kws={'linewidth': 2})

plt.show()

