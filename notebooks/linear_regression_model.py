
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load dataset
df = pd.read_csv("dataset/Housing.csv")

# Step 2: Convert categorical columns into numerical columns
df_encoded = pd.get_dummies(df, drop_first=True, dtype=int)

# Step 3: Separate input features and target
X = df_encoded.drop("price", axis=1)
y = df_encoded["price"]

# Step 4: Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Create and train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Predict house prices
y_pred = model.predict(X_test)

# Step 7: Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("LINEAR REGRESSION MODEL RESULTS")
print("--------------------------------")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Step 8: Plot actual vs predicted prices
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted House Prices")

# Reference line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.savefig("outputs/actual_vs_predicted.png")
plt.show()

# Step 9: Save trained model
joblib.dump(model, "models/house_price_model.pkl")

# Save feature names for future predictions
joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")

print("\nModel saved successfully!")
print("EDA and Linear Regression completed!")
