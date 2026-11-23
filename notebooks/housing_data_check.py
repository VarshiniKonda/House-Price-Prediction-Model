
import pandas as pd

# Load the housing dataset
df = pd.read_csv("dataset/Housing.csv")

# Display first 5 rows
print("FIRST 5 ROWS:")
print(df.head())

# Display dataset information
print("\nDATASET INFORMATION:")
print(df.info())

# Check missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())

# Display number of rows and columns
print("\nDATASET SHAPE:")
print(df.shape)
