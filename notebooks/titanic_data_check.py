
import pandas as pd

# Load Titanic dataset
df = pd.read_csv("dataset/cleaned_titanic.csv")

print("FIRST 5 ROWS:")
print(df.head())

print("\nCOLUMN NAMES:")
print(df.columns.tolist())

print("\nDATASET INFORMATION:")
df.info()

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nDATASET SHAPE:")
print(df.shape)

print("\nTARGET COLUMN CHECK:")
if "Survived" in df.columns:
    print("Survived column found!")
else:
    print("Survived column not found!")
