
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("dataset/Housing.csv")

# Basic statistics
print("DATASET STATISTICS:")
print(df.describe())

# Average house price
print("\nAVERAGE HOUSE PRICE:")
print(df["price"].mean())

# Average area
print("\nAVERAGE HOUSE AREA:")
print(df["area"].mean())

# House price distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["price"], kde=True)
plt.title("House Price Distribution")
plt.xlabel("House Price")
plt.ylabel("Frequency")
plt.savefig("outputs/house_price_distribution.png")
plt.show()

# Area vs Price
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="area", y="price")
plt.title("Area vs House Price")
plt.xlabel("Area")
plt.ylabel("Price")
plt.savefig("outputs/area_vs_price.png")
plt.show()

print("\nEDA COMPLETED SUCCESSFULLY!")
