
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load Titanic dataset
df = pd.read_csv("dataset/cleaned_titanic.csv")

features = [
    "Pclass", "Sex", "Age", "SibSp",
    "Parch", "Fare", "Embarked_C",
    "Embarked_Q", "Embarked_S"
]

X = df[features]
y = df["Survived"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Test different max_depth values
depth_values = [1, 2, 3, 4, 5, 6, 7, 8, 10, None]

print("DECISION TREE MAX_DEPTH COMPARISON")
print("----------------------------------")

for depth in depth_values:
    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"Max Depth: {depth}, Accuracy: {accuracy * 100:.2f}%")

print("\nMAX_DEPTH TUNING COMPLETED!")
