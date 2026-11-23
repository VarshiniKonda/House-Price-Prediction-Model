
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load dataset
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

# Decision Tree
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)

# KNN with feature scaling
knn_model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])

knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred)

# Compare results
print("CLASSIFICATION MODEL COMPARISON")
print("--------------------------------")
print("Decision Tree Accuracy:", round(dt_accuracy * 100, 2), "%")
print("Scaled KNN Accuracy:", round(knn_accuracy * 100, 2), "%")

if dt_accuracy > knn_accuracy:
    print("\nDecision Tree performed better.")
elif knn_accuracy > dt_accuracy:
    print("\nScaled KNN performed better.")
else:
    print("\nBoth models performed equally.")
