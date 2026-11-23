
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Decision Tree
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)
dt_accuracy = accuracy_score(y_test, dt.predict(X_test))

# Scaled KNN
knn = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])
knn.fit(X_train, y_train)
knn_accuracy = accuracy_score(y_test, knn.predict(X_test))

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_accuracy = accuracy_score(y_test, rf.predict(X_test))

# Display comparison
results = pd.DataFrame({
    "Model": ["Decision Tree", "Scaled KNN", "Random Forest"],
    "Accuracy (%)": [
        round(dt_accuracy * 100, 2),
        round(knn_accuracy * 100, 2),
        round(rf_accuracy * 100, 2)
    ]
})

print("FINAL CLASSIFICATION MODEL COMPARISON")
print("--------------------------------------")
print(results.to_string(index=False))

best_model = results.loc[results["Accuracy (%)"].idxmax()]

print("\nBest Model:", best_model["Model"])
print("Best Accuracy:", best_model["Accuracy (%)"], "%")
