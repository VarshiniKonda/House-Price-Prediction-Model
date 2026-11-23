
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load dataset
df = pd.read_csv("dataset/cleaned_titanic.csv")

# Step 2: Select input features
features = [
    "Pclass", "Sex", "Age", "SibSp",
    "Parch", "Fare", "Embarked_C",
    "Embarked_Q", "Embarked_S"
]

X = df[features]
y = df["Survived"]

# Step 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 4: Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("TITANIC LOGISTIC REGRESSION RESULTS")
print("-----------------------------------")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Accuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100, "%")

print("\nCONFUSION MATRIX:")
print(cm)

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))

# Step 7: Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Titanic Confusion Matrix")
plt.savefig("outputs/titanic_confusion_matrix.png")
plt.show()

# Step 8: Save model
joblib.dump(model, "models/titanic_logistic_model.pkl")
joblib.dump(features, "models/titanic_feature_columns.pkl")

print("\nTitanic model saved successfully!")
