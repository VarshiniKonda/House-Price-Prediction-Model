
import matplotlib.pyplot as plt

models = ["Decision Tree", "Scaled KNN", "Random Forest"]
accuracies = [79.33, 81.56, 82.12]

plt.figure(figsize=(9, 5))

bars = plt.bar(models, accuracies)

plt.title("Titanic Classification Model Comparison")
plt.xlabel("Classification Models")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)

for bar, accuracy in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{accuracy}%",
        ha="center"
    )

plt.tight_layout()
plt.savefig("outputs/model_comparison.png")
plt.show()

print("Model comparison graph saved successfully!")
