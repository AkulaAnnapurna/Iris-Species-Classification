from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
CHARTS = OUT / "charts"
MODEL_DIR = OUT / "model"
for folder in (OUT, CHARTS, MODEL_DIR):
    folder.mkdir(parents=True, exist_ok=True)


def main():
    sns.set_theme(style="whitegrid")

    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    feature_names = list(iris.feature_names)
    target_names = list(iris.target_names)
    df["species"] = df["target"].map(dict(enumerate(target_names)))

    print("\\n=== IRIS DATASET OVERVIEW ===")
    print("Shape:", df.shape)
    print("\\nFirst five rows:\\n", df.head())
    print("\\nMissing values:\\n", df.isnull().sum())
    print("\\nSpecies counts:\\n", df["species"].value_counts())
    df.to_csv(OUT / "iris_dataset.csv", index=False)

    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="species", hue="species", legend=False)
    plt.title("Number of Samples per Iris Species")
    plt.xlabel("Species")
    plt.ylabel("Number of samples")
    plt.tight_layout()
    plt.savefig(CHARTS / "species_distribution.png", dpi=300)
    plt.close()

    pair = sns.pairplot(df, vars=feature_names, hue="species", diag_kind="hist", corner=True)
    pair.fig.suptitle("Iris Feature Relationships by Species", y=1.02)
    pair.savefig(CHARTS / "feature_pairplot.png", dpi=250)
    plt.close("all")

    X = df[feature_names]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    results = []
    fitted = {}
    print("\\n=== MODEL ACCURACY ===")
    for name, model in models.items():
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        score = accuracy_score(y_test, prediction)
        results.append({"Model": name, "Accuracy": score})
        fitted[name] = model
        print(f"{name}: {score:.3f} ({score * 100:.1f}%)")

    result_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
    result_df.to_csv(OUT / "model_comparison.csv", index=False)
    best_name = result_df.iloc[0]["Model"]
    best_model = fitted[best_name]
    best_prediction = best_model.predict(X_test)

    report = classification_report(y_test, best_prediction, target_names=target_names)
    print(f"\\nBest model: {best_name}\\n\\n{report}")
    (OUT / "classification_report.txt").write_text(
        f"Best model: {best_name}\\n\\n{report}", encoding="utf-8"
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(data=result_df, x="Model", y="Accuracy", hue="Model", legend=False)
    plt.ylim(0, 1.05)
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.xlabel("Model")
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(CHARTS / "model_accuracy_comparison.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, best_prediction, display_labels=target_names,
        cmap="Blues", colorbar=False, ax=ax
    )
    ax.set_title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig(CHARTS / "confusion_matrix.png", dpi=300)
    plt.close()

    joblib.dump({
        "model": best_model,
        "feature_names": feature_names,
        "target_names": target_names,
        "model_name": best_name,
    }, MODEL_DIR / "best_iris_model.joblib")

    sample = X_test.iloc[[0]]
    predicted = int(best_model.predict(sample)[0])
    actual = int(y_test.iloc[0])
    print("\\n=== SAMPLE PREDICTION ===")
    print("Predicted species:", target_names[predicted])
    print("Actual species:", target_names[actual])
    print("\\nProject completed. Check the outputs folder.")


if __name__ == "__main__":
    main()
