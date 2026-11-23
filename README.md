# Iris Species Classification

A beginner-friendly machine learning project that classifies Iris flowers into Setosa, Versicolor, and Virginica.

## What this project does
- Loads the built-in Iris dataset from scikit-learn.
- Explores the data and checks missing values.
- Creates species-distribution and feature-relationship visualizations.
- Trains Logistic Regression, Decision Tree, and Random Forest classifiers.
- Compares accuracy and evaluates the best model.
- Saves a confusion matrix, classification report, and trained model.

## Project structure
```text
Iris-Species-Classification/
├── data/
├── outputs/
│   ├── charts/
│   └── model/
├── src/
│   └── iris_classification.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Run in VS Code on Windows

Open this folder in VS Code and open Terminal.

1. Create an environment:
```powershell
python -m venv .venv
```

2. Activate it:
```powershell
.\.venv\Scripts\Activate.ps1
```
If PowerShell blocks activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```
Then activate again.

3. Install libraries:
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

4. Run:
```powershell
python src/iris_classification.py
```

## Generated outputs
- `outputs/iris_dataset.csv`
- `outputs/model_comparison.csv`
- `outputs/classification_report.txt`
- `outputs/charts/species_distribution.png`
- `outputs/charts/feature_pairplot.png`
- `outputs/charts/model_accuracy_comparison.png`
- `outputs/charts/confusion_matrix.png`
- `outputs/model/best_iris_model.joblib`

## Dataset
The script uses the built-in Iris dataset included with scikit-learn. It has 150 samples, four numeric measurements, and three species classes.
