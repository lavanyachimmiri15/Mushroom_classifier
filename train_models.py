"""Train and evaluate five classifiers on the UCI Mushroom dataset.

Each model is a scikit-learn Pipeline (one-hot encoder + classifier). Its main
hyperparameter is tuned with 5-fold stratified cross-validation on the training
split only. The tuned model is then evaluated once on the untouched 20% test split.

Outputs:
    test_data.csv           20% stratified hold-out split (features + target)
    model/*.joblib          fitted scikit-learn pipelines (one per model)
    model/metrics.csv       test-set metrics, tuned params, and CV F1 per model
    model/metadata.json     feature list, target info, model files, sklearn version

Usage:
    python train_models.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "agaricus-lepiota.csv"
TEST_FILE = BASE_DIR / "test_data.csv"
MODEL_DIR = BASE_DIR / "model"
TARGET = "output"
POSITIVE = "p"  # poisonous is treated as the positive class

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {DATA_FILE.name}: {df.shape[0]} rows, {df.shape[1]} columns")
    print("Class balance:", df[TARGET].value_counts().to_dict())
    missing = (df == "?").sum().loc[lambda s: s > 0]
    print("'?' values:", missing.to_dict())
    print("Duplicate rows:", int(df.duplicated().sum()))
    return df


def preprocess(df: pd.DataFrame):
    # veil-type has a single value in every row -> no information.
    constant_cols = [c for c in df.columns if df[c].nunique() == 1]
    print("Dropping constant columns:", constant_cols)
    df = df.drop(columns=constant_cols)
    # '?' in stalk-root is kept as its own category (missingness may carry signal).
    features = [c for c in df.columns if c != TARGET]
    X = df[features]
    y = (df[TARGET] == POSITIVE).astype(int)
    return X, y, features


def make_pipeline(estimator) -> Pipeline:
    return Pipeline([
        ("encode", OneHotEncoder(handle_unknown="ignore")),
        ("clf", estimator),
    ])


def build_searches() -> dict:
    """Return (pipeline, param_grid) per model. Grids target each model's key knob."""
    return {
        "Logistic Regression": (
            make_pipeline(LogisticRegression(max_iter=1000, random_state=SEED)),
            {"clf__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "Decision Tree": (
            make_pipeline(DecisionTreeClassifier(random_state=SEED)),
            {"clf__max_depth": [3, 4, 5, 6, 7, 8, None]},
        ),
        "KNN": (
            make_pipeline(KNeighborsClassifier()),
            {"clf__n_neighbors": [3, 5, 7, 9, 11, 15]},
        ),
        # MultinomialNB suits non-negative one-hot inputs better than GaussianNB.
        "Naive Bayes": (
            make_pipeline(MultinomialNB()),
            {"clf__alpha": [0.01, 0.1, 0.5, 1.0]},
        ),
        # No probability=True (deprecated in sklearn 1.9); AUC uses decision_function.
        "Support Vector Machine": (
            make_pipeline(SVC(kernel="rbf", random_state=SEED)),
            {"clf__C": [0.1, 1.0, 10.0]},
        ),
    }


def positive_scores(model, X) -> np.ndarray:
    """Continuous scores for the positive class, used for AUC."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def evaluate(model, X, y) -> dict:
    pred = model.predict(X)
    return {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, positive_scores(model, X)),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }


def main() -> None:
    print("scikit-learn", sklearn.__version__)
    MODEL_DIR.mkdir(exist_ok=True)

    X, y, features = preprocess(load_data())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")

    test_df = X_test.copy()
    test_df[TARGET] = np.where(y_test == 1, "p", "e")
    test_df.to_csv(TEST_FILE, index=False)
    print(f"Saved {TEST_FILE.name}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results, tuning = {}, {}
    for name, (pipeline, grid) in build_searches().items():
        search = GridSearchCV(pipeline, grid, cv=cv, scoring="f1", n_jobs=-1)
        search.fit(X_train, y_train)
        model = search.best_estimator_
        best = search.best_index_
        cv_mean = search.cv_results_["mean_test_score"][best]
        cv_std = search.cv_results_["std_test_score"][best]
        params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}

        results[name] = evaluate(model, X_test, y_test)
        tuning[name] = {"Best params": json.dumps(params), "CV F1 (mean)": cv_mean,
                        "CV F1 (std)": cv_std}
        joblib.dump(model, MODEL_DIR / MODEL_FILES[name])

        pred = model.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        print("\n" + "=" * 60)
        print(f"{name}  ->  saved model/{MODEL_FILES[name]}")
        print(f"Best params: {params}   5-fold CV F1 = {cv_mean:.4f} +/- {cv_std:.4f}")
        print(f"Confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")
        print(classification_report(y_test, pred, target_names=["edible", "poisonous"], digits=4))

    metrics_table = pd.DataFrame(results).T.round(4)
    metrics_table.index.name = "ML Model Name"
    tuning_table = pd.DataFrame(tuning).T
    tuning_table[["CV F1 (mean)", "CV F1 (std)"]] = (
        tuning_table[["CV F1 (mean)", "CV F1 (std)"]].astype(float).round(4)
    )
    metrics_table.join(tuning_table).to_csv(MODEL_DIR / "metrics.csv")

    metadata = {
        "features": features,
        "target": TARGET,
        "positive_label": POSITIVE,
        "models": MODEL_FILES,
        "sklearn_version": sklearn.__version__,
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print("\nTest-set comparison table:")
    print(metrics_table.to_string())
    print("\nTuning summary (5-fold CV on training split):")
    print(tuning_table.to_string())

    # Sanity check: reload saved models and score the saved test CSV.
    reloaded = pd.read_csv(TEST_FILE)
    y_reloaded = (reloaded[TARGET] == POSITIVE).astype(int)
    print("\nReload check (accuracy on test_data.csv):")
    for name, fname in MODEL_FILES.items():
        m = joblib.load(MODEL_DIR / fname)
        acc = accuracy_score(y_reloaded, m.predict(reloaded[features]))
        print(f"  {name:<25} {acc:.4f}")


if __name__ == "__main__":
    main()
