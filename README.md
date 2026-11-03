# Mushroom Edibility Classification – ML Assignment 2

## a. Problem statement

Predict whether a gilled mushroom is **edible** or **poisonous** from its physical traits (cap, gill, stalk, veil, ring, spore print, population, and habitat). This is a binary classification task. Five classifiers are trained on the same data and compared on Accuracy, AUC, Precision, Recall, F1, and MCC. An interactive Streamlit app loads the saved models and evaluates them on an uploaded test CSV.

Because eating a poisonous mushroom can be fatal, **poisonous (`p`) is treated as the positive class**. Recall on the poisonous class (how many poisonous mushrooms we catch) is the most safety-critical metric.

## b. Dataset description

| Item | Value |
|---|---|
| Source | UCI Machine Learning Repository – *Mushroom* (agaricus-lepiota): https://archive.ics.uci.edu/dataset/73/mushroom (DOI [10.24432/C5959T](https://doi.org/10.24432/C5959T), CC BY 4.0) |
| Origin | Hypothetical samples of 23 gilled mushroom species (Agaricus and Lepiota families) from the Audubon Society Field Guide. Species of unknown edibility were merged into the poisonous class. |
| Task | Binary classification |
| Instances | 8,124 |
| Features | 22 categorical features (21 used after preprocessing) |
| Target | `output` (the UCI class label): `e` = edible (4,208, 51.8%), `p` = poisonous (3,916, 48.2%) |
| Missing values | `stalk-root` has 2,480 `?` values |

Features: `cap-shape`, `cap-surface`, `cap-color`, `bruises`, `odor`, `gill-attachment`, `gill-spacing`, `gill-size`, `gill-color`, `stalk-shape`, `stalk-root`, `stalk-surface-above-ring`, `stalk-surface-below-ring`, `stalk-color-above-ring`, `stalk-color-below-ring`, `veil-type`, `veil-color`, `ring-number`, `ring-type`, `spore-print-color`, `population`, `habitat`. Every feature is coded as a single letter.

Preprocessing:
- `veil-type` has only one value in the whole dataset, so it is dropped.
- `?` in `stalk-root` is kept as its own category rather than imputed.
- All features are one-hot encoded inside each model's scikit-learn `Pipeline` (`OneHotEncoder(handle_unknown="ignore")`). The saved models therefore accept the raw letter-coded CSV.
- 80/20 stratified train/test split (`random_state=42`): 6,499 training rows and 1,625 test rows. The test split is saved as `test_data.csv`.

## c. GitHub repository link

https://github.com/lavanyachimmiri15/Mushroom_classifier

Live Streamlit app: https://mushroomclassifier-y9xvgq4i5jnkgssqubi4w5.streamlit.app/

## d. Models used

All five models use the same preprocessing pipeline and the same train/test split. Each model's main hyperparameter was tuned with `GridSearchCV` (5-fold stratified CV, F1 scoring) on the **training split only**. The test split was used once, for the final evaluation below.

| Model | Search grid | Selected | 5-fold CV F1 (train) |
|---|---|---|---|
| Logistic Regression | `C` in {0.01, 0.1, 1, 10} | `C=10` | 0.9995 ± 0.0010 |
| Decision Tree | `max_depth` in {3, 4, 5, 6, 7, 8, None} | `max_depth=5` | 0.9997 ± 0.0006 |
| KNN | `n_neighbors` in {3, 5, 7, 9, 11, 15} | `n_neighbors=3` | 0.9995 ± 0.0010 |
| Naive Bayes (Multinomial) | `alpha` in {0.01, 0.1, 0.5, 1.0} | `alpha=0.01` | 0.9904 ± 0.0025 |
| Support Vector Machine (RBF) | `C` in {0.1, 1, 10} | `C=10` | 1.0000 ± 0.0000 |

Notes:
- Multinomial NB is used rather than Gaussian NB because one-hot features are non-negative indicators, not continuous Gaussian variables.
- The SVM is trained without `probability=True`, which is deprecated in scikit-learn 1.9. Its AUC is computed from `decision_function` scores. All other models use `predict_proba`.

### Comparison table (test set, 1,625 rows; positive class = poisonous)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Decision Tree | 0.9988 | 0.9990 | 1.0000 | 0.9974 | 0.9987 | 0.9975 |
| KNN | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Naive Bayes | 0.9895 | 0.9994 | 0.9910 | 0.9872 | 0.9891 | 0.9791 |
| Support Vector Machine | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Confusion matrices (842 edible, 783 poisonous in the test set):

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| Logistic Regression | 842 | 0 | 0 | 783 |
| Decision Tree | 842 | 0 | 2 | 781 |
| KNN | 842 | 0 | 0 | 783 |
| Naive Bayes | 835 | 7 | 10 | 773 |
| Support Vector Machine | 842 | 0 | 0 | 783 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Perfect on the test set (MCC 1.0) and 0.9995 CV F1, so the one-hot classes are almost exactly linearly separable. The largest coefficients match domain intuition: `odor=n` (none) pushes strongly toward edible, while `odor=f` (foul) and `spore-print-color=r` (green) push toward poisonous. It is the most interpretable of the perfect models and the cheapest to train and run. |
| Decision Tree | Strong (MCC 0.9975), but it is the only tuned model besides Naive Bayes that misclassifies any poisonous mushrooms: 2 are labeled edible. CV selected `max_depth=5` (12 leaves), which makes it easy to read as rules. Root split is `odor=n`; top importances are `odor=n` (0.62), `stalk-root=c` (0.18), and `stalk-surface-below-ring=y` (0.10). The shallow depth trades a small number of rare cases for a much simpler rule set. |
| KNN | Perfect on the test set with k=3 (CV F1 0.9995). Mushrooms with similar trait profiles almost always share a label, so a small neighborhood in one-hot space is enough. KNN stores the whole training set, so prediction is the slowest here (~0.18 s for 1,625 rows vs. under 0.01 s for LR/DT/NB). |
| Naive Bayes | Lowest scores (Accuracy 0.9895, MCC 0.9791), with 10 poisonous mushrooms labeled edible and 7 false alarms. Tuning helped noticeably: `alpha=0.01` beat the default `alpha=1.0` (test accuracy 0.9458 before tuning). AUC is high (0.9994), but lowering the decision threshold does not remove the missed poisonous cases (recall 0.9872 at 0.5 and at 0.3). Treating each trait as independent evidence limits how well it can model trait combinations. |
| Support Vector Machine | Perfect on the test set and the only model with perfect CV F1 on every fold (1.0000 ± 0.0000). The RBF kernel captures trait interactions that the linear and independence-based models only approximate. It uses 770 of the 6,499 training rows as support vectors and is the slowest to train (~0.44 s vs. ~0.02 s for the others), which is irrelevant at this scale. |
| **Overall, Winner for your dataset?** | **Support Vector Machine.** It is perfect on the test set and is the only model that is also perfect across every cross-validation fold, i.e. it never labeled a poisonous mushroom as edible. Logistic Regression and KNN tie on the test set but show small CV errors. Logistic Regression is the best choice when interpretability matters. The dataset is close to deterministic: a single rule on `odor` alone (almond, anise, or none means edible) is 98.5% accurate. So differences between models come down to a handful of hard cases, and Recall/MCC on the poisonous class are the most informative metrics. |

## Additional information

### Project structure

```
.
├── app.py                  # Streamlit app (loads saved models, never retrains)
├── train_models.py         # preprocessing, training, evaluation, model export
├── model_training.ipynb    # notebook walkthrough: EDA, training, evaluation, plots
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/config.toml  # forest-green app theme
├── agaricus-lepiota.csv    # full dataset
├── test_data.csv           # 20% stratified hold-out split (features + target)
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    ├── metrics.csv         # test metrics, tuned params, CV F1
    └── metadata.json       # feature list, target, model files, sklearn version
```

### Streamlit app features

- **Single-mushroom prediction:** pick the 21 traits from labeled dropdowns (e.g. "almond (a)") and get an edible/poisonous prediction from the selected model, with its probability (or SVM margin) and a vote from all five models.
- CSV upload for test data. A sample `test_data.csv` can be downloaded, or used automatically if nothing is uploaded.
- Model selection dropdown covering all five classifiers.
- Evaluation metrics: Accuracy, AUC, Precision, Recall, F1, MCC.
- Confusion matrix and classification report.
- Side-by-side comparison of all models on the uploaded data, plus a downloadable predictions CSV.
- Upload validation: required columns, a valid target column, and a row limit.

The uploaded CSV must contain the 21 feature columns above (excluding `veil-type`) plus the target column `output` (or `class`, as in public copies of the dataset), with values `p`/`e` or `1`/`0`. Extra columns such as `veil-type` are ignored.

### Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_models.py   # optional: retrain and regenerate model/ and test_data.csv
streamlit run app.py
```

> The saved models were trained with scikit-learn 1.9.1 (NumPy 2.5, pandas 3.1). `requirements.txt` pins scikit-learn exactly, because joblib models must be loaded with the version that saved them; the other libraries use version ranges. The project installs on Python 3.11 to 3.15, so any of these can be selected under *Advanced settings* on Streamlit Community Cloud.
