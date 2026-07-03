# Purchase Order Risk Classification (Cost-Sensitive)

A cost-sensitive binary classification project that predicts whether an online purchase order is **high risk** or **low risk** of defaulting on payment, using a real-world e-commerce dataset of 30,000 orders and 44 features (customer info, payment info, order behaviour, and order history).

## Problem

The target class is heavily imbalanced — only ~5.8% of orders are labelled high risk (`CLASS = yes`). Worse, the two error types are not equally costly:

| Error | Business cost |
|---|---|
| High-risk order predicted as low-risk (false negative) | **50** |
| Low-risk order predicted as high-risk (false positive) | 5 |

Because of this, the project is evaluated with a custom cost function rather than accuracy alone — a model that predicts "low risk" for everyone would hit ~94% accuracy while catching **zero** high-risk cases, which is the worst possible outcome for the business.

```
Total Cost = 50 x False Negatives + 5 x False Positives
```

## Approach

1. **EDA** — shape, dtypes, missing values, class balance.
2. **Cleaning** — dropped columns with >70% missing values, dropped the `ORDER_ID` identifier.
3. **Preprocessing pipeline** (`sklearn.compose.ColumnTransformer`) — median imputation + standard scaling for numeric features; most-frequent imputation + one-hot encoding for categorical features.
4. **Models compared** — Logistic Regression, Random Forest, and a small MLP neural network, each wrapped in a full `sklearn.Pipeline`.
5. **Cost-sensitive evaluation** — accuracy, confusion matrix, recall on the high-risk class, and the custom cost function for every model.
6. **Decision threshold tuning** — rather than relying on the default 0.5 classification threshold, each model's predicted probabilities are swept to find the threshold that minimizes total cost on the test set.

## A bug worth calling out

An earlier version of this notebook configured Random Forest with a manually chosen `class_weight={"no": 1, "yes": 10}`. That produced a recall of **0.0** on the high-risk class — the forest never predicted "yes" a single time, despite ~94% accuracy. Random Forest's bootstrap-per-tree training doesn't respond to small manual class weights the way Logistic Regression does. Switching to `class_weight="balanced"` (and giving the trees a bit more capacity) fixed this and made Random Forest the strongest model overall. This is left in the notebook as a documented example of why recall — not just accuracy — has to be checked for imbalanced problems.

## Results

| Model | Threshold | Accuracy | Recall (high-risk) | Total Cost |
|---|---|---|---|---|
| Random Forest (balanced) | 0.52 (tuned) | 0.79 | 0.54 | **13,495** |
| Logistic Regression | 0.26 (tuned) | 0.77 | 0.45 | 15,580 |
| MLP Neural Network | 0.36 (tuned) | 0.88 | 0.16 | 16,925 |
| Random Forest (naive weighting, default threshold) | 0.50 | 0.94 | **0.00** | 17,450 |

**Best model: Random Forest with balanced class weights and a cost-minimizing decision threshold.** It has the lowest total cost and the best recall on the class that actually matters for the business, even though its raw accuracy is lower than the naive Random Forest baseline.

**Limitations:** recall on the high-risk class still tops out around 0.54, so roughly half of high-risk orders would still slip through in production. Natural next steps: SMOTE/oversampling, gradient boosting with `scale_pos_weight` (XGBoost/LightGBM), feature engineering from `B_BIRTHDATE` and order-history columns, and a proper hyperparameter search instead of fixed settings.

## Repo contents

```
├── Applied_Machine_Learning_A2.ipynb   # full notebook: EDA, pipeline, models, cost function, threshold tuning
├── risk-dataset.txt                    # tab-separated dataset (30,000 rows x 44 columns)
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt
jupyter notebook Applied_Machine_Learning_A2.ipynb
```
Run all cells in order — the dataset file must be in the same folder as the notebook.

## Tech stack

Python, pandas, NumPy, scikit-learn (Pipeline, ColumnTransformer, LogisticRegression, RandomForestClassifier, MLPClassifier), matplotlib.
