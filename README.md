# Telco Customer Churn Prediction

An end-to-end machine learning project that predicts which telecom customers are likely to churn, so a retention team can contact them first. It includes an analysis notebook, a modular training and prediction pipeline, and a Flask web app.

**Dataset:** [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn): 7,043 customers, about 26.5% churn.

## Results

All scores are on a held-out test set (1,409 customers, 374 churners). The models were compared with 5-fold cross-validated PR-AUC on the training data, because the classes are imbalanced.

| Model | CV PR-AUC | Test PR-AUC | Test ROC-AUC |
|---|---|---|---|
| XGBoost (baseline) | 0.6627 | 0.6553 | 0.8417 |
| Logistic Regression | 0.6600 | 0.6326 | 0.8413 |
| Random Forest | 0.6553 | 0.6479 | 0.8417 |
| LightGBM | 0.6537 | 0.6472 | 0.8359 |
| **XGBoost (tuned with Optuna)** | **0.6722** | **0.6632** | **0.8470** |

- The four baseline models are within noise of each other (CV std about 0.02), so a simple logistic regression is nearly as good as the boosted trees.
- Engineered features (tenure group, add-on count) gave no gain beyond noise and were not used.
- Tuning gave a small gain (+0.0095 CV PR-AUC).

**Choosing the decision threshold** (selected on out-of-fold training predictions with F2, which favors recall):

| Threshold (test set) | Churners caught | Customers contacted | Precision |
|---|---|---|---|
| Default 0.5 | 79.9% | 42.0% | 50.5% |
| **Tuned 0.368** | **87.2%** | **50.8%** | **45.5%** |

At the tuned threshold the model catches 326 of the 374 churners by contacting 50.8% of customers.

## App demo

The Flask app has two pages: a single-customer form and a batch page that scores a whole CSV, sorts it by risk, and lets you download the results.

**Batch prediction on the held-out test split (1,409 customers): 326 of 374 churners caught by flagging 50.8% of customers**

![Batch prediction](images/batch_test_split.png)

**Single customer: high risk**

![High risk customer](images/single_high_risk.png)

**Single customer: low risk**

![Low risk customer](images/single_low_risk.png)

## Approach

1. **Cleaning:** converted blank `TotalCharges` strings to numbers, dropped `customerID`, mapped `Churn` to 0/1.
2. **EDA:** churn rate by contract, internet service and payment method; tenure and charges distributions; correlation with churn.
3. **Preprocessing:** scikit-learn `ColumnTransformer` (median imputation and scaling for numeric columns, imputation and one-hot encoding for categorical columns), placed inside each model pipeline so it is refitted on every CV fold (no leakage).
4. **Models:** Logistic Regression, Random Forest, LightGBM and XGBoost, compared by cross-validated PR-AUC.
5. **Tuning:** XGBoost tuned with Optuna (50 trials) on cross-validated PR-AUC, using the training set only.
6. **Threshold:** chosen on out-of-fold training predictions, and the test set was used once for the final report.
7. **Interpretation:** SHAP summary plots for the final model.

## Key drivers (SHAP)

Month-to-month contracts, short tenure, higher monthly charges, fiber optic internet, no online security or tech support, and electronic check payment push predictions toward churn. Two-year contracts and long tenure push toward staying. SHAP shows what the model relies on, not what causes churn.

## Project structure

```
customerchurnprediction/
├── app.py                          Flask app (single customer + batch CSV)
├── templates/                      base.html, home.html, batch.html
├── notebook/
│   ├── telco_churn_prediction.ipynb
│   └── data/                       Telco customer churn CSV
├── src/customer_chrun/
│   ├── components/
│   │   ├── data_ingestion.py       read CSV, stratified train/test split
│   │   ├── data_transformation.py  cleaning + preprocessor
│   │   └── model_trainer.py        models, Optuna tuning, threshold, evaluation
│   ├── pipelines/
│   │   ├── train_pipeline.py       runs the full training flow
│   │   └── prediction_pipeline.py  loads the saved model and predicts
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
├── images/                         app screenshots
├── requirements.txt
└── setup.py
```

## How to run

```bash
# 1. Create and activate an environment, then install the packages
conda create -p ven python=3.11.9 
conda activate ven/
pip install -r requirements.txt

# 2. Train the model (about 2-3 minutes). Creates artifacts/model.pkl
python -m src.customer_chrun.pipelines.training_pipeline

# 3. Start the app
python app.py
```

Open `http://127.0.0.1:5000`. The **Batch (CSV)** tab has a "Try the sample data" button that scores the held-out test split.

The modular pipeline reproduces the notebook's results (same cross-validated and test scores).

## Tech stack

Python, pandas, scikit-learn, XGBoost, LightGBM, Optuna, SHAP, Flask.


## Author

Amit Kumar · [GitHub](https://github.com/your-username) · [LinkedIn](https://www.linkedin.com/in/your-profile)