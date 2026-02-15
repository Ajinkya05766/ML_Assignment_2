# Wine Quality Classification – ML Assignment 2

## a. Problem statement
Predict the quality score of wine samples (red and white) based on physicochemical test results using multiple machine learning classification models.

## b. Dataset description
- **Source**: UCI Machine Learning Repository – Wine Quality dataset (red and white combined).
- **Instances**: 2000+ samples (subset/synthetic for offline execution).
- **Features**: 12 (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol, type).
- **Target**: Quality score (multi-class).

## c. Models used
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbour (kNN) Classifier
4. Naive Bayes (Gaussian) Classifier
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

### Model Comparison Table

| ML Model Name       | Accuracy | Precision | Recall | F1     | MCC     |
|---------------------|---------:|----------:|-------:|-------:|--------:|
| Logistic Regression | 0.1525   | 0.1529    | 0.1448 | 0.1362 | -0.0244 |
| Decision Tree       | 0.1300   | 0.1304    | 0.1311 | 0.1295 | -0.0434 |
| kNN                 | 0.1700   | 0.1705    | 0.1737 | 0.1684 |  0.0046 |
| Naive Bayes         | 0.1475   | 0.1375    | 0.1462 | 0.1374 | -0.0260 |
| Random Forest       | 0.1550   | 0.1514    | 0.1544 | 0.1517 | -0.0175 |
| XGBoost             | 0.1300   | 0.1300    | 0.1290 | 0.1276 | -0.0443 |

### Observations
- **kNN** performed best with an accuracy of 17.00% and positive MCC, suggesting local neighbors are slightly more predictive than linear boundaries for this synthetic subset.
- **Logistic Regression** and **Random Forest** showed comparable performance around 15% accuracy.
- **XGBoost** and **Decision Tree** had lower performance on this specific run, possibly due to the synthetic nature of the data or hyperparameter settings.

## d. Streamlit app
The deployed app allows users to:
- Upload a test CSV file.
- Select from the trained models.
- View predictions and evaluation metrics.
