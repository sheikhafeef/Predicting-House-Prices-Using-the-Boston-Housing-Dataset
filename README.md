# Predicting-House-Prices-Using-the-Boston-Housing-Dataset
Build a regression model from scratch to predict house prices using the Boston Housing Dataset.

>Project Overview:

This project aims to predict housing prices using the Boston Housing Dataset. We implement regression models from scratch, including:

Linear Regression

Decision Tree (used as a Random Forest)

Gradient Boosting (XGBoost-like)

>Install Required Libraries:

pip install pandas numpy matplotlib seaborn scikit-learn

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score

pandas for handling datasets.

numpy for array operations.

matplotlib.pyplot and seaborn for visualization.

train_test_split from sklearn.model_selection for splitting the dataset.

mean_squared_error and r2_score from sklearn.metrics for evaluating models.

>Load and Explore the Dataset:

data = pd.read_csv('Boston (1).csv')

print(data.head())

Loads the dataset

Displays the first few rows

>Observation:

The dataset contains various features (CRIM, ZN, INDUS, etc.) and a target variable (MEDV - housing price).

The table confirms that the dataset is structured correctly.

>Handle Missing Data:

data = data.dropna()

Removes missing values for cleaner data.

>Normalize the Dataset:

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

data[data.columns] = scaler.fit_transform(data[data.columns])

Scales features to the range [0,1] for better model performance.

> Split Dataset into Features and Target:

X = data.drop('MEDV', axis=1)

y = data['MEDV']

X: Independent features

y: Target (house price)

>Train/Test Split:

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

  X, y, test_size=0.2, random_state=42
)

80% Training | 20% Testing

Ensures reproducibility with random_state=42

>Model Implementations:


>inear Regression:

Implements Gradient Descent

Trains weights to minimize the error

>Decision Tree:

Custom tree splits data based on feature thresholds

Captures non-linear patterns better than linear regression

>Gradient Boosting:

Trains multiple decision trees sequentially

Each new tree corrects errors from previous trees

>Model Evaluation Metrics:

RMSE (Root Mean Square Error): Measures prediction error

R² Score: Indicates how well the model explains variance in the target variable

>Feature Importance Plot:

importance = pd.Series(rf_model.tree[0], index=X_train.columns)

importance.sort_values(ascending=False).plot(kind='bar', figsize=(10, 5))

plt.title("Feature Importance - Random Forest")

plt.show()

>Observation:

The feature importance plot is unexpected.

All features are given equal importance, likely due to an implementation issue in the decision tree model.

>Observations from Output:

>Linear Regression:

Best model in this implementation

RMSE: 0.1357 → Indicates reasonable error

R²: 0.4917 → Explains about 49% of variance in house prices

Performs better than decision trees

>Decision Tree:

Underperforms badly

Negative R² (-0.0134) → Worse than a random model

Possible issue in feature selection or splitting

>Gradient Boosting:

Poor performance

RMSE: 0.1901 → Higher error than Linear Regression

R²: 0.0016 → Model almost does not explain variance in data
