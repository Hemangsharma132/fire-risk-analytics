# 🔥 FireRisk Analytics: Forest Fire Weather Index Prediction

## 📌 Project Overview

In this project, I analyzed meteorological data from the Algerian Forest Fires dataset and developed a Machine Learning model to predict the Fire Weather Index (FWI).

The main objective of this project was to understand the relationship between weather conditions and forest fire behavior, perform Exploratory Data Analysis (EDA), prepare the dataset through feature engineering, and compare different regression algorithms to identify a suitable model for FWI prediction.

I used Python and its data analysis and Machine Learning libraries to perform data cleaning, visualization, feature selection, model training, evaluation, and model serialization.

## 🎯 Objectives

- Analyze the Algerian Forest Fires dataset and understand its structure.
- Perform data cleaning and handle missing values.
- Conduct Exploratory Data Analysis to identify patterns and relationships.
- Visualize the distribution of variables and analyze correlations.
- Perform feature engineering and prepare data for Machine Learning.
- Select relevant features and investigate multicollinearity.
- Train and compare multiple regression models.
- Evaluate model performance using Mean Absolute Error (MAE) and R² score.
- Save the trained Ridge Regression model and StandardScaler using Pickle.

## 📂 Dataset Description

The project uses the Algerian Forest Fires dataset, which contains meteorological observations collected from two regions of Algeria:

- Bejaia
- Sidi Bel-abbes

The original dataset contains 244 observations collected between June and September 2012.

### Important Features

| Feature | Description |
|---|---|
| Temperature | Maximum daily temperature in Celsius |
| RH | Relative humidity |
| Ws | Wind speed |
| Rain | Daily rainfall |
| FFMC | Fine Fuel Moisture Code |
| DMC | Duff Moisture Code |
| DC | Drought Code |
| ISI | Initial Spread Index |
| BUI | Buildup Index |
| FWI | Fire Weather Index |
| Classes | Fire or not-fire classification |
| Region | Region identifier |

The dataset also includes date-related information such as day, month, and year.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Pickle
- Jupyter Notebook

## 🔍 Project Workflow

### 1. Dataset Loading and Initial Analysis

I started by importing the required Python libraries and loading the original dataset using Pandas.

I examined the dataset using functions such as:

- `head()` to inspect the first few records.
- `info()` to understand column names, data types, and non-null values.
- `describe()` to examine the statistical properties of numerical features.
- `isnull().sum()` to identify missing values.

This initial analysis helped me understand the structure and quality of the dataset before proceeding with data preparation.

### 2. Data Cleaning and Preprocessing

After examining the dataset, I performed data cleaning and preprocessing.

The main steps included:

- Identifying missing values.
- Assigning a region identifier to distinguish observations from the two regions.
- Removing rows containing missing values.
- Removing an additional row identified during dataset preparation.
- Cleaning column names by removing unnecessary spaces.
- Converting date-related and weather-related columns to appropriate data types.
- Saving the cleaned dataset as a CSV file for subsequent analysis and model training.

These steps helped prepare the dataset for EDA and Machine Learning.

### 3. Exploratory Data Analysis (EDA)

I performed EDA to understand the distribution of variables and identify relationships within the dataset.

#### Feature Distribution

I used histograms to examine the distributions of numerical features, including temperature, relative humidity, wind speed, rainfall, and fire-weather indices.

This helped me understand the range and distribution of the observations.

#### Fire and Not-Fire Analysis

I examined the distribution of the `Classes` column to understand the number of fire and not-fire observations.

I also converted the categorical target labels into numerical values for analysis:

- Fire: 1
- Not Fire: 0

I visualized the class distribution to understand how the observations were divided between these two categories.

#### Correlation Analysis

I calculated correlations between numerical variables and visualized them using a heatmap.

This analysis helped me investigate relationships between meteorological variables and fire-weather indices.

#### Outlier Analysis

I used box plots to examine the distribution of selected numerical features and identify potential outliers.

#### Monthly Fire Analysis

I analyzed the distribution of fire and not-fire observations across different months and examined the two regions separately.

This helped me explore seasonal patterns in the available observations.

### 4. Feature Engineering and Feature Selection

After completing the initial EDA, I prepared the dataset for regression modeling.

The main steps included:

- Removing day, month, and year from the model inputs.
- Encoding the categorical `Classes` feature into numerical values.
- Separating the independent features from the dependent target variable.
- Using the `FWI` column as the prediction target.
- Splitting the dataset into training and testing sets.
- Examining correlations between the independent variables.
- Identifying highly correlated features using a correlation threshold of 0.85.
- Removing selected highly correlated features from the training and testing feature sets.

The purpose of correlation-based feature selection was to investigate multicollinearity and reduce redundant information among the predictors.

### 5. Train-Test Split

I divided the dataset into training and testing sets using Scikit-learn's `train_test_split()` function.

The configuration used was:

- Test size: 25%
- Random state: 42

The training data was used to fit the regression models, while the testing data was used to evaluate their predictions on held-out observations.

### 6. Feature Scaling Using StandardScaler

I applied `StandardScaler` to standardize the input features.

The scaler was fitted on the training data and then applied to both the training and testing sets.

Standardization transforms features based on their training-set means and standard deviations.

This is particularly useful for regularized regression algorithms such as Ridge, Lasso, and ElasticNet, which are sensitive to feature scales.

I also used box plots to visualize the feature distributions before and after scaling.

### 7. Model Training and Comparison

I trained and evaluated four regression algorithm families to compare their performance in predicting the Fire Weather Index.

#### Linear Regression

I used Linear Regression as a baseline model to examine the relationship between the input features and FWI.

#### Lasso Regression

I trained a Lasso Regression model, which uses L1 regularization and can shrink some feature coefficients to zero.

I also experimented with LassoCV to select the regularization parameter through cross-validation.

#### Ridge Regression

I trained Ridge Regression, which uses L2 regularization to penalize large coefficients.

I also used RidgeCV to select the regularization parameter through cross-validation.

#### ElasticNet Regression

I trained ElasticNet Regression, which combines L1 and L2 regularization.

I also experimented with ElasticNetCV to tune the regularization parameter using cross-validation.

### 8. Model Evaluation

I evaluated the regression models using the following metrics:

#### Mean Absolute Error (MAE)

MAE measures the average absolute difference between the actual FWI values and the values predicted by the model.

A lower MAE indicates that predictions are closer to the actual values on average.

#### R² Score

The R² score measures how well the model explains the variation in the target variable relative to a baseline prediction.

A higher R² score generally indicates better predictive performance on the evaluated dataset.

I also created scatter plots comparing actual and predicted FWI values to visually inspect model performance.

The model evaluation process helped me compare the regression algorithms and examine their predictive behavior.

### 9. Final Model Selection

After experimenting with the regression algorithms and their cross-validated variants, I selected RidgeCV as the model to save for subsequent use.

RidgeCV performs cross-validation to select a suitable regularization parameter.

The final saved model is intended to predict FWI from the selected and standardized input features.

The numerical evaluation results should be taken from the executed model-training notebook.

### 10. Model Serialization Using Pickle

After training the selected model and fitting the scaler, I saved both objects using Python's Pickle module.

The saved files are:

- `ridge.pkl` — the trained RidgeCV regression model.
- `scaler.pkl` — the fitted StandardScaler.

Saving these objects makes it possible to load the trained model and scaler later without repeating the complete training process.

For future predictions, the input features must use the same preprocessing steps and feature order as those used during training.

## 📁 Project Structure

```text
fire-risk-analytics/
│
├── 01_EDA_and_Feature_Engineering.ipynb
├── 02_Model_Training.ipynb
├── Algerian_forest_fires_cleaned_dataset.csv
├── ridge.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
