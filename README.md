# CardioSense 

# Technical Skills Used

Programming Language: Python

Libraries and Frameworks:
- Data Manipulation: Pandas and NumPy for handling and preprocessing the dataset.
- Machine Learning: Scikit-learn, XGBoost, and LightGBM for classification models and hyperparameter tuning.
- Imbalanced Data Handling: Imbalanced-learn (SMOTE, ADASYN, Balanced Bagging) to address class imbalance issues.
- Model Evaluation: Scikit-learn metrics like accuracy, precision, recall, F1 score, ROC AUC, and classification reports.
- Visualization: Matplotlib for plotting feature importances and ROC curves.

Techniques and Concepts:
- Data Preprocessing: Handling missing values, encoding categorical variables, and standardizing numerical features.
- Feature Engineering: Feature importance analysis and selection using Random Forest.
- Ensemble Learning: Voting Classifier and Balanced Bagging for model performance improvement.
- Hyperparameter Tuning: GridSearchCV for optimizing Random Forest hyperparameters.
- Class Imbalance Handling: Synthetic sampling techniques like SMOTE and ADASYN.

# Features and Program Capabilities
Data Preprocessing:
- Handles missing values by imputing mean for numeric and mode for categorical data.
- Encodes categorical variables using Label Encoding.
- Standardizes numerical features for better model performance.

Class Imbalance Resolution:
- Uses SMOTE (Synthetic Minority Oversampling Technique) and ADASYN to balance the dataset, ensuring that the models are not biased toward the majority class.

Feature Selection:
- Analyzes feature importance using Random Forest and selects the top 10 features for model training.

Machine Learning Models (Trains multiple models):
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost
- LightGBM
- Balanced Bagging Classifier
- Implements a Voting Classifier for ensemble learning.

Hyperparameter Tuning:
- Uses GridSearchCV for fine-tuning hyperparameters in Random Forest to optimize model performance.

Evaluation Metrics (Calculates and displays metrics for each model):
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC
- Provides a comprehensive classification report for all models.

Model Comparison:
- Compares all models based on their evaluation metrics and identifies the best-performing model using the F1 score.

Visualization:
- Plots feature importances.
- Generates and overlays ROC curves for models that provide probability predictions.

Result Summarization:
- Outputs a DataFrame summarizing performance metrics for all models.
- Highlights the best-performing model based on the highest F1 score.

# What the Program Can Do

Predict Heart Disease:
- Trains various machine learning models to predict heart disease outcomes based on patient data.

Handle Imbalanced Data:
- Ensures balanced training for minority and majority classes using advanced sampling techniques like SMOTE and ADASYN.

Evaluate and Compare Models:
- Provides detailed metrics for model evaluation and comparison.
- Highlights the best model for heart disease prediction.

Optimize Performance:
- Performs hyperparameter tuning for improved accuracy and reliability of predictions.

Interactive Insights:
- Visualizes key aspects like feature importance and ROC curves for better interpretability.

Support for Decision-Making:
- Offers actionable insights for selecting the most suitable model for predicting heart disease, ideal for healthcare analysts and researchers.
