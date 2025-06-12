# CardioSense 

Cardiovascular diseases (CVDs) are the leading cause of death globally, responsible for over 17.9 million deaths each year, accounting for 32% of all global deaths.
(Source: World Health Organization, 2023)

Early detection is critical, but in many healthcare systems, patient data is underutilized and models lack accuracy due to imbalanced datasets, poor interpretability, and limited clinical integration. Medical professionals and researchers need robust, interpretable, and equitable tools to help identify patients at high risk before symptoms become life-threatening.

# Why This Program Matters

CardioSense bridges the gap between clinical data and predictive healthcare by leveraging machine learning to detect heart disease risk. Unlike black-box models or generic classifiers, CardioSense:
- Handles real-world imbalanced data.
- Tunes for accuracy and interpretability.
- Helps decision-makers compare models transparently.
- Provides visual insights for clinical understanding.

This makes it ideal for hospitals, healthcare researchers, medical AI teams, and public health analysts seeking to reduce diagnostic errors and improve early detection rates.

# What the Program Does
- Predict Heart Disease: Trains and compares multiple machine learning models using patient data (e.g., cholesterol levels, blood pressure, chest pain type, etc.)
- Outputs the likelihood of heart disease for each patient.
- Handle Imbalanced Data: Uses SMOTE and ADASYN to balance class distribution (e.g., healthy vs. diseased), improving model fairness and real-world usability.
- Evaluate and Compare Models: Calculates key metrics suc as Accuracy, Precision, Recall, F1 Score, ROC AUC
- Produces a side-by-side summary table and classification report.
- Identifies the best-performing model for heart disease prediction using F1 score as the primary metric.
- Optimize Performance: Implements GridSearchCV for hyperparameter tuning of Random Forest and other models.
- Generate Interactive Insights: Visualizes feature importance (what variables matter most for prediction).
- Plots ROC curves to show trade-offs in classification performance.

Notes: 
- Recall (Sensitivity): Out of all actual patients with heart disease, how many did the model correctly find?
- F1 Score: How balanced is precision and recall? Useful when the dataset is imbalanced. High F1 = a model that’s both precise and good at finding actual cases.
- ROC AUC (Receiver Operating Characteristic – Area Under Curve): How well can the model distinguish between patients with and without heart disease?
- ROC Curve plots how recall and false positive rate trade off at different thresholds.
- AUC is a score from 0 to 1 — higher = better.
  

# 👨‍⚕️ Target Audience
- Healthcare Data Scientists building ML models for diagnostics.
- Clinical Researchers studying risk factors for heart disease.
- Hospitals & Clinics integrating predictive tools into electronic health record (EHR) systems.
- Public Health Agencies prioritizing preventative care strategies.
- Medical AI Engineers creating interpretable and ethical ML systems for deployment.
  

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
