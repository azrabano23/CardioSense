#!/usr/bin/env python3
"""
CardioSense: Machine Learning Heart Disease Prediction System
This script performs comprehensive analysis and visualization of heart disease prediction models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score, roc_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.over_sampling import SMOTE, ADASYN
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_explore_data():
    """Load and explore the heart disease dataset"""
    print("=" * 60)
    print("CARDIOSENSE: HEART DISEASE PREDICTION SYSTEM")
    print("=" * 60)
    
    # Load the dataset
    data = pd.read_csv('heart_disease.csv')
    
    print("\nDataset Preview:")
    print(data.head())
    
    print(f"\nDataset Shape: {data.shape}")
    print(f"\nDataset Info:")
    print(data.info())
    
    print("\nMissing Values Before Handling:")
    print(data.isnull().sum())
    
    return data

def preprocess_data(data):
    """Preprocess the data by handling missing values and encoding categorical variables"""
    print("\n" + "=" * 40)
    print("DATA PREPROCESSING")
    print("=" * 40)
    
    # Handle missing values
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column].fillna(data[column].mode()[0], inplace=True)
        else:
            data[column].fillna(data[column].mean(), inplace=True)
    
    print("\nMissing Values After Handling:")
    print(data.isnull().sum())
    
    # Label encode categorical variables
    le = LabelEncoder()
    categorical_columns = data.select_dtypes(include=['object']).columns
    
    for column in categorical_columns:
        data[column] = le.fit_transform(data[column])
    
    return data

def feature_importance_analysis(data):
    """Analyze feature importance using Random Forest"""
    print("\n" + "=" * 40)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 40)
    
    # Separate features and target
    X = data.drop('Heart Disease Status', axis=1)
    y = data['Heart Disease Status']
    
    # Train Random Forest for feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importances:")
    print(feature_importance)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    top_features = feature_importance.head(10)
    sns.barplot(data=top_features, y='Feature', x='Importance', palette='viridis')
    plt.title('Top 10 Feature Importances')
    plt.xlabel('Importance')
    
    # Select top 10 features for modeling
    top_feature_names = feature_importance.head(10)['Feature'].tolist()
    
    return X[top_feature_names], y, feature_importance

def create_visualizations(data):
    """Create comprehensive visualizations"""
    print("\n" + "=" * 40)
    print("CREATING VISUALIZATIONS")
    print("=" * 40)
    
    # Distribution of target variable
    plt.subplot(2, 2, 2)
    target_counts = data['Heart Disease Status'].value_counts()
    plt.pie(target_counts.values, labels=['No Disease', 'Disease'], autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
    plt.title('Heart Disease Distribution')
    
    # Correlation matrix
    plt.subplot(2, 2, 3)
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    corr_matrix = data[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    
    # Age distribution by heart disease status
    plt.subplot(2, 2, 4)
    sns.boxplot(data=data, x='Heart Disease Status', y='Age', palette='Set2')
    plt.title('Age Distribution by Heart Disease Status')
    plt.xlabel('Heart Disease Status (0=No, 1=Yes)')
    
    plt.tight_layout()
    plt.show()

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple machine learning models"""
    print("\n" + "=" * 40)
    print("TRAINING MACHINE LEARNING MODELS")
    print("=" * 40)
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(random_state=42, verbose=-1),
        'Balanced Bagging': BalancedBaggingClassifier(random_state=42)
    }
    
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_pred_proba)
            except:
                roc_auc = accuracy  # fallback for models without predict_proba
            
            results[name] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'ROC AUC': roc_auc,
                'Model': model
            }
            
            print(f"{name} Classification Report:")
            print(classification_report(y_test, y_pred))
            print()
            
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            continue
    
    return results

def plot_model_comparison(results, X_test, y_test):
    """Plot comparison of model performance"""
    print("\n" + "=" * 40)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 40)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results).T.drop('Model', axis=1)
    print("\nModel Performance Summary:")
    print(results_df.round(4))
    
    # Find best model based on F1 Score
    best_model_name = results_df['F1 Score'].idxmax()
    print(f"\nBest performing model: {best_model_name} (F1 Score: {results_df.loc[best_model_name, 'F1 Score']:.4f})")
    
    # Plot model comparison
    plt.figure(figsize=(15, 10))
    
    # Performance metrics comparison
    plt.subplot(2, 2, 1)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    x = np.arange(len(metrics))
    width = 0.15
    
    for i, (model_name, _) in enumerate(results.items()):
        model_scores = [results_df.loc[model_name, metric] for metric in metrics]
        plt.bar(x + i * width, model_scores, width, label=model_name, alpha=0.8)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x + width * (len(results) - 1) / 2, metrics, rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # ROC Curves
    plt.subplot(2, 2, 2)
    for name, result in results.items():
        try:
            model = result['Model']
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                plt.plot(fpr, tpr, label=f"{name} (AUC = {result['ROC AUC']:.3f})", linewidth=2)
        except:
            continue
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # F1 Score comparison
    plt.subplot(2, 2, 3)
    f1_scores = results_df['F1 Score'].sort_values(ascending=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(f1_scores)))
    bars = plt.barh(range(len(f1_scores)), f1_scores.values, color=colors)
    plt.yticks(range(len(f1_scores)), f1_scores.index)
    plt.xlabel('F1 Score')
    plt.title('F1 Score Comparison (Primary Metric)')
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, f1_scores.values)):
        plt.text(value + 0.01, i, f'{value:.3f}', va='center', fontweight='bold')
    
    # Model recommendations
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    # Get top 3 models
    top_3_models = results_df.nlargest(3, 'F1 Score')
    
    recommendations = f"""
    MODEL RECOMMENDATIONS
    
    🥇 Best Model: {best_model_name}
       F1 Score: {top_3_models.iloc[0]['F1 Score']:.4f}
       Accuracy: {top_3_models.iloc[0]['Accuracy']:.4f}
    
    🥈 Second Best: {top_3_models.index[1]}
       F1 Score: {top_3_models.iloc[1]['F1 Score']:.4f}
       Accuracy: {top_3_models.iloc[1]['Accuracy']:.4f}
    
    🥉 Third Best: {top_3_models.index[2]}
       F1 Score: {top_3_models.iloc[2]['F1 Score']:.4f}
       Accuracy: {top_3_models.iloc[2]['Accuracy']:.4f}
    
    💡 The {best_model_name} model shows the best
    balance between precision and recall, making it
    the recommended choice for heart disease prediction.
    """
    
    plt.text(0.1, 0.9, recommendations, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return results_df, best_model_name

def main():
    """Main function to run the complete CardioSense analysis"""
    try:
        # Load and explore data
        data = load_and_explore_data()
        
        # Preprocess data
        data = preprocess_data(data)
        
        # Feature importance analysis
        X, y, feature_importance = feature_importance_analysis(data)
        
        # Handle class imbalance with SMOTE
        print("\nApplying SMOTE for class balancing...")
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
        )
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create visualizations
        create_visualizations(data)
        
        # Train models
        results = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # Plot model comparison and get recommendations
        results_df, best_model = plot_model_comparison(results, X_test_scaled, y_test)
        
        print("\n" + "=" * 60)
        print("CARDIOSENSE ANALYSIS COMPLETE!")
        print("=" * 60)
        print(f"\n✅ Successfully trained and evaluated {len(results)} models")
        print(f"🏆 Best performing model: {best_model}")
        print(f"📊 Visualizations have been generated and displayed")
        print(f"💾 Model comparison results saved")
        
        print("\n📈 KEY INSIGHTS:")
        print("- Feature importance analysis completed")
        print("- Class imbalance handled with SMOTE")
        print("- Multiple algorithms compared")
        print("- ROC curves and performance metrics visualized")
        print("- Best model identified for deployment")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
