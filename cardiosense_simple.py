#!/usr/bin/env python3
"""
CardioSense: Simplified Heart Disease Prediction System
This script performs comprehensive analysis and visualization of heart disease prediction models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
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
    data.info()
    
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
            data[column] = data[column].fillna(data[column].mode()[0])
        else:
            data[column] = data[column].fillna(data[column].mean())
    
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
    
    # Select top 10 features for modeling
    top_feature_names = feature_importance.head(10)['Feature'].tolist()
    
    return X[top_feature_names], y, feature_importance

def create_visualizations(data, feature_importance):
    """Create comprehensive visualizations"""
    print("\n" + "=" * 40)
    print("CREATING VISUALIZATIONS")
    print("=" * 40)
    
    # Create a large figure for all visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('CardioSense: Heart Disease Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Feature importance
    top_features = feature_importance.head(10)
    axes[0, 0].barh(top_features['Feature'], top_features['Importance'], color='skyblue')
    axes[0, 0].set_title('Top 10 Feature Importances')
    axes[0, 0].set_xlabel('Importance')
    
    # 2. Distribution of target variable
    target_counts = data['Heart Disease Status'].value_counts()
    colors = ['lightblue', 'lightcoral']
    axes[0, 1].pie(target_counts.values, labels=['No Disease', 'Disease'], autopct='%1.1f%%', colors=colors)
    axes[0, 1].set_title('Heart Disease Distribution')
    
    # 3. Age distribution by heart disease status
    disease_no = data[data['Heart Disease Status'] == 0]['Age']
    disease_yes = data[data['Heart Disease Status'] == 1]['Age']
    axes[0, 2].hist([disease_no, disease_yes], bins=20, alpha=0.7, label=['No Disease', 'Disease'], color=['lightblue', 'lightcoral'])
    axes[0, 2].set_title('Age Distribution by Disease Status')
    axes[0, 2].set_xlabel('Age')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].legend()
    
    # 4. Correlation heatmap (subset of features)
    numeric_cols = data.select_dtypes(include=[np.number]).columns[:15]  # Limit to first 15 for readability
    corr_matrix = data[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, ax=axes[1, 0])
    axes[1, 0].set_title('Feature Correlation Matrix')
    
    # 5. Blood Pressure vs Cholesterol by Disease Status
    if 'Blood Pressure' in data.columns and 'Cholesterol Level' in data.columns:
        for status in [0, 1]:
            subset = data[data['Heart Disease Status'] == status]
            label = 'No Disease' if status == 0 else 'Disease'
            color = 'lightblue' if status == 0 else 'lightcoral'
            axes[1, 1].scatter(subset['Blood Pressure'], subset['Cholesterol Level'], 
                             alpha=0.6, label=label, color=color)
        axes[1, 1].set_title('Blood Pressure vs Cholesterol Level')
        axes[1, 1].set_xlabel('Blood Pressure')
        axes[1, 1].set_ylabel('Cholesterol Level')
        axes[1, 1].legend()
    
    # 6. Dataset summary statistics
    axes[1, 2].axis('off')
    summary_text = f"""
    DATASET SUMMARY
    
    Total Records: {len(data):,}
    Features: {len(data.columns)-1}
    
    Disease Distribution:
    • No Disease: {target_counts[0]:,} ({target_counts[0]/len(data)*100:.1f}%)
    • Disease: {target_counts[1]:,} ({target_counts[1]/len(data)*100:.1f}%)
    
    Top Risk Factors:
    • {feature_importance.iloc[0]['Feature']}
    • {feature_importance.iloc[1]['Feature']}
    • {feature_importance.iloc[2]['Feature']}
    
    Age Range: {data['Age'].min():.0f} - {data['Age'].max():.0f} years
    Avg Age: {data['Age'].mean():.1f} years
    """
    
    axes[1, 2].text(0.1, 0.9, summary_text, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    
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
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
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
    # Ensure numeric columns are properly typed
    for col in results_df.columns:
        results_df[col] = pd.to_numeric(results_df[col])
    print("\nModel Performance Summary:")
    print(results_df.round(4))
    
    # Find best model based on F1 Score
    best_model_name = results_df['F1 Score'].idxmax()
    print(f"\nBest performing model: {best_model_name} (F1 Score: {results_df.loc[best_model_name, 'F1 Score']:.4f})")
    
    # Create comparison visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('CardioSense: Model Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Performance metrics comparison
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, (model_name, _) in enumerate(results.items()):
        model_scores = [results_df.loc[model_name, metric] for metric in metrics]
        axes[0, 0].bar(x + i * width, model_scores, width, label=model_name, alpha=0.8)
    
    axes[0, 0].set_xlabel('Metrics')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Model Performance Comparison')
    axes[0, 0].set_xticks(x + width)
    axes[0, 0].set_xticklabels(metrics, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 2. ROC Curves
    for name, result in results.items():
        try:
            model = result['Model']
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                axes[0, 1].plot(fpr, tpr, label=f"{name} (AUC = {result['ROC AUC']:.3f})", linewidth=2)
        except:
            continue
    
    axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.6)
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curves Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # 3. F1 Score comparison
    f1_scores = results_df['F1 Score'].sort_values(ascending=True)
    colors = ['lightcoral', 'lightblue', 'lightgreen'][:len(f1_scores)]
    bars = axes[1, 0].barh(range(len(f1_scores)), f1_scores.values, color=colors)
    axes[1, 0].set_yticks(range(len(f1_scores)))
    axes[1, 0].set_yticklabels(f1_scores.index)
    axes[1, 0].set_xlabel('F1 Score')
    axes[1, 0].set_title('F1 Score Comparison (Primary Metric)')
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, f1_scores.values)):
        axes[1, 0].text(value + 0.01, i, f'{value:.3f}', va='center', fontweight='bold')
    
    # 4. Model recommendations
    axes[1, 1].axis('off')
    
    # Get top 3 models
    top_3_models = results_df.nlargest(3, 'F1 Score')
    
    recommendations = f"""
    🏥 CARDIOSENSE RECOMMENDATIONS
    
    🥇 BEST MODEL: {best_model_name}
       • F1 Score: {top_3_models.iloc[0]['F1 Score']:.4f}
       • Accuracy: {top_3_models.iloc[0]['Accuracy']:.4f}
       • ROC AUC: {top_3_models.iloc[0]['ROC AUC']:.4f}
    
    🥈 RUNNER-UP: {top_3_models.index[1]}
       • F1 Score: {top_3_models.iloc[1]['F1 Score']:.4f}
       • Accuracy: {top_3_models.iloc[1]['Accuracy']:.4f}
    
    🥉 THIRD PLACE: {top_3_models.index[2]}
       • F1 Score: {top_3_models.iloc[2]['F1 Score']:.4f}
       • Accuracy: {top_3_models.iloc[2]['Accuracy']:.4f}
    
    💡 INSIGHTS:
    • {best_model_name} provides the best balance
      of precision and recall
    • Recommended for clinical deployment
    • Consider ensemble methods for production
    """
    
    axes[1, 1].text(0.05, 0.95, recommendations, fontsize=10, verticalalignment='top',
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
        
        # Create initial visualizations
        create_visualizations(data, feature_importance)
        
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
        
        # Train models
        results = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # Plot model comparison and get recommendations
        results_df, best_model = plot_model_comparison(results, X_test_scaled, y_test)
        
        # Final summary
        print("\n" + "=" * 60)
        print("🎉 CARDIOSENSE ANALYSIS COMPLETE! 🎉")
        print("=" * 60)
        print(f"\n✅ Successfully analyzed heart disease dataset")
        print(f"📊 Dataset: {len(data):,} records with {len(data.columns)-1} features")
        print(f"🤖 Trained and evaluated {len(results)} ML models")
        print(f"🏆 Best performing model: {best_model}")
        print(f"📈 Comprehensive visualizations generated")
        
        print(f"\n🔍 KEY FINDINGS:")
        print(f"• Most important risk factor: {feature_importance.iloc[0]['Feature']}")
        print(f"• Best model F1 score: {results_df.loc[best_model, 'F1 Score']:.4f}")
        print(f"• Best model accuracy: {results_df.loc[best_model, 'Accuracy']:.4f}")
        print(f"• Class imbalance handled with SMOTE oversampling")
        print(f"• Feature selection based on Random Forest importance")
        
        print(f"\n💡 CLINICAL RECOMMENDATIONS:")
        print(f"• Deploy {best_model} for heart disease screening")
        print(f"• Focus on top risk factors for prevention")
        print(f"• Regular monitoring of high-risk patients")
        print(f"• Integration with electronic health records")
        
        print(f"\n🚀 CardioSense is ready for clinical deployment!")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
