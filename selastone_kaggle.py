# ============================================================================
# SELASTONE LOAN DEFAULT PREDICTION - JUPYTER NOTEBOOK
# Using Loan_Default.csv from Kaggle
# ============================================================================

# %% CELL 1: IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import pickle
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
sns.set_style("whitegrid")
print("✓ All imports successful")

# %% CELL 2: LOAD DATA
csv_path = '/mnt/user-data/uploads/Loan_Default.csv'

df = pd.read_csv(csv_path)
print(f"✓ Data loaded: {df.shape}")
print(f"\nDataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\nFirst rows:")
print(df.head())

# %% CELL 3: EXPLORE & CLEAN
print("\n" + "="*70)
print("DATA EXPLORATION & CLEANING")
print("="*70)

# Check target variable
print(f"\nTarget Variable (Status):")
print(df['Status'].value_counts())
print(f"Default Rate: {df['Status'].mean():.2%}")

# Remove ID column (not a feature)
df = df.drop(['ID'], axis=1)

# Missing values
print(f"\nMissing Values Summary:")
missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
print(missing_pct[missing_pct > 0].head(15))

# Drop columns with >40% missing
drop_cols = missing_pct[missing_pct > 40].index.tolist()
print(f"\nDropping {len(drop_cols)} columns with >40% missing:")
print(drop_cols)
df = df.drop(columns=drop_cols)

print(f"\nDataset shape after cleaning: {df.shape}")

# %% CELL 4: FEATURE ENGINEERING
print("\n" + "="*70)
print("FEATURE ENGINEERING")
print("="*70)

# Extract numeric and categorical columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Remove target from features
if 'Status' in numeric_cols:
    numeric_cols.remove('Status')

print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")
print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols}")

# Fill missing numeric values with median
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# Fill missing categorical values with mode
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

print(f"\n✓ Missing values filled")

# Create derived features
df['loan_to_income'] = df['loan_amount'] / (df['income'] + 1)
df['loan_to_property'] = df['loan_amount'] / (df['property_value'] + 1)
df['credit_to_income'] = df['Credit_Score'] / (df['income'] + 1)

# Add new features to numeric columns
new_features = ['loan_to_income', 'loan_to_property', 'credit_to_income']
numeric_cols.extend(new_features)

print(f"✓ Created {len(new_features)} derived features")
print(f"✓ Total numeric features: {len(numeric_cols)}")

# %% CELL 5: PREPARE FEATURES FOR MODELING
print("\n" + "="*70)
print("PREPARE FEATURES")
print("="*70)

# Separate X and y
X = df[numeric_cols + categorical_cols].copy()
y = df['Status'].copy()

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target distribution:")
print(f"  0 (No Default): {(y == 0).sum():,}")
print(f"  1 (Default): {(y == 1).sum():,}")

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

print(f"✓ Encoded {len(categorical_cols)} categorical columns")

# Handle outliers (clip at 1st and 99th percentiles)
for col in numeric_cols:
    q1 = X[col].quantile(0.01)
    q99 = X[col].quantile(0.99)
    X[col] = X[col].clip(q1, q99)

print(f"✓ Handled outliers")
print(f"\nFinal features: {X.columns.tolist()}")

# %% CELL 6: TRAIN-TEST SPLIT
print("\n" + "="*70)
print("TRAIN-TEST SPLIT")
print("="*70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"  Default rate: {y_train.mean():.2%}")
print(f"\nTest set: {X_test.shape}")
print(f"  Default rate: {y_test.mean():.2%}")

feature_names = X.columns.tolist()

# %% CELL 7: SCALE FEATURES
print("\n" + "="*70)
print("SCALE FEATURES")
print("="*70)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Features scaled (mean=0, std=1)")
print(f"Training set shape: {X_train_scaled.shape}")
print(f"Test set shape: {X_test_scaled.shape}")

# %% CELL 8: HANDLE CLASS IMBALANCE
print("\n" + "="*70)
print("HANDLE CLASS IMBALANCE (SMOTE)")
print("="*70)

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print(f"Before SMOTE:")
print(f"  Shape: {X_train_scaled.shape}")
print(f"  Defaults: {y_train.sum():,} ({y_train.mean():.2%})")

print(f"\nAfter SMOTE:")
print(f"  Shape: {X_train_balanced.shape}")
print(f"  Defaults: {y_train_balanced.sum():,} ({y_train_balanced.mean():.2%})")

# %% CELL 9: TRAIN XGBOOST
print("\n" + "="*70)
print("TRAIN XGBOOST MODEL")
print("="*70)

# Calculate scale_pos_weight for imbalance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss',
    verbosity=0
)

print(f"Training XGBoost with scale_pos_weight={scale_pos_weight:.2f}...")
xgb_model.fit(X_train_balanced, y_train_balanced)

print(f"✓ XGBoost model trained")
print(f"  Trees: {xgb_model.n_estimators}")
print(f"  Max depth: {xgb_model.max_depth}")

# %% CELL 10: EVALUATE MODEL
print("\n" + "="*70)
print("MODEL EVALUATION")
print("="*70)

# Training predictions
y_train_pred = xgb_model.predict(X_train_scaled)
y_train_pred_proba = xgb_model.predict_proba(X_train_scaled)[:, 1]

# Test predictions
y_test_pred = xgb_model.predict(X_test_scaled)
y_test_pred_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]

# Metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

train_f1 = f1_score(y_train, y_train_pred)
test_f1 = f1_score(y_test, y_test_pred)

train_auc = roc_auc_score(y_train, y_train_pred_proba)
test_auc = roc_auc_score(y_test, y_test_pred_proba)

print("Training Metrics:")
print(f"  Accuracy: {train_accuracy:.4f}")
print(f"  F1-Score: {train_f1:.4f}")
print(f"  AUC-ROC:  {train_auc:.4f}")

print("\nTest Metrics:")
print(f"  Accuracy: {test_accuracy:.4f}")
print(f"  F1-Score: {test_f1:.4f}")
print(f"  AUC-ROC:  {test_auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
print(f"\nConfusion Matrix:")
print(f"  True Negatives:  {cm[0,0]:,}")
print(f"  False Positives: {cm[0,1]:,}")
print(f"  False Negatives: {cm[1,0]:,}")
print(f"  True Positives:  {cm[1,1]:,}")

print(f"\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=['No Default', 'Default']))

# %% CELL 11: VISUALIZATIONS
print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=['No Default', 'Default'],
            yticklabels=['No Default', 'Default'], cbar=False)
axes[0, 0].set_title('Confusion Matrix', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Actual')
axes[0, 0].set_xlabel('Predicted')

# 2. Feature Importance
importances = xgb_model.feature_importances_
top_n = 15
top_idx = np.argsort(importances)[-top_n:]
axes[0, 1].barh(range(len(top_idx)), importances[top_idx], color='steelblue')
axes[0, 1].set_yticks(range(len(top_idx)))
axes[0, 1].set_yticklabels([feature_names[i] for i in top_idx], fontsize=9)
axes[0, 1].set_xlabel('Importance')
axes[0, 1].set_title(f'Top {top_n} Features', fontsize=12, fontweight='bold')

# 3. Default Distribution
default_counts = y.value_counts()
axes[1, 0].bar(['No Default', 'Default'], [default_counts[0], default_counts[1]], 
               color=['green', 'red'], alpha=0.7, edgecolor='black', linewidth=1.5)
axes[1, 0].set_title('Default Distribution', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Count')
for i, v in enumerate([default_counts[0], default_counts[1]]):
    axes[1, 0].text(i, v + 1000, f'{v:,}', ha='center', fontweight='bold')

# 4. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
axes[1, 1].plot(fpr, tpr, linewidth=2.5, label=f'AUC = {test_auc:.4f}', color='steelblue')
axes[1, 1].plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
axes[1, 1].set_xlabel('False Positive Rate')
axes[1, 1].set_ylabel('True Positive Rate')
axes[1, 1].set_title('ROC Curve', fontsize=12, fontweight='bold')
axes[1, 1].legend(loc='lower right')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/loan_default_results.png', dpi=100, bbox_inches='tight')
plt.show()
print("✓ Visualizations saved to: /mnt/user-data/outputs/loan_default_results.png")

# %% CELL 12: FEATURE IMPORTANCE
print("\n" + "="*70)
print("FEATURE IMPORTANCE")
print("="*70)

feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 Features:")
print(feature_importance.head(20).to_string(index=False))

# Save to CSV
output_dir = Path('/mnt/user-data/outputs')
output_dir.mkdir(exist_ok=True)
feature_importance.to_csv(output_dir / 'feature_importance.csv', index=False)
print(f"\n✓ Feature importance saved to: feature_importance.csv")

# %% CELL 13: SAVE MODELS
print("\n" + "="*70)
print("SAVING MODELS")
print("="*70)

# Save model
with open(output_dir / 'xgboost_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

# Save scaler
with open(output_dir / 'scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save feature names
with open(output_dir / 'feature_names.json', 'w') as f:
    json.dump(feature_names, f)

# Save metadata
metadata = {
    'test_auc': float(test_auc),
    'test_f1': float(test_f1),
    'test_accuracy': float(test_accuracy),
    'train_auc': float(train_auc),
    'train_f1': float(train_f1),
    'train_accuracy': float(train_accuracy),
    'train_samples': len(X_train),
    'test_samples': len(X_test),
    'num_features': len(feature_names),
    'default_rate': float(y.mean())
}
with open(output_dir / 'metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Models saved to {output_dir}:")
print(f"  - xgboost_model.pkl")
print(f"  - scaler.pkl")
print(f"  - feature_names.json")
print(f"  - metadata.json")

# %% CELL 14: PREDICTION FUNCTION
print("\n" + "="*70)
print("PREDICTION FUNCTION")
print("="*70)

class LoanDefaultPredictor:
    """Predict loan default probability"""
    
    def __init__(self, model_path, scaler_path, features_path):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        with open(features_path) as f:
            self.features = json.load(f)
    
    def predict(self, customer_dict):
        """
        Predict default probability
        
        Args:
            customer_dict: dict with feature values
            
        Returns:
            dict with risk_score, category, prediction
        """
        df = pd.DataFrame([customer_dict])
        
        # Add missing features with 0
        for feat in self.features:
            if feat not in df.columns:
                df[feat] = 0
        
        # Select features in correct order
        df = df[self.features]
        
        # Scale
        df_scaled = self.scaler.transform(df)
        
        # Predict
        risk_score = float(self.model.predict_proba(df_scaled)[0, 1])
        prediction = int(risk_score >= 0.5)
        
        # Categorize
        if risk_score < 0.3:
            category = 'Low'
        elif risk_score < 0.6:
            category = 'Medium'
        else:
            category = 'High'
        
        return {
            'risk_score': risk_score,
            'category': category,
            'prediction': 'Default' if prediction == 1 else 'No Default'
        }

# Initialize predictor
predictor = LoanDefaultPredictor(
    output_dir / 'xgboost_model.pkl',
    output_dir / 'scaler.pkl',
    output_dir / 'feature_names.json'
)

print("✓ Predictor initialized")

# %% CELL 15: TEST PREDICTIONS
print("\n" + "="*70)
print("TEST PREDICTIONS")
print("="*70)

# Use actual test samples
print("\nExample 1: Low-risk customer")
test_sample_1 = X_test.iloc[0].to_dict()
result_1 = predictor.predict(test_sample_1)
print(f"  Risk Score: {result_1['risk_score']:.2%}")
print(f"  Category: {result_1['category']}")
print(f"  Prediction: {result_1['prediction']}")

print("\nExample 2: Random customer")
test_sample_2 = X_test.iloc[100].to_dict()
result_2 = predictor.predict(test_sample_2)
print(f"  Risk Score: {result_2['risk_score']:.2%}")
print(f"  Category: {result_2['category']}")
print(f"  Prediction: {result_2['prediction']}")

print("\nExample 3: High-risk customer (if default exists in test set)")
default_indices = np.where(y_test == 1)[0]
if len(default_indices) > 0:
    high_risk_idx = default_indices[0]
    test_sample_3 = X_test.iloc[high_risk_idx].to_dict()
    result_3 = predictor.predict(test_sample_3)
    print(f"  Risk Score: {result_3['risk_score']:.2%}")
    print(f"  Category: {result_3['category']}")
    print(f"  Prediction: {result_3['prediction']}")

# %% CELL 16: SUMMARY REPORT
print("\n" + "="*70)
print("SELASTONE LOAN DEFAULT PREDICTION - COMPLETE")
print("="*70)

print(f"\n📊 MODEL PERFORMANCE:")
print(f"  Test Accuracy: {test_accuracy:.4f}")
print(f"  Test F1-Score: {test_f1:.4f}")
print(f"  Test AUC-ROC:  {test_auc:.4f}")

print(f"\n📁 DATASET INFO:")
print(f"  Total records: {len(df):,}")
print(f"  Default rate: {y.mean():.2%}")
print(f"  Training samples: {len(X_train):,}")
print(f"  Test samples: {len(X_test):,}")
print(f"  Total features: {len(feature_names)}")

print(f"\n📁 OUTPUT FILES:")
print(f"  Location: {output_dir}")
print(f"  - xgboost_model.pkl (trained model)")
print(f"  - scaler.pkl (feature scaler)")
print(f"  - feature_names.json (feature list)")
print(f"  - metadata.json (model metrics)")
print(f"  - feature_importance.csv (top features)")
print(f"  - loan_default_results.png (visualizations)")

print(f"\n✅ Model ready for deployment!")
print(f"\n🚀 To make predictions:")
print(f"   result = predictor.predict(customer_dict)")
print(f"   # Returns: risk_score, category, prediction")
