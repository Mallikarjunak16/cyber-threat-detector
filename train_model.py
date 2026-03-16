import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import os
import kagglehub

def preprocess_data(df, label_encoders=None, is_training=True):
    df = df.copy()
    
    # 1. Feature Engineering: Log Transformation
    skewed_features = ['dur', 'sbytes', 'dbytes', 'spkts', 'dpkts']
    for feat in skewed_features:
        if feat in df.columns:
            df[feat] = np.log1p(df[feat])
            
    # 2. Feature Engineering: Interaction Features (User Specified Formulas)
    # dur is the duration column in UNSW-NB15
    if 'dur' in df.columns:
        df['byte_rate'] = df['sbytes'] / (df['dur'] + 0.00001)
        df['packet_density'] = df['spkts'] / (df['dur'] + 0.00001)
    
    # Drop unnecessary columns
    drop_cols = ['id', 'label', 'attack_cat']
    y = df['label'] if 'label' in df.columns else None
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    # Encode categorical columns
    categorical_cols = ['proto', 'service', 'state']
    if is_training:
        label_encoders = {}
        for col in categorical_cols:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
        return X, y, label_encoders
    else:
        for col, le in label_encoders.items():
            if col in X.columns:
                X[col] = X[col].astype(str).map(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
        return X, y

def train_and_save_model():
    dataset_path = kagglehub.dataset_download('mrwellsdavid/unsw-nb15')
    csv_path = f"{dataset_path}/UNSW_NB15_testing-set.csv"
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    X, y, label_encoders = preprocess_data(df, is_training=True)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split for meta-model training
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 3. Class Balancing with SMOTE
    print("Applying SMOTE for class balancing...")
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    # 4. Build Optimized Stacking Ensemble Classifier
    print("Building High-Performance Stacking Ensemble...")
    # Increased depth and estimators for 99% accuracy target
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1)),
        ('xgb', XGBClassifier(n_estimators=200, max_depth=10, learning_rate=0.1, random_state=42, eval_metric='logloss')),
        ('lgbm', LGBMClassifier(n_estimators=200, max_depth=10, learning_rate=0.1, random_state=42, verbose=-1))
    ]
    
    ensemble_model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=3,
        n_jobs=-1
    )
    
    print("Training Final Optimized Ensemble Model...")
    ensemble_model.fit(X_train_sm, y_train_sm)
    
    # 5. Train Isolation Forest (Unsupervised)
    print("Training Isolation Forest...")
    iso_model = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    iso_model.fit(X_scaled)
    
    # Save artifacts
    joblib.dump(ensemble_model, 'ensemble_model.pkl')
    joblib.dump(iso_model, 'iso_forest.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    
    # Save accuracy for HUD badge
    test_accuracy = ensemble_model.score(X_test, y_test)
    print(f"Optimized Ensemble Accuracy: {test_accuracy:.4%}")
    with open('model_accuracy.txt', 'w') as f:
        f.write(f"{test_accuracy:.4%}")
    
    print("World-class optimized ensemble model saved successfully.")

if __name__ == "__main__":
    train_and_save_model()
