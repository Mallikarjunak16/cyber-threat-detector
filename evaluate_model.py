import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import os
import kagglehub

def final_evaluate():
    # Define paths
    dataset_path = kagglehub.dataset_download('mrwellsdavid/unsw-nb15')
    csv_path = f"{dataset_path}/UNSW_NB15_testing-set.csv"
    model_path = 'ensemble_model.pkl'
    scaler_path = 'scaler.pkl'
    encoder_path = 'label_encoders.pkl'

    print(f"Loading data and artifacts...")
    df = pd.read_csv(csv_path)
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoders = joblib.load(encoder_path)

    # 1. Feature Engineering: Log Transformation & Interaction Features
    skewed_features = ['dur', 'sbytes', 'dbytes', 'spkts', 'dpkts']
    for feat in skewed_features:
        if feat in df.columns:
            df[feat] = np.log1p(df[feat])
            
    if 'dur' in df.columns:
        df['byte_rate'] = (df['sbytes'] + df['dbytes']) / (df['dur'] + 1e-6)
        df['packet_density'] = (df['spkts'] + df['dpkts']) / (df['dur'] + 1e-6)

    # Save actual labels
    y_true = df['label'].values
    
    # Preprocess
    drop_cols = ['id', 'label', 'attack_cat']
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    # Encode categorical columns
    for col, le in label_encoders.items():
        if col in X.columns:
            X[col] = X[col].astype(str).map(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
            
    # Scale features
    X_scaled = scaler.transform(X)
    
    print("Running Optimized Ensemble Model Evaluation...")
    y_pred = model.predict(X_scaled)
    
    # Calculate Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # FPR
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn)

    # Output results
    print("\n" + "="*50)
    print("      OPTIMIZED ENSEMBLE EVALUATION REPORT")
    print("="*50)
    print(f"{'Metric':<25} | {'Value':<10}")
    print("-" * 40)
    print(f"{'Global Accuracy':<25} | {accuracy:.4%}")
    print(f"{'Precision':<25} | {precision:.4%}")
    print(f"{'Recall (Detection Rate)':<25} | {recall:.4%}")
    print(f"{'F1-Score':<25} | {f1:.4%}")
    print(f"{'False Positive Rate (FPR)':<25} | {fpr:.4%}")
    print("="*50)
    
    # Save results for app.py
    with open('model_stats.txt', 'w') as f:
        f.write(f"Accuracy: {accuracy:.4%}\n")
        f.write(f"Precision: {precision:.4%}\n")
        f.write(f"F1-Score: {f1:.4%}\n")
        f.write(f"FPR: {fpr:.4%}\n")
    
    print("\n[ CLASSIFICATION REPORT ]")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Attack']))

    if accuracy > 0.99:
        print("\n🏆 PLATINUM STANDARD: Accuracy > 99%. PS-AIML-02 Target Achieved.")
    elif accuracy > 0.95:
        print("\n🥇 GOLD STANDARD: High accuracy and robust performance.")

if __name__ == "__main__":
    final_evaluate()
