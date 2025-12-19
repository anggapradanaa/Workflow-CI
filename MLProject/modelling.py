import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix)
import mlflow
import mlflow.lightgbm
import json
import os
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """Load preprocessed data"""
    print("=" * 60)
    print("CI/CD Pipeline - Loading Data")
    print("=" * 60)
    
    X_train = pd.read_csv('namadataset_preprocessing/X_train.csv')
    X_test = pd.read_csv('namadataset_preprocessing/X_test.csv')
    y_train = pd.read_csv('namadataset_preprocessing/y_train.csv').values.ravel()
    y_test = pd.read_csv('namadataset_preprocessing/y_test.csv').values.ravel()
    
    print(f"Data loaded successfully!")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, X_test, y_train, y_test):
    """Train LightGBM model"""
    print("\n" + "=" * 60)
    print("Training Model - CI/CD Pipeline")
    print("=" * 60)
    
    # Set experiment name
    mlflow.set_experiment("diabetes-lgbm-ci")
    
    # Enable autolog for automatic logging
    mlflow.lightgbm.autolog(log_models=True, log_input_examples=True)
    
    with mlflow.start_run(run_name="lgbm_ci_run") as run:
        print("\nMLflow tracking enabled (Local)")
        
        # Initialize model with optimized parameters for CI
        model = LGBMClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=20,
            min_child_samples=15,
            subsample=0.85,
            colsample_bytree=0.95,
            reg_alpha=0.15,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1
        )
        
        # Train model
        print("\nTraining LightGBM...")
        model.fit(X_train, y_train)
        print("Training completed!")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Log additional metrics
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1_score", f1)
        mlflow.log_metric("test_auc_roc", auc)
        
        # Log parameters
        mlflow.log_param("ci_pipeline", "github_actions")
        mlflow.log_param("model_type", "LightGBM")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        print("\n" + "=" * 60)
        print("Model Performance")
        print("=" * 60)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"AUC-ROC:   {auc:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"TN: {tn}  FP: {fp}")
        print(f"FN: {fn}  TP: {tp}")
        
        # Get run info
        run_id = run.info.run_id
        artifact_uri = run.info.artifact_uri
        
        print("\n" + "=" * 60)
        print("MLflow Tracking Info")
        print("=" * 60)
        print(f"Run ID: {run_id}")
        print(f"Artifact URI: {artifact_uri}")
        
        # Register model
        model_name = "diabetes-lgbm-ci-model"
        print(f"\nRegistering model: {model_name}")
        
        model_uri = f"runs:/{run_id}/model"
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )
        
        print(f"Model registered successfully!")
        print(f"  Name: {registered_model.name}")
        print(f"  Version: {registered_model.version}")
        
        # Save metrics to JSON for artifacts
        metrics_dict = {
            'run_id': run_id,
            'model_name': model_name,
            'model_version': registered_model.version,
            'metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'auc_roc': float(auc)
            },
            'confusion_matrix': {
                'true_negative': int(tn),
                'false_positive': int(fp),
                'false_negative': int(fn),
                'true_positive': int(tp)
            }
        }
        
        # Create artifacts directory
        os.makedirs('artifacts', exist_ok=True)
        
        # Save metrics
        with open('artifacts/metrics.json', 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        print("\nMetrics saved to artifacts/metrics.json")
        
        # Save model info
        model_info = {
            'model_name': model_name,
            'model_version': registered_model.version,
            'run_id': run_id,
            'artifact_uri': artifact_uri,
            'model_uri': model_uri
        }
        
        with open('artifacts/model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        print("Model info saved to artifacts/model_info.json")
        
        return model, metrics_dict, model_name, registered_model.version


def main():
    """Main CI/CD training pipeline"""
    print("\n" + "=" * 60)
    print("CI/CD PIPELINE - MODEL TRAINING")
    print("=" * 60)
    print("Project: Diabetes Prediction with LightGBM")
    print("Pipeline: GitHub Actions")
    print("=" * 60)
    
    try:
        # Load data
        X_train, X_test, y_train, y_test = load_data()
        
        # Train model
        model, metrics, model_name, model_version = train_model(
            X_train, X_test, y_train, y_test
        )
        
        print("\n" + "=" * 60)
        print("CI/CD PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nSummary:")
        print(f"  Model trained and registered")
        print(f"  Model name: {model_name}")
        print(f"  Model version: {model_version}")
        print(f"  Metrics saved to artifacts/")
        print(f"  Ready for Docker build")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())