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
    
    X_train = pd.read_csv('diabetes_preprocessing/X_train.csv')
    X_test = pd.read_csv('diabetes_preprocessing/X_test.csv')
    y_train = pd.read_csv('diabetes_preprocessing/y_train.csv').values.ravel()
    y_test = pd.read_csv('diabetes_preprocessing/y_test.csv').values.ravel()
    
    print(f"Data loaded successfully!")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 60)
    print("Training Model - CI/CD Pipeline")
    print("=" * 60)

    mlflow.set_experiment("diabetes-lgbm-ci")
    
    # 🔑 Check if we're running inside MLflow Projects
    run_id = os.environ.get("MLFLOW_RUN_ID")
    
    if run_id:
        # ✅ Running via MLflow Projects - use existing run
        print(f"\n✓ Running inside MLflow Projects")
        print(f"  Using existing run ID: {run_id}")
        active_run = mlflow.active_run()
        if active_run is None:
            print("  ERROR: No active run found, but MLFLOW_RUN_ID is set")
            raise RuntimeError("Inconsistent MLflow state")
    else:
        # ✅ Running manually - create new run
        print(f"\n✓ Running manually (not via MLflow Projects)")
        print(f"  Creating new MLflow run...")
        mlflow.start_run()
        active_run = mlflow.active_run()
        run_id = active_run.info.run_id
        print(f"  Created run ID: {run_id}")

    # Enable autologging
    mlflow.lightgbm.autolog(log_models=True, log_input_examples=True)

    # Train model
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

    print("\nTraining LightGBM...")
    model.fit(X_train, y_train)
    print("Training completed!")

    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    # Log metrics
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall", recall)
    mlflow.log_metric("test_f1_score", f1)
    mlflow.log_metric("test_auc_roc", auc)

    # Log params
    mlflow.log_param("ci_pipeline", "github_actions")
    mlflow.log_param("model_type", "LightGBM")

    print(f"\n📊 Model Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")

    # Register model
    model_name = "diabetes-lgbm-ci-model"
    model_uri = f"runs:/{run_id}/model"

    print(f"\n📦 Registering model: {model_name}")
    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )
    print(f"  ✓ Model registered as version {registered_model.version}")

    # Save artifacts
    os.makedirs("artifacts", exist_ok=True)

    metrics_data = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc_roc": auc
    }

    model_info_data = {
        "model_name": model_name,
        "model_version": registered_model.version,
        "run_id": run_id,
        "model_uri": model_uri
    }

    with open("artifacts/metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=2)

    with open("artifacts/model_info.json", "w") as f:
        json.dump(model_info_data, f, indent=2)

    print(f"\n💾 Artifacts saved:")
    print(f"  - artifacts/metrics.json")
    print(f"  - artifacts/model_info.json")

    # Don't end the run here - let MLflow Projects handle it
    # If we started it manually, we'll end it in main()
    
    return model, metrics_data, model_name, registered_model.version
  

def main():
    """Main CI/CD training pipeline"""
    print("\n" + "=" * 60)
    print("CI/CD PIPELINE - MODEL TRAINING")
    print("=" * 60)
    print("Project: Diabetes Prediction with LightGBM")
    print("Pipeline: GitHub Actions")
    print("=" * 60)
    
    # Check if running via MLflow Projects
    is_mlflow_projects = os.environ.get("MLFLOW_RUN_ID") is not None
    
    try:
        # Load data
        X_train, X_test, y_train, y_test = load_data()
        
        # Train model
        model, metrics, model_name, model_version = train_model(
            X_train, X_test, y_train, y_test
        )
        
        print("\n" + "=" * 60)
        print("✅ CI/CD PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\n📋 Summary:")
        print(f"  Model Name:    {model_name}")
        print(f"  Model Version: {model_version}")
        print(f"  Accuracy:      {metrics['accuracy']:.4f}")
        print(f"  F1 Score:      {metrics['f1_score']:.4f}")
        print(f"  AUC-ROC:       {metrics['auc_roc']:.4f}")
        print("\n📁 Artifacts:")
        print(f"  - Saved to artifacts/ directory")
        print(f"  - MLflow run logged successfully")
        print(f"  - Ready for Docker build")
        print("=" * 60)
        
        # End run if we started it manually
        if not is_mlflow_projects and mlflow.active_run():
            mlflow.end_run()
            print("\n✓ MLflow run ended")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # End run on error if we started it manually
        if not is_mlflow_projects and mlflow.active_run():
            mlflow.end_run(status="FAILED")
        
        return 1


if __name__ == "__main__":
    exit(main())
