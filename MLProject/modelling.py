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

    # Set experiment (aman walau belum ada)
    mlflow.set_experiment("diabetes-lgbm-ci")

    # Aktifkan autolog
    mlflow.lightgbm.autolog(
        log_models=True,
        log_input_examples=True
    )
  
    with mlflow.start_run(nested=True, run_name="lgbm_ci_run") as run:
        print("\nMLflow tracking enabled (CI)")
        print(f"Run ID: {run.info.run_id}")

        # =========================
        # 1. Inisialisasi model
        # =========================
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

        # =========================
        # 2. Training
        # =========================
        print("\nTraining LightGBM...")
        model.fit(X_train, y_train)
        print("Training completed!")

        # =========================
        # 3. Evaluasi
        # =========================
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        # =========================
        # 4. Logging manual
        # =========================
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1_score", f1)
        mlflow.log_metric("test_auc_roc", auc)

        mlflow.log_param("ci_pipeline", "github_actions")
        mlflow.log_param("model_type", "LightGBM")

        # =========================
        # 5. Register model
        # =========================
        run_id = run.info.run_id
        artifact_uri = run.info.artifact_uri

        model_name = "diabetes-lgbm-ci-model"
        model_uri = f"runs:/{run_id}/model"

        print(f"\nRegistering model: {model_name}")
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )

        print("Model registered successfully!")
        print(f"  Version: {registered_model.version}")

        # =========================
        # 6. Simpan info ke artifacts/
        # =========================
        os.makedirs("artifacts", exist_ok=True)

        model_info = {
            "model_name": model_name,
            "model_version": registered_model.version,
            "run_id": run_id,
            "artifact_uri": artifact_uri,
            "model_uri": model_uri
        }

        with open("artifacts/model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)

        print("Model info saved to artifacts/model_info.json")

        return model, model_info, model_name, registered_model.version
  

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
