import numpy as np
import pandas as pd
import os
import json
from pandas import DataFrame
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

# load embeddings
def load_data(emb_path='embeddings.npy', labels_path='labels.csv'):
  """Load previous embeddings generated"""
  embeddings = np.load(emb_path) 
  labels_df = pd.read_csv(labels_path)
  if "index" in labels_df.columns:
    labels_df = labels_df.sort_values('index')
  return embeddings, labels_df

def labels_to_binary(labels_df: DataFrame):
  if "label" not in labels_df.columns:
    raise ValueError("labels.csv debe tener columa label")
  uniques = labels_df['label'].unique().tolist()
  if "me" in uniques:
    pos_label = 'me'
  else: 
    pos_label = uniques[0]
  y = (labels_df['label'] == pos_label).astype(int).to_numpy()
  return y, pos_label

def train() -> None:
  X, labels_df = load_data()
  y, pos_label = labels_to_binary(labels_df)

  X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
  )

  # Scaler para ajustar solo con train
  scaler = StandardScaler()
  scaler.fit(X_train)
  X_train_s = scaler.transform(X_train)
  X_val_s = scaler.transform(X_val)

  # Modelo logistic regression
  model = LogisticRegression(max_iter=200, solver='lbfgs')
  model.fit(X_train_s, y_train)

  y_pred = model.predict(X_val_s)
  y_proba = model.predict_proba(X_val_s)[:, 1]

  acc = float(accuracy_score(y_val, y_pred))
  auc = float(roc_auc_score(y_val, y_proba))

  tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
  class_report = classification_report(y_val, y_pred, output_dict=True)

  metrics = {
    "n_samples": int(X.shape[0]),
    "n_train": int(X_train.shape[0]),
    "n_val": int(X_val.shape[0]),
    "accuracy": acc,
    "roc_auc": auc,
    "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    "classification_report": class_report
  }

  dump(model, os.path.join("models", "model.joblib"))
  dump(scaler, os.path.join("models", "sclaer.joblib"))

  with open(os.path.join("models", "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

  print("Entrenamiento terminado")
  print(f"Accuracy: {acc:.4f}, ROC AUC: {auc:.4f}")
  print("Modelos guardados en: models")

if __name__ == '__main__':
  train()