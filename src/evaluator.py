import joblib
import numpy as np
import pandas as pd
import os
import json
from sklearn.metrics import (
  confusion_matrix, roc_curve, 
  precision_recall_curve, auc
)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas import DataFrame

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

def evaluate():
  # load data
  model = joblib.load('models/model.joblib')
  scaler = joblib.load('models/sclaer.joblib')
  X = np.load('embeddings.npy')
  df = pd.read_csv('labels.csv')
  y, pos_label = labels_to_binary(df)
  random_state = 42
  os.makedirs('reports', exist_ok=True)

  # dividir 
  _, X_val, _, y_val = train_test_split(
    X, y, test_size=0.2, random_state=random_state, stratify=y
  )

  X_val_scaled = scaler.transform(X_val)

  y_proba = model.predict_proba(X_val_scaled)[:, 1] 
  y_pred_default = (y_proba >= 0.5).astype(int)


  fpr, tpr, roc_thresholds = roc_curve(y_val, y_proba)
  roc_auc = auc(fpr, tpr)
  precision, recall, pr_thresholds = precision_recall_curve(y_val, y_proba)
  f1_scores = 2 * precision * recall / (precision + recall + 1e-6)
  best_idx = np.argmax(f1_scores)
  best_tau = pr_thresholds[best_idx]
  best_f1 = f1_scores[best_idx]

  cm = confusion_matrix(y_val, y_pred_default)

  results = {
    "roc_auc": float(roc_auc),
    "best_tau": float(best_tau),
    "best_f1": float(best_f1),
    "confusion_matrix": cm.tolist(),
  }

  os.makedirs("reports", exist_ok=True)

  with open(os.path.join("reports", "evaluation.json"), "w") as f:
    json.dump(results, f, indent=4)

  print(f"\nAUC: {roc_auc:.4f}")
  print(f"Mejor τ (según F1): {best_tau:.3f}")
  print(f"F1 máximo: {best_f1:.4f}")

  # Graficar
  plt.figure(figsize=(5, 5))
  plt.imshow(cm, cmap="Blues")
  plt.title("Matriz de confusión (τ=0.5)")
  plt.xlabel("Predicción")
  plt.ylabel("Real")
  for i in range(2):
      for j in range(2):
          plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
  plt.savefig(os.path.join("reports", "confusion_matrix.png"))
  plt.close()

  # Curva ROC
  plt.figure()
  plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
  plt.plot([0, 1], [0, 1], "k--")
  plt.xlabel("Tasa de falsos positivos")
  plt.ylabel("Tasa de verdaderos positivos")
  plt.title("Curva ROC")
  plt.legend()
  plt.savefig(os.path.join("reports", "roc_curve.png"))
  plt.close()

  # Curva Precision-Recall
  plt.figure()
  plt.plot(recall, precision, label=f"F1 max = {best_f1:.2f}")
  plt.xlabel("Recall")
  plt.ylabel("Precision")
  plt.title("Curva Precision-Recall")
  plt.legend()
  plt.savefig(os.path.join("reports", "precision_recall_curve.png"))
  plt.close()

  print("Resultados guardados en reports/")

if __name__ == '__main__':
  evaluate()