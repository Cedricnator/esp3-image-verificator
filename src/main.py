from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from flask import Flask, request
from PIL import Image, UnidentifiedImageError

torch.set_grad_enabled(False)

app = Flask(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mtcnn = MTCNN(
  image_size=160,
  margin=0,
  keep_all=False,
  min_face_size=20,
  device=DEVICE,
)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)

model = joblib.load(MODELS_DIR / "model.joblib")
scaler = joblib.load(MODELS_DIR / "sclaer.joblib")

DEFAULT_THRESHOLD = 0.5


@app.get("/")
def root() -> Dict[str, str]:
  return {"message": "Face verification service is running"}


@app.get("/health")
def health() -> Dict[str, str]:
  return {"status": "ok"}


def extract_embedding(image: Image.Image) -> Tuple[np.ndarray | None, str | None]:
  face = mtcnn(image)
  if face is None:
    return None, "No face detected in the image"

  with torch.no_grad():
    embedding_tensor = resnet(face.unsqueeze(0).to(DEVICE))

  embedding = embedding_tensor.cpu().numpy().reshape(1, -1)
  return embedding, None


def predict_label(embedding: np.ndarray) -> Dict[str, Any]:
  embedding_scaled = scaler.transform(embedding)
  proba = float(model.predict_proba(embedding_scaled)[0, 1])
  label = "me" if proba >= DEFAULT_THRESHOLD else "not_me"
  return {
    "label": label,
    "probability": proba,
    "threshold": DEFAULT_THRESHOLD,
  }


@app.post("/verify")
def verify() -> Tuple[Any, int] | Dict[str, Any]:
  if "image" not in request.files:
    return {"error": "Missing 'image' file in form-data"}, 400

  file_storage = request.files["image"]
  if file_storage.filename == "":
    return {"error": "Empty filename"}, 400

  try:
    image = Image.open(file_storage.stream).convert("RGB")
  except UnidentifiedImageError:
    return {"error": "Invalid image file"}, 400

  embedding, error = extract_embedding(image)
  if embedding is None:
    return {"error": error}, 422

  result = predict_label(embedding)
  return result, 200