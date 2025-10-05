import pandas as pd
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import os
import torch
import numpy as np
from pathlib import Path


mtcnn = MTCNN(
  image_size=160, 
  margin=0, 
  keep_all=False,
  min_face_size=20,
)

resnet = InceptionResnetV1(pretrained='vggface2').eval()


def find_files(base: Path):
  return {
    folder: sorted((base / folder).glob("*"))
    for folder in ("me", "not_me")
  }
  
def crop_faces(output_dir=Path('data/cropped')):
  repo_root = Path(__file__).resolve().parents[2]
  data_dir = repo_root / "data"
  output_dir = Path(output_dir) if output_dir else data_dir / "cropped"  
  output_dir.mkdir(parents=True, exist_ok=True)
  files = find_files(data_dir)
  for label, paths in sorted(files.items()):
    save_path = os.path.join(output_dir, label)
    os.makedirs(save_path, exist_ok=True)
    print(f"Procesando carpeta: {label}")
    for file in paths:
      try:
        img = Image.open(file).convert("RGB")
      except Exception as e:
        print(f"No se pudo encontrar la imagen, {file}: {e}")
        continue
      face = mtcnn(img)
      if face is None:
        print(f"No se detecto un rostro en: {file}")
        continue

      # convertir el tensor a imagen
      face_img = Image.fromarray((face.permute(1, 2, 0).numpy() * 255).astype("uint8"))
      save_name = Path(file).stem + "_crop.jpg"      
      face_img.save(Path(save_path) / save_name)  
  print("✅ Recortes generados en data/cropped/")

if __name__ == "__main__":
  crop_faces()
