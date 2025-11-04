import pandas as pd
import numpy as np
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from pathlib import Path
import torch
from torchvision import transforms

resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Transform para normalizar las imágenes como espera el modelo
transform = transforms.Compose([
  transforms.Resize((160, 160)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def find_files(base: Path):
  return {
    folder: sorted((base / folder).glob("*"))
    for folder in ("me", "not_me")
  }

def make_embeddings():
  repo_root = Path(__file__).resolve().parents[2]
  data_dir = repo_root / "data" / "cropped"
  files = find_files(data_dir)
  embeddings = []
  labels = []
  
  skipped = 0
  processed = 0

  for label, paths in files.items():
    print(f"Procesando carpeta '{label}': {len(paths)} imágenes")
    for file in paths:
      try:
        img = Image.open(file).convert("RGB")
        # Aplicar transformaciones 
        face_tensor = transform(img).unsqueeze(0)
        
        # Generar embedding directamente
        with torch.no_grad():
          emb = resnet(face_tensor).detach().numpy().flatten()
        
        embeddings.append(emb)
        labels.append({"index": len(embeddings) - 1, "filename": str(file), "label": label})
        processed += 1
      except Exception as e:
        print(f"  ✗ Error procesando {file.name}: {e}")
        skipped += 1
        continue

  print(f"\nProcesadas: {processed} imágenes")
  if skipped > 0:
    print(f"Omitidas: {skipped} imágenes")

  np.save("embeddings.npy", np.array(embeddings))
  pd.DataFrame(labels).to_csv("labels.csv", index=False)
  print(f"Embeddings guardados: {len(embeddings)} vectores")
  print(f"   - 'me': {sum(1 for label in labels if label['label'] == 'me')}")
  print(f"   - 'not_me': {sum(1 for label in labels if label['label'] == 'not_me')}")

if __name__ == '__main__':
  make_embeddings()