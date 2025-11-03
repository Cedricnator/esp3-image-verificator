import pandas as pd
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
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

def make_embeddings():
  repo_root = Path(__file__).resolve().parents[2]
  data_dir = repo_root / "data" / "cropped"
  files = find_files(data_dir)
  embeddings = []
  labels = []

  for label, paths in files.items():
    for file in paths:
      img = Image.open(file).convert("RGB")
      face = mtcnn(img)
      if face is None:
          continue
      emb = resnet(face.unsqueeze(0)).detach().numpy().flatten()
      embeddings.append(emb)
      labels.append({"index": len(embeddings) - 1, "filename": str(file), "label": label})

  np.save("embeddings.npy", np.array(embeddings))
  pd.DataFrame(labels).to_csv("labels.csv", index=False)

if __name__ == '__main__':
  make_embeddings()