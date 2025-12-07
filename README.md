# ESP3 Image Verificator

Descripción
- Servicio de verificación facial binaria ("me" vs "not_me") basado en facenet-pytorch (MTCNN + InceptionResnetV1) y un clasificador (Logistic Regression).
- Provee endpoint HTTP POST `/verify` que recibe una imagen (multipart/form-data, campo `image`) y devuelve etiqueta y probabilidad.

Requisitos
- Python 3.9+ (recomendado)
- Recomendado usar virtualenv/pyenv
- Dependencias principales: torch, facenet-pytorch, flask, flask-cors, scikit-learn, joblib, pillow

Instalación (desde la raíz del repo)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Ejecutar en desarrollo (Flask)
```bash
# desde la raíz del repositorio
export FLASK_APP=src.main:app
flask run --host=0.0.0.0 --port=33211
```

Ejecutar en producción (Gunicorn)
```bash
# desde la raíz del repositorio
gunicorn -w 2 -b 0.0.0.0:33211 src.main:app --reload
```

Ejemplo de petición
- curl
```bash
curl -X POST http://localhost:33211/verify -F "image=@/ruta/imagen.jpg"
```

Privacidad y ética (resumen)
- No almacenar imágenes sensibles sin consentimiento.
- Limitar acceso al servicio en producción (autenticación, HTTPS).
- Documentar tasa de falsos positivos/negativos y posibles sesgos en los datos.

Estructura del repo (resumen)
- src/ : código fuente (main.py, scripts, train.py)
- data/ : datos originales
- data/cropped/ : imágenes recortadas por etiqueta
- models/ : modelo y scaler serializados
- requirements.txt


Mejoras sugeridas
- Añadir autenticación y límites de tasa.
- Persistencia segura de logs y trazabilidad de predicciones.
- Pruebas unitarias y CI.
- Soporte batch y endpoints asincrónicos para throughput.

