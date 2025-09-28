# Krishisamruddhi – Crop Disease Detection (Django + PyTorch)

An AI-powered web application to detect crop diseases from leaf images. Users can upload a photo of a crop leaf, and the app predicts the disease class along with a confidence score.

## Features
- Image upload interface for crop leaf analysis
- Server-side inference using a CNN (PyTorch)
- Top-1 predicted class and confidence display
- Simple, responsive UI with two pages: `home` and `detect`

## Tech Stack
- Backend: Django 5
- ML Inference: PyTorch, TorchVision, Pillow
- Frontend: Django Templates, CSS
- Storage: SQLite (development)
- Static files: WhiteNoise (configured)

## Dataset
- Training data: 20k Multi-Class Crop Disease Images
- Source: Kaggle — [`20k-multi-class-crop-disease-images`](https://www.kaggle.com/datasets/jawadali1045/20k-multi-class-crop-disease-images/data)

## Model
- Architecture: Simple CNN defined in `disease_detection/ml_model/model.py`
- Weights: `disease_detection/ml_model/model.pth`
- Input transforms: Resize to 64×64, convert to tensor
- Output classes: 42 crop health/disease categories (see `CLASS_NAMES` in `disease_detection/predictor.py`)
- Reported performance: ~80% accuracy (top-1)

## Project Structure
```
src/
  manage.py
  krishisamruddhi/
    settings.py
    urls.py
    wsgi.py
    asgi.py
  disease_detection/
    urls.py
    views.py
    predictor.py
    models.py
    admin.py
    ml_model/
      model.py
      model.pth
    templates/
      disease_detection/
        home.html
        detect.html
  static/
    css/style.css
    images/background.jpg
  media/            # uploaded images will be stored here
  requirements.txt
```

## Setup
1. Create and activate a virtual environment.
   - Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
   - macOS/Linux (bash):
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

2. Install dependencies:
   ```bash
   pip install -r src/requirements.txt
   ```

3. Apply migrations and collect static (optional for dev):
   ```bash
   python src/manage.py migrate
   python src/manage.py collectstatic --noinput
   ```

4. Run the development server:
   ```bash
   python src/manage.py runserver
   ```

5. Open the app at `http://127.0.0.1:8000/`.

## Usage
- Go to Home → "Crop Disease Detection".
- Upload a clear image of a crop leaf (JPG/PNG).
- Submit to view predicted disease and confidence.

## Key Endpoints
- `/` → Home
- `/detect/` → Image upload and prediction

## Configuration Notes
- Media (uploaded files): `MEDIA_ROOT` is `src/media/` and served in DEBUG mode.
- Static files: WhiteNoise configured; `STATICFILES_DIRS` includes `src/static/`, `STATIC_ROOT` is `src/staticfiles/`.
- Allowed hosts: `*` in development. Restrict in production.

## Development Tips
- The predictor loads the model on import for efficient inference: see `disease_detection/predictor.py`.
- Update or retrain the model by replacing `model.pth` and ensuring class order matches `CLASS_NAMES`.
- If running on CPU-only environments, current code maps tensors to CPU (`map_location='cpu'`).

## License
This project is for educational purposes. Check dataset licensing on Kaggle before commercial use.
