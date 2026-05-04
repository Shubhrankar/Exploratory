# Django NLP Lab Integration Setup

This folder contains the core files needed to integrate your fine-tuned NLLB and IndicTrans2 models into a modern Django web application.

## Files Included:
1. `services.py` - Contains the `TranslationService` Singleton. This prevents Out-Of-Memory (OOM) errors by ensuring the 1GB+ models are loaded into GPU memory exactly once on server startup, rather than every time someone requests a translation.
2. `views.py` - Contains the Django API endpoint (`/api/translate/`) and the view to render the main HTML page.
3. `urls.py` - Contains the routing paths for your views.
4. `index.html` - A premium, dark-mode frontend utilizing glassmorphism UI principles.

## How to use this on Lightning AI:

### 1. Scaffold a fresh Django project (if you don't have one):
Run this in your terminal:
```bash
pip install django
django-admin startproject nlp_lab
cd nlp_lab
python manage.py startapp mt_app
```

### 2. Move the provided files into the `mt_app` folder:
Copy the files I provided into your new `mt_app` directory:
- `services.py` -> `nlp_lab/mt_app/services.py`
- `views.py` -> `nlp_lab/mt_app/views.py`
- `urls.py` -> `nlp_lab/mt_app/urls.py`

Create a templates folder and move `index.html`:
- `mkdir -p nlp_lab/mt_app/templates/mt_app/`
- `index.html` -> `nlp_lab/mt_app/templates/mt_app/index.html`

### 3. Update the Model Paths:
In `services.py`, ensure `BASE_PATH` points to the directory where your `nllb_finetuned_hi_bn` and `indictrans2_finetuned_hi_bn` folders are stored on your Lightning AI workspace. 

### 4. Wire up Django Configuration:
In `nlp_lab/nlp_lab/settings.py`:
- Add `'mt_app'` to your `INSTALLED_APPS` list.

In `nlp_lab/nlp_lab/urls.py`:
- Add the app routes:
```python
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('mt_app.urls')),
]
```

### 5. Run the Server!
```bash
python manage.py runserver 0.0.0.0:8000
```
Then use Lightning AI's port forwarding/preview feature to open the website in your browser!
