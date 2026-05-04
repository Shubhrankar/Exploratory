# Bengali to Hindi Translation Ensemble - Django App

This is a Django web application that integrates the fine-tuned machine translation models (NLLB-600M and IndicTrans2-320M) and uses COMET-Kiwi to perform a Quality Estimation-based ensemble, returning the best output.

## Setup on Lightning AI

1. Open your Lightning AI studio terminal.
2. Make sure you place the `nllb_finetuned_bn_hi` and `indictrans2_finetuned_bn_hi` folders inside the `mt_web_app` directory (next to `manage.py`). If not found, the app will fall back to downloading the base models from Hugging Face.
3. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
4. Authenticate with Hugging Face (required for COMET-Kiwi):
   ```bash
   huggingface-cli login
   ```
5. Run the Django migrations:
   ```bash
   python manage.py migrate
   ```
6. Start the server:
   ```bash
   python manage.py runserver 0.0.0.0:8000
   ```
7. Open the URL provided by Lightning AI to access the web app.

**Note:** The models are loaded lazily on the first translation request.
