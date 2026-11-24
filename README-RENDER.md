# Render deployment overlay

Add these files to the root of the existing Emotion-recognition repository. Keep `emojitranslator.py` and `fer2013_mini_XCEPTION.102-0.66.hdf5` unchanged.

Render Web Service settings:
- Build: `pip install -r requirements.txt`
- Start: `gunicorn app:app --workers 1 --threads 1 --timeout 120`
- Health check: `/health`
- Python: 3.11.9
- Region: Singapore
- Recommended RAM: at least 2 GB (TensorFlow and MediaPipe are memory-intensive).

The browser obtains each visitor's webcam permission and sends reduced JPEG frames to the API. HTTPS is required for browser camera access; Render supplies HTTPS.

Local run: `pip install -r requirements.txt` then `python app.py`, and open `http://127.0.0.1:10000`.
