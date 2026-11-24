import base64, os, threading
import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, jsonify, render_template, request
from tensorflow.keras.models import load_model

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL = load_model(os.path.join(ROOT, "fer2013_mini_XCEPTION.102-0.66.hdf5"), compile=False)
LABELS = {0:"angry", 1:"disgust", 2:"fear", 3:"happy", 4:"sad", 5:"surprise", 6:"neutral"}
EMOJI = {"angry":"😠", "disgust":"🤢", "fear":"😨", "happy":"😊", "sad":"😢", "surprise":"😲", "neutral":"😐"}
mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
    min_detection_confidence=0.5, min_tracking_confidence=0.5)
lock = threading.Lock()
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

@app.get("/")
def home():
    return render_template("index.html")

@app.get("/health")
def health():
    return jsonify(status="ok", model_loaded=MODEL is not None)

@app.post("/api/predict")
def predict():
    data = request.get_json(silent=True) or {}
    image = data.get("image", "")
    try:
        raw = base64.b64decode(image.split(",", 1)[1], validate=True)
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    except (ValueError, TypeError, IndexError):
        return jsonify(error="Invalid webcam frame"), 400
    if frame is None:
        return jsonify(error="Could not decode webcam frame"), 400
    h, w = frame.shape[:2]
    if w > 1280 or h > 960:
        return jsonify(error="Camera frame is too large"), 400
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with lock:
        result = mesh.process(rgb)
        if not result.multi_face_landmarks:
            return jsonify(face_detected=False, message="No face detected")
        xs = [int(p.x*w) for p in result.multi_face_landmarks[0].landmark]
        ys = [int(p.y*h) for p in result.multi_face_landmarks[0].landmark]
        x1, y1 = max(0, min(xs)-20), max(0, min(ys)-20)
        x2, y2 = min(w, max(xs)+20), min(h, max(ys)+20)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return jsonify(face_detected=False, message="Face area unavailable")
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64))
        model_input = np.reshape(resized / 255.0, (1, 64, 64, 1))
        probabilities = MODEL.predict(model_input, verbose=0)[0]
    index = int(np.argmax(probabilities))
    emotion = LABELS[index]
    return jsonify(face_detected=True, emotion=emotion, emoji=EMOJI[emotion],
        confidence=round(float(probabilities[index])*100, 1),
        scores={LABELS[i]:round(float(probabilities[i])*100, 1) for i in LABELS})

@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Frame too large"), 413

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
