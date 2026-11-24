const video = document.querySelector("#video");
const canvas = document.querySelector("#capture");
const ctx = canvas.getContext("2d");
const start = document.querySelector("#start");
const stop = document.querySelector("#stop");
const placeholder = document.querySelector("#placeholder");
const indicator = document.querySelector("header small") || document.querySelector("#model-indicator");

let stream = null;
let running = false;
let busy = false;
let timer = null;

let blazefaceModel = null;
let emotionModel = null;
let clientModelReady = false;

const names = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"];
const emojiMap = {
  angry: "😠",
  disgust: "🤢",
  fear: "😨",
  happy: "😊",
  sad: "😢",
  surprise: "😲",
  neutral: "😐"
};

// Hidden 64x64 canvas for emotion model input
const faceCanvas = document.createElement("canvas");
faceCanvas.width = 64;
faceCanvas.height = 64;
const faceCtx = faceCanvas.getContext("2d", { willReadFrequently: true });

function scores(data = {}) {
  document.querySelector("#scores").innerHTML = names
    .map(
      n =>
        `<div class="score"><span>${n}</span><div class="track"><i style="width:${
          data[n] || 0
        }%"></i></div><b>${data[n] === undefined ? "—" : data[n].toFixed(1) + "%"}</b></div>`
    )
    .join("");
}

function noFace(msg = "Move into the camera frame") {
  document.querySelector("#emoji").textContent = "🔎";
  document.querySelector("#emotion").textContent = "No face";
  document.querySelector("#hint").textContent = msg;
  document.querySelector("#confidence").textContent = "—";
  document.querySelector("#bar").style.width = "0";
  scores();
}

async function initModels() {
  if (indicator) indicator.textContent = "● LOADING AI MODEL...";
  try {
    if (typeof tf !== "undefined" && typeof blazeface !== "undefined") {
      const [bf, em] = await Promise.all([
        blazeface.load(),
        tf.loadLayersModel("static/model/model.json")
      ]);
      blazefaceModel = bf;
      emotionModel = em;

      // Warmup inference
      tf.tidy(() => {
        const dummy = tf.zeros([1, 64, 64, 1]);
        em.predict(dummy);
      });

      clientModelReady = true;
      if (indicator) indicator.textContent = "● AI MODEL READY";
      console.log("Client-side models initialized successfully.");
      return;
    }
  } catch (err) {
    console.warn("Client model initialization error:", err);
  }
  if (indicator) indicator.textContent = "● SERVER MODE READY";
}

async function analyse() {
  if (!running || busy || video.readyState < 2) return;
  busy = true;
  try {
    if (clientModelReady && blazefaceModel && emotionModel) {
      // Direct in-browser inference using BlazeFace and TensorFlow.js
      const predictions = await blazefaceModel.estimateFaces(video, false);
      if (!predictions || predictions.length === 0) {
        noFace("Move into the camera frame");
        return;
      }

      const face = predictions[0];
      const [x1_raw, y1_raw] = face.topLeft;
      const [x2_raw, y2_raw] = face.bottomRight;
      const rawW = x2_raw - x1_raw;
      const rawH = y2_raw - y1_raw;

      const padX = rawW * 0.2;
      const padY = rawH * 0.2;
      const x1 = Math.max(0, x1_raw - padX);
      const y1 = Math.max(0, y1_raw - padY);
      const x2 = Math.min(video.videoWidth, x2_raw + padX);
      const y2 = Math.min(video.videoHeight, y2_raw + padY);
      const cropW = x2 - x1;
      const cropH = y2 - y1;

      if (cropW <= 0 || cropH <= 0) {
        noFace("Face area unavailable");
        return;
      }

      // Crop face onto 64x64 canvas
      faceCtx.drawImage(video, x1, y1, cropW, cropH, 0, 0, 64, 64);
      const imgData = faceCtx.getImageData(0, 0, 64, 64);
      const pixels = imgData.data;

      // Convert RGB to grayscale and normalize [0, 1]
      const gray = new Float32Array(64 * 64);
      for (let i = 0; i < 64 * 64; i++) {
        const offset = i * 4;
        gray[i] =
          (0.299 * pixels[offset] +
            0.587 * pixels[offset + 1] +
            0.114 * pixels[offset + 2]) /
          255.0;
      }

      // Run prediction
      const probs = tf.tidy(() => {
        const tensor = tf.tensor4d(gray, [1, 64, 64, 1]);
        return emotionModel.predict(tensor).dataSync();
      });

      let bestIdx = 0;
      let bestProb = probs[0];
      for (let i = 1; i < probs.length; i++) {
        if (probs[i] > bestProb) {
          bestProb = probs[i];
          bestIdx = i;
        }
      }

      const emotion = names[bestIdx];
      const emoji = emojiMap[emotion] || "😐";
      const confidence = Math.round(bestProb * 1000) / 10;
      const scoresObj = {};
      names.forEach((name, i) => {
        scoresObj[name] = Math.round(probs[i] * 1000) / 10;
      });

      document.querySelector("#emoji").textContent = emoji;
      document.querySelector("#emotion").textContent = emotion;
      document.querySelector("#hint").textContent = "Expression classification result";
      document.querySelector("#confidence").textContent = confidence + "%";
      document.querySelector("#bar").style.width = confidence + "%";
      scores(scoresObj);
    } else {
      // Fallback to server API /api/predict
      const w = 480;
      const h = Math.round((w * video.videoHeight) / video.videoWidth);
      canvas.width = w;
      canvas.height = h;
      ctx.drawImage(video, 0, 0, w, h);
      const r = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: canvas.toDataURL("image/jpeg", 0.68) })
      });
      const d = await r.json();
      if (!r.ok) throw Error(d.error || "Prediction failed");
      if (!d.face_detected) {
        noFace(d.message);
        return;
      }
      document.querySelector("#emoji").textContent = d.emoji;
      document.querySelector("#emotion").textContent = d.emotion;
      document.querySelector("#hint").textContent = "Expression classification result";
      document.querySelector("#confidence").textContent = d.confidence + "%";
      document.querySelector("#bar").style.width = d.confidence + "%";
      scores(d.scores);
    }
  } catch (e) {
    document.querySelector("#hint").textContent = e.message;
  } finally {
    busy = false;
  }
}

start.onclick = async () => {
  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw Error("Camera requires HTTPS or localhost.");
    }
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } }
    });
    video.srcObject = stream;
    await video.play();
    video.style.display = "block";
    placeholder.style.display = "none";
    running = true;
    start.disabled = true;
    stop.disabled = false;
    document.querySelector("#status").textContent = "● LIVE";
    analyse();
    timer = setInterval(analyse, 250);
  } catch (e) {
    document.querySelector("#hint").textContent =
      e.message || "Camera permission is required";
  }
};

stop.onclick = () => {
  running = false;
  clearInterval(timer);
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = null;
  video.srcObject = null;
  video.style.display = "none";
  placeholder.style.display = "flex";
  start.disabled = false;
  stop.disabled = true;
  document.querySelector("#status").textContent = "● STANDBY";
  document.querySelector("#emoji").textContent = "◎";
  document.querySelector("#emotion").textContent = "Waiting";
  document.querySelector("#hint").textContent = "Start your camera to see your result";
  document.querySelector("#confidence").textContent = "—";
  document.querySelector("#bar").style.width = "0";
  scores();
};

window.addEventListener("beforeunload", () => {
  if (stream) stream.getTracks().forEach(t => t.stop());
});

scores();
initModels();
