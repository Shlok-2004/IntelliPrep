import os
import cv2
import time
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model


# =========================================================
# LOAD MODELS ONCE (IMPORTANT FOR PERFORMANCE)
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "emotion_recognition_model.h5")
FACE_MODEL = os.path.join(BASE_DIR, "face_landmarker.task")

emotion_model = load_model(MODEL_PATH, compile=False)
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path=FACE_MODEL)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)

face_landmarker = vision.FaceLandmarker.create_from_options(options)


# =========================================================
# MAIN HR ANALYSIS FUNCTION
# =========================================================
def run_hr_video_analysis(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("Uploaded video could not be opened")

    emotion_count = {e: 0 for e in EMOTIONS}
    eye_contact_frames = 0
    total_frames = 0

    start_time = time.time()

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30  # fallback

    frame_count = 0
    processed_frames = 0
    MAX_PROCESSED_FRAMES = 120  # limit to 120 sampled frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Process only 1 frame per second
        if frame_count % fps != 0:
            continue

        processed_frames += 1
        total_frames += 1

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = face_landmarker.detect(mp_image)

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            xs = [int(l.x * w) for l in landmarks]
            ys = [int(l.y * h) for l in landmarks]

            x1, y1 = max(0, min(xs)), max(0, min(ys))
            x2, y2 = min(w, max(xs)), min(h, max(ys))

            face = frame[y1:y2, x1:x2]

            if face.size > 0:
                face = cv2.resize(face, (48, 48))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face = face.astype("float32") / 255.0
                face = face.reshape(1, 48, 48, 1)

                preds = emotion_model.predict(face, verbose=0)
                emotion = EMOTIONS[np.argmax(preds)]
                emotion_count[emotion] += 1

            # Eye contact approximation (nose centered)
            nose_x = landmarks[1].x
            if 0.45 < nose_x < 0.55:
                eye_contact_frames += 1

        # Stop early if enough frames processed
        if processed_frames >= MAX_PROCESSED_FRAMES:
            break

    cap.release()

    duration = time.time() - start_time

    # =========================================================
    # SCORING LOGIC
    # =========================================================
    eye_contact_score = (eye_contact_frames / max(1, total_frames)) * 100

    dominant_emotion = max(emotion_count, key=emotion_count.get)

    emotion_weights = {
        'Happy': 100,
        'Neutral': 85,
        'Surprise': 75,
        'Fear': 60,
        'Sad': 50,
        'Disgust': 40,
        'Angry': 30
    }

    total_emotion_frames = sum(emotion_count.values())
    emotion_score = 0

    if total_emotion_frames > 0:
        for emo, count in emotion_count.items():
            emotion_score += emotion_weights[emo] * (count / total_emotion_frames)
    else:
        emotion_score = 50

    emotion_changes = sum(1 for v in emotion_count.values() if v > 0)
    confidence_score = max(50, 100 - emotion_changes * 5)

    final_hr_score = (
        0.4 * emotion_score +
        0.35 * eye_contact_score +
        0.25 * confidence_score
    )

    return {
        "duration_sec": round(duration, 2),
        "eye_contact_score": round(eye_contact_score, 2),
        "emotion_score": round(emotion_score, 2),
        "confidence_score": round(confidence_score, 2),
        "final_hr_score": round(final_hr_score, 2),
        "dominant_emotion": dominant_emotion,
        "emotion_distribution": emotion_count
    }
