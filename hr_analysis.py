import os
import time
import numpy as np

# Suppress MediaPipe/TF C++ logs
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('absl').setLevel('ERROR')


# =========================================================
# LOAD MODELS ONCE (IMPORTANT FOR PERFORMANCE)
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TFLITE_MODEL_PATH = os.path.join(BASE_DIR, "emotion_recognition_model.tflite")
FACE_MODEL = os.path.join(BASE_DIR, "face_landmarker.task")

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Lazy-loaded globals
_emotion_interpreter = None
_face_landmarker = None

def _get_emotion_interpreter():
    global _emotion_interpreter
    if _emotion_interpreter is None:
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            import tensorflow.lite as tflite
        
        _emotion_interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)
        _emotion_interpreter.allocate_tensors()
    return _emotion_interpreter

def _get_face_landmarker():
    global _face_landmarker
    if _face_landmarker is None:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=FACE_MODEL),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )
        _face_landmarker = vision.FaceLandmarker.create_from_options(options)
    return _face_landmarker

def cleanup_hr_models():
    global _emotion_interpreter, _face_landmarker
    if _face_landmarker is not None:
        try:
            _face_landmarker.close()
        except Exception:
            pass
    _emotion_interpreter = None
    _face_landmarker = None
    import gc
    gc.collect()


# =========================================================
# MAIN HR ANALYSIS FUNCTION
# =========================================================
def run_hr_video_analysis(video_path):
    import mediapipe as mp
    import cv2

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
    MAX_PROCESSED_FRAMES = 15  # STRICT LIMIT to prevent Render 120s Worker Timeout

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Render Free Tier optimization: Process only 1 frame every 2 seconds
        if frame_count % (fps * 2) != 0:
            continue

        processed_frames += 1
        total_frames += 1

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = _get_face_landmarker().detect(mp_image)

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

                interpreter = _get_emotion_interpreter()
                input_idx = interpreter.get_input_details()[0]['index']
                output_idx = interpreter.get_output_details()[0]['index']
                
                interpreter.set_tensor(input_idx, face)
                interpreter.invoke()
                preds = interpreter.get_tensor(output_idx)
                
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
