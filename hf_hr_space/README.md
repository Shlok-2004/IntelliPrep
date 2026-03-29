---
title: IntelliPrep HR Analysis
emoji: 🎥
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

# IntelliPrep HR Video Analysis Microservice

A lightweight FastAPI microservice that processes HR interview video recordings for IntelliPrep.

## What it does

- Accepts `.mp4` video uploads via HTTP POST
- Runs MediaPipe face landmark detection
- Runs TFLite emotion recognition model
- Returns JSON scores: eye contact, emotion, confidence, final HR score

## API Endpoint

**POST** `/analyze`

- Body: `multipart/form-data` with field `video` (mp4 file)
- Returns: JSON with `final_hr_score`, `eye_contact_score`, `emotion_score`, `confidence_score`, `dominant_emotion`, `emotion_distribution`

## Models

Place the following model files in the root of this Space:
- `emotion_recognition_model.tflite`
- `face_landmarker.task`
