"""
IntelliPrep HR Analysis — FastAPI Microservice
Runs on Hugging Face Spaces (Docker)
"""
import os
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

from hr_analysis import run_hr_video_analysis

app = FastAPI(
    title="IntelliPrep HR Analysis API",
    description="Analyzes HR interview videos for emotion, eye contact, and confidence.",
    version="1.0.0"
)


@app.get("/")
def root():
    return {"status": "ok", "message": "IntelliPrep HR Analysis Service is running."}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/analyze")
async def analyze_video(video: UploadFile = File(...)):
    """
    Accepts an mp4 video upload and returns HR analysis scores.
    """
    if not video.filename.endswith((".mp4", ".webm", ".avi", ".mov")):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload an mp4, webm, avi, or mov file."
        )

    # Save video to a temp file for OpenCV
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        contents = await video.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result = run_hr_video_analysis(tmp_path)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    # HuggingFace Spaces expects the app to listen on port 7860
    uvicorn.run(app, host="0.0.0.0", port=7860)
