"""FastAPI service for frame-by-frame OpenCV watermark inpainting."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI(title="GeminiClean API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    # Allow all origins and methods so the frontend works both via local file:// and HTTP
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

HTML_FILE = Path(__file__).parent / "index.html"

@app.get("/")
@app.get("/index.html")
async def serve_index():
    if HTML_FILE.exists():
        return FileResponse(HTML_FILE, media_type="text/html")
    raise HTTPException(status_code=404, detail="index.html not found")

VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".avi", ".mkv"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def cleanup(path: Path) -> None:
    path.unlink(missing_ok=True)


def watermark_boxes(width: int, height: int, position: str) -> list[tuple[int, int, int, int]]:
    """Return conservative corner boxes, including the visible Gemini sparkle."""
    side = min(width, height)
    size = max(96, round(side * 0.18))
    margin = max(12, round(side * 0.02))
    options = {
        "bottom-right": [(width - size - margin, height - size - margin)],
        "bottom-left": [(margin, height - size - margin)],
        "top-right": [(width - size - margin, margin)],
        "top-left": [(margin, margin)],
        "all-corners": [
            (width - size - margin, height - size - margin),
            (margin, height - size - margin),
            (width - size - margin, margin),
            (margin, margin),
        ],
    }
    if position not in options:
        raise HTTPException(status_code=422, detail="Unsupported watermark position.")
    return [(max(0, x), max(0, y), size, size) for x, y in options[position]]


def inpaint_frame(frame: np.ndarray, position: str, radius: int) -> np.ndarray:
    height, width = frame.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    for x, y, box_width, box_height in watermark_boxes(width, height, position):
        mask[y : min(height, y + box_height), x : min(width, x + box_width)] = 255
    return cv2.inpaint(frame, mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)


def get_ffmpeg_binary() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg") or "ffmpeg"


def mux_audio(silent_video: Path, original: Path, output: Path) -> bool:
    """Encode H.264 video and copy an optional original audio stream."""
    ffmpeg_bin = get_ffmpeg_binary()
    command = [
        ffmpeg_bin, "-y", "-i", str(silent_video), "-i", str(original),
        "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(output),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


@app.post("/api/remove-watermark")
async def remove_watermark(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    position: str = Form("bottom-right"),
    inpaint_radius: int = Form(3),
):
    suffix = Path(file.filename or "upload").suffix.lower()
    if suffix not in VIDEO_EXTENSIONS | IMAGE_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Supported types: PNG, JPG, WebP, MP4, WebM, MOV, AVI, MKV.")
    if not 1 <= inpaint_radius <= 15:
        raise HTTPException(status_code=422, detail="Inpaint radius must be between 1 and 15.")

    job_dir = Path(tempfile.mkdtemp(prefix="gemini-clean-"))
    source = job_dir / f"source{suffix}"
    with source.open("wb") as target:
        shutil.copyfileobj(file.file, target)

    safe_stem = Path(file.filename or "media").stem.replace("/", "_").replace("\\", "_")
    output = job_dir / f"cleaned-{safe_stem}{'.mp4' if suffix in VIDEO_EXTENSIONS else '.png'}"

    try:
        if suffix in IMAGE_EXTENSIONS:
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if image is None:
                raise HTTPException(status_code=400, detail="The image could not be decoded.")
            if not cv2.imwrite(str(output), inpaint_frame(image, position, inpaint_radius)):
                raise RuntimeError("Could not write cleaned image.")
            media_type = "image/png"
        else:
            capture = cv2.VideoCapture(str(source))
            fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if not capture.isOpened() or width < 1 or height < 1:
                raise HTTPException(status_code=400, detail="The video could not be decoded.")
            silent = job_dir / "inpainted-video.mp4"
            writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError("Could not create video encoder.")
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                writer.write(inpaint_frame(frame, position, inpaint_radius))
            capture.release()
            writer.release()
            if not mux_audio(silent, source, output):
                shutil.move(str(silent), str(output))
            media_type = "video/mp4"
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise

    background_tasks.add_task(shutil.rmtree, job_dir, True)
    return FileResponse(output, media_type=media_type, filename=output.name, background=background_tasks)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8080)

