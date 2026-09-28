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


def get_roi_boxes(
    width: int,
    height: int,
    position: str = "bottom-right",
    box_scale: float = 0.12,
    custom_box: tuple[float, float, float, float] | None = None,
) -> list[tuple[int, int, int, int]]:
    """Return ROI boxes where watermark/sparkle is located, supporting aspect ratios & custom coordinates."""
    if custom_box and len(custom_box) == 4 and custom_box[2] > 0 and custom_box[3] > 0:
        bx, by, bw, bh = custom_box
        # Support normalized (0.0 to 1.0) coordinates
        if bw <= 1.0 and bh <= 1.0:
            px = int(bx * width)
            py = int(by * height)
            pw = int(bw * width)
            ph = int(bh * height)
        else:
            px, py, pw, ph = int(bx), int(by), int(bw), int(bh)
        px = max(0, min(width - 1, px))
        py = max(0, min(height - 1, py))
        pw = max(10, min(width - px, pw))
        ph = max(10, min(height - py, ph))
        return [(px, py, pw, ph)]

    # Aspect ratio aware standard presets
    is_tall = height > width * 1.15
    is_wide = width > height * 1.15
    scale_factor = box_scale / 0.12

    if is_tall:
        size_w = max(24, round(width * 0.16 * scale_factor))
        size_h = max(24, round(height * 0.14 * scale_factor))
        offset_x = round(width * 0.08)
        offset_y = round(height * 0.06)
    elif is_wide:
        size_w = max(24, round(width * 0.14 * scale_factor))
        size_h = max(24, round(height * 0.18 * scale_factor))
        offset_x = round(width * 0.04)
        offset_y = round(height * 0.04)
    else:
        # Square / 1:1 or moderate
        size_w = max(24, round(width * 0.22 * scale_factor))
        size_h = max(24, round(height * 0.22 * scale_factor))
        offset_x = round(width * 0.08)
        offset_y = round(height * 0.08)

    options = {
        "bottom-right": [(width - size_w - offset_x, height - size_h - offset_y, size_w, size_h)],
        "bottom-left": [(offset_x, height - size_h - offset_y, size_w, size_h)],
        "top-right": [(width - size_w - offset_x, offset_y, size_w, size_h)],
        "top-left": [(offset_x, offset_y, size_w, size_h)],
        "all-corners": [
            (width - size_w - offset_x, height - size_h - offset_y, size_w, size_h),
            (offset_x, height - size_h - offset_y, size_w, size_h),
            (width - size_w - offset_x, offset_y, size_w, size_h),
            (offset_x, offset_y, size_w, size_h),
        ],
    }
    target = options.get(position, options["bottom-right"])
    return [(max(0, x), max(0, y), min(width - max(0, x), w), min(height - max(0, y), h)) for x, y, w, h in target]


def detect_sparkle_mask(
    frame: np.ndarray,
    position: str = "bottom-right",
    box_scale: float = 0.12,
    sensitivity: float = 0.6,
    custom_box: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """Isolates neutral white sparkle pixels while strictly protecting colored objects like shoes."""
    height, width = frame.shape[:2]
    full_mask = np.zeros((height, width), dtype=np.uint8)
    boxes = get_roi_boxes(width, height, position, box_scale, custom_box)

    for bx, by, bw, bh in boxes:
        if bw <= 0 or bh <= 0:
            continue
        roi = frame[by : by + bh, bx : bx + bw]
        if roi.size == 0:
            continue

        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)

        chroma = np.sqrt((a_chan.astype(float) - 128) ** 2 + (b_chan.astype(float) - 128) ** 2)

        k = max(15, (min(bw, bh) // 4) * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        tophat_l = cv2.morphologyEx(l_chan, cv2.MORPH_TOPHAT, kernel)
        l_bg_med = cv2.medianBlur(l_chan, k)
        l_diff = np.maximum(0, l_chan.astype(float) - l_bg_med.astype(float))

        sparkle_metric = np.maximum(tophat_l.astype(float), l_diff)
        is_sparkle_color = chroma < 55.0

        thresh_val = max(5, int(18 - sensitivity * 14))
        sparkle = (sparkle_metric > thresh_val) & is_sparkle_color
        full_mask[by : by + bh, bx : bx + bw] = sparkle.astype(np.uint8) * 255

    return full_mask


def process_clean_frame(
    frame: np.ndarray,
    position: str = "bottom-right",
    mode: str = "smart",
    algorithm: str = "ns",
    box_scale: float = 0.12,
    sensitivity: float = 0.6,
    radius: int = 2,
    custom_box: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """Cleans the watermark while keeping 100% of the surrounding background (shoes, carpet, floor) intact."""
    height, width = frame.shape[:2]
    cleaned = frame.copy()
    boxes = get_roi_boxes(width, height, position, box_scale, custom_box)

    for bx, by, bw, bh in boxes:
        if bw <= 0 or bh <= 0:
            continue

        roi = cleaned[by : by + bh, bx : bx + bw]
        if roi.size == 0:
            continue

        if mode == "tight-box":
            box_mask = np.full((bh, bw), 255, dtype=np.uint8)
            cleaned_roi = cv2.inpaint(roi, box_mask, inpaintRadius=max(1, min(7, radius)), flags=cv2.INPAINT_NS)
            cleaned[by : by + bh, bx : bx + bw] = cleaned_roi
            continue

        # Convert ROI to LAB color space
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)

        chroma = np.sqrt((a_chan.astype(float) - 128) ** 2 + (b_chan.astype(float) - 128) ** 2)

        k = max(15, (min(bw, bh) // 4) * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        tophat_l = cv2.morphologyEx(l_chan, cv2.MORPH_TOPHAT, kernel)
        l_bg_med = cv2.medianBlur(l_chan, k)
        l_diff = np.maximum(0, l_chan.astype(float) - l_bg_med.astype(float))

        sparkle_metric = np.maximum(tophat_l.astype(float), l_diff)
        is_sparkle_color = chroma < 55.0

        thresh_val = max(5, int(18 - sensitivity * 14))
        star_mask = ((sparkle_metric > thresh_val) & is_sparkle_color).astype(np.uint8) * 255

        if cv2.countNonZero(star_mask) > 5:
            dilated_mask = cv2.dilate(star_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
            deglow_l = np.where(dilated_mask > 0, l_bg_med, l_chan)
            lab_deglow = cv2.merge([deglow_l, a_chan, b_chan])
            deglow_roi = cv2.cvtColor(lab_deglow, cv2.COLOR_LAB2BGR)

            flags = cv2.INPAINT_NS if algorithm.lower() in ("ns", "navier_stokes", "smart") else cv2.INPAINT_TELEA
            cleaned_roi = cv2.inpaint(deglow_roi, dilated_mask, inpaintRadius=max(1, min(4, radius)), flags=flags)
            cleaned[by : by + bh, bx : bx + bw] = cleaned_roi

    return cleaned


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
    mode: str = Form("smart"),
    algorithm: str = Form("ns"),
    box_scale: float = Form(0.12),
    sensitivity: float = Form(0.6),
    inpaint_radius: int = Form(2),
    box_x: float = Form(None),
    box_y: float = Form(None),
    box_w: float = Form(None),
    box_h: float = Form(None),
):
    suffix = Path(file.filename or "upload").suffix.lower()
    if suffix not in VIDEO_EXTENSIONS | IMAGE_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Supported types: PNG, JPG, WebP, MP4, WebM, MOV, AVI, MKV.")
    if not 1 <= inpaint_radius <= 15:
        raise HTTPException(status_code=422, detail="Inpaint radius must be between 1 and 15.")

    custom_box = None
    if box_x is not None and box_y is not None and box_w is not None and box_h is not None:
        if box_w > 0 and box_h > 0:
            custom_box = (box_x, box_y, box_w, box_h)

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
            cleaned = process_clean_frame(
                image,
                position=position,
                mode=mode,
                algorithm=algorithm,
                box_scale=box_scale,
                sensitivity=sensitivity,
                radius=inpaint_radius,
                custom_box=custom_box,
            )
            if not cv2.imwrite(str(output), cleaned):
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
                cleaned_frame = process_clean_frame(
                    frame,
                    position=position,
                    mode=mode,
                    algorithm=algorithm,
                    box_scale=box_scale,
                    sensitivity=sensitivity,
                    radius=inpaint_radius,
                    custom_box=custom_box,
                )
                writer.write(cleaned_frame)

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

