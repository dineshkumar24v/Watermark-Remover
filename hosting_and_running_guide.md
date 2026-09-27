# Gemini Watermark Remover — Running & Hosting Guide

The **Gemini Watermark Remover** consists of a modern Tailwind/HTML5 frontend ([index.html](file:///e:/AI-YouTube-Factory/gemini%20watermark%20remover%20project/index.html)) and a high-performance Python FastAPI backend ([main.py](file:///e:/AI-YouTube-Factory/gemini%20watermark%20remover%20project/main.py)) powered by OpenCV for image/video inpainting.

---

### Option 1: Running the Complete App (Backend + Frontend) — Recommended

The Python server hosts both the API and the web interface in a single process:

1. **Install Dependencies (if not already installed):**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the Server:**
   ```bash
   python main.py
   ```
   *Or with live-reload:*
   ```bash
   python -m uvicorn main:app --host 127.0.0.1 --port 8080 --reload
   ```

3. **Open in Browser:**
   Visit [http://127.0.0.1:8080](http://127.0.0.1:8080) in your web browser.

---

### Option 2: Running the Frontend Standalone

If you only want to open the HTML interface directly:

1. **Direct Browser Open:** Double-click [index.html](file:///e:/AI-YouTube-Factory/gemini%20watermark%20remover%20project/index.html) or open it in Chrome/Edge/Firefox.
2. **Local Static Server:**
   ```bash
   npx serve .
   ```
   *Note: For backend processing features (OpenCV inpainting and video reconstruction), ensure the FastAPI backend is running on port `8080`.*

---

### Option 3: Hosting It Online

#### 1. Frontend Hosting (Free Static Hosting)
* **Vercel / Netlify / GitHub Pages:** Deploy [index.html](file:///e:/AI-YouTube-Factory/gemini%20watermark%20remover%20project/index.html) directly by dragging and dropping the folder into Vercel or Netlify Drop.

#### 2. Full-Stack Hosting (API + Frontend)
* **Render / Railway / Fly.io / VPS:** Deploy `main.py` and `requirements.txt` with start command:
  ```bash
  uvicorn main:app --host 0.0.0.0 --port $PORT
  ```