# app.py
import io
from typing import Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from rembg import remove, new_session

app = FastAPI(title="High-Quality Background Removal API", version="1.0.0")

# CORS (open by default; tighten for prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Model sessions cache to avoid reloading per request
_SESSION_CACHE = {}

def get_session(model_name: str):
    """
    Cache rembg sessions by model name.
    Available models include:
      - "isnet-general-use" (default, very good)
      - "u2net_human_seg"   (humans)
      - "u2net"
    """
    sess = _SESSION_CACHE.get(model_name)
    if sess is None:
        sess = new_session(model_name)
        _SESSION_CACHE[model_name] = sess
    return sess

def read_image_from_upload(upload: UploadFile) -> np.ndarray:
    """Read an uploaded image into an RGB numpy array."""
    data = upload.file.read()
    if not data:
        raise ValueError("Empty image upload.")
    arr = np.frombuffer(data, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Unable to decode image. Ensure it's a valid JPG/PNG.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def feather_alpha(a: np.ndarray, r: float = 1.5) -> np.ndarray:
    """Optional softening of alpha edges."""
    if r is None or r <= 0:
        return a
    return cv2.GaussianBlur(a, (0, 0), r)

def parse_solid_rgb(solid_str: Optional[str]) -> Optional[Tuple[int, int, int]]:
    """Parse 'R,G,B' into a tuple or return None."""
    if not solid_str:
        return None
    parts = [p.strip() for p in solid_str.split(",")]
    if len(parts) != 3:
        raise ValueError("solid must be 'R,G,B' (e.g., '255,255,255').")
    r, g, b = map(int, parts)
    for v in (r, g, b):
        if v < 0 or v > 255:
            raise ValueError("solid RGB values must be in 0..255.")
    return (r, g, b)

def composite(img_rgb: np.ndarray, alpha: np.ndarray, bg_rgb: Optional[np.ndarray]) -> np.ndarray:
    """
    If bg_rgb is None -> return RGBA (transparent background).
    Else -> return composited RGB (on background).
    """
    a = (alpha.astype(np.float32) / 255.0)[..., None]
    if bg_rgb is None:
        rgba = np.dstack([img_rgb, (a * 255).astype(np.uint8)])
        return rgba
    if bg_rgb.shape[:2] != img_rgb.shape[:2]:
        bg_rgb = cv2.resize(bg_rgb, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    comp = (img_rgb * a + bg_rgb * (1 - a)).astype(np.uint8)
    return comp

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/remove-bg")
def remove_bg(
    file: UploadFile = File(..., description="Foreground image (JPG/PNG)"),
    # Optional background image to composite with (JPG/PNG)
    bg: Optional[UploadFile] = File(None, description="Background image (JPG/PNG)"),
    # Alternatively specify a solid color, e.g., "255,255,255"
    solid: Optional[str] = Form(None, description="Solid RGB 'R,G,B' (e.g., '255,255,255')"),
    # Or blur the original background by an odd kernel size (e.g., 31)
    blur: Optional[int] = Form(0, description="Odd kernel size for background blur (e.g., 31)"),
    # Edge softening radius in px
    feather: Optional[float] = Form(1.5, description="Edge feather radius (px)"),
    # rembg model to use
    model: Optional[str] = Form("isnet-general-use", description="isnet-general-use | u2net_human_seg | u2net"),
    # Force output format? "png" (with alpha) or "jpg" (no alpha). Default auto.
    output_format: Optional[str] = Form(None, description="'png' or 'jpg' (auto if omitted)"),
):
    """
    Returns the processed image:
      - If no background is provided (no bg/solid/blur), returns a PNG with transparency.
      - If a background is provided (bg/solid/blur), returns a JPG (unless output_format='png').
    """
    try:
        # Read input foreground
        img_rgb = read_image_from_upload(file)

        # Prepare background (priority: bg file > solid > blur)
        bg_rgb = None
        if bg is not None:
            bg_rgb = read_image_from_upload(bg)
        else:
            solid_rgb = parse_solid_rgb(solid)
            if solid_rgb:
                bg_rgb = np.full_like(img_rgb, solid_rgb, dtype=np.uint8)
            elif blur and isinstance(blur, int) and blur > 0:
                if blur % 2 == 0:
                    # Make it odd
                    blur += 1
                bg_rgb = cv2.GaussianBlur(img_rgb, (blur, blur), 0)

        # Matting via rembg (returns RGBA)
        session = get_session(model)
        rgba = remove(
            img_rgb,
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_structure_size=3,
            post_process_mask=True,
        )

        alpha = feather_alpha(rgba[..., 3], feather if feather is not None else 0.0)
        out = composite(img_rgb, alpha, bg_rgb)

        # Decide output format + encode
        wants_png = (output_format or "").lower() == "png"
        wants_jpg = (output_format or "").lower() == "jpg"

        if bg_rgb is None and not wants_jpg:
            # Transparent
            out_bgra = cv2.cvtColor(out, cv2.COLOR_RGBA2BGRA)
            ok, buf = cv2.imencode(".png", out_bgra)
            if not ok:
                return JSONResponse(status_code=500, content={"error": "Failed to encode PNG."})
            return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/png",
                                     headers={"Content-Disposition": 'inline; filename="cut.png"'})
        else:
            # RGB on background (JPG by default)
            if out.shape[2] == 4:
                out = cv2.cvtColor(out, cv2.COLOR_RGBA2RGB)
            ext = ".png" if wants_png else ".jpg"
            mime = "image/png" if wants_png else "image/jpeg"
            ok, buf = cv2.imencode(ext, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            if not ok:
                return JSONResponse(status_code=500, content={"error": f"Failed to encode {ext}."})
            filename = f"result{ext}"
            return StreamingResponse(io.BytesIO(buf.tobytes()), media_type=mime,
                                     headers={"Content-Disposition": f'inline; filename="{filename}"'})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
