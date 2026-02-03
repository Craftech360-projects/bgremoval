import io
import traceback

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from rembg import remove, new_session

app = FastAPI(title="Background Removal API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Best available model — cached once on first request
_session = None

def get_session():
    global _session
    if _session is None:
        _session = new_session("bria-rmbg")
    return _session


def guided_filter(guide: np.ndarray, src: np.ndarray,
                   radius: int, eps: float) -> np.ndarray:
    """
    Edge-preserving guided filter using standard cv2.boxFilter.
    Identical to cv2.ximgproc.guidedFilter but needs no contrib build.
    Both inputs must be float32, single-channel.
    """
    ksize = (2 * radius + 1, 2 * radius + 1)
    mean_I = cv2.boxFilter(guide, cv2.CV_32F, ksize)
    mean_p = cv2.boxFilter(src, cv2.CV_32F, ksize)
    mean_Ip = cv2.boxFilter(guide * src, cv2.CV_32F, ksize)
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(guide * guide, cv2.CV_32F, ksize)
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)
    return mean_a * guide + mean_b


def refine_alpha(raw_alpha: np.ndarray, guide_rgb: np.ndarray) -> np.ndarray:
    """
    Best-quality alpha refinement:
    two-pass guided filter (coarse + fine) → morphological cleanup →
    edge-zone bilateral smoothing → harden confident regions.
    """
    guide_gray = cv2.cvtColor(guide_rgb, cv2.COLOR_RGB2GRAY)
    alpha_f = raw_alpha.astype(np.float32) / 255.0
    guide_f = guide_gray.astype(np.float32) / 255.0

    # Pass 1: large radius for global consistency
    alpha_f = guided_filter(guide_f, alpha_f, radius=16, eps=1e-4)
    # Pass 2: small radius for fine edge detail (hair, fur, etc.)
    alpha_f = guided_filter(guide_f, alpha_f, radius=4, eps=5e-5)

    alpha = np.clip(alpha_f * 255, 0, 255).astype(np.uint8)

    # Morphological cleanup — fill tiny holes, remove specks
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel, iterations=1)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel, iterations=1)

    # Bilateral smoothing only in the FG/BG transition zone
    fg = (alpha > 240).astype(np.uint8)
    bg = (alpha < 15).astype(np.uint8)
    edge_zone = cv2.dilate(
        1 - fg - bg,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=2,
    )
    smoothed = cv2.bilateralFilter(alpha, d=9, sigmaColor=50, sigmaSpace=50)
    mask = edge_zone > 0
    alpha[mask] = smoothed[mask]

    # Harden confident regions
    alpha[raw_alpha > 245] = 255
    alpha[raw_alpha < 10] = 0

    return alpha


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/remove-bg")
def remove_bg(
    file: UploadFile = File(..., description="Image (JPG/PNG)"),
):
    """Send an image, get back a PNG with the background removed."""
    try:
        # Decode upload
        data = file.file.read()
        if not data:
            return JSONResponse(status_code=400, content={"error": "Empty file."})
        arr = np.frombuffer(data, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return JSONResponse(status_code=400,
                                content={"error": "Invalid image. Send a JPG or PNG."})
        img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Remove background with best model + alpha matting
        rgba = remove(
            img_rgb,
            session=get_session(),
            alpha_matting=True,
            alpha_matting_foreground_threshold=270,
            alpha_matting_background_threshold=5,
            alpha_matting_erode_size=5,
            post_process_mask=True,
        )
        raw_alpha = rgba[..., 3]

        # Refine edges
        alpha = refine_alpha(raw_alpha, img_rgb)

        # Build RGBA output
        out = np.dstack([img_rgb, alpha])
        out_bgra = cv2.cvtColor(out, cv2.COLOR_RGBA2BGRA)
        ok, buf = cv2.imencode(".png", out_bgra)
        if not ok:
            return JSONResponse(status_code=500,
                                content={"error": "Failed to encode PNG."})

        return StreamingResponse(
            io.BytesIO(buf.tobytes()),
            media_type="image/png",
            headers={"Content-Disposition": 'inline; filename="cut.png"'},
        )

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=400, content={"error": str(e)})
