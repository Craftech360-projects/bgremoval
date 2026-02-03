import runpod
import cv2
import numpy as np
import base64
import traceback
import os
from rembg import remove, new_session

# --- MODEL LOADING ---
_session = None

def get_session():
    global _session
    if _session is None:
        u2net_home = os.getenv('U2NET_HOME', '/root/.u2net')
        model_path = os.path.join(u2net_home, "birefnet-general.onnx")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Run builder.py first.")
        
        print(f"Loading BiRefNet from: {model_path}")
        
        # model_name="birefnet-general": Uses the High-Res Pipeline
        # model_path=model_path: Forces local file usage
        _session = new_session(model_name="birefnet-general", **{"model_path": model_path})
        
    return _session

# --- IMAGE PROCESSING ---

def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
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
    guide_gray = cv2.cvtColor(guide_rgb, cv2.COLOR_RGB2GRAY)
    alpha_f = raw_alpha.astype(np.float32) / 255.0
    guide_f = guide_gray.astype(np.float32) / 255.0

    alpha_f = guided_filter(guide_f, alpha_f, radius=16, eps=1e-4)
    alpha_f = guided_filter(guide_f, alpha_f, radius=4, eps=5e-5)

    alpha = np.clip(alpha_f * 255, 0, 255).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel, iterations=1)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel, iterations=1)

    fg = (alpha > 240).astype(np.uint8)
    bg = (alpha < 15).astype(np.uint8)
    edge_zone = cv2.dilate(1 - fg - bg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
    smoothed = cv2.bilateralFilter(alpha, d=9, sigmaColor=50, sigmaSpace=50)
    
    mask = edge_zone > 0
    alpha[mask] = smoothed[mask]
    alpha[raw_alpha > 245] = 255
    alpha[raw_alpha < 10] = 0

    return alpha

def process_image(input_bytes):
    arr = np.frombuffer(input_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image")
    
    img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    rgba = remove(
        img_rgb,
        session=get_session(),
        alpha_matting=True,
        alpha_matting_foreground_threshold=270,
        alpha_matting_background_threshold=5,
        alpha_matting_erode_size=5,
        post_process_mask=True,
    )
    
    alpha = refine_alpha(rgba[..., 3], img_rgb)
    out = np.dstack([img_rgb, alpha])
    
    out_bgra = cv2.cvtColor(out, cv2.COLOR_RGBA2BGRA)
    ok, buf = cv2.imencode(".png", out_bgra)
    if not ok:
        raise ValueError("Failed to encode PNG")
    return buf.tobytes()

# --- ENTRY POINT ---

def handler(job):
    job_input = job['input']
    if 'image' not in job_input:
        return {"error": "No 'image' provided"}
    try:
        image_data = base64.b64decode(job_input['image'])
        result_bytes = process_image(image_data)
        
        result_b64 = base64.b64encode(result_bytes).decode('utf-8')
        
        return {
            "image": result_b64
        }

    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}

runpod.serverless.start({"handler": handler})