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

def detect_hair_and_fine_details(img_rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Detect hair strands and fine details using multi-scale analysis."""
    
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # Multi-scale Gaussian derivatives for hair detection
    hair_responses = []
    scales = [1.0, 1.5, 2.0, 3.0]  # Different scales for different hair thickness
    
    for sigma in scales:
        # Gaussian derivatives at different orientations
        for angle in [0, 30, 60, 90, 120, 150]:  # 6 orientations
            # Create oriented filter
            kernel_size = max(3, int(6 * sigma + 1))
            if kernel_size % 2 == 0:
                kernel_size += 1
                
            # Gabor-like filter for hair detection
            kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma, 
                                      np.radians(angle), 2*np.pi/3, 0.5, 0, ktype=cv2.CV_32F)
            
            response = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, kernel)
            hair_responses.append(np.abs(response))
    
    # Combine all responses - maximum response across scales and orientations
    hair_mask = np.maximum.reduce(hair_responses)
    
    # Normalize and threshold
    hair_mask = (hair_mask - hair_mask.min()) / (hair_mask.max() - hair_mask.min() + 1e-8)
    hair_regions = hair_mask > 0.3
    
    return hair_regions.astype(np.uint8) * 255

def preserve_fine_gaps(alpha: np.ndarray, original_img: np.ndarray) -> np.ndarray:
    """Preserve small gaps and fine details between objects."""
    
    # Convert to LAB for better perceptual processing
    lab = cv2.cvtColor(original_img, cv2.COLOR_RGB2LAB)
    
    # Multi-scale edge detection for gap preservation
    edges_fine = cv2.Canny(lab[:,:,0], 20, 60)  # Fine edges
    edges_medium = cv2.Canny(lab[:,:,0], 40, 120)  # Medium edges
    
    # Morphological operations to detect thin structures
    kernel_thin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    kernel_gaps = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # Detect thin structures (potential gaps)
    thin_structures = cv2.morphologyEx(edges_fine, cv2.MORPH_CLOSE, kernel_thin)
    gap_candidates = cv2.morphologyEx(thin_structures, cv2.MORPH_OPEN, kernel_gaps)
    
    # Enhance alpha in gap regions where background should show through
    alpha_enhanced = alpha.copy().astype(np.float32)
    
    # For pixels in gap regions with medium alpha, reduce alpha to preserve gaps
    gap_regions = (gap_candidates > 0) & (alpha > 50) & (alpha < 200)
    alpha_enhanced[gap_regions] *= 0.7  # Make gaps more transparent
    
    # For pixels in fine edge regions with high alpha, slightly reduce to show detail
    fine_edge_regions = (edges_fine > 0) & (alpha > 180)
    alpha_enhanced[fine_edge_regions] *= 0.95
    
    return np.clip(alpha_enhanced, 0, 255).astype(np.uint8)

def multi_scale_detail_enhancement(alpha: np.ndarray, original_img: np.ndarray) -> np.ndarray:
    """Multi-scale processing to preserve details at different levels."""
    
    h, w = alpha.shape
    scales = [1.0, 0.75, 0.5]  # Process at different resolutions
    enhanced_alphas = []
    
    for scale in scales:
        if scale == 1.0:
            scaled_img = original_img
            scaled_alpha = alpha
        else:
            new_h, new_w = int(h * scale), int(w * scale)
            scaled_img = cv2.resize(original_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            scaled_alpha = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Process at this scale
        gray = cv2.cvtColor(scaled_img, cv2.COLOR_RGB2GRAY)
        
        # Adaptive bilateral filtering based on local texture
        texture = cv2.Laplacian(gray, cv2.CV_64F)
        texture_strength = np.abs(texture)
        texture_normalized = texture_strength / (texture_strength.max() + 1e-8)
        
        # Stronger filtering in low-texture regions, weaker in high-texture (detail) regions
        processed_alpha = scaled_alpha.copy().astype(np.float32)
        
        for i in range(scaled_alpha.shape[0]):
            for j in range(scaled_alpha.shape[1]):
                if texture_normalized[i, j] > 0.5:  # High texture - preserve details
                    # Minimal processing for detail preservation
                    continue
                else:  # Low texture - can be smoothed more
                    # Apply stronger smoothing
                    roi_size = 5
                    i_start, i_end = max(0, i-roi_size//2), min(scaled_alpha.shape[0], i+roi_size//2+1)
                    j_start, j_end = max(0, j-roi_size//2), min(scaled_alpha.shape[1], j+roi_size//2+1)
                    roi = scaled_alpha[i_start:i_end, j_start:j_end]
                    processed_alpha[i, j] = np.mean(roi)
        
        # Resize back if needed
        if scale != 1.0:
            processed_alpha = cv2.resize(processed_alpha, (w, h), interpolation=cv2.INTER_CUBIC)
        
        enhanced_alphas.append(processed_alpha)
    
    # Combine scales - give more weight to finer scales for detail preservation
    weights = [0.5, 0.3, 0.2]  # Favor original scale
    final_alpha = np.zeros_like(alpha, dtype=np.float32)
    
    for alpha_scaled, weight in zip(enhanced_alphas, weights):
        final_alpha += alpha_scaled * weight
    
    return np.clip(final_alpha, 0, 255).astype(np.uint8)

def texture_aware_refinement(alpha: np.ndarray, original_img: np.ndarray) -> np.ndarray:
    """Refine alpha based on local texture patterns for hair and fine details."""
    
    # Convert to different color spaces for better analysis
    gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(original_img, cv2.COLOR_RGB2HSV)
    
    # Local Binary Pattern for texture analysis
    def compute_lbp(img, radius=1, n_points=8):
        """Simple LBP implementation for texture detection."""
        lbp = np.zeros_like(img)
        for i in range(radius, img.shape[0] - radius):
            for j in range(radius, img.shape[1] - radius):
                center = img[i, j]
                pattern = 0
                for p in range(n_points):
                    angle = 2 * np.pi * p / n_points
                    x = int(round(i + radius * np.cos(angle)))
                    y = int(round(j + radius * np.sin(angle)))
                    if img[x, y] >= center:
                        pattern |= (1 << p)
                lbp[i, j] = pattern
        return lbp
    
    # Compute texture features
    lbp = compute_lbp(gray)
    texture_variance = cv2.Laplacian(gray, cv2.CV_64F)
    texture_variance = np.abs(texture_variance)
    
    # Detect directional structures (likely hair)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_direction = np.arctan2(sobel_y, sobel_x)
    
    # Enhance alpha based on texture analysis
    alpha_refined = alpha.copy().astype(np.float32)
    
    # High texture regions (likely hair/fur) - preserve more detail
    high_texture = texture_variance > np.percentile(texture_variance, 70)
    
    # Strong directional gradients (hair strands)
    strong_gradients = gradient_magnitude > np.percentile(gradient_magnitude, 80)
    
    # Combine conditions for hair/detail regions
    detail_regions = high_texture | strong_gradients
    
    # For detail regions, enhance contrast in alpha to preserve fine structures
    detail_mask = detail_regions & (alpha > 50) & (alpha < 240)
    
    # Enhance contrast in detail regions
    alpha_refined[detail_mask] = np.where(
        alpha_refined[detail_mask] > 127,
        np.minimum(alpha_refined[detail_mask] * 1.1, 255),
        np.maximum(alpha_refined[detail_mask] * 0.9, 0)
    )
    
    return np.clip(alpha_refined, 0, 255).astype(np.uint8)

def precision_alpha_refinement(alpha: np.ndarray, original_img: np.ndarray) -> np.ndarray:
    """Ultra-high precision alpha refinement for maximum accuracy."""
    
    # Convert to float for precision processing
    alpha_float = alpha.astype(np.float64) / 255.0
    img_lab = cv2.cvtColor(original_img, cv2.COLOR_RGB2LAB)
    
    # Step 1: Trimap generation for better accuracy
    # Definite foreground (high alpha)
    fg_mask = alpha > 240
    # Definite background (very low alpha) 
    bg_mask = alpha < 15
    # Unknown region (needs refinement)
    unknown_mask = ~(fg_mask | bg_mask)
    
    # Step 2: Local variance-based refinement
    # Calculate local color variance to detect object boundaries
    gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY).astype(np.float64)
    local_var = cv2.Laplacian(gray, cv2.CV_64F)
    local_var = cv2.GaussianBlur(np.abs(local_var), (5, 5), 1.0)
    
    # Step 3: Edge-aware alpha matting
    # Use color gradients to guide alpha refinement
    grad_x = cv2.Sobel(img_lab[:,:,0], cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_lab[:,:,0], cv2.CV_64F, 0, 1, ksize=3)
    color_gradient = np.sqrt(grad_x**2 + grad_y**2)
    
    # Step 4: Iterative alpha refinement in unknown regions
    refined_alpha = alpha_float.copy()
    
    for iteration in range(3):  # 3 iterations for precision
        # Bilateral filtering with edge preservation
        refined_alpha = cv2.bilateralFilter(
            (refined_alpha * 255).astype(np.uint8), 
            15, 80, 80
        ).astype(np.float64) / 255.0
        
        # Enhance boundaries based on color gradients
        boundary_strength = color_gradient / (color_gradient.max() + 1e-8)
        alpha_grad_x = cv2.Sobel(refined_alpha, cv2.CV_64F, 1, 0, ksize=3)
        alpha_grad_y = cv2.Sobel(refined_alpha, cv2.CV_64F, 0, 1, ksize=3)
        alpha_gradient = np.sqrt(alpha_grad_x**2 + alpha_grad_y**2)
        
        # Sharpen alpha where there are strong color boundaries
        sharpening_factor = 1 + (boundary_strength * 0.3)
        refined_alpha = refined_alpha * sharpening_factor
        
        # Preserve definite foreground and background
        refined_alpha[fg_mask] = 1.0
        refined_alpha[bg_mask] = 0.0
        
        refined_alpha = np.clip(refined_alpha, 0, 1)
    
    # Step 5: Fine detail and hair preservation
    alpha_uint8 = np.clip(refined_alpha * 255, 0, 255).astype(np.uint8)
    
    # Multi-scale detail enhancement
    alpha_uint8 = multi_scale_detail_enhancement(alpha_uint8, original_img)
    
    # Texture-aware refinement for hair and fine structures
    alpha_uint8 = texture_aware_refinement(alpha_uint8, original_img)
    
    # Preserve fine gaps between objects
    alpha_uint8 = preserve_fine_gaps(alpha_uint8, original_img)
    
    # Step 6: Sub-pixel accuracy enhancement with guided filtering
    refined_alpha = alpha_uint8.astype(np.float64) / 255.0
    epsilon = 0.0005  # Smaller epsilon for more precision
    radius = 6        # Smaller radius to preserve fine details
    
    mean_I = cv2.boxFilter(gray / 255.0, cv2.CV_64F, (radius, radius))
    mean_alpha = cv2.boxFilter(refined_alpha, cv2.CV_64F, (radius, radius))
    mean_Ialpha = cv2.boxFilter((gray / 255.0) * refined_alpha, cv2.CV_64F, (radius, radius))
    cov_Ialpha = mean_Ialpha - mean_I * mean_alpha
    
    mean_II = cv2.boxFilter((gray / 255.0) * (gray / 255.0), cv2.CV_64F, (radius, radius))
    var_I = mean_II - mean_I * mean_I
    
    a = cov_Ialpha / (var_I + epsilon)
    b = mean_alpha - a * mean_I
    
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (radius, radius))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (radius, radius))
    
    final_alpha = mean_a * (gray / 255.0) + mean_b
    
    # Step 7: Final hair enhancement pass
    final_alpha_uint8 = np.clip(final_alpha * 255, 0, 255).astype(np.uint8)
    hair_mask = detect_hair_and_fine_details(original_img, final_alpha_uint8)
    
    # Enhance alpha in detected hair regions
    hair_regions = (hair_mask > 100) & (final_alpha_uint8 > 30) & (final_alpha_uint8 < 250)
    final_alpha_enhanced = final_alpha_uint8.copy().astype(np.float32)
    final_alpha_enhanced[hair_regions] *= 1.05  # Slight enhancement
    
    return np.clip(final_alpha_enhanced, 0, 255).astype(np.uint8)

def multi_model_ensemble(img_rgb: np.ndarray, models: list) -> np.ndarray:
    """Use multiple models and combine results for higher accuracy."""
    
    alpha_maps = []
    
    for model_name in models:
        try:
            session = get_session(model_name)
            rgba = remove(
                img_rgb, 
                session=session,
                alpha_matting=True,
                alpha_matting_foreground_threshold=270,
                alpha_matting_background_threshold=5,
                alpha_matting_erode_structure_size=2,
                post_process_mask=True
            )
            alpha_maps.append(rgba[..., 3].astype(np.float32))
        except Exception as e:
            print(f"Model {model_name} failed: {e}")
            continue
    
    if not alpha_maps:
        raise ValueError("All models failed")
    
    if len(alpha_maps) == 1:
        return alpha_maps[0].astype(np.uint8)
    
    # Weighted ensemble - give more weight to consistent predictions
    alpha_stack = np.stack(alpha_maps, axis=-1)
    
    # Calculate pixel-wise variance to find consistent regions
    alpha_mean = np.mean(alpha_stack, axis=-1)
    alpha_var = np.var(alpha_stack, axis=-1)
    
    # High confidence where variance is low
    confidence = 1.0 / (1.0 + alpha_var / 100.0)
    
    # Weighted average based on confidence
    weights = confidence[..., np.newaxis]
    weighted_sum = np.sum(alpha_stack * weights, axis=-1)
    weight_sum = np.sum(weights, axis=-1).squeeze()
    
    ensemble_alpha = weighted_sum / weight_sum
    return np.clip(ensemble_alpha, 0, 255).astype(np.uint8)

def enhance_transparency(alpha: np.ndarray, original_img: np.ndarray, method: str = "advanced") -> np.ndarray:
    """Enhanced transparency processing for cleaner cutouts."""
    if method == "basic":
        return cv2.GaussianBlur(alpha, (0, 0), 1.5)
    
    # Advanced multi-step alpha refinement
    enhanced_alpha = alpha.copy().astype(np.float32)
    
    # Step 1: Edge detection and enhancement
    edges = cv2.Canny(alpha, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Step 2: Smooth transitions near edges
    edge_mask = edges_dilated > 0
    smooth_alpha = cv2.GaussianBlur(enhanced_alpha, (5, 5), 1.0)
    enhanced_alpha[edge_mask] = smooth_alpha[edge_mask]
    
    # Step 3: Preserve fine details (hair, fur)
    gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    detail_mask = cv2.Laplacian(gray, cv2.CV_64F)
    detail_mask = np.abs(detail_mask) > 10
    
    # Enhance alpha in high-detail areas
    fine_detail_regions = detail_mask & (alpha > 100) & (alpha < 240)
    enhanced_alpha[fine_detail_regions] *= 1.1
    
    # Step 4: Anti-aliasing for smoother edges
    enhanced_alpha = cv2.bilateralFilter(
        enhanced_alpha.astype(np.uint8), 9, 75, 75
    ).astype(np.float32)
    
    # Step 5: Gradient smoothing in transition areas
    transition_mask = (alpha > 20) & (alpha < 235)
    if np.any(transition_mask):
        grad_x = cv2.Sobel(enhanced_alpha, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(enhanced_alpha, cv2.CV_32F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Smooth high-gradient areas more
        high_grad = gradient_magnitude > 30
        smooth_regions = transition_mask & high_grad
        if np.any(smooth_regions):
            very_smooth = cv2.GaussianBlur(enhanced_alpha, (7, 7), 2.0)
            enhanced_alpha[smooth_regions] = very_smooth[smooth_regions]
    
    return np.clip(enhanced_alpha, 0, 255).astype(np.uint8)

def feather_alpha(a: np.ndarray, r: float = 1.5) -> np.ndarray:
    """Basic alpha feathering - kept for compatibility."""
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
    feather: Optional[float] = Form(0.0, description="Edge feather radius (px)"),
    # Enhanced transparency mode
    enhance_mode: Optional[str] = Form("advanced", description="basic | advanced | precision (ultra-high accuracy)"),
    # Use multiple models for ensemble (higher accuracy but slower)
    use_ensemble: Optional[bool] = Form(False, description="Use multiple models for maximum accuracy"),
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

        # High-accuracy matting processing
        if use_ensemble:
            # Multi-model ensemble for maximum accuracy
            ensemble_models = ["isnet-general-use", "u2net_human_seg", "u2net"]
            if model not in ensemble_models:
                ensemble_models.append(model)
            raw_alpha = multi_model_ensemble(img_rgb, ensemble_models)
        else:
            # Single model processing with optimized parameters
            session = get_session(model)
            
            # Ultra-high quality parameters for maximum accuracy
            matting_params = {
                "alpha_matting": True,
                "alpha_matting_foreground_threshold": 270,
                "alpha_matting_background_threshold": 5,
                "alpha_matting_erode_structure_size": 2,
                "post_process_mask": True,
            }
            
            rgba = remove(img_rgb, session=session, **matting_params)
            raw_alpha = rgba[..., 3]

        # Ultra-high precision transparency processing
        enhancement_method = enhance_mode if enhance_mode in ["basic", "advanced", "precision"] else "advanced"
        
        if enhancement_method == "precision":
            # Maximum accuracy processing
            alpha = precision_alpha_refinement(raw_alpha, img_rgb)
        else:
            alpha = enhance_transparency(raw_alpha, img_rgb, method=enhancement_method)
        
        # Apply additional feathering if requested
        if feather and feather > 0:
            alpha = feather_alpha(alpha, feather)
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
