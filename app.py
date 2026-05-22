import base64
import binascii
from io import BytesIO
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image, ImageColor, ImageOps

from person_aware_bg_remove import (
    DEFAULT_ALPHA_MODEL,
    remove_background_from_image,
)


app = FastAPI(title="Person-aware background removal API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


IMAGE_FIELD_NAMES = (
    "image",
    "file",
    "photo",
    "capture",
    "camera",
    "cameraImage",
    "capturedImage",
)

BACKGROUND_FIELD_NAMES = (
    "background",
    "backgroundImage",
    "bg",
    "bgImage",
    "selectedBackground",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/remove-bg")
async def remove_bg(request: Request):
    form = await request.form()
    foreground_upload = find_upload(form, IMAGE_FIELD_NAMES)

    if foreground_upload is None:
        foreground_upload = first_upload(form)

    if foreground_upload is None:
        raise HTTPException(
            status_code=400,
            detail="No image upload found in multipart form data.",
        )

    foreground = open_image(await foreground_upload.read())
    background_value = find_background_value(form, foreground_upload)

    yolo_model = str(form.get("yolo_model") or form.get("yoloModel") or "yolo11x-seg.pt")
    alpha_model = str(form.get("alpha_model") or form.get("alphaModel") or DEFAULT_ALPHA_MODEL)
    conf = parse_float(form.get("conf"), default=0.25)
    all_people = parse_bool(form.get("all_people") or form.get("allPeople"), default=True)
    background = await read_background(background_value)

    try:
        cutout, _, _, _, _ = remove_background_from_image(
            image=foreground,
            yolo_model=yolo_model,
            alpha_model=alpha_model,
            conf=conf,
            all_people=all_people,
        )
        output = composite_on_background(cutout, background)
    except RuntimeError as exc:
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    buffer = BytesIO()
    output.save(buffer, format="PNG")
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="image/png",
        headers={"Content-Disposition": 'inline; filename="removed-background.png"'},
    )


def find_upload(form: Any, names: tuple[str, ...]):
    for name in names:
        value = form.get(name)
        if is_upload(value):
            return value
    return None


def first_upload(form: Any):
    for value in form.values():
        if is_upload(value):
            return value
    return None


def find_background_value(form: Any, foreground_upload):
    for name in BACKGROUND_FIELD_NAMES:
        value = form.get(name)
        if value is not None and value is not foreground_upload:
            return value

    uploads = [value for value in form.values() if is_upload(value)]
    for upload in uploads:
        if upload is not foreground_upload:
            return upload

    return None


def is_upload(value: Any) -> bool:
    return hasattr(value, "filename") and hasattr(value, "read")


def open_image(data: bytes) -> Image.Image:
    try:
        image = Image.open(BytesIO(data))
        return ImageOps.exif_transpose(image).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Uploaded image is not valid.") from exc


async def read_background(value: Any) -> Image.Image | str | None:
    if is_upload(value):
        data = await value.read()
        return open_image(data)
    if isinstance(value, str):
        return value
    return None


def composite_on_background(cutout: Image.Image, background_value: Any) -> Image.Image:
    background = resolve_background(background_value, cutout.size)
    if background is None:
        return cutout

    background.alpha_composite(cutout)
    return background


def resolve_background(value: Any, size: tuple[int, int]) -> Image.Image | None:
    if value is None:
        return None

    if is_upload(value):
        raise HTTPException(
            status_code=400,
            detail="Internal background upload was not decoded.",
        )

    if isinstance(value, Image.Image):
        return resize_cover(value.convert("RGBA"), size)

    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text or text.lower() in {"none", "null", "transparent"}:
        return None

    data_url_image = image_from_data_url(text)
    if data_url_image is not None:
        return resize_cover(data_url_image.convert("RGBA"), size)

    try:
        color = ImageColor.getrgb(text)
    except ValueError:
        return None

    return Image.new("RGBA", size, color + (255,))


def image_from_data_url(value: str) -> Image.Image | None:
    if not value.startswith("data:image/"):
        return None

    try:
        _, encoded = value.split(",", 1)
        return Image.open(BytesIO(base64.b64decode(encoded))).convert("RGBA")
    except (ValueError, binascii.Error, OSError):
        return None


def resize_cover(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    target_width, target_height = size
    image_width, image_height = image.size
    scale = max(target_width / image_width, target_height / image_height)
    resized = image.resize(
        (round(image_width * scale), round(image_height * scale)),
        Image.Resampling.LANCZOS,
    )
    left = (resized.width - target_width) // 2
    top = (resized.height - target_height) // 2
    return resized.crop((left, top, left + target_width, top + target_height))


def parse_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8765)
