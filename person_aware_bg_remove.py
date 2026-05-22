import argparse
import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from ultralytics import YOLO


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_ALPHA_MODEL = "rembg:isnet-general-use"


@lru_cache(maxsize=2)
def load_hf_segmentation_model(model_name: str):
    token = os.environ.get("HF_TOKEN")
    return AutoModelForImageSegmentation.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=token,
    ).eval().to(DEVICE)


@lru_cache(maxsize=4)
def load_rembg_session(model_name: str):
    from rembg import new_session

    return new_session(model_name)


@lru_cache(maxsize=2)
def load_yolo_model(model_name: str):
    return YOLO(model_name)


def get_hf_alpha(image: Image.Image, model) -> np.ndarray:
    original_size = image.size

    transform = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prediction = model(input_tensor)[-1].sigmoid().cpu()[0].squeeze()

    alpha = transforms.ToPILImage()(prediction).resize(
        original_size,
        Image.Resampling.LANCZOS,
    )
    return np.array(alpha, dtype=np.uint8)


def get_rembg_alpha(image: Image.Image, model_name: str) -> np.ndarray:
    from rembg import remove

    session = load_rembg_session(model_name)
    mask = remove(image, session=session, only_mask=True)
    return np.array(mask.convert("L"), dtype=np.uint8)


def get_alpha(image: Image.Image, alpha_model: str) -> tuple[np.ndarray, str]:
    if alpha_model.startswith("rembg:"):
        model_name = alpha_model.split(":", 1)[1]
        return get_rembg_alpha(image, model_name), f"rembg:{model_name}"

    if alpha_model.startswith("hf:"):
        model_name = alpha_model.split(":", 1)[1]
    else:
        model_name = alpha_model

    try:
        model = load_hf_segmentation_model(model_name)
        return get_hf_alpha(image, model), f"hf:{model_name}"
    except OSError as exc:
        if "gated repo" not in str(exc).lower() and "403" not in str(exc):
            raise
        raise RuntimeError(
            "\n".join(
                [
                    f"Cannot access Hugging Face model '{model_name}'.",
                    "That model is gated for your account.",
                    "Use the default non-gated model instead:",
                    "  python person_aware_bg_remove.py image.jpg -o output_no_bg.png --debug-dir debug",
                    "Or request access on Hugging Face, then set HF_TOKEN and run:",
                    f"  python person_aware_bg_remove.py image.jpg -o output_no_bg.png --alpha-model hf:{model_name}",
                ]
            )
        ) from exc


def get_person_mask_yolo(
    image: Image.Image,
    model_name: str = "yolo11x-seg.pt",
    conf: float = 0.25,
    include_all_people: bool = True,
) -> np.ndarray:
    model = load_yolo_model(model_name)
    result = model.predict(np.array(image), conf=conf, verbose=False)[0]

    width, height = image.size
    mask = np.zeros((height, width), dtype=np.uint8)

    if result.masks is None or result.boxes is None:
        return mask

    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    person_indexes = np.where(classes == 0)[0]
    if len(person_indexes) == 0:
        return mask

    masks = result.masks.data.detach().cpu().numpy()
    selected_indexes = person_indexes

    if not include_all_people:
        areas = [masks[i].sum() for i in person_indexes]
        selected_indexes = [person_indexes[int(np.argmax(areas))]]

    for index in selected_indexes:
        person_mask = masks[index]
        person_mask = cv2.resize(
            person_mask,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
        mask[person_mask > 0.5] = 255

    return clean_binary_mask(mask)


def clean_binary_mask(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def combine_person_mask_with_alpha(
    person_mask: np.ndarray,
    rmbg_alpha: np.ndarray,
    erode_iterations: int = 2,
    dilate_iterations: int = 2,
    kernel_size: int = 7,
    blur_size: int = 5,
) -> np.ndarray:
    person_mask = person_mask.astype(np.uint8)
    rmbg_alpha = rmbg_alpha.astype(np.uint8)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    inner_person = cv2.erode(person_mask, kernel, iterations=erode_iterations)
    outer_person = cv2.dilate(person_mask, kernel, iterations=dilate_iterations)

    final_alpha = np.zeros_like(rmbg_alpha)

    # The center of the detected person is protected from RMBG false negatives.
    final_alpha[inner_person > 0] = 255

    # RMBG is used only in the boundary band, where soft hair/clothing edges matter.
    edge_band = (outer_person > 0) & (inner_person == 0)
    final_alpha[edge_band] = rmbg_alpha[edge_band]

    final_alpha[outer_person == 0] = 0

    if blur_size > 0:
        if blur_size % 2 == 0:
            blur_size += 1
        final_alpha = cv2.GaussianBlur(final_alpha, (blur_size, blur_size), 0)

    return final_alpha


def save_rgba(image: Image.Image, alpha: np.ndarray, output_path: str) -> None:
    output = image.copy()
    output.putalpha(Image.fromarray(alpha, mode="L"))
    output.save(output_path)


def remove_background_from_image(
    image: Image.Image,
    yolo_model: str,
    alpha_model: str,
    conf: float,
    all_people: bool,
) -> tuple[Image.Image, np.ndarray, np.ndarray, np.ndarray, str]:
    source_alpha, alpha_source = get_alpha(image, alpha_model)
    person_mask = get_person_mask_yolo(
        image,
        model_name=yolo_model,
        conf=conf,
        include_all_people=all_people,
    )

    if person_mask.max() == 0:
        final_alpha = source_alpha
        print(f"No person mask found. Saved {alpha_source}-only result.")
    else:
        final_alpha = combine_person_mask_with_alpha(person_mask, source_alpha)

    output = image.copy()
    output.putalpha(Image.fromarray(final_alpha, mode="L"))
    return output, source_alpha, person_mask, final_alpha, alpha_source


def remove_background(
    input_path: str,
    output_path: str,
    yolo_model: str,
    alpha_model: str,
    conf: float,
    all_people: bool,
    debug_dir,
) -> None:
    image = ImageOps.exif_transpose(Image.open(input_path)).convert("RGB")
    output, source_alpha, person_mask, final_alpha, _ = remove_background_from_image(
        image=image,
        yolo_model=yolo_model,
        alpha_model=alpha_model,
        conf=conf,
        all_people=all_people,
    )

    save_rgba(image, final_alpha, output_path)

    if debug_dir:
        debug_path = Path(debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)
        Image.fromarray(source_alpha).save(debug_path / "source_alpha.png")
        Image.fromarray(person_mask).save(debug_path / "person_mask.png")
        Image.fromarray(final_alpha).save(debug_path / "final_alpha.png")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Person-aware background removal using YOLO person segmentation plus soft alpha matting."
    )
    parser.add_argument("input", help="Input image path")
    parser.add_argument(
        "-o",
        "--output",
        default="output_no_bg.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--yolo-model",
        default="yolo11x-seg.pt",
        help="Ultralytics segmentation model. Use yolo11x-seg.pt for quality or yolo11n-seg.pt for speed.",
    )
    parser.add_argument(
        "--alpha-model",
        default=DEFAULT_ALPHA_MODEL,
        help=(
            "Alpha model. Default is rembg:isnet-general-use. "
            "Use hf:briaai/RMBG-2.0 only after Hugging Face access is approved."
        ),
    )
    parser.add_argument(
        "--rmbg-model",
        default=None,
        help="Deprecated alias for --alpha-model hf:<model-name>",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Person segmentation confidence threshold",
    )
    parser.add_argument(
        "--single-person",
        action="store_true",
        help="Keep only the largest detected person",
    )
    parser.add_argument(
        "--debug-dir",
        default=None,
        help="Optional folder for source_alpha.png, person_mask.png, and final_alpha.png",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    alpha_model = args.alpha_model
    if args.rmbg_model:
        alpha_model = f"hf:{args.rmbg_model}"

    remove_background(
        input_path=args.input,
        output_path=args.output,
        yolo_model=args.yolo_model,
        alpha_model=alpha_model,
        conf=args.conf,
        all_people=not args.single_person,
        debug_dir=args.debug_dir,
    )
