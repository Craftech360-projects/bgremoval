# Person-aware background removal

This project removes image backgrounds while protecting the detected person body.
It fixes the common failure where a dress or shirt disappears because it has a
similar color to the background.

The script uses two signals:

- `rembg:isnet-general-use` for a soft alpha matte by default.
- `YOLO segmentation` for a hard person mask.

The final alpha keeps the inside of the detected person fully visible, uses RMBG
only around the edge band for soft hair/clothing boundaries, and removes
everything outside the person area.

## Install

```bash
pip install -r requirements.txt
```

For best speed, install a CUDA-enabled PyTorch build that matches your GPU.

## Run

```bash
python person_aware_bg_remove.py captured_image.jpg -o output_no_bg.png --debug-dir debug
```

## Run as backend for React

The frontend can call:

```text
POST http://127.0.0.1:8765/remove-bg
```

Start the backend with:

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8765
```

The endpoint accepts multipart `FormData`. It looks for the foreground image in
common fields like `image`, `file`, `photo`, `capture`, or `cameraImage`. It can
also accept an optional background in fields like `background`,
`backgroundImage`, `bg`, or `selectedBackground`.

The response is a PNG image. If no background is sent, the PNG keeps
transparency. If a background image, data URL, or CSS color is sent, the backend
composites the cutout onto it before returning the PNG.

Quality-first mode uses `yolo11x-seg.pt` by default. For faster but weaker
results:

```bash
python person_aware_bg_remove.py captured_image.jpg -o output_no_bg.png --yolo-model yolo11n-seg.pt
```

If the image contains multiple people, all are kept by default. To keep only the
largest detected person:

```bash
python person_aware_bg_remove.py captured_image.jpg -o output_no_bg.png --single-person
```

## Optional RMBG-2.0

`briaai/RMBG-2.0` is a gated Hugging Face model. If you do not have access, you
will get a `403 Forbidden` error. The default command above avoids that by using
`rembg:isnet-general-use`.

After Hugging Face grants access to `briaai/RMBG-2.0`, log in or set `HF_TOKEN`,
then run:

```bash
python person_aware_bg_remove.py captured_image.jpg -o output_no_bg.png --alpha-model hf:briaai/RMBG-2.0
```

## Why this works better than alpha matting alone

Alpha matting alone can treat clothes as background when the clothing color and
background color are close. The person mask is used as a foreground guarantee:

- Inner person area: alpha is forced to `255`.
- Person boundary band: RMBG alpha is used for natural soft edges.
- Outside the person mask: alpha is forced to `0`.

Use the files in `debug/` to inspect each stage:

- `source_alpha.png`
- `person_mask.png`
- `final_alpha.png`
