from pathlib import Path
from urllib.parse import urlparse
import base64
import io
from PIL import Image
import requests
from typing import Union

def open_image_from_any(src: str, *, timeout: int = 10) -> Image.Image:
    """
    Open an image from a file path, URL, or base-64 string with Pillow.

    Parameters
    ----------
    src : str
        The image source.  It can be:
          • path to an image on disk  
          • http(s) URL  
          • plain base-64 or data-URI base-64
    timeout : int, optional
        HTTP timeout (s) when downloading from a URL.

    Returns
    -------
    PIL.Image.Image
    """
    parsed = urlparse(src)

    # 1) Detect a URL ----------------------------------------------------------------
    if parsed.scheme in {"http", "https"}:
        # --- requests version
        resp = requests.get(src, timeout=timeout)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content))

        # --- urllib version (uncomment if you can’t pip-install requests)
        # with urllib_request.urlopen(src, timeout=timeout) as fp:
        #     return Image.open(fp)

    # 2) Detect a base-64 string ------------------------------------------------------
    #    • data-URI style:  "data:image/png;base64,……"
    #    • bare base-64    :  "iVBORw0KGgoAAAANSUhEUgAABVYA…"
    try:
        # Strip header if present
        if src.startswith("data:"):
            header, b64 = src.split(",", 1)
        else:
            b64 = src

        # “validate=True” quickly rejects non-b64 text without decoding everything
        img_bytes = base64.b64decode(b64, validate=True)
        return Image.open(io.BytesIO(img_bytes))

    except (base64.binascii.Error, ValueError):
        # Not base-64 → fall through to path handling
        pass

    # 3) Treat it as a local file path ----------------------------------------------
    path = Path(src).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")
    return Image.open(path)


def image_to_data_uri(img: Union[Image.Image, str, dict], fmt=None) -> str:
    if isinstance(img, dict):
        if "bytes" in img:
            img = img["bytes"]

    if isinstance(img, Image.Image):
        # Try to detect format from PIL Image first
        detected_fmt = img.format or fmt or "PNG"
        buf = io.BytesIO()
        img.save(buf, format=detected_fmt)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/{detected_fmt.lower()};base64,{b64}"
    elif isinstance(img, str):
        return img
    elif isinstance(img, bytes):
        # Try to detect format from magic bytes
        detected_fmt = fmt or detect_image_format_from_bytes(img)
        return f"data:image/{detected_fmt.lower()};base64,{base64.b64encode(img).decode('utf-8')}"
    else:
        raise ValueError(f"Invalid image type: {type(img)}")

def detect_image_format_from_bytes(img_bytes: bytes) -> str:
    """Detect image format from bytes using magic numbers"""
    if len(img_bytes) < 4:
        return "PNG"  # Default fallback
    
    # Check magic bytes for common formats
    if img_bytes.startswith(b'\xff\xd8\xff'):
        return "JPEG"
    elif img_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
        return "PNG"
    elif img_bytes.startswith(b'GIF87a') or img_bytes.startswith(b'GIF89a'):
        return "GIF"
    elif img_bytes.startswith(b'RIFF') and img_bytes[8:12] == b'WEBP':
        return "WEBP"
    elif img_bytes.startswith(b'BM'):
        return "BMP"
    else:
        return "PNG"  # Default fallback

def image_to_pil(img: Union[Image.Image, str, dict]) -> Image.Image:
    if isinstance(img, str):
        return open_image_from_any(img)
    elif isinstance(img, dict):
        return open_image_from_any(img["bytes"])
    else:
        return img