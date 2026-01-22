from typing import Dict, List
from pathlib import Path
from urllib.parse import urlparse
import base64
import io
from PIL import Image
import requests
from typing import Union
from typing import Optional


def open_image_from_any(src: str | Image.Image, *, timeout: int = 10) -> Image.Image:
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
    if isinstance(src, Image.Image):
        return src

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
        # Check if it's already a data URI
        if img.startswith("data:image/"):
            return img

        # Check if it's a URL
        parsed = urlparse(img)
        if parsed.scheme in {"http", "https"}:
            # Fetch the image from URL
            resp = requests.get(img, timeout=10)
            resp.raise_for_status()
            img_bytes = resp.content
            # Detect format from magic bytes
            detected_fmt = fmt or detect_image_format_from_bytes(img_bytes)
            return f"data:image/{detected_fmt.lower()};base64,{base64.b64encode(img_bytes).decode('utf-8')}"
        else:
            # Could be a file path or base64 string - use open_image_from_any to handle both
            try:
                pil_image = open_image_from_any(img)
                # Convert PIL image to data URI
                detected_fmt = pil_image.format or fmt or "PNG"
                buf = io.BytesIO()
                pil_image.save(buf, format=detected_fmt)
                b64 = base64.b64encode(buf.getvalue()).decode()
                return f"data:image/{detected_fmt.lower()};base64,{b64}"
            except Exception:
                # If open_image_from_any fails, return as is (might be raw base64)
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
    if img_bytes.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    elif img_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    elif img_bytes.startswith(b"GIF87a") or img_bytes.startswith(b"GIF89a"):
        return "GIF"
    elif img_bytes.startswith(b"RIFF") and img_bytes[8:12] == b"WEBP":
        return "WEBP"
    elif img_bytes.startswith(b"BM"):
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


def _is_jupyter() -> bool:
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if not shell:
            return False
        if "IPKernelApp" in shell.config:
            return True  # Jupyter Notebook or JupyterLab
        if shell.__class__.__name__ == "ZMQInteractiveShell":
            return True
        return False
    except Exception:
        return False


def _unicode_half_block(image: Image.Image, width: int):
    # if columns is None:
    #     try:
    #         columns = shutil.get_terminal_size((80, 24)).columns
    #     except Exception:
    #         columns = 80

    img = image.convert("RGB")
    target_w = min(width, img.width)
    aspect = img.height / img.width if img.width else 1.0
    target_h = int(round((aspect * target_w) * 2))
    img = img.resize((target_w, target_h), Image.BICUBIC)
    print(f"Image resized to {target_w}x{target_h}")

    reset = "\033[0m"
    for y in range(0, img.height, 2):
        line = ""
        for x in range(img.width):
            top = img.getpixel((x, y))
            bottom = img.getpixel((x, y + 1)) if y + 1 < img.height else (0, 0, 0)
            line += f"\033[38;2;{top[0]};{top[1]};{top[2]}m\033[48;2;{bottom[0]};{bottom[1]};{bottom[2]}m▀"
        print(line + reset)


def _jupyter_display(image: Image.Image):
    from IPython.display import display

    display(image)


def display_image(
    path_or_image: str | Image.Image,
    method: str = "auto",
    width: Optional[str] = 80,
):
    """
    Display an image intelligently depending on environment.
    Supports: Jupyter, Kitty, iTerm2, WezTerm, or plain terminals (Unicode fallback).
    """
    image = open_image_from_any(path_or_image)

    chosen = method
    if method == "auto":
        if _is_jupyter():
            chosen = "jupyter"
        else:
            chosen = "unicode"

    try:
        if chosen == "jupyter":
            _jupyter_display(image)
        elif chosen == "unicode":
            _unicode_half_block(image, width=width)
        else:
            raise ValueError(f"Unknown method: {chosen}")
    except Exception:
        if chosen != "unicode":
            try:
                _unicode_half_block(image, width=width)
            except Exception:
                pass
        raise


def display_messages(messages: List[Dict]):
    for i, message in enumerate(messages):
        print(f"{'=' * 40} Turn {i} {'=' * 40}")
        role = message["role"]
        print(f"{role}: ", end="")

        content = message["content"]
        if isinstance(content, str):
            print(content)
        elif isinstance(content, list):
            for item in content:
                if item["type"] == "text":
                    print(item["text"])
                elif item["type"] == "image":
                    display_image(item["image"])
                elif item["type"] == "image_url":
                    display_image(item["image_url"])
                else:
                    raise ValueError(f"Invalid message type: {item['type']}")
        else:
            print(content)

        if "tool_calls" in message:
            print("Tool calls: ", end="")
            print(message["tool_calls"])
