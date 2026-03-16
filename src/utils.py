"""
utils.py — Utility Functions (v2)
==================================
Updated to support quality-mode comparisons, enhanced visualisation,
and richer image info reporting.
"""

import os
import sys
import urllib.request
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Optional


# ── Model download URLs ───────────────────────────────────────────────────────
MODEL_URLS = {
    "colorization_deploy_v2.prototxt": (
        "https://raw.githubusercontent.com/richzhang/colorization/"
        "caffe/colorization/models/colorization_deploy_v2.prototxt"
    ),
    "colorization_release_v2.caffemodel": (
    "https://huggingface.co/spaces/BilalSardar/Black-N-White-To-Color/"
    "resolve/main/colorization_release_v2.caffemodel"
),
    "pts_in_hull.npy": (
        "https://raw.githubusercontent.com/richzhang/colorization/"
        "caffe/colorization/resources/pts_in_hull.npy"
    ),
}

MODELS_DIR = Path(__file__).parent.parent / "models"


# =============================================================================
# Image I/O
# =============================================================================

def load_image(image_path: str) -> np.ndarray:
    """Load an image from disk as BGR. Raises if file is missing or unreadable."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"OpenCV could not decode: {image_path}")
    return img


def save_image(image_bgr: np.ndarray, output_path: str) -> None:
    """Save a BGR image to disk, creating parent directories as needed."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if not cv2.imwrite(output_path, image_bgr):
        raise IOError(f"Failed to save image: {output_path}")
    print(f"[✓] Saved: {output_path}")


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode an image from raw bytes (Streamlit upload)."""
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode uploaded image bytes.")
    return img


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """BGR (OpenCV) → RGB (Matplotlib / Streamlit / PIL)."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def to_display_gray(image_bgr: np.ndarray) -> np.ndarray:
    """Return the input image converted to 3-channel grayscale BGR."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# =============================================================================
# Visualisation
# =============================================================================

def show_comparison(
    original_bgr:  np.ndarray,
    colorized_bgr: np.ndarray,
    title:    str           = "AI Image Colorization",
    save_path: Optional[str] = None,
) -> None:
    """
    Display side-by-side original (grayscale) vs colourised image with Matplotlib.
    Optionally saves the figure if save_path is given.
    """
    fig = plt.figure(figsize=(12, 5), facecolor="#0f0f1a")
    fig.suptitle(title, fontsize=15, color="white", fontweight="bold")
    gs  = gridspec.GridSpec(1, 2, wspace=0.04)

    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(cv2.cvtColor(to_display_gray(original_bgr), cv2.COLOR_BGR2RGB))
    ax1.set_title("Original (Grayscale)", color="#aaaaaa", fontsize=11, pad=8)
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(bgr_to_rgb(colorized_bgr))
    ax2.set_title("AI Colorized", color="#00d2ff", fontsize=11, pad=8)
    ax2.axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[✓] Comparison saved: {save_path}")

    plt.show()


def show_quality_comparison(
    original_bgr:      np.ndarray,
    colorized_fast:    np.ndarray,
    colorized_hq:      np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """
    Three-way comparison: Original | Fast | High Quality.
    Useful during development to evaluate the improvement from guided filtering.
    """
    fig = plt.figure(figsize=(18, 5), facecolor="#0f0f1a")
    fig.suptitle("Quality Comparison: Fast vs High Quality", fontsize=14,
                 color="white", fontweight="bold")
    gs = gridspec.GridSpec(1, 3, wspace=0.03)

    panels = [
        (to_display_gray(original_bgr), "Original (Grayscale)", "#aaaaaa"),
        (colorized_fast,                "Fast Mode",            "#ffb347"),
        (colorized_hq,                  "High Quality Mode",    "#00d2ff"),
    ]

    for i, (img, label, colour) in enumerate(panels):
        ax = fig.add_subplot(gs[i])
        ax.imshow(bgr_to_rgb(img))
        ax.set_title(label, color=colour, fontsize=11, pad=8)
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[✓] Quality comparison saved: {save_path}")

    plt.show()


def create_side_by_side(
    original_bgr:  np.ndarray,
    colorized_bgr: np.ndarray,
    separator_px:  int = 4,
) -> np.ndarray:
    """
    Create a single BGR image: grayscale on the left, colorised on the right,
    separated by a thin white line. Useful for Streamlit display.
    """
    gray_bgr = to_display_gray(original_bgr)
    sep = np.ones((original_bgr.shape[0], separator_px, 3), dtype=np.uint8) * 220
    return np.hstack([gray_bgr, sep, colorized_bgr])


# =============================================================================
# Colour quality metrics
# =============================================================================

def measure_saturation(image_bgr: np.ndarray) -> dict:
    """
    Compute mean and std of the chrominance (saturation) of a BGR image in LAB.

    Returns a dict with 'mean_chroma' and 'std_chroma'.
    Higher mean_chroma → more colourful image.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a   = lab[:, :, 1] - 128.0
    b   = lab[:, :, 2] - 128.0
    chroma = np.sqrt(a**2 + b**2)
    return {
        "mean_chroma": float(chroma.mean()),
        "std_chroma":  float(chroma.std()),
        "max_chroma":  float(chroma.max()),
    }


# =============================================================================
# Image helpers
# =============================================================================

def validate_image(image: np.ndarray) -> None:
    """Basic sanity checks — raises ValueError if image is invalid."""
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Image must be a non-None NumPy array.")
    if image.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D array; got shape {image.shape}.")
    if image.size == 0:
        raise ValueError("Image array is empty.")


def resize_for_display(image: np.ndarray, max_dim: int = 800) -> np.ndarray:
    """
    Resize image so the longest side ≤ max_dim, preserving aspect ratio.
    Returns the original if already within limit.
    """
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / max(h, w)
    return cv2.resize(image, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA)


def get_image_info(image: np.ndarray) -> dict:
    """Return a dict of image properties (shape, dtype, value range, channels)."""
    return {
        "shape":    image.shape,
        "dtype":    str(image.dtype),
        "min":      int(image.min()),
        "max":      int(image.max()),
        "channels": image.shape[2] if image.ndim == 3 else 1,
        "megapixels": round(image.shape[0] * image.shape[1] / 1e6, 2),
    }


# =============================================================================
# Model downloader
# =============================================================================

def _progress_hook(fname: str):
    """Return a reporthook closure that prints a progress bar for urllib."""
    def hook(count, block_size, total_size):
        if total_size <= 0:
            return
        pct   = min(int(count * block_size * 100 / total_size), 100)
        bar   = "█" * (pct // 2) + "░" * (50 - pct // 2)
        sys.stdout.write(f"\r  [{bar}] {pct:3d}%  {fname}")
        sys.stdout.flush()
    return hook


def download_models(models_dir: Optional[str] = None, force: bool = False) -> bool:
    """
    Download all three model files if not already present.
    Returns True if all files exist after the call.
    """
    target = Path(models_dir) if models_dir else MODELS_DIR
    target.mkdir(parents=True, exist_ok=True)

    all_ok = True
    for fname, url in MODEL_URLS.items():
        dest = target / fname
        if dest.exists() and not force:
            print(f"[✓] Already present: {fname}")
            continue
        print(f"\n[↓] {fname}")
        print(f"    {url}")
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )},
            )
            with urllib.request.urlopen(req) as response:
                total   = int(response.headers.get("Content-Length", 0))
                written = 0
                hook    = _progress_hook(fname)
                block   = 8192
                with open(dest, "wb") as f:
                    while True:
                        chunk = response.read(block)
                        if not chunk:
                            break
                        f.write(chunk)
                        written += len(chunk)
                        hook(written // block, block, total)
            print()
        except Exception as exc:
            print(f"\n[✗] Download failed: {exc}")
            all_ok = False

    return all_ok


def check_models(models_dir: Optional[str] = None) -> dict:
    """Return {filename: bool_exists} for each required model file."""
    target = Path(models_dir) if models_dir else MODELS_DIR
    return {name: (target / name).exists() for name in MODEL_URLS}
