"""
colorize.py — Refactored Colorization Engine (v2)
==================================================
Improvements over v1:
  1. Guided Filter Upsampling  — upsamples (a,b) using the sharp L channel as an
     edge guide, preventing color from bleeding across object boundaries.
  2. ColorizeOptions dataclass  — all quality knobs in one place with Fast /
     Balanced / High Quality presets.
  3. Tiled inference  — for images larger than a threshold, tiles are colorized
     independently and blended with smooth Gaussian weights to avoid seams.
  4. Vibrance boost  — non-linear saturation lift that targets dull pixels most,
     preserving already-vivid areas.
  5. float32 precision throughout; only cast to uint8 at the very end.

Why guided filter?
  Bilinear resize of (a,b) from 224x224 up to e.g. 1920x1080 smears color
  across edges (blue sky bleeds into the dark tree line below it).
  The guided filter uses the full-resolution L channel — which knows every
  edge — to steer the upsampling: color only propagates to neighbours that
  share similar lightness.

  He et al. (2013) guided filter formula:
      a_k = Cov(I, p) / (Var(I) + eps)
      b_k = mean(p)   - a_k * mean(I)
      output q_i = mean(a)_i * I_i + mean(b)_i
  where I = guide (L channel), p = source channel to upsample, eps = reg term.
"""

from __future__ import annotations
import cv2
import numpy as np
import os
from dataclasses import dataclass
from typing import Optional, Tuple

# ── Model file paths ──────────────────────────────────────────────────────────
_SRC_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.normpath(os.path.join(_SRC_DIR, "..", "models"))
PROTOTXT    = os.path.join(MODELS_DIR, "colorization_deploy_v2.prototxt")
CAFFEMODEL  = os.path.join(MODELS_DIR, "colorization_release_v2.caffemodel")
CLUSTER_PTS = os.path.join(MODELS_DIR, "pts_in_hull.npy")

MODEL_INPUT_SIZE = 224   # fixed input size for Zhang et al. network


# =============================================================================
# Configuration dataclass
# =============================================================================

@dataclass
class ColorizeOptions:
    """
    All quality and processing settings for colorization.

    Attributes:
        guided_filter_radius : Guided filter window radius. Larger = smoother
                               colour transitions. Range 1-32. Default 8.
                               Set to 0 to disable guided filter entirely.
        guided_filter_eps    : Regularisation term. Larger = more smoothing
                               (weak edges treated as flat). Default 1e-2.
        vibrance_strength    : Non-linear sat boost (0=off, 1=full).
                               Lifts dull colours more than vivid ones.
                               Range 0.0-2.0.  Default 0.6.
        saturation_scale     : Linear saturation multiplier applied after
                               vibrance. 1.0 = no change.  Default 1.1.
        bilateral_smoothing  : Bilateral filter on final image — reduces
                               colour noise in flat areas. Default True.
        bilateral_d          : Bilateral filter neighbourhood diameter.
        use_tiling           : Enable tiled inference for large images.
        tile_threshold       : Images with max(H,W) > this are tiled.
        tile_size            : Tile size in pixels (square).
        tile_overlap         : Overlap between adjacent tiles.
        high_res_upsample    : Interpolation flag for initial (a,b) resize
                               before guided filter. LANCZOS4 by default.
    """
    guided_filter_radius: int   = 8
    guided_filter_eps:    float = 1e-2
    vibrance_strength:    float = 0.6
    saturation_scale:     float = 1.1
    bilateral_smoothing:  bool  = True
    bilateral_d:          int   = 9
    use_tiling:           bool  = True
    tile_threshold:       int   = 600
    tile_size:            int   = 400
    tile_overlap:         int   = 64
    high_res_upsample:    int   = cv2.INTER_LANCZOS4

    @classmethod
    def fast(cls) -> "ColorizeOptions":
        """Speed-first: no tiling, no guided filter, minimal enhancement."""
        return cls(
            guided_filter_radius=0,
            vibrance_strength=0.3,
            saturation_scale=1.0,
            bilateral_smoothing=False,
            use_tiling=False,
        )

    @classmethod
    def balanced(cls) -> "ColorizeOptions":
        """Default balanced quality/speed tradeoff."""
        return cls()

    @classmethod
    def high_quality(cls) -> "ColorizeOptions":
        """Maximum quality: larger guided filter, stronger enhancement, tiling."""
        return cls(
            guided_filter_radius=16,
            guided_filter_eps=5e-3,
            vibrance_strength=0.9,
            saturation_scale=1.2,
            bilateral_smoothing=True,
            bilateral_d=11,
            use_tiling=True,
            tile_threshold=500,
            tile_size=350,
            tile_overlap=80,
        )


# =============================================================================
# Model loading
# =============================================================================

def load_model() -> cv2.dnn_Net:
    """
    Load the pretrained Zhang et al. (2016) colorization network.

    The 313 quantised (a,b) colour-cluster centres (pts_in_hull.npy) are
    injected as fixed blobs into the 'class8_ab' and 'conv8_313_rh' layers
    so the network can convert bin-probability outputs into actual (a,b) values.

    Returns:
        net : Configured OpenCV DNN network, ready for inference.
    Raises:
        FileNotFoundError if any model file is missing.
    """
    missing = [p for p in (PROTOTXT, CAFFEMODEL, CLUSTER_PTS) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing model files:\n" +
            "\n".join("  • " + p for p in missing) +
            "\n\nRun:  python download_models.py"
        )

    net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)

    pts = np.load(CLUSTER_PTS)                         # (313, 2)
    pts = pts.transpose().reshape(2, 313, 1, 1)        # (2, 313, 1, 1)

    net.getLayer(net.getLayerId("class8_ab")).blobs = [
        pts.astype(np.float32)
    ]
    net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [
        np.full([1, 313], 2.606, dtype=np.float32)     # annealed rebalancing
    ]
    return net


# =============================================================================
# Guided filter  (He et al. 2013 — no opencv-contrib required)
# =============================================================================

def _guided_filter(guide: np.ndarray, src: np.ndarray,
                   radius: int, eps: float) -> np.ndarray:
    """
    Fast guided filter using box-filter approximation.

    guide and src must both be float32.
    guide should be normalised to [0, 1] (the L channel / 255).

    The filter finds linear local models q_i = a_i*I_i + b_i such that q
    follows the guide's edges while smoothing src's low-frequency noise.
    """
    ksize   = (2 * radius + 1, 2 * radius + 1)
    guide_f = guide.astype(np.float32)
    src_f   = src.astype(np.float32)

    # Local means
    mean_I  = cv2.boxFilter(guide_f,          cv2.CV_32F, ksize)
    mean_p  = cv2.boxFilter(src_f,            cv2.CV_32F, ksize)
    mean_Ip = cv2.boxFilter(guide_f * src_f,  cv2.CV_32F, ksize)
    mean_II = cv2.boxFilter(guide_f * guide_f, cv2.CV_32F, ksize)

    # Covariance / variance
    cov_Ip = mean_Ip - mean_I * mean_p   # Cov(I, p)
    var_I  = mean_II - mean_I * mean_I   # Var(I)

    # Linear model coefficients
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    # Average over overlapping windows
    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)

    return mean_a * guide_f + mean_b


def _guided_upsample_ab(
    l_full:   np.ndarray,          # (H, W) float32, cv2 scale [0..255]
    ab_low:   np.ndarray,          # (2, h, w) model output range ~[-110..110]
    target_h: int,
    target_w: int,
    opts:     ColorizeOptions,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Upsample (a,b) to original resolution using the L channel as edge guide.

    1. Lanczos resize to target size (better starting point than bilinear)
    2. Normalise L to [0,1] guide
    3. Apply guided filter independently to a and b channels
    """
    # Step 1: Initial high-quality resize
    a_up = cv2.resize(ab_low[0], (target_w, target_h),
                      interpolation=opts.high_res_upsample).astype(np.float32)
    b_up = cv2.resize(ab_low[1], (target_w, target_h),
                      interpolation=opts.high_res_upsample).astype(np.float32)

    if opts.guided_filter_radius < 1:
        return a_up, b_up   # guided filter disabled in fast mode

    # Step 2+3: Guided filter with L as edge guide
    guide = (l_full / 255.0).astype(np.float32)
    a_up  = _guided_filter(guide, a_up, opts.guided_filter_radius, opts.guided_filter_eps)
    b_up  = _guided_filter(guide, b_up, opts.guided_filter_radius, opts.guided_filter_eps)

    return a_up, b_up


# =============================================================================
# Colour enhancement
# =============================================================================

def _apply_vibrance(lab_float: np.ndarray,
                    strength: float,
                    sat_scale: float) -> np.ndarray:
    """
    Non-linear vibrance boost in LAB space.

    Unlike plain saturation scaling, vibrance applies a stronger boost to
    *less-saturated* pixels — lifting dull areas while leaving vivid colours
    unchanged.  This avoids the blown-out look of naive saturation multiplication.

    Boost formula:
        chroma  = sqrt(a^2 + b^2)                    current saturation
        boost   = 1 + strength * (1 - chroma/128)^2  larger for low-sat pixels
        a_new   = a * boost * sat_scale
        b_new   = b * boost * sat_scale

    Args:
        lab_float : Float32 LAB image with cv2 scale (L:[0,255], a/b:[0,255]).
        strength  : Vibrance strength (0 = off, 1 = full).
        sat_scale : Linear saturation multiplier on top of vibrance.
    """
    lab = lab_float.copy()

    # Centre a,b around zero (cv2 stores them offset by 128)
    a_c = lab[:, :, 1].astype(np.float32) - 128.0
    b_c = lab[:, :, 2].astype(np.float32) - 128.0

    chroma = np.sqrt(a_c ** 2 + b_c ** 2)          # [0, ~180]

    # Non-linear boost: quadratic ramp from 1+strength (gray) → 1 (max chroma)
    boost = 1.0 + strength * ((1.0 - chroma / 128.0) ** 2)

    lab[:, :, 1] = np.clip(a_c * boost * sat_scale + 128.0, 0, 255)
    lab[:, :, 2] = np.clip(b_c * boost * sat_scale + 128.0, 0, 255)
    return lab


# =============================================================================
# Pipeline stages
# =============================================================================

def _ensure_bgr3(img: np.ndarray) -> np.ndarray:
    """Guarantee input is 3-channel BGR."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _preprocess(image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    BGR → LAB, extract full-res L, build 224x224 DNN input blob.

    Returns
    -------
    l_full        Float32 L channel at original resolution, cv2 scale [0..255].
    blob          (1, 1, 224, 224) float32 blob with mean 50 subtracted.
    original_size (H, W) of the input image.
    """
    H, W  = image_bgr.shape[:2]
    lab   = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_full = lab[:, :, 0]                             # retain at full resolution

    # Normalise: cv2 L [0,255] → model L [0,100] → subtract mean 50
    l_norm  = l_full / 255.0 * 100.0
    l_small = cv2.resize(l_norm, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
    l_input = l_small - 50.0                          # mean-centre

    blob = cv2.dnn.blobFromImage(l_input)             # (1, 1, 224, 224)
    return l_full, blob, (H, W)


def _infer(net: cv2.dnn_Net, blob: np.ndarray) -> np.ndarray:
    """Forward pass.  Returns ab of shape (2, 224, 224)."""
    net.setInput(blob)
    return net.forward()[0]   # remove batch dim


def _postprocess(
    l_full:        np.ndarray,
    ab_low:        np.ndarray,
    original_size: Tuple[int, int],
    opts:          ColorizeOptions,
) -> np.ndarray:
    """
    Build the final BGR image from model (a,b) output.

    1. Guided-filter upsample (a,b) to original size
    2. Shift (a,b) from model range to cv2 LAB range [0,255]
    3. Stack [L, a, b] as float LAB
    4. Vibrance boost
    5. Convert LAB -> BGR
    6. Optional bilateral denoising
    """
    H, W = original_size

    # 1. Edge-guided upsampling
    a_up, b_up = _guided_upsample_ab(l_full, ab_low, H, W, opts)

    # 2. Shift: model output is ~[-128,127]; cv2 LAB stores a,b in [0,255]
    a_cv2 = np.clip(a_up + 128.0, 0, 255)
    b_cv2 = np.clip(b_up + 128.0, 0, 255)
    l_cv2 = np.clip(l_full, 0, 255)

    # 3. Float LAB image
    lab_f = np.stack([l_cv2, a_cv2, b_cv2], axis=2).astype(np.float32)

    # 4. Vibrance
    if opts.vibrance_strength > 0 or opts.saturation_scale != 1.0:
        lab_f = _apply_vibrance(lab_f, opts.vibrance_strength, opts.saturation_scale)

    # 5. LAB -> BGR
    result = cv2.cvtColor(lab_f.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # 6. Bilateral smoothing (reduces colour noise, preserves edges)
    if opts.bilateral_smoothing:
        result = cv2.bilateralFilter(result, d=opts.bilateral_d,
                                     sigmaColor=35, sigmaSpace=35)
    return result


# =============================================================================
# Tiled inference for high-resolution images
# =============================================================================

def _tile_blend_weights(h: int, w: int, overlap: int) -> np.ndarray:
    """
    (H, W, 1) weight map for one tile: 1.0 in the centre, linear ramp
    down to 0 at each border within the overlap margin.
    Overlapping tiles weighted by these maps blend seamlessly.
    """
    wmap = np.ones((h, w), dtype=np.float32)
    o = max(1, overlap)
    for i in range(min(o, h)):
        t = i / o
        wmap[i,  :]  = np.minimum(wmap[i,  :],  t)
        wmap[-1-i, :] = np.minimum(wmap[-1-i, :], t)
    for j in range(min(o, w)):
        t = j / o
        wmap[:,  j] = np.minimum(wmap[:,  j],  t)
        wmap[:, -1-j] = np.minimum(wmap[:, -1-j], t)
    return wmap[:, :, np.newaxis]


def _colorize_single(image_bgr: np.ndarray,
                     net: cv2.dnn_Net,
                     opts: ColorizeOptions) -> np.ndarray:
    """Colorize one image or tile — no tiling logic."""
    l_full, blob, size = _preprocess(image_bgr)
    ab_low             = _infer(net, blob)
    return _postprocess(l_full, ab_low, size, opts)


def _colorize_tiled(image_bgr: np.ndarray,
                    net: cv2.dnn_Net,
                    opts: ColorizeOptions) -> np.ndarray:
    """
    Colorize a high-res image via overlapping tiles with smooth blending.

    Why tiling improves quality:
        At 224x224 the model has to represent the *entire* image semantics.
        A 2000x1500 photo squished to 224x224 loses critical local detail.
        Feeding tile-sized crops lets the model work at a scale where it can
        recognise textures, faces, vegetation, etc., producing richer colours.

    Blending strategy:
        Every tile is assigned a weight map (linear ramp at borders, 1 at centre).
        Accumulated colour / accumulated weight = smooth blend with no seams.
    """
    H, W = image_bgr.shape[:2]
    ts, ov = opts.tile_size, opts.tile_overlap
    stride = max(1, ts - ov)

    acc_color = np.zeros((H, W, 3), dtype=np.float32)
    acc_wgt   = np.zeros((H, W, 1), dtype=np.float32)

    ys = list(range(0, max(1, H - ts), stride)) + [max(0, H - ts)]
    xs = list(range(0, max(1, W - ts), stride)) + [max(0, W - ts)]
    # De-duplicate while preserving order
    ys = list(dict.fromkeys(ys))
    xs = list(dict.fromkeys(xs))

    for y0 in ys:
        for x0 in xs:
            y1 = min(y0 + ts, H)
            x1 = min(x0 + ts, W)
            tile = image_bgr[y0:y1, x0:x1]

            tile_result = _colorize_single(tile, net, opts)
            wmap        = _tile_blend_weights(y1 - y0, x1 - x0, ov)

            acc_color[y0:y1, x0:x1] += tile_result.astype(np.float32) * wmap
            acc_wgt[y0:y1, x0:x1]   += wmap

    result = acc_color / np.maximum(acc_wgt, 1e-6)
    return result.clip(0, 255).astype(np.uint8)


# =============================================================================
# Public API
# =============================================================================

def colorize_image(
    image_bgr: np.ndarray,
    net:       cv2.dnn_Net,
    opts:      Optional[ColorizeOptions] = None,
) -> np.ndarray:
    """
    Colorize a grayscale image end-to-end.

    Automatically chooses tiled or single-pass inference based on image size
    and the opts.use_tiling / opts.tile_threshold settings.

    Args:
        image_bgr : BGR image from cv2.imread (grayscale-valued but 3-channel).
        net       : Model returned by load_model().
        opts      : Quality settings; defaults to ColorizeOptions.balanced().

    Returns:
        Colorized BGR image, same spatial dimensions as input.
    """
    if opts is None:
        opts = ColorizeOptions.balanced()

    image_bgr = _ensure_bgr3(image_bgr)
    H, W = image_bgr.shape[:2]

    if opts.use_tiling and max(H, W) > opts.tile_threshold:
        return _colorize_tiled(image_bgr, net, opts)
    return _colorize_single(image_bgr, net, opts)


def image_to_grayscale(image_bgr: np.ndarray) -> np.ndarray:
    """Return a 3-channel BGR image rendered as grayscale."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
