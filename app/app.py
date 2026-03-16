"""
app.py — Streamlit Web Interface (v2)
======================================
Run with:
    streamlit run app/app.py

New in v2:
  • Quality mode selector: Fast / Balanced / High Quality
  • Vibrance + saturation sliders with live preview
  • Colour metrics panel (mean chroma before vs after)
  • Tiling info badge for high-res images
  • Clean dark UI with labelled enhancement controls
"""

import os, sys, time, io
import cv2
import numpy as np
import streamlit as st
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from colorize import load_model, colorize_image, ColorizeOptions
from utils    import (
    load_image_from_bytes, bgr_to_rgb, download_models, check_models,
    resize_for_display, get_image_info, measure_saturation, to_display_gray,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Image Colorization",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #0d0d1a; color: #dde1f0; }
[data-testid="stSidebar"] {
    background: linear-gradient(170deg, #12122b 0%, #0d1a2e 100%);
    border-right: 1px solid #1e2a45;
}
.hero {
    text-align: center; padding: 1.8rem 1rem 1rem;
    background: linear-gradient(135deg, #12122b, #0d1a2e);
    border-radius: 14px; margin-bottom: 1.5rem;
    border: 1px solid #1e2a45;
}
.hero h1 {
    font-size: 2.4rem; margin: 0;
    background: linear-gradient(90deg, #00d2ff, #c77dff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p { color: #7a88b0; font-size: 0.95rem; margin-top: .4rem; }
.badge {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 700; margin: 2px;
}
.badge-ok     { background: #00b894; color: #fff; }
.badge-miss   { background: #d63031; color: #fff; }
.badge-tile   { background: #0984e3; color: #fff; }
.badge-fast   { background: #fdcb6e; color: #1a1a1a; }
.badge-hq     { background: #6c5ce7; color: #fff; }
.metric-card {
    background: #12122b; border: 1px solid #1e2a45;
    border-radius: 10px; padding: .9rem 1.2rem; text-align: center;
}
.metric-val { font-size: 1.5rem; font-weight: 700; color: #00d2ff; }
.metric-lbl { font-size: 0.75rem; color: #7a88b0; margin-top: 2px; }
.img-cap { text-align:center; font-size:.8rem; color:#7a88b0; margin-top:4px; }
hr { border-color: #1e2a45; }
.stButton > button {
    background: linear-gradient(90deg,#00d2ff,#c77dff);
    color: #0d0d1a; border: none; border-radius: 8px;
    font-weight: 800; padding: .45rem 1.8rem; width: 100%;
}
</style>
""", unsafe_allow_html=True)


# ── Cached model ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model():
    return load_model()


def to_png_bytes(image_bgr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(bgr_to_rgb(image_bgr)).save(buf, format="PNG")
    return buf.getvalue()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.markdown("---")

    # Model status
    st.markdown("### 📦 Model Files")
    model_status = check_models()
    all_present  = all(model_status.values())
    for fname, ok in model_status.items():
        cls = "badge-ok" if ok else "badge-miss"
        lbl = "Ready" if ok else "Missing"
        st.markdown(f'<span class="badge {cls}">{lbl}</span> `{fname}`',
                    unsafe_allow_html=True)

    if not all_present:
        st.markdown("---")
        if st.button("⬇️ Download Models"):
            with st.spinner("Downloading (~130 MB)…"):
                ok = download_models()
            (st.success if ok else st.error)(
                "Done — refresh the page." if ok else "Download failed. Check connection."
            )

    st.markdown("---")

    # ── Quality preset ────────────────────────────────────────────────────────
    st.markdown("### 🎚️ Quality Mode")
    quality_mode = st.radio(
    "Select quality mode",
    ["⚡ Fast", "⚖️ Balanced", "💎 High Quality"],
    label_visibility="collapsed",   # hides label visually but satisfies accessibility
        index=1,
        help=(
            "Fast: no guided filter, no tiling (~0.3s)\n"
            "Balanced: guided filter r=8, tiling >600px (~1–3s)\n"
            "High Quality: guided filter r=16, aggressive tiling (~3–8s)"
        ),
    )

    st.markdown("---")

    # ── Fine-tune sliders ─────────────────────────────────────────────────────
    st.markdown("### 🌈 Enhancement")

    vibrance = st.slider(
        "Vibrance",
        min_value=0.0, max_value=2.0, value=0.6, step=0.05,
        help="Non-linear saturation boost — targets dull pixels most.\n"
             "0 = off, 1 = standard, 2 = very vivid.",
    )
    saturation = st.slider(
        "Saturation scale",
        min_value=0.5, max_value=2.5, value=1.1, step=0.05,
        help="Linear multiplier applied on top of vibrance.\n"
             "1.0 = no change, >1 = more saturated.",
    )
    guided_r = st.slider(
        "Guided filter radius",
        min_value=0, max_value=32, value=8, step=1,
        help="Larger = smoother colour boundaries.\n"
             "0 disables the guided filter (faster but more colour bleed).",
    )

    st.markdown("---")

    # ── Display ───────────────────────────────────────────────────────────────
    st.markdown("### 🖥️ Display")
    max_disp = st.slider("Max display size (px)", 300, 1200, 650, 50)

    st.markdown("---")
    st.markdown("""
**AI Image Colorization v2**
- 🧠 Zhang et al. (2016)
- 🔬 Guided filter upsampling
- 🔲 Tiled high-res inference
- 🌈 Vibrance enhancement
    """)


# ── Build ColorizeOptions from sidebar ────────────────────────────────────────
if   "Fast"         in quality_mode:
    opts = ColorizeOptions.fast()
elif "High Quality" in quality_mode:
    opts = ColorizeOptions.high_quality()
else:
    opts = ColorizeOptions.balanced()

# Override with manual sliders
opts.vibrance_strength    = vibrance
opts.saturation_scale     = saturation
opts.guided_filter_radius = guided_r


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🎨 AI Image Colorization</h1>
  <p>Upload a black &amp; white photo — guided deep learning brings it to life</p>
</div>
""", unsafe_allow_html=True)


# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload a grayscale or B&W image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

sample_path = os.path.join(PROJECT_ROOT, "images", "sample_bw.jpg")
use_sample  = False

if not uploaded:
    if os.path.exists(sample_path):
        c1, _ = st.columns([1, 4])
        with c1:
            if st.button("🖼️ Try Sample Image"):
                use_sample = True
    else:
        st.info("Upload an image above, or place a test image at `images/sample_bw.jpg`.")


# ── Main colorization block ────────────────────────────────────────────────────
if uploaded or use_sample:

    # Load
    if use_sample:
        image_bgr = cv2.imread(sample_path)
        filename  = "sample_bw.jpg"
    else:
        image_bgr = load_image_from_bytes(uploaded.read())  # type: ignore[union-attr]
        filename  = uploaded.name                            # type: ignore[union-attr]

    if image_bgr is None:
        st.error("❌ Could not decode the image.")
        st.stop()

    if not all_present:
        st.error("❌ Model files missing — use the sidebar to download them.")
        st.stop()

    # Load model
    with st.spinner("Loading model…"):
        try:
            net = get_model()
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()

    # Detect high-res & show tiling badge
    H, W = image_bgr.shape[:2]
    will_tile = opts.use_tiling and max(H, W) > opts.tile_threshold
    mode_info_col, _ = st.columns([3, 1])
    with mode_info_col:
        badges = ""
        if "Fast" in quality_mode:
            badges += '<span class="badge badge-fast">⚡ Fast</span>'
        elif "High Quality" in quality_mode:
            badges += '<span class="badge badge-hq">💎 High Quality</span>'
        else:
            badges += '<span class="badge badge-ok">⚖️ Balanced</span>'
        if will_tile:
            badges += f'<span class="badge badge-tile">🔲 Tiled ({opts.tile_size}px tiles)</span>'
        st.markdown(badges, unsafe_allow_html=True)

    # Run
    with st.spinner("🎨 Colorizing…"):
        t0 = time.time()
        colorized_bgr = colorize_image(image_bgr, net, opts)
        elapsed = time.time() - t0

    st.success(f"✅ Done in **{elapsed:.2f}s** — {W}×{H}px image")

    # ── Metrics ───────────────────────────────────────────────────────────────
    st.markdown("---")
    sat_before = measure_saturation(to_display_gray(image_bgr))
    sat_after  = measure_saturation(colorized_bgr)

    m1, m2, m3, m4, m5 = st.columns(5)
    cards = [
        (f"{W}×{H}", "Resolution"),
        (f"{get_image_info(image_bgr)['megapixels']} MP", "Megapixels"),
        (f"{elapsed:.2f}s", "Process time"),
        (f"{sat_before['mean_chroma']:.1f}", "Chroma (before)"),
        (f"{sat_after['mean_chroma']:.1f}", "Chroma (after)"),
    ]
    for col, (val, lbl) in zip([m1, m2, m3, m4, m5], cards):
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-val">{val}</div>'
            f'<div class="metric-lbl">{lbl}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Side-by-side display ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔍 Result")

    orig_disp  = resize_for_display(image_bgr,    max_disp)
    color_disp = resize_for_display(colorized_bgr, max_disp)

    col1, col2 = st.columns(2)
    with col1:
        st.image(bgr_to_rgb(to_display_gray(orig_disp)), width='stretch')
        st.markdown('<p class="img-cap">⬛ Original (Grayscale)</p>', unsafe_allow_html=True)
    with col2:
        st.image(bgr_to_rgb(color_disp), width='stretch')
        st.markdown('<p class="img-cap">🌈 AI Colorized</p>', unsafe_allow_html=True)

    # ── Download ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ⬇️ Download")

    base = os.path.splitext(filename)[0]
    d1, d2 = st.columns(2)

    with d1:
        st.download_button(
            "📥 Download Colorized Image",
            data=to_png_bytes(colorized_bgr),
            file_name=f"{base}_colorized.png",
            mime="image/png",
            width='stretch',
        )
    with d2:
        gray_bgr = to_display_gray(image_bgr)
        sep      = np.ones((H, 6, 3), dtype=np.uint8) * 180
        combo    = np.hstack([gray_bgr, sep, colorized_bgr])
        st.download_button(
            "📥 Download Comparison",
            data=to_png_bytes(combo),
            file_name=f"{base}_comparison.png",
            mime="image/png",
            width='stretch',
        )

    # ── Explainer ─────────────────────────────────────────────────────────────
    with st.expander("🧠 What's new in v2? (click to expand)"):
        st.markdown("""
**v2 Improvements over v1:**

| Feature | v1 | v2 |
|---|---|---|
| (a,b) upsampling | Bilinear resize | **Guided filter** (edge-aware) |
| Saturation | None | **Vibrance boost** (non-linear) |
| Large images | Squish to 224px | **Tiled inference** + blend |
| Precision | uint8 throughout | **float32** until final cast |
| Interpolation | BILINEAR | **LANCZOS4** |
| Noise reduction | None | **Bilateral filter** on output |

**Guided Filter (the key improvement):**
The model always outputs a 224×224 colour map.  Simply resizing that to
1920×1080 smears colour across edges.  The guided filter uses the *full-resolution*
L channel as a structural guide: colour only spreads to neighbours with similar
lightness values, so a blue sky stays above a dark tree line instead of bleeding into it.

**Vibrance vs Saturation:**
Plain saturation multiplication makes vivid pixels over-saturated and blown-out.
Vibrance applies a *stronger boost to grey/dull pixels* and barely touches
already-colourful ones — the same algorithm used in Adobe Lightroom's Vibrance slider.

**Tiled Inference:**
For a 2000×1500 photo squished to 224×224, the model must encode ALL scene
semantics in those 224 pixels — it loses texture, local contrast, and fine detail.
Tiling feeds 400×400 crops with 64px overlap so the model always sees a
scale-appropriate view.  Blend weights (linear ramp at borders → 1 at centre)
stitch tiles invisibly.
        """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#4a5578;font-size:.75rem;">'
    'AI Image Colorization v2 — BE Final Year Project | '
    'Zhang et al. (2016) + Guided Filter + Tiled Inference'
    '</p>',
    unsafe_allow_html=True,
)
