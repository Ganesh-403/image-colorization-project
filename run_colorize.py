"""
run_colorize.py — Command-Line Colorization Script
===================================================
Colorize a single image from the command line without Streamlit.
Useful for testing and quick batch processing.

Usage:
    python run_colorize.py --input images/sample_bw.jpg
    python run_colorize.py --input images/sample_bw.jpg --output results/out.jpg --show
"""

import os
import sys
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import cv2
from colorize import load_model, colorize_image
from utils    import load_image, save_image, show_comparison, validate_image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Colorize a black & white image using deep learning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to the input grayscale image."
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Path to save the colorized output. If not given, saves to results/ folder."
    )
    parser.add_argument(
        "--show", "-s", action="store_true",
        help="Display the side-by-side comparison using Matplotlib."
    )
    parser.add_argument(
        "--save-comparison", action="store_true",
        help="Save the side-by-side comparison figure as an image."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 55)
    print("  AI Image Colorization — CLI Runner")
    print("=" * 55)

    # ── Load image ────────────────────────────────────────────────────────────
    print(f"\n[1/4] Loading image: {args.input}")
    image_bgr = load_image(args.input)
    validate_image(image_bgr)
    print(f"      Shape: {image_bgr.shape}")

    # ── Load model ────────────────────────────────────────────────────────────
    print("\n[2/4] Loading pretrained colorization model...")
    try:
        net = load_model()
        print("      Model loaded successfully.")
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        sys.exit(1)

    # ── Colorize ──────────────────────────────────────────────────────────────
    print("\n[3/4] Running colorization...")
    start = time.time()
    colorized_bgr = colorize_image(image_bgr, net)
    elapsed = time.time() - start
    print(f"      Done in {elapsed:.2f}s.")

    # ── Save output ───────────────────────────────────────────────────────────
    if args.output:
        output_path = args.output
    else:
        base = os.path.splitext(os.path.basename(args.input))[0]
        output_path = os.path.join("results", f"{base}_colorized.jpg")

    print(f"\n[4/4] Saving output...")
    save_image(colorized_bgr, output_path)

    # Optional: save comparison figure
    if args.save_comparison:
        comp_path = output_path.replace("_colorized", "_comparison")
        show_comparison(image_bgr, colorized_bgr, save_path=comp_path)

    # ── Display ───────────────────────────────────────────────────────────────
    if args.show:
        show_comparison(image_bgr, colorized_bgr,
                        title=f"Colorization Result — {os.path.basename(args.input)}")

    print("\n✅ All done!")


if __name__ == "__main__":
    main()
