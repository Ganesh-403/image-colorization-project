"""
download_models.py — Automatic Model File Downloader
=====================================================
Run this script ONCE before running the application.
It downloads the three required model files into the /models directory.

Usage:
    python download_models.py
    python download_models.py --force    # re-download even if files exist

Files downloaded:
    - colorization_deploy_v2.prototxt   (~10 KB)  — CNN architecture definition
    - colorization_release_v2.caffemodel (~130 MB) — pretrained model weights
    - pts_in_hull.npy                   (~3 KB)   — 313 quantized color cluster centers
"""

import sys
import argparse

# Ensure /src is importable
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from utils import download_models, check_models


def main():
    parser = argparse.ArgumentParser(description="Download colorization model files.")
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-download even if files already exist."
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  AI Image Colorization — Model Downloader")
    print("=" * 60)

    # Check current status
    status = check_models()
    print("\n📦 Current model file status:")
    for fname, present in status.items():
        icon = "✓" if present else "✗"
        print(f"  [{icon}] {fname}")

    if all(status.values()) and not args.force:
        print("\n✅ All model files are already present. Nothing to download.")
        print("   Use --force to re-download anyway.")
        return

    print("\n⬇️  Starting download...\n")
    success = download_models(force=args.force)

    if success:
        print("\n" + "=" * 60)
        print("✅ All model files downloaded successfully!")
        print("   You can now run: streamlit run app/app.py")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ One or more downloads failed.")
        print("   Check your internet connection and try again.")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
